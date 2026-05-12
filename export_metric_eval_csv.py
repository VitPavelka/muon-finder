from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from candidate_labels import load_binary_labels
from wdf_io import load_dataset


@dataclass(frozen=True)
class MetricRule:
	name: str
	threshold_low: float
	threshold_high: float
	reverse: bool


def _read_labels(path: Path) -> Dict[Tuple[int, int, int], bool]:
	return {key: bool(value) for key, value in load_binary_labels(path).items()}


def _parse_metric_list(text: str) -> List[str]:
	items = [item.strip() for item in str(text).split(",")]
	return [item for item in items if item]


def _parse_override_specs(values: Optional[Sequence[str]]) -> Dict[str, Tuple[float, float]]:
	out: Dict[str, Tuple[float, float]] = {}
	for raw in values or ():
		item = str(raw).strip()
		if not item:
			continue
		if "=" not in item:
			raise ValueError(f"Invalid --threshold item: {item!r}. Expected feature=low:high")
		name, rhs = item.split("=", 1)
		parts = [p.strip() for p in rhs.split(":")]
		if len(parts) != 2:
			raise ValueError(f"Invalid --threshold item: {item!r}. Expected feature=low:high")
		out[name.strip()] = (float(parts[0]), float(parts[1]))
	return out


def _parse_reverse_specs(values: Optional[Sequence[str]]) -> Dict[str, bool]:
	out: Dict[str, bool] = {}
	for raw in values or ():
		item = str(raw).strip()
		if not item:
			continue
		if "=" in item:
			name, rhs = item.split("=", 1)
			flag = rhs.strip().lower()
			if flag in {"1", "true", "yes", "y", "on"}:
				out[name.strip()] = True
			elif flag in {"0", "false", "no", "n", "off"}:
				out[name.strip()] = False
			else:
				raise ValueError(f"Invalid --reverse item: {item!r}. Use feature=true/false")
		else:
			out[item] = True
	return out


def _metric_row_from_corr(corr_rows: Sequence[dict], metric: str) -> Optional[dict]:
	return next((row for row in corr_rows if str(row.get("feature")) == metric), None)


def _load_corr_rows(path: Optional[Path]) -> List[dict]:
	if path is None:
		return []
	data = json.loads(path.read_text(encoding="utf-8"))
	rows = data.get("correlations", [])
	if not isinstance(rows, list):
		raise ValueError(f"{path}: correlations is not a list")
	return rows


def _infer_reverse_from_corr_row(row: dict) -> Optional[bool]:
	block = row.get("empirical_purity_intervals", {})
	if not isinstance(block, dict):
		return None
	intervals = block.get("intervals", [])
	if not isinstance(intervals, list):
		return None
	muon_pure = [iv for iv in intervals if isinstance(iv, dict) and str(iv.get("kind")) == "muon_pure"]
	nonmuon_pure = [iv for iv in intervals if isinstance(iv, dict) and str(iv.get("kind")) == "nonmuon_pure"]
	if not muon_pure or not nonmuon_pure:
		return None
	best_mu = max(muon_pure, key=lambda iv: (float(iv.get("n_muon", 0)), float(iv.get("muon_fraction", 0.0))))
	best_non = max(nonmuon_pure, key=lambda iv: (float(iv.get("n_total", 0)), float(iv.get("nonmuon_fraction", 0.0))))
	mu_center = 0.5 * (float(best_mu.get("left", np.nan)) + float(best_mu.get("right", np.nan)))
	non_center = 0.5 * (float(best_non.get("left", np.nan)) + float(best_non.get("right", np.nan)))
	if not (np.isfinite(mu_center) and np.isfinite(non_center)):
		return None
	return bool(mu_center < non_center)


def _thresholds_from_corr_row(row: dict) -> Optional[Tuple[float, float]]:
	block = row.get("empirical_purity_intervals", {})
	if not isinstance(block, dict):
		return None
	intervals = block.get("intervals", [])
	if not isinstance(intervals, list):
		return None
	mixed = [iv for iv in intervals if isinstance(iv, dict) and str(iv.get("kind")) == "mixed"]
	if not mixed:
		return None
	best = max(
		mixed,
		key=lambda iv: (
			float(iv.get("n_total", 0)),
			float(iv.get("right", 0.0)) - float(iv.get("left", 0.0)),
		),
	)
	left = float(best.get("left", np.nan))
	right = float(best.get("right", np.nan))
	if not (np.isfinite(left) and np.isfinite(right)):
		return None
	return (left, right)


def _resolve_metric_rules(
		metrics: Sequence[str],
		corr_rows: Sequence[dict],
		threshold_overrides: Dict[str, Tuple[float, float]],
		reverse_overrides: Dict[str, bool],
) -> List[MetricRule]:
	rules: List[MetricRule] = []
	for metric in metrics:
		row = _metric_row_from_corr(corr_rows, metric)
		if metric in threshold_overrides:
			low, high = threshold_overrides[metric]
		else:
			if row is None:
				raise ValueError(
					f"No thresholds for feature {metric!r}. Provide --threshold {metric}=low:high or --corr-json."
				)
			thresholds = _thresholds_from_corr_row(row)
			if thresholds is None:
				raise ValueError(
					f"Could not infer thresholds for feature {metric!r} from corr.json. "
					f"Provide --threshold {metric}=low:high."
				)
			low, high = thresholds
		if low > high:
			low, high = high, low

		if metric in reverse_overrides:
			reverse = bool(reverse_overrides[metric])
		else:
			reverse = bool(_infer_reverse_from_corr_row(row)) if row is not None else False

		rules.append(MetricRule(name=metric, threshold_low=float(low), threshold_high=float(high), reverse=reverse))
	return rules


def _category_for_value(value: float, rule: MetricRule) -> int:
	if not np.isfinite(value):
		return -1
	if rule.reverse:
		if value > rule.threshold_high:
			return 0
		if value > rule.threshold_low:
			return 1
		return 2
	if value < rule.threshold_low:
		return 0
	if value < rule.threshold_high:
		return 1
	return 2


def _iter_spike_rows(per_spectrum: Sequence[dict]) -> Iterable[Tuple[int, int, int, dict]]:
	for spectrum_index, spec in enumerate(per_spectrum):
		y = int(spec.get("y", -1))
		x = int(spec.get("x", -1))
		spikes = spec.get("spikes", [])
		if not isinstance(spikes, list):
			continue
		for spike in spikes:
			yield spectrum_index, y, x, spike


def _candidate_sort_key(row: dict) -> Tuple[float, int]:
	pos = float(row["candidate_position_cm1"])
	peak_index = int(row["peak_index"])
	return (pos, peak_index)


def build_rows(
		debug_report: dict,
		x_axis: np.ndarray,
		labels: Dict[Tuple[int, int, int], bool],
		rules: Sequence[MetricRule],
		labeled_only: bool,
) -> List[dict]:
	per_spectrum = debug_report.get("per_spectrum", [])
	if not isinstance(per_spectrum, list):
		raise ValueError("debug report does not contain per_spectrum list")

	by_spectrum: Dict[int, List[dict]] = {}
	for spectrum_index, y, x, spike in _iter_spike_rows(per_spectrum):
		peak_index = int(spike.get("peak_index", -1))
		if peak_index < 0 or peak_index >= x_axis.size:
			continue
		label_key = (y, x, peak_index)
		is_muon = labels.get(label_key, None)
		if labeled_only and is_muon is None:
			continue

		row = {
			"spectrum_index": int(spectrum_index),
			"peak_index": int(peak_index),
			"candidate_position_cm1": float(x_axis[peak_index]),
			"is_muon": is_muon,
		}
		category_score = 0
		score_score = 0.0
		for rule in rules:
			value = float(spike.get(rule.name, np.nan))
			category = _category_for_value(value, rule)
			row[f"{rule.name}_category"] = category
			row[f"{rule.name}_score"] = value
			if category >= 0:
				category_score += category
			if np.isfinite(value):
				score_score += value
		row["category_score"] = int(category_score)
		row["score_score"] = float(score_score)
		by_spectrum.setdefault(int(spectrum_index), []).append(row)

	out_rows: List[dict] = []
	for spectrum_index in sorted(by_spectrum):
		candidates = sorted(by_spectrum[spectrum_index], key=_candidate_sort_key)
		for candidate_index, row in enumerate(candidates):
			row["candidate_index"] = int(candidate_index)
			out_rows.append(row)
	return out_rows


def _write_csv(path: Path, rows: Sequence[dict], rules: Sequence[MetricRule], delimiter: str) -> None:
	fieldnames = [
		"spectrum_index",
		"candidate_index",
		"candidate_position_cm1",
	]
	for rule in rules:
		fieldnames.append(f"{rule.name}_category")
	for rule in rules:
		fieldnames.append(f"{rule.name}_score")
	fieldnames.extend([
		"category_score",
		"score_score",
		"is_muon",
	])

	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
		writer.writeheader()
		for row in rows:
			out = {name: row.get(name, "") for name in fieldnames}
			if out["is_muon"] is None:
				out["is_muon"] = ""
			elif isinstance(out["is_muon"], bool):
				out["is_muon"] = "True" if out["is_muon"] else "False"
			writer.writerow(out)


def _load_x_axis(path: Path) -> np.ndarray:
	try:
		ds = load_dataset(path)
		return np.asarray(ds.x_axis, dtype=float)
	except Exception:
		pass

	if path.suffix.lower() == ".npz":
		with np.load(path, allow_pickle=True) as data:
			if "x_axis" in data.files:
				return np.asarray(data["x_axis"], dtype=float)

	raise ValueError(f"Could not load x_axis from {path}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Export per-candidate metric categories and scores to CSV.")
	parser.add_argument("--input", type=Path, required=True, help="Input dataset (.wdf or .npz) for x-axis lookup.")
	parser.add_argument("--report", type=Path, required=True, help="debug.json/debug report with per_spectrum spikes.")
	parser.add_argument("--labels-csv", type=Path, required=True, help="labels.csv with y,x,peak_index,is_muon.")
	parser.add_argument("--metrics", type=str, required=True, help="Comma-separated feature names.")
	parser.add_argument("--corr-json", type=Path, default=None, help="Optional corr.json for thresholds and reverse inference.")
	parser.add_argument(
		"--threshold",
		dest="thresholds",
		action="append",
		default=None,
		help="Manual threshold override: feature=low:high. May be repeated.",
	)
	parser.add_argument(
		"--reverse",
		dest="reverses",
		action="append",
		default=None,
		help="Manual reverse override: feature or feature=true/false. May be repeated.",
	)
	parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path.")
	parser.add_argument("--delimiter", type=str, default=",", help="CSV delimiter, default ','.")
	parser.add_argument(
		"--labeled-only",
		action="store_true",
		help="Export only candidates present in labels.csv.",
	)
	args = parser.parse_args()

	if len(str(args.delimiter)) != 1:
		raise ValueError("--delimiter must be a single character")

	metrics = _parse_metric_list(args.metrics)
	if not metrics:
		raise ValueError("No metrics were provided.")

	corr_rows = _load_corr_rows(args.corr_json)
	threshold_overrides = _parse_override_specs(args.thresholds)
	reverse_overrides = _parse_reverse_specs(args.reverses)
	rules = _resolve_metric_rules(metrics, corr_rows, threshold_overrides, reverse_overrides)

	report = json.loads(args.report.read_text(encoding="utf-8"))
	x_axis = _load_x_axis(args.input)
	labels = _read_labels(args.labels_csv)
	rows = build_rows(
		debug_report=report,
		x_axis=x_axis,
		labels=labels,
		rules=rules,
		labeled_only=bool(args.labeled_only),
	)
	_write_csv(args.out_csv, rows, rules, delimiter=str(args.delimiter))

	print(f"Exported {len(rows)} rows to {args.out_csv}")
	for rule in rules:
		print(
			f"{rule.name}: low={rule.threshold_low:.6g} high={rule.threshold_high:.6g} reverse={rule.reverse}"
		)


if __name__ == "__main__":
	main()
