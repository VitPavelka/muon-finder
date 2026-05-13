from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from candidate_labels import LABEL_CLASS_COLORS, load_binary_labels, load_label_classes
from feature_discrimination import GWS_SOURCE_PREFIX_BY_MODE
from wdf_io import load_dataset
from muon_decision import annotate_feature_dict_with_muon_rule_v3
from muon_decision import annotate_feature_dict_with_spike_score_v5
from muon_decision import get_muon_rule_v3_metric_thresholds


GWS_SOURCE_METRIC_PREFIXES = tuple(
	sorted(
		{
			str(prefix)
			for prefix in GWS_SOURCE_PREFIX_BY_MODE.values()
			if str(prefix).strip()
		}
	)
)
EXPERIMENT_METRIC_PREFIXES = (
	"recdw",
	"rucdw",
	"decdw",
	"raw_edge",
	"mg_edge",
	"th_w7",
	"th_w9",
	"th_w11",
	"raw_ball",
	"mg_ball",
	"raw_exp",
	"mg_exp",
)
EDGE_SIMPLE_METRIC_PREFIXES = (
	"width_",
	"width_ratio_",
	"raw_edge_width_",
	"raw_edge_ctx_width_",
	"raw_edge_base_expansion_rate",
	"raw_edge_ctx_base_expansion_rate",
	"raw_edge_intensity_per_width_loss_",
	"raw_edge_broadening_per_intensity_",
	"raw_edge_profile_",
	"mg_edge_width_",
	"mg_edge_base_expansion_rate",
	"mg_edge_intensity_per_width_loss_",
	"mg_edge_broadening_per_intensity_",
	"mg_edge_profile_",
	"mg_edge_root_width_",
)
EDGE_ROOT_METRIC_PREFIXES = (
	"raw_edge_width_at_0",
	"raw_edge_width_at_5",
	"raw_edge_width_at_10",
	"raw_edge_width_at_15",
	"raw_edge_width_at_20",
	"raw_edge_width_at_25",
	"raw_edge_root_width_",
	"raw_edge_ctx_width_at_0",
	"raw_edge_ctx_width_at_5",
	"raw_edge_ctx_width_at_10",
	"raw_edge_ctx_width_at_15",
	"raw_edge_ctx_width_at_20",
	"raw_edge_ctx_width_at_25",
	"raw_edge_ctx_root_width_",
	"mg_edge_width_at_5",
	"mg_edge_width_at_10",
	"mg_edge_width_at_15",
	"mg_edge_width_at_20",
	"mg_edge_width_at_25",
	"mg_edge_root_width_",
)
EDGE_ROOT_CTX_METRIC_PREFIXES = (
	"recdw_at_5",
	"recdw_at_10",
	"recdw_at_15",
	"recdw_at_20",
	"recdw_at_25",
	"decdw_at_5",
	"decdw_at_10",
	"decdw_at_15",
	"decdw_at_20",
	"decdw_at_25",
	"raw_edge_ctx_width_at_0",
	"raw_edge_ctx_width_at_5",
	"raw_edge_ctx_width_at_10",
	"raw_edge_ctx_width_at_15",
	"raw_edge_ctx_width_at_20",
	"raw_edge_ctx_width_at_25",
	"raw_edge_ctx_root_width_",
)
EDGE_CTX_METRIC_PREFIXES = ("recdw_", "rucdw_", "decdw_", "raw_edge_ctx_", "raw_edge_dense_ctx_")
EDGE_DENSE_METRIC_PREFIXES = ("recdw_", "rucdw_", "decdw_", "raw_edge_dense_", "raw_edge_dense_ctx_", "raw_edge_ctx_dense_", "mg_edge_dense_")
EDGE_TH_METRIC_PREFIXES = (
	"th_width_",
	"th_base_expansion_rate",
	"th_points_to_",
	"th_w7_width_",
	"th_w7_base_expansion_rate",
	"th_w7_points_to_",
	"th_w9_width_",
	"th_w9_base_expansion_rate",
	"th_w9_points_to_",
	"th_w11_width_",
	"th_w11_base_expansion_rate",
	"th_w11_points_to_",
)
EDGE_METRIC_KEYWORD_PREFIXES = {
	"edge": EDGE_SIMPLE_METRIC_PREFIXES + EDGE_DENSE_METRIC_PREFIXES + EDGE_TH_METRIC_PREFIXES,
	"edge-simple": EDGE_SIMPLE_METRIC_PREFIXES,
	"edge_level": EDGE_SIMPLE_METRIC_PREFIXES,
	"edge-level": EDGE_SIMPLE_METRIC_PREFIXES,
	"edge-root": EDGE_ROOT_METRIC_PREFIXES,
	"edge_root": EDGE_ROOT_METRIC_PREFIXES,
	"edge-root-ctx": EDGE_ROOT_CTX_METRIC_PREFIXES,
	"edge_root_ctx": EDGE_ROOT_CTX_METRIC_PREFIXES,
	"raw-edge-root": EDGE_ROOT_METRIC_PREFIXES,
	"raw_edge_root": EDGE_ROOT_METRIC_PREFIXES,
	"raw-edge-root-ctx": EDGE_ROOT_CTX_METRIC_PREFIXES,
	"raw_edge_root_ctx": EDGE_ROOT_CTX_METRIC_PREFIXES,
	"mg-edge-root": EDGE_ROOT_METRIC_PREFIXES,
	"mg_edge_root": EDGE_ROOT_METRIC_PREFIXES,
	"raman-veto-edge": EDGE_ROOT_METRIC_PREFIXES,
	"raman_veto_edge": EDGE_ROOT_METRIC_PREFIXES,
	"edge_ctx": EDGE_CTX_METRIC_PREFIXES,
	"edge-ctx": EDGE_CTX_METRIC_PREFIXES,
	"raw_edge_ctx": EDGE_CTX_METRIC_PREFIXES,
	"raw-edge-ctx": EDGE_CTX_METRIC_PREFIXES,
	"edge_dense": EDGE_DENSE_METRIC_PREFIXES,
	"edge-dense": EDGE_DENSE_METRIC_PREFIXES,
	"edge_th": EDGE_TH_METRIC_PREFIXES,
	"edge-th": EDGE_TH_METRIC_PREFIXES,
	"th_edge": EDGE_TH_METRIC_PREFIXES,
	"th-edge": EDGE_TH_METRIC_PREFIXES,
	"edge-th-w7": ("th_w7_width_", "th_w7_base_expansion_rate", "th_w7_points_to_"),
	"edge-th-w9": ("th_w9_width_", "th_w9_base_expansion_rate", "th_w9_points_to_"),
	"edge-th-w11": ("th_w11_width_", "th_w11_base_expansion_rate", "th_w11_points_to_"),
}


def _load_config(path: Optional[Path]) -> Tuple[Dict[str, object], Optional[Path]]:
	if path is None:
		return {}, None
	cfg_path = Path(path)
	cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
	if not isinstance(cfg, dict):
		raise ValueError(f"Config must be a JSON object: {cfg_path}")
	return cfg, cfg_path.parent


def _config_path_value(cfg: Dict[str, object], key: str, base_dir: Optional[Path]) -> Optional[Path]:
	value = cfg.get(key)
	if value is None or str(value).strip() == "":
		return None
	path = Path(str(value))
	if not path.is_absolute() and base_dir is not None:
		path = base_dir / path
	return path


def _kmeans_thresholds_1d(x: np.ndarray, k: int = 3, iters: int = 60) -> Tuple[float, float]:
	vals = np.asarray(x[np.isfinite(x)], dtype=float)
	if vals.size < k:
		q = np.quantile(vals, [1.0 / 3.0, 2.0 / 3.0]) if vals.size > 0 else np.array([0.0, 0.0])
		return float(q[0]), float(q[1])

	centroids = np.quantile(vals, np.linspace(0.1, 0.9, k)).astype(float)
	for _ in range(max(1, int(iters))):
		dist = np.abs(vals[:, None] - centroids[None, :])
		lab = np.argmin(dist, axis=1)
		new_c = centroids.copy()
		for i in range(k):
			m = vals[lab == i]
			if m.size > 0:
				new_c[i] = float(np.mean(m))
		if np.allclose(new_c, centroids):
			break
		centroids = new_c

	centroids = np.sort(centroids)
	return float((centroids[0] + centroids[1]) * 0.5), float((centroids[1] + centroids[2]) * 0.5)


def _class_name(v: float, t_low: float, t_high: float, reverse: bool = False) -> str:
	if not np.isfinite(v):
		return "invalid"
	if bool(reverse):
		if v > t_high:
			return "no-muon"
		if v > t_low:
			return "maybe-muon"
		return "ok-muon"
	if v < t_low:
		return "no-muon"
	if v < t_high:
		return "maybe-muon"
	return "ok-muon"


def _class_color(v: float, t_low: float, t_high: float, reverse: bool = False) -> str:
	c = _class_name(v, t_low, t_high, reverse=reverse)
	if c == "no-muon":
		return "#1f77b4"
	if c == "maybe-muon":
		return "#ff7f0e"
	if c == "ok-muon":
		return "#d62728"
	return "#7f7f7f"

def _fmt_opt(v: object) -> str:
	if v is None:
		return "nan"
	try:
		vf = float(v)
	except Exception:
		return str(v)
	if not np.isfinite(vf):
		return "nan"
	return f"{vf:.4f}"


def _fmt_signal_label(v: object) -> str:
	if v is None:
		return "nan"
	try:
		vf = float(v)
	except Exception:
		return str(v)
	if not np.isfinite(vf):
		return "nan"
	if vf == 1.0 or vf == -1.0:
		return str(int(vf))
	return f"{vf:.4f}"


def _ensure_metric_available(spike: Optional[Dict[str, object]], metric: str) -> Dict[str, object]:
	if not isinstance(spike, dict):
		return {}
	row = dict(spike)
	if str(metric) in row:
		return row
	if str(metric).startswith("muon_rule_v3_"):
		row.update(annotate_feature_dict_with_muon_rule_v3(row))
	if str(metric).startswith("ss5") or str(metric) in {"spike_score_v5"}:
		row.update(annotate_feature_dict_with_spike_score_v5(row))
	return row


def _uniform_bin_edges(vals: np.ndarray, target_bins: int = 72) -> np.ndarray:
	x = np.asarray(vals[np.isfinite(vals)], dtype=float)
	if x.size == 0:
		return np.linspace(0.0, 1.0, max(2, int(target_bins) + 1))
	vmin = float(np.min(x))
	vmax = float(np.max(x))
	if not np.isfinite(vmin) or not np.isfinite(vmax):
		return np.linspace(0.0, 1.0, max(2, int(target_bins) + 1))
	if vmin == vmax:
		span = max(abs(vmin) * 0.05, 1.0)
		vmin -= 0.5 * span
		vmax += 0.5 * span
	return np.linspace(vmin, vmax, max(2, int(target_bins) + 1))


def _load_corr_hint(path: Optional[Path], metric: str) -> str:
	if path is None or not path.exists():
		return ""
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return ""
	arr = data.get('correlations', [])
	if not isinstance(arr, list):
		return ""
	for row in arr:
		if str(row.get('feature')) == metric:
			p = _fmt_opt(row.get('pearson'))
			s = _fmt_opt(row.get('spearman'))
			a = _fmt_opt(row.get('auc_feature_oriented', row.get('auc_feature_raw')))

			return f"corr.json: pearson{p} spearman={s} auc={a}"
	return ""


def _load_corr_metric_row(path: Optional[Path], metric: str) -> Optional[Dict[str, object]]:
	if path is None or not path.exists():
		return None
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return None
	arr = data.get("correlations", [])
	if not isinstance(arr, list):
		return None
	for row in arr:
		if isinstance(row, dict) and str(row.get("feature", "")) == str(metric):
			return row
	return None


def _resolve_metric_keyword(metric: str, corr_json_path: Optional[Path]) -> str:
	"""Resolve prefix-style metric keywords via corr.json feature ranking."""
	metric_name = str(metric).strip()
	prefix = ""
	prefixes: Tuple[str, ...] = ()
	if metric_name.endswith("_*"):
		prefix = metric_name[:-1]
	elif metric_name in EDGE_METRIC_KEYWORD_PREFIXES:
		prefixes = tuple(EDGE_METRIC_KEYWORD_PREFIXES[metric_name])
	elif metric_name in GWS_SOURCE_METRIC_PREFIXES:
		prefix = metric_name + "_"
	elif metric_name in EXPERIMENT_METRIC_PREFIXES:
		prefix = metric_name + "_"
	if (
		not prefixes
		and
		prefix not in GWS_SOURCE_METRIC_PREFIXES
		and prefix.rstrip("_") not in GWS_SOURCE_METRIC_PREFIXES
		and prefix.rstrip("_") not in EXPERIMENT_METRIC_PREFIXES
	):
		return metric_name
	if not prefixes and prefix.rstrip("_") in GWS_SOURCE_METRIC_PREFIXES + EXPERIMENT_METRIC_PREFIXES and not prefix.endswith("_"):
		prefix = prefix + "_"
	if corr_json_path is None or not corr_json_path.exists():
		raise ValueError(f"Metric keyword '{metric_name}' requires --corr-json to resolve a concrete feature.")
	try:
		data = json.loads(corr_json_path.read_text(encoding="utf-8"))
	except Exception as exc:
		raise ValueError(f"Failed to read corr.json for metric keyword '{metric_name}': {exc}") from exc
	rows = data.get("correlations", [])
	if not isinstance(rows, list):
		raise ValueError(f"corr.json has no correlations list needed for metric keyword '{metric_name}'.")
	matches: List[Dict[str, object]] = []
	for row in rows:
		if not isinstance(row, dict):
			continue
		feature = str(row.get("feature", "")).strip()
		if (prefixes and feature.startswith(prefixes)) or ((not prefixes) and feature.startswith(prefix)):
			matches.append(row)
	if not matches:
		raise ValueError(f"No corr.json features matched metric keyword '{metric_name}'.")
	matches.sort(
		key=lambda row: float(row.get("auc_feature_oriented", row.get("auc_feature_raw", float("-inf")))),
		reverse=True,
	)
	best = str(matches[0].get("feature", "")).strip()
	if not best:
		raise ValueError(f"Metric keyword '{metric_name}' resolved to an empty feature name.")
	print(f"[metric keyword] {metric_name} -> {best}")
	print(f"[metric keyword matches] {metric_name}: {', '.join(str(row.get('feature', '')).strip() for row in matches[:24])}")
	return best


def _load_corr_purity_defaults(path: Optional[Path], metric: str) -> Tuple[Optional[float], Optional[float], Optional[bool], str]:
	"""Load threshold defaults from corr.json empirical purity intervals."""
	rule_thresholds = get_muon_rule_v3_metric_thresholds(metric)
	if path is None or not path.exists():
		if rule_thresholds is not None:
			low, high, reverse = rule_thresholds
			return float(low), float(high), bool(reverse), "rule defaults"
		return None, None, None, ""
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return None, None, None, ""
	rows = data.get("correlations", [])
	if not isinstance(rows, list):
		return None, None, None, ""
	for row in rows:
		if str(row.get("feature")) != metric:
			continue
		block = row.get("empirical_purity_intervals", {})
		if not isinstance(block, dict):
			return None, None, None, ""
		intervals = block.get("intervals", [])
		if not isinstance(intervals, list):
			return None, None, None, ""
		mixed = [iv for iv in intervals if isinstance(iv, dict) and str(iv.get("kind")) == "mixed"]
		muon_pure = [iv for iv in intervals if isinstance(iv, dict) and str(iv.get("kind")) == "muon_pure"]
		nonmuon_pure = [iv for iv in intervals if isinstance(iv, dict) and str(iv.get("kind")) == "nonmuon_pure"]
		t_low = None
		t_high = None
		if mixed:
			best_mixed = max(mixed, key=lambda iv: (float(iv.get("n_total", 0)), float(iv.get("right", 0.0)) - float(iv.get("left", 0.0))))
			t_low = float(best_mixed.get("left", np.nan))
			t_high = float(best_mixed.get("right", np.nan))
		reverse = None
		if muon_pure and nonmuon_pure:
			best_mu = max(muon_pure, key=lambda iv: (float(iv.get("n_muon", 0)), float(iv.get("muon_fraction", 0.0))))
			best_non = max(nonmuon_pure, key=lambda iv: (float(iv.get("n_total", 0)), float(iv.get("nonmuon_fraction", 0.0))))
			mu_center = 0.5 * (float(best_mu.get("left", np.nan)) + float(best_mu.get("right", np.nan)))
			non_center = 0.5 * (float(best_non.get("left", np.nan)) + float(best_non.get("right", np.nan)))
			if np.isfinite(mu_center) and np.isfinite(non_center):
				reverse = bool(mu_center < non_center)
		hint = ""
		if t_low is not None and t_high is not None:
			hint = f"purity defaults: thr-low={t_low:.4g} thr-high={t_high:.4g}"
			if reverse is not None:
				hint += f" reverse={reverse}"
		if (t_low is None or t_high is None) and rule_thresholds is not None:
			t_low = float(rule_thresholds[0])
			t_high = float(rule_thresholds[1])
			reverse = bool(rule_thresholds[2])
			hint = "rule defaults"
		return t_low, t_high, reverse, hint
	if rule_thresholds is not None:
		low, high, reverse = rule_thresholds
		return float(low), float(high), bool(reverse), "rule defaults"
	return None, None, None, ""


def main() -> None:
	parser = argparse.ArgumentParser(description="Interactive metric-threshold viewer for muon candidate peaks")
	parser.add_argument("--config", type=Path, default=None, help="Optional shared config.json; supplies input_path, debug_report_path, labels_csv, and optionally corr_json_path.")
	parser.add_argument("--input", type=Path, default=None, help="Input dataset path (.wdf/.npz); overrides config input_path.")
	parser.add_argument("--report", type=Path, default=None, help="Path to debug_report/debug.json; overrides config debug_report_path.")
	parser.add_argument("--labels-csv", type=Path, default=None, help="labels.csv with y,x,peak_index,is_muon; overrides config labels_csv.")
	parser.add_argument("--metric", type=str, required=True, help="Metric/feature in spike dict (e.g. spike_score_v1)")
	parser.add_argument("--corr-json", type=Path, default=None, help="Optional corr.json from debug_stats.py for metric hint")
	parser.add_argument("--threshold-low", type=float, default=None, help="Lower threshold for no/maybe split")
	parser.add_argument("--threshold-high", type=float, default=None, help="Upper threshold for maybe/ok split")
	parser.add_argument("--thr-l", type=float, default=None, help="Alias for --threshold-low")
	parser.add_argument("--thr-h", type=float, default=None, help="Alias for --threshold-high")
	parser.add_argument("--reverse", action="store_true", help="Invert histogram x-axis direction.")
	parser.add_argument("--label-mode", choices=["binary", "ternary"], default=None, help="Label display mode; default comes from config label_mode or binary.")
	parser.add_argument("--show-classes", type=str, default="muon,raman,noise", help="Ternary mode: comma-separated label classes to show.")
	parser.add_argument("--include-unknown", action="store_true", help="Ternary mode: also show unknown labels in gray.")
	parser.add_argument("--unlabeled-as-noise", action="store_true", help="Ternary mode: treat unknown labels as noise.")
	args = parser.parse_args()
	cfg, cfg_base = _load_config(args.config)
	if args.label_mode is None:
		args.label_mode = str(cfg.get("label_mode", "binary")).strip().lower()
	if args.input is None:
		args.input = _config_path_value(cfg, "input_path", cfg_base)
	if args.report is None:
		args.report = _config_path_value(cfg, "debug_report_path", cfg_base)
	if args.labels_csv is None:
		args.labels_csv = _config_path_value(cfg, "labels_csv", cfg_base)
	if args.corr_json is None:
		args.corr_json = (
			_config_path_value(cfg, "corr_json_path", cfg_base)
			or _config_path_value(cfg, "corr_json", cfg_base)
		)
	missing_paths = []
	if args.input is None:
		missing_paths.append("--input or config input_path")
	if args.report is None:
		missing_paths.append("--report or config debug_report_path")
	if args.labels_csv is None:
		missing_paths.append("--labels-csv or config labels_csv")
	if missing_paths:
		raise ValueError("Missing required path(s): " + ", ".join(missing_paths))
	args.metric = _resolve_metric_keyword(args.metric, args.corr_json)

	# Avoid conflict with Matplotlib toolbar history buildings:
	# default keymap uses left/right back/forward, which clashes with spectrum navigation.
	for _km, _drop in (("keymap.back", "left"), ("keymap.forward", "right")):
		try:
			mpl.rcParams[_km] = [k for k in mpl.rcParams.get(_km, []) if str(k).lower() != _drop]
		except Exception:
			pass

	report = json.loads(Path(args.report).read_text(encoding="utf-8"))
	per = list(report.get("per_spectrum", []))
	if not per:
		raise ValueError("No per_spectrum in debug_report.json")

	label_mode = str(args.label_mode).strip().lower()
	if label_mode == "ternary":
		label_classes = load_label_classes(
			Path(args.labels_csv),
			include_unknown=bool(args.include_unknown),
			unlabeled_as_noise=bool(args.unlabeled_as_noise),
		)
		labels = {key: (1 if cls == "muon" else 0) for key, cls in label_classes.items()}
		show_classes = [c.strip().lower() for c in str(args.show_classes).split(",") if c.strip()]
		if bool(args.include_unknown) and "unknown" not in show_classes:
			show_classes.append("unknown")
	else:
		label_classes = {}
		labels = load_binary_labels(Path(args.labels_csv))
		show_classes = []
	if not labels:
		raise ValueError("labels.csv has no rows")
	ds = load_dataset(Path(args.input))
	x_axis = ds.x_axis
	raw = ds.spectra

	label_coords = sorted({(int(y), int(x)) for (y, x, _p) in labels.keys()})
	per_by_coord: Dict[Tuple[int, int], Dict] = {
		(int(spec.get('y', -1)), int(spec.get('x', -1))): spec
		for spec in per
	}

	rows: List[Dict] = []
	mu_vals: List[float] = []
	non_vals: List[float] = []
	for (y, x) in label_coords:
		spec = per_by_coord.get((y, x), {'y': y, 'x': x, 'spikes': []})
		spikes = list(spec.get('spikes', []))
		selected = []
		for sp in spikes:
			sp = _ensure_metric_available(sp, args.metric)
			if not sp:
				continue
			p = int(sp.get('peak_index', -1))
			k = (y, x, p)
			if k not in labels:
				continue
			mv = float(sp.get(args.metric, np.nan))
			lbl = int(labels[k])
			row = dict(sp)
			row['metric_value'] = mv
			row['is_muon'] = lbl
			row['label_class'] = str(label_classes.get(k, "muon" if lbl == 1 else "non-muon"))
			selected.append(row)
			if np.isfinite(mv):
				(mu_vals if lbl == 1 else non_vals).append(mv)
		selected.sort(key=lambda sp: int(sp.get('peak_index', -1)))
		rows.append({'y': y, 'x': x, 'spikes': selected})

	vals = np.asarray(mu_vals + non_vals, dtype=float)
	if vals.size == 0:
		raise ValueError(f"Metric '{args.metric}' has no finite values in selected rows")
	class_vals: Dict[str, List[float]] = {cls: [] for cls in ("muon", "raman", "noise", "unknown")}
	if label_mode == "ternary":
		for spec in rows:
			for sp in list(spec.get("spikes", [])):
				mv = float(sp.get("metric_value", np.nan))
				cls = str(sp.get("label_class", "unknown")).lower()
				if np.isfinite(mv):
					class_vals.setdefault(cls, []).append(mv)

	scatter_points: List[Dict[str, float]] = []
	for spec_idx, spec in enumerate(rows):
		for sp in list(spec.get('spikes', [])):
			if not isinstance(sp, dict):
				continue
			mv = float(sp.get('metric_value', np.nan))
			if not np.isfinite(mv):
				continue
			scatter_points.append(
				{
					"spectrum_idx": float(spec_idx),
					"metric_value": mv,
					"is_muon": float(int(sp.get('is_muon', 0))),
					"label_class": str(sp.get("label_class", "muon" if int(sp.get("is_muon", 0)) == 1 else "non-muon")),
					"peak_index": float(int(sp.get('peak_index', -1))),
				}
			)
	scatter_x = np.arange(len(scatter_points), dtype=float)
	scatter_y = np.asarray([float(p["metric_value"]) for p in scatter_points], dtype=float)
	scatter_lbl = np.asarray([int(p["is_muon"]) for p in scatter_points], dtype=int)
	scatter_classes = np.asarray([str(p.get("label_class", "")) for p in scatter_points], dtype=str)
	scatter_spec_idx = np.asarray([int(p["spectrum_idx"]) for p in scatter_points], dtype=int)

	corr_low, corr_high, corr_reverse, corr_purity_hint = _load_corr_purity_defaults(args.corr_json, args.metric)
	t_low = float(args.threshold_low) if args.threshold_low is not None else None
	t_high = float(args.threshold_high) if args.threshold_high is not None else None
	if args.thr_l is not None:
		t_low = float(args.thr_l)
	if args.thr_h is not None:
		t_high = float(args.thr_h)
	if t_low is None and corr_low is not None:
		t_low = float(corr_low)
	if t_high is None and corr_high is not None:
		t_high = float(corr_high)
	if t_low is None or t_high is None:
		auto_low, auto_high = _kmeans_thresholds_1d(vals, k=3)
		if t_low is None:
			t_low = auto_low
		if t_high is None:
			t_high = auto_high
	if t_low > t_high:
		t_low, t_high = t_high, t_low
	reverse = bool(args.reverse) or bool(corr_reverse)

	print(f"[{args.metric}] thresholds: low={t_low:.6g}, high={t_high:.6g}, reverse={bool(reverse)}")

	state = {'i': 0}
	fig = plt.figure(figsize=(15, 8))
	ax = fig.add_axes((0.06, 0.42, 0.91, 0.52))
	ax_hist = fig.add_axes((0.06, 0.08, 0.42, 0.22))
	ax_scatter = fig.add_axes((0.55, 0.08, 0.42, 0.22))
	info_txt = ax.text(
		0.01,
		0.99,
		"",
		transform=ax.transAxes,
		va="top",
		ha="left",
		family="monospace",
		fontsize=10,
		bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
	)

	corr_hint = _load_corr_hint(args.corr_json, args.metric)
	corr_row = _load_corr_metric_row(args.corr_json, args.metric)
	hist_state: Dict[str, object] = {'bins': None}

	def _draw_hist() -> None:
		ax_hist.clear()
		bins = _uniform_bin_edges(vals, target_bins=72)
		if label_mode == "ternary":
			for cls in show_classes:
				arr = np.asarray(class_vals.get(cls, []), dtype=float)
				if arr.size:
					ax_hist.hist(arr, bins=bins, alpha=0.42, color=LABEL_CLASS_COLORS.get(cls, "#7f7f7f"))
		else:
			if non_vals:
				ax_hist.hist(non_vals, bins=bins, alpha=0.45, color="#1f77b4")
			if mu_vals:
				ax_hist.hist(mu_vals, bins=bins, alpha=0.45, color="#d62728")
		ax_hist.axvline(t_low, color="#ff7f0e", linestyle="--", linewidth=1.8)
		ax_hist.axvline(t_high, color="#d62728", linestyle="--", linewidth=1.8)
		ax_hist.set_title(f"{args.metric} histogram")
		ax_hist.grid(alpha=0.25)
		ax_hist.set_xlabel(args.metric)
		ax_hist.set_ylabel("Count")
		if bool(reverse):
			ax_hist.invert_xaxis()
		hist_state['bins'] = bins

	def _draw_scatter(current_spec_idx: int) -> None:
		ax_scatter.clear()
		if scatter_y.size == 0:
			ax_scatter.set_title(f"{args.metric} scatter overview")
			ax_scatter.grid(alpha=0.25)
			return
		mask_non = scatter_lbl == 0
		mask_mu = scatter_lbl == 1
		current_mask = scatter_spec_idx == int(current_spec_idx)
		if label_mode == "ternary":
			for cls in show_classes:
				mask_cls = scatter_classes == cls
				if np.any(mask_cls):
					ax_scatter.scatter(
						scatter_x[mask_cls],
						scatter_y[mask_cls],
						s=24,
						alpha=0.75,
						color=LABEL_CLASS_COLORS.get(cls, "#7f7f7f"),
					)
		elif np.any(mask_non):
			ax_scatter.scatter(
				scatter_x[mask_non],
				scatter_y[mask_non],
				s=24,
				alpha=0.75,
				color="#1f77b4",
			)
		if label_mode != "ternary" and np.any(mask_mu):
			ax_scatter.scatter(
				scatter_x[mask_mu],
				scatter_y[mask_mu],
				s=24,
				alpha=0.75,
				color="#d62728",
			)
		ax_scatter.axhline(t_low, color="#ff7f0e", linestyle="--", linewidth=1.5)
		ax_scatter.axhline(t_high, color="#d62728", linestyle="--", linewidth=1.5)
		if np.any(current_mask):
			ax_scatter.scatter(
				scatter_x[current_mask],
				scatter_y[current_mask],
				s=96,
				facecolors="none",
				edgecolors="black",
				linewidths=1.5,
				zorder=5,
			)
		ax_scatter.set_title(f"{args.metric} scatter overview")
		ax_scatter.set_xlabel("Labeled candidate index")
		ax_scatter.set_ylabel("Metric value")
		ax_scatter.grid(alpha=0.25)

	def _draw_main() -> None:
		i = int(state['i'])
		spec = rows[i]
		y = int(spec['y'])
		x = int(spec['x'])
		spikes = list(spec.get('spikes', []))
		_draw_hist()
		_draw_scatter(i)

		ax.clear()
		signal = raw[y, x, :].astype(float)
		ax.plot(x_axis, signal, color="#1f77b4", linewidth=1.1)

		n_no = 0
		n_maybe = 0
		n_ok = 0
		for sp_idx, sp in enumerate(spikes):
			p = int(sp.get('peak_index', -1))
			if p < 0 or p >= x_axis.size:
				continue
			mv = float(sp.get("metric_value", np.nan))
			col = _class_color(mv, t_low, t_high, reverse=bool(reverse))
			cn = _class_name(mv, t_low, t_high, reverse=bool(reverse))
			if cn == "no-muon":
				n_no += 1
			elif cn == "maybe-muon":
				n_maybe += 1
			elif cn == "ok-muon":
				n_ok += 1
			xp = float(x_axis[p])
			ax.axvline(xp, linestyle="--", linewidth=1.5, color=col, alpha=0.92)
			label = _fmt_signal_label(mv)
			side = -1 if (sp_idx % 2 == 0) else 1
			x_offset = side * (10 + 3 * min(len(label), 5))
			y_frac = 0.92 - 0.11 * (sp_idx % 3)
			ax.annotate(
				label,
				xy=(xp, y_frac),
				xycoords=ax.get_xaxis_transform(),
				xytext=(x_offset, 0),
				textcoords="offset points",
				color=col,
				fontsize=10,
				rotation=80,
				va="center",
				ha="right" if side < 0 else "left",
				bbox={
					"boxstyle": "round,pad=0.12",
					"facecolor": "white",
					"edgecolor": "none",
					"alpha": 0.82,
				},
				clip_on=False,
				zorder=3,
			)

		n_mu = sum(1 for sp in spikes if isinstance(sp, dict) and int(sp.get('is_muon', 0)) == 1)
		n_non = sum(1 for sp in spikes if isinstance(sp, dict) and int(sp.get('is_muon', 0)) == 0)

		ax.set_title(
			f"Metric viewer | idx {i+1}/{len(rows)} | y={y}, x={x} | spikes={len(spikes)} | no={n_no}, maybe={n_maybe}, ok={n_ok}",
		)
		ax.set_xlabel("Raman shift / cm$^{-1}$")
		ax.set_ylabel("Intensity")
		ax.grid(alpha=0.23)
		if bool(reverse):
			bins = hist_state.get('bins')
			if isinstance(bins, np.ndarray) and bins.size >= 2:
				for sp in spikes:
					if not isinstance(sp, dict):
						continue
					mv = float(sp.get("metric_value", np.nan))
					if not np.isfinite(mv):
						continue
					bi = int(np.clip(np.digitize([mv], bins)[0] - 1, 0, bins.size - 2))
					x0 = float(bins[bi])
					x1 = float(bins[bi + 1])
					ax_hist.axvspan(x0, x1, color="#2ca02c", alpha=0.10, zorder=0)
		else:
			bins = hist_state.get('bins')
			if isinstance(bins, np.ndarray) and bins.size >= 2:
				for sp in spikes:
					if not isinstance(sp, dict):
						continue
					mv = float(sp.get("metric_value", np.nan))
					if not np.isfinite(mv):
						continue
					bi = int(np.clip(np.digitize([mv], bins)[0] - 1, 0, bins.size - 2))
					x0 = float(bins[bi])
					x1 = float(bins[bi + 1])
					ax_hist.axvspan(x0, x1, color="#2ca02c", alpha=0.10, zorder=0)

		auc_txt = _fmt_opt(None if corr_row is None else corr_row.get("auc_feature_oriented", corr_row.get("auc_feature_raw")))
		info = [
			f"metric: {args.metric}",
			f"auc: {auc_txt}",
			f"label_mode: {label_mode}",
			f"threshold low: {t_low:.6g}",
			f"threshold high: {t_high:.6g}",
			f"reverse mode: {bool(reverse)}",
			"",
			f"total non-muon: {len(non_vals)}",
			f"total muon: {len(mu_vals)}",
			f"scatter candidates: {scatter_y.size}",
			f"labeled spikes in spectrum: {len(spikes)}",
			f"labels muon/non: {n_mu}/{n_non}",
		]
		if label_mode == "ternary":
			info += [f"total {cls}: {len(class_vals.get(cls, []))}" for cls in show_classes]
		if len(spikes) == 0:
			info += ["", "No labeled candidates in this spectrum."]
		if bool(reverse):
			info += ["", "Reverse semantics:", " lower / more negative metric => more muon-like"]
		if corr_hint:
			info += ["", corr_hint]
		if corr_purity_hint:
			info += [corr_purity_hint]
		info_txt.set_text("\n".join(info))
		fig.canvas.draw_idle()

	def on_key(event) -> None:
		key = (event.key or "").lower()
		if key in ("right", "n"):
			state['i'] = min(len(rows) - 1, int(state['i']) + 1)
			_draw_main()
		elif key in ("left", "p"):
			state['i'] = max(0, int(state['i']) - 1)
			_draw_main()
		elif key == "q":
			plt.close(fig)

	def _nearest_scatter_spectrum_idx(event) -> Optional[int]:
		if scatter_y.size == 0 or event.inaxes is not ax_scatter:
			return None
		if event.x is None or event.y is None:
			return None
		points_px = ax_scatter.transData.transform(np.column_stack([scatter_x, scatter_y]))
		click_px = np.array([float(event.x), float(event.y)], dtype=float)
		d2 = np.sum((points_px - click_px) ** 2, axis=1)
		if d2.size == 0 or not np.any(np.isfinite(d2)):
			return None
		best = int(np.nanargmin(d2))
		return int(scatter_spec_idx[best])

	def on_click(event) -> None:
		if event.inaxes is not ax_scatter:
			return
		spec_idx = _nearest_scatter_spectrum_idx(event)
		if spec_idx is None:
			return
		state['i'] = int(np.clip(spec_idx, 0, len(rows) - 1))
		_draw_main()

	fig.canvas.mpl_connect("key_press_event", on_key)
	fig.canvas.mpl_connect("button_press_event", on_click)
	_draw_hist()
	_draw_main()
	print("Controls: Left/Right arrows (or p/n) to browse spectra, click scatter to jump, q to quit.")
	plt.show()


if __name__ == '__main__':
	main()
