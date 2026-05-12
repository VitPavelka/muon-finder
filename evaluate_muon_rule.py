from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

from candidate_labels import binary_from_label_class, normalize_label_class, parse_optional_binary_label
from muon_decision import classify_muon_rule_v3


def _detect_delimiter(sample: str) -> str:
	return ";" if sample.count(";") > sample.count(",") else ","


def _iter_rows(path: Path) -> Iterable[Dict[str, str]]:
	text = path.read_text(encoding="utf-8")
	if not text.strip():
		return []
	delimiter = _detect_delimiter(text[:4096])
	reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
	return list(reader)


def _get_float(row: Dict[str, str], *names: str) -> float:
	for name in names:
		if name in row and str(row[name]).strip() != "":
			return float(row[name])
	raise KeyError(f"Missing numeric column. Tried: {names}")


def _get_label(row: Dict[str, str]) -> bool:
	lbl = parse_optional_binary_label(row.get("is_muon"))
	if lbl is None and str(row.get("label_class", "")).strip():
		lbl = binary_from_label_class(normalize_label_class(row.get("label_class")))
	if lbl is not None:
		return bool(lbl)
	raise ValueError("Input CSV must contain is_muon column with boolean/0/1 values.")


def _fp_fn_row(row: Dict[str, str], decision: Dict[str, object]) -> Dict[str, object]:
	return {
		"spectrum_index": row.get("spectrum_index", ""),
		"candidate_index": row.get("candidate_index", ""),
		"candidate_position": row.get("candidate_position_cm1", row.get("candidate_position", "")),
		"spike_score_v1": decision["spike_score_v1"],
		"pce_negpref_t098_evidence_signed": decision["pce_negpref_t098_evidence_signed"],
		"gws_evidence_signed": decision["gws_evidence_signed"],
		"ss_category_v3": decision["ss_category_v3"],
		"pce_category_v3": decision["pce_category_v3"],
		"gws_category_v3": decision["gws_category_v3"],
		"muon_rule_v3_decision": decision["muon_rule_v3_decision"],
		"muon_rule_v3_reason": decision["muon_rule_v3_reason"],
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate muon decision rule from exported metric CSV.")
	parser.add_argument("--input", type=Path, required=True, help="Input CSV with exported metric scores and labels.")
	parser.add_argument("--rule", type=str, default="muon_rule_v3", help="Decision rule name. Currently only muon_rule_v3 is supported.")
	args = parser.parse_args()

	rule = str(args.rule).strip().lower()
	if rule != "muon_rule_v3":
		raise ValueError(f"Unsupported rule: {args.rule!r}")

	rows = list(_iter_rows(args.input))
	if not rows:
		raise ValueError("Input CSV is empty.")

	decision_counts = {"auto_muon": 0, "maybe_muon": 0, "no_muon": 0}
	reason_counts: Dict[str, int] = {}
	tp = fp = fn = tn = 0
	maybe_mu = 0
	maybe_non = 0
	fp_rows: List[Dict[str, object]] = []
	fn_rows: List[Dict[str, object]] = []

	for row in rows:
		ss = _get_float(row, "spike_score_v1_score", "spike_score_v1")
		pce = _get_float(row, "pce_negpref_t098_evidence_signed_score", "pce_negpref_t098_evidence_signed")
		gws = _get_float(row, "gws_evidence_signed_score", "gws_evidence_signed")
		is_muon = _get_label(row)
		result = classify_muon_rule_v3(ss, pce, gws)
		decision = {
			"spike_score_v1": ss,
			"pce_negpref_t098_evidence_signed": pce,
			"gws_evidence_signed": gws,
			"muon_rule_v3_decision": result.decision,
			"muon_rule_v3_reason": result.reason,
			"muon_rule_v3_score": result.score,
			"ss_category_v3": result.ss_category,
			"pce_category_v3": result.pce_category,
			"gws_category_v3": result.gws_category,
		}
		decision_counts[result.decision] += 1
		reason_counts[result.reason] = int(reason_counts.get(result.reason, 0) + 1)
		if result.decision == "maybe_muon":
			if is_muon:
				maybe_mu += 1
			else:
				maybe_non += 1
		is_auto = result.decision == "auto_muon"
		if is_auto and is_muon:
			tp += 1
		elif is_auto and not is_muon:
			fp += 1
			fp_rows.append(_fp_fn_row(row, decision))
		elif (not is_auto) and is_muon:
			fn += 1
			fn_rows.append(_fp_fn_row(row, decision))
		else:
			tn += 1

	print(f"rule={rule}")
	print("counts:")
	for key in ("auto_muon", "maybe_muon", "no_muon"):
		print(f"  {key}: {decision_counts[key]}")
	print("confusion_auto_muon:")
	print(f"  tp={tp} fp={fp} fn={fn} tn={tn}")
	print(f"  precision={tp / max(tp + fp, 1):.6f}")
	print(f"  recall={tp / max(tp + fn, 1):.6f}")
	print(f"  specificity={tn / max(tn + fp, 1):.6f}")
	print("maybe_muon_labels:")
	print(f"  muon={maybe_mu} non_muon={maybe_non}")
	print("reason_summary:")
	for reason, count in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0])):
		print(f"  {reason}: {count}")

	def _print_table(name: str, items: List[Dict[str, object]]) -> None:
		print(f"{name}:")
		if not items:
			print("  <none>")
			return
		for item in items:
			print(
				"  "
				f"spectrum_index={item['spectrum_index']} "
				f"candidate_index={item['candidate_index']} "
				f"candidate_position={item['candidate_position']} "
				f"ss={item['spike_score_v1']:.6g} "
				f"pce={item['pce_negpref_t098_evidence_signed']:.6g} "
				f"gws={item['gws_evidence_signed']:.6g} "
				f"ss_cat={item['ss_category_v3']} "
				f"pce_cat={item['pce_category_v3']} "
				f"gws_cat={item['gws_category_v3']} "
				f"decision={item['muon_rule_v3_decision']} "
				f"reason={item['muon_rule_v3_reason']}"
			)

	_print_table("false_positives", fp_rows)
	_print_table("false_negatives", fn_rows)


if __name__ == "__main__":
	main()
