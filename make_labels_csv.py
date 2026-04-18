from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _flatten_candidates(report: Dict[str, Any]) -> List[Dict[str, int]]:
	rows: List[Dict[str, int]] = []
	for spec in report.get("per_spectrum", []):
		y = int(spec.get("y", -1))
		x = int(spec.get("x", -1))
		for sp in spec.get("spikes", []):
			peak_index = int(sp.get("peak_index", -1))
			rows.append({"y": y, "x": x, "peak_index": peak_index})
	return rows


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Generate labels.csv template from debug_report.json candidates."
	)
	parser.add_argument("--report", type=Path, required=True, help="Path to debug_report.json")
	parser.add_argument("--out-csv", type=Path, required=True, help="Output labels CSV path")
	args = parser.parse_args()

	report = json.loads(args.report.read_text(encoding="utf-8"))
	rows = _flatten_candidates(report)
	if not rows:
		raise ValueError("No candidates found in report['per_spectrum'][*]['spikes'].")

	# de-duplicate exact candidate keys while preserving stable sort
	unique = sorted({(r['y'], r['x'], r['peak_index']) for r in rows})

	with args.out_csv.open(mode="w", encoding="utf-8", newline="") as csvfile:
		w = csv.writer(csvfile)
		w.writerow(["y", "x", "peak_index", "is_muon"])
		for y, x, peak in unique:
			w.writerow([y, x, peak, ""])

	print(f"Wrote {len(unique)} candidate rows to {args.out_csv}")
	print("Fill the last column (is muon) manually: 1=muon, 0=non-muon")

	
if __name__ == "__main__":
	main()
