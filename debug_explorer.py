from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _pick_spectrum(
		per: List[Dict[str, Any]],
		spectrum_index: Optional[int],
		y: Optional[int],
		x: Optional[int],
) -> Dict[str, Any]:
	if y is not None and x is not None:
		for row in per:
			if int(row.get("y", -1)) == int(y) and int(row.get("x", -1)) == int(x):
				return row
		raise ValueError(f"Spectrum (y={y}, x={x}) not found in per_spectrum.")
	if spectrum_index is None:
		spectrum_index = 0
	if spectrum_index < 0 or spectrum_index >= len(per):
		raise ValueError(f"spectrum_index out of range: {spectrum_index}, n={len(per)}")
	return per[int(spectrum_index)]


def main() -> None:
	parser = argparse.ArgumentParser(description="Explore per-spectrum debug report parameters.")
	parser = argparse.ArgumentParser(
		description="Explore candidate-spike parameters inside a single per_spectrum entry."
	)
	parser.add_argument("--report", type=Path, required=True, help="Path to the report.json")
	parser.add_argument("--param", type=str, default="muon_score", help="Spike parameter to plot")
	parser.add_argument("--spectrum-index", type=int, default=0, help="Index in per_spectrum list")
	parser.add_argument("--y", type=int, default=None, help="Select spectrum by y (requires --x)")
	parser.add_argument("--x", type=int, default=None, help="Select spectrum by x (requires --y)")
	parser.add_argument(
		"--x-axis",
		type=str,
		default="candidate_index",
		choices=["candidate_index", "peak_index", "peak_position_cm1"],
		help="Which x-axis to use for candidate points."
	)
	args = parser.parse_args()

	if (args.y is None) ^ (args.x is None):
		raise ValueError("Use both --y and --x together or neither.")

	data = json.loads(Path(args.report).read_text(encoding="utf-8"))
	per = data.get("per_spectrum", [])
	if not per:
		raise ValueError("Report has no per_spectrum entries. Enable debug_include_per_spectrum=true.")

	row = _pick_spectrum(per, args.spectrum_index, args.y, args.x)
	spikes = row.get("spikes", [])
	if not spikes:
		raise ValueError("Selected spectrum has no spike candidates in report.")

	y_vals = np.array([float(s.get(args.param, np.nan)) for s in spikes], dtype=float)
	if args.x_axis == "candidate_index":
		x_vals = np.arange(len(spikes), dtype=float)
		x_label = "N (candidate index in selected spectrum)"
	else:
		x_vals = np.array([float(s.get(args.x_axis, np.nan)) for s in spikes], dtype=float)
		x_label = args.x_axis

	plt.figure(figsize=(8,4))
	plt.plot(x_vals, y_vals, marker="o", linestyle="-", markersize=4)
	plt.xlabel(x_label)
	plt.ylabel(args.param)
	plt.title(
		f"Candidates in spectrum y={row.get('y')} x={row.get('x')} | "
		f"n_candidates={len(spikes)}"
	)
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
	