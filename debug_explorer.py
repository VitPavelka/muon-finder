from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _extract_value(spikes: List[Dict[str, Any]], param: str, reduce: str) -> float:
	if not spikes:
		return float("nan")
	vals = [s.get(param) for s in spikes if isinstance(s.get(param), (int, float))]
	if not vals:
		return float("nan")
	arr = np.asarray(vals, dtype=float)
	if reduce == "max":
		return float(np.max(arr))
	if reduce == "mean":
		return float(np.mean(arr))
	return float(arr[0])  # first


def main() -> None:
	parser = argparse.ArgumentParser(description="Explore per-spectrum debug report parameters.")
	parser.add_argument("--report", type=Path, required=True, help="Path to debug_report.json")
	parser.add_argument("--param", type=str, default="muon_score", help="Spike parameter to plot")
	parser.add_argument("--reduce", type=str, default="max", choices=["max", "mean", "first"])
	parser.add_argument("--only-has-spikes", action="store_true", help="Plot only spectra with n_spikes > 0")
	args = parser.parse_args()

	data = json.loads(Path(args.report).read_text(encoding="utf-8"))
	per = data.get("per_spectrum", [])
	if not per:
		raise ValueError("Report has no per_spectrum entries. Enable debug_include_per_spectrum=true.")

	x = []
	y = []
	for i, row in enumerate(per):
		n_spikes = int(row.get("n_spikes", 0))
		if args.only_has_spikes and n_spikes <= 0:
			continue
		val = _extract_value(row.get("spikes", []), args.param, args.reduce)
		x.append(i)
		y.append(val)

	plt.figure(figsize=(10, 4))
	plt.plot(x, y, marker="o", linestyle="-", markersize=3)
	plt.xlabel("N (index per_spectrum)")
	plt.ylabel(f"{args.param} [{args.reduce}]")
	plt.title(f"Debug explorer: {args.param} vs N")
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()

