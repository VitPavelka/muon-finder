from __future__ import annotations

import argparse
import json
from pathlib import Path
from tkinter import font
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Plot selected candidate parameter either in one spectrum or as multi-spectrum panels."
	)
	parser.add_argument("--report", type=Path, required=True, help="Path to the debug_report.json")
	parser.add_argument("--param", type=str, default="muon_score", help="Spike parameter to plot")
	parser.add_argument(
		"--x-axis",
		type=str,
		default="candidate_index",
		choices=["candidate_index", "peak_index", "peak_position_cm1"],
		help="Which x-axis to use for candidate points."
	)
	parser.add_argument(
		"--mode",
		type=str,
		default="candidates_in_spectrum",
		choices=["candidates_in_spectrum", "multi_spectra_panels"],
		help="Single spectrum or all-spectra panel mode."
	)
	parser.add_argument("--y", type=int, default=None, help="Select spectrum by y (requires --x)")
	parser.add_argument("--x", type=int, default=None, help="Select spectrum by x (requires --y)")
	parser.add_argument(
		"--max-spectra",
		type=int,
		default=500,
		help="Max number of spectra to display in spectra overview mode."
	)
	args = parser.parse_args()

	data = json.loads(Path(args.report).read_text(encoding="utf-8"))
	per = data.get("per_spectrum", [])
	if not per:
		raise ValueError("Report has no per_spectrum entries. Enable debug_include_per_spectrum=true.")

	if args.mode == "candidates_in_spectrum":
		if args.y is None or args.x is None:
			raise ValueError("For candidates_in_spectrum, provide both --y and --x.")
		row = None
		for cand in per:
			if int(cand.get("y", -1)) == int(args.y) and int(cand.get("x", -1)) == int(args.x):
				row = cand
				break
		if row is None:
			raise ValueError(f"Spectrum (y={args.y}, x={args.x}) not found in per_spectrum.")
		spikes = row.get("spikes", [])
		if not spikes:
			raise ValueError("Selected spectrum has no candidates.")

		y_vals = np.array([float(s.get(args.param, np.nan)) for s in spikes], dtype=float)
		if args.x_axis == "candidate_index":
			x_vals = np.arange(len(spikes), dtype=float)
			x_label = "N (candidate index in selected spectrum)"
		else:
			x_vals = np.array([float(s.get(args.x_axis, np.nan)) for s in spikes], dtype=float)
			x_label = args.x_axis

		plt.figure(figsize=(8, 4))
		plt.plot(x_vals, y_vals, marker="o", linestyle="-", markersize=4)
		plt.xlabel(x_label)
		plt.ylabel(args.param)
		plt.title(f"Spectrum y={row.get('y')} x={row.get('x')} | n_candidates={len(spikes)}")
		plt.grid(alpha=0.3)
		plt.grid(alpha=0.3)
		plt.tight_layout()
		plt.show()
		return

	# mode = multi_spectra_panels
	rows = per
	n_pan = len(rows)
	n_cols = int(np.ceil(np.sqrt(n_pan)))
	n_rows = int(np.ceil(n_pan / n_cols))
	fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.8 * n_rows), squeeze=False)
	flat = axs.ravel()

	for i, row in enumerate(rows):
		ax = flat[i]
		spikes = row.get("spikes", [])
		if spikes:
			y_vals = np.array([float(s.get(args.param, np.nan)) for s in spikes], dtype=float)
			if args.x_axis == "candidate_index":
				x_vals = np.arange(len(spikes), dtype=float)
				x_label = "candidate idx"
			else:
				x_vals = np.array([float(s.get(args.x_axis, np.nan)) for s in spikes], dtype=float)
				x_label = args.x_axis
			ax.plot(x_vals, y_vals, marker="o", linestyle="-", markersize=3)
			ax.set_xlabel(x_label, fontsize=8)
			ax.set_ylabel(args.param, fontsize=8)
		else:
			ax.text(0.5, 0.5, "no candidates", ha="center", va="center", transform=ax.transAxes)
		ax.set_title(f"y={row.get('y')} x={row.get('x')}", fontsize=9)
		ax.grid(alpha=0.25)

	for j in range(n_pan, len(flat)):
		flat[j].axis("off")

	fig.suptitle(f"All spectra panels: {args.param}", fontsize=12)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
