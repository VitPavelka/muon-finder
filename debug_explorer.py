from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _split_threshold_mask(y_vals: np.ndarray, threshold_value: float, threshold_mode: str) -> np.ndarray:
	if threshold_mode == "above":
		return y_vals > threshold_value
	if threshold_mode == "below":
		return y_vals < threshold_value
	if threshold_mode == "abs_above":
		return np.abs(y_vals) > threshold_value
	raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")


def _plot_series(
		ax: plt.Axes,
		x_vals: np.ndarray,
		y_vals: np.ndarray,
		threshold_value: float | None,
		threshold_mode: str,
) -> None:
	if threshold_value is None:
		ax.plot(x_vals, y_vals, marker="o", linestyle="-", markersize=3, color="C0")
		return

	mask_hi = _split_threshold_mask(y_vals, float(threshold_value), threshold_mode)
	mask_lo = ~mask_hi
	ax.plot(x_vals, y_vals, linestyle="-", linewidth=1.0, color="C0", alpha=0.6)
	if np.any(mask_lo):
		ax.scatter(x_vals[mask_lo], y_vals[mask_lo], s=16, color="C0", zorder=3)
	if np.any(mask_hi):
		ax.scatter(x_vals[mask_hi], y_vals[mask_hi], s=22, color="C3", zorder=4)


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
		default=100,
		help="Maximum number of spectra panels to show in multi_spectra_panels mode."
	)
	parser.add_argument(
		"--start-index",
		type=int,
		default=0,
		help="Start index for panel slicing in multi_spectra_panels mode."
	)
	parser.add_argument("--threshold-value", type=float, default=None, help="Highlight point threshold for y-values.")
	parser.add_argument(
		"--threshold-mode",
		type=str,
		default="above",
		choices=["above", "below", "abs_above"],
		help="How to compare y-values against threshold-value."
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
		_plot_series(
			plt.gca(),
			x_vals=x_vals,
			y_vals=y_vals,
			threshold_value=args.threshold_value,
			threshold_mode=str(args.threshold_mode),
		)
		plt.xlabel(x_label)
		plt.ylabel(args.param)
		plt.title(f"Spectrum y={row.get('y')} x={row.get('x')} | n_candidates={len(spikes)}")
		if args.threshold_value is not None:
			plt.axhline(float(args.threshold_value), linestyle="--", color="gray", linewidth=1.0, alpha=0.7)
		plt.grid(alpha=0.3)
		plt.grid(alpha=0.3)
		plt.tight_layout()
		plt.show()
		return

	# mode = multi_spectra_panels
	rows = per
	start = max(0, int(args.start_index))
	if start > 0:
		rows = rows[start:]
	if args.max_spectra is not None and int(args.max_spectra) > 0:
		rows = rows[: int(args.max_spectra)]
	if not rows:
		raise ValueError("No spectra left after applying --start-index/--max-spectra slicing.")
	n_pan = len(rows)
	n_cols = int(np.ceil(np.sqrt(n_pan)))
	n_rows = int(np.ceil(n_pan / n_cols))
	fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.1 * n_rows), squeeze=False)
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
			_plot_series(
				ax,
				x_vals=x_vals,
				y_vals=y_vals,
				threshold_value=args.threshold_value,
				threshold_mode=str(args.threshold_mode),
			)
			ax.set_xlabel(x_label, fontsize=8)
			ax.set_ylabel(args.param, fontsize=8)
			if args.threshold_value is not None:
				ax.axhline(
					float(args.threshold_value),
					linestyle="--",
					linewidth=0.9,
					color="gray",
					alpha=0.7
				)
		else:
			ax.text(0.5, 0.5, "no candidates", ha="center", va="center", transform=ax.transAxes)
		ax.set_title(f"y={row.get('y')} x={row.get('x')}", fontsize=9)
		ax.grid(alpha=0.25)

	for j in range(n_pan, len(flat)):
		flat[j].axis("off")

	fig.suptitle(
		f"All spectra panels: {args.param} | shown={n_pan} start={start}", fontsize=12
	)
	# plt.tight_layout()
	fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.92, wspace=0.35, hspace=0.55)
	plt.show()


if __name__ == "__main__":
	main()
