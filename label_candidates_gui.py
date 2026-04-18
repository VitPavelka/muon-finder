from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from wdf_io import load_dataset


def _load_existing_labels(path: Optional[Path]) -> Dict[Tuple[int, int, int], int]:
	if path is None or not path.exists():
		return {}
	out: Dict[Tuple[int, int, int], int] = {}
	with path.open("r", encoding="utf-8", newline="") as f:
		r = csv.DictReader(f)
		for row in r:
			k = (int(row["y"]), int(row["x"]), int(row['peak_index']))
			out[k] = 1 if int(row['is_muon']) else 0
	return out


def _save_labels(path: Path, rows: List[Dict], selected: Dict[Tuple[int, int, int], bool]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="") as f:
		w = csv.writer(f)
		w.writerow(["y", "x", "peak_index", "is_muon"])
		for spec in rows:
			y = int(spec['y'])
			x = int(spec['x'])
			for sp in spec.get("spikes", []):
				p = int(sp['peak_index'])
				w.writerow([y, x, p, 1 if selected.get((y, x, p), False) else 0])


def main() -> None:
	parser = argparse.ArgumentParser(description="Interactive candidate labeler (click spike lines to toggle muon=1).")
	parser.add_argument("--report", type=Path, required=True, help="Path to debug_report.json")
	parser.add_argument("--input", type=Path, required=True, help="Input dataset path (.wdf/.npz) for raw spectra plotting")
	parser.add_argument("--out-csv", type=Path, required=True, help="Output labels CSV path")
	parser.add_argument("--start-index", type=int, default=0, help="Start spectrum index in per_spectrum")
	parser.add_argument("--max-spectra", type=int, default=None, help="Limit number of spectra in labeling session")
	parser.add_argument("--existing-labels-csv", type=Path, default=None, help="Optional existing labels CSV to preload")
	args = parser.parse_args()

	report = json.loads(args.report.read_text(encoding="utf-8"))
	rows_all: List[Dict] = list(report.get("per_spectrum", []))
	if not rows_all:
		raise ValueError("Report has no per_spectrum entries.")

	start = max(0, int(args.start_index))
	rows = rows_all[start:]
	if args.max_spectra is not None and int(args.max_spectra) > 0:
		rows = rows[: int(args.max_spectra)]
	if not rows:
		raise ValueError("No spectra to label after applying start/max filters.")

	ds = load_dataset(args.input)
	x_axis = ds.x_axis
	raw = ds.spectra

	pre = _load_existing_labels(args.existing_labels_csv)
	selected: Dict[Tuple[int, int, int], bool] = {
		k: bool(v)
		for k, v in pre.items()
	}

	state = {"i": 0}
	fig, ax = plt.subplots(1, 1, figsize=(11, 5))
	plt.subplots_adjust(bottom=0.18)

	def _draw(preserve_view: bool = False) -> None:
		xlim_prev = ax.get_xlim()
		ylim_prev = ax.get_ylim()
		ax.clear()
		i = int(state['i'])
		spec = rows[i]
		y = int(spec['y'])
		x = int(spec['x'])
		spikes = spec.get('spikes', [])

		spec_raw = raw[y, x, :].astype(float)
		ax.plot(x_axis, spec_raw, color="#1f77b4", linewidth=1.2, label="raw")

		for sp in spikes:
			p = int(sp['peak_index'])
			s = int(sp.get('start', p))
			e = int(sp.get('end', p))
			key = (y, x, p)
			is_on = bool(selected.get(key, False))
			pcol = "red" if is_on else "#7f7f7f"
			alpha = 0.95 if is_on else 0.55
			ax.axvline(float(x_axis[p]), linestyle="--", linewidth=1.4, color=pcol, alpha=alpha)
			if 0 <= s < x_axis.size and 0 <= e < x_axis.size:
				x0 = min(float(x_axis[s]), float(x_axis[e]))
				x1 = max(float(x_axis[s]), float(x_axis[e]))
				ax.axvspan(x0, x1, color="green", alpha=0.07, zorder=0)

		n_sel = sum(
			1 for sp in spikes
			if selected.get((y, x, int(sp['peak_index'])), False)
		)
		ax.set_title(
			f"Labeler | idx{i+1}/{len(rows)} | y={y} x={x} | candidates={len(spikes)} | selected_muons={n_sel}",
		)
		ax.set_xlabel("Raman shift/cm$^{-1}$")
		ax.set_ylabel("Intensity")
		ax.grid(alpha=0.25)
		ax.legend(loc="best")
		if preserve_view:
			ax.set_xlim(xlim_prev)
			ax.set_ylim(ylim_prev)
		fig.canvas.draw_idle()

	def _toggle_nearest(click_x: float) -> None:
		i = int(state['i'])
		spec = rows[i]
		y = int(spec['y'])
		x = int(spec['x'])
		spikes = spec.get('spikes', [])
		if not spikes:
			return

		peak_x = np.array([float(x_axis[int(sp['peak_index'])]) for sp in spikes], dtype=float)
		j = int(np.argmin(np.abs(peak_x - click_x)))
		tol = 0.01 * (float(np.max(x_axis)) - float(np.min(x_axis)))
		if abs(float(peak_x[j]) - float(click_x)) > tol:
			return
		p = int(spikes[j]['peak_index'])
		k = (y, x, p)
		selected[k] = not bool(selected.get(k, False))
		_draw(preserve_view=True)

	def _next() -> None:
		state['i'] = min(len(rows) - 1, int(state['i']) + 1)
		_draw()

	def _prev() -> None:
		state['i'] = max(0, int(state['i']) - 1)
		_draw()

	def on_click(event) -> None:
		if event.inaxes != ax or event.xdata is None:
			return
		_toggle_nearest(float(event.xdata))

	def on_key(event) -> None:
		key = (event.key or "").lower()
		if key in ("n", "right"):
			_next()
		elif key in ("p", "left"):
			_prev()
		elif key == "s":
			_save_labels(args.out_csv, rows, selected)
			print(f"Saved labels -> {args.out_csv}")
		elif key == "q":
			_save_labels(args.out_csv, rows, selected)
			print(f"Saved labels -> {args.out_csv}")
			plt.close(fig)

	fig.canvas.mpl_connect("button_press_event", on_click)
	fig.canvas.mpl_connect("key_press_event", on_key)
	_draw()

	print("Controls: click near a candidate line to toggle label; n/p = next/prev; s = save; q = save+quit")
	plt.show()


if __name__ == "__main__":
	main()
