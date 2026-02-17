# viewer.py
from __future__ import annotations

from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from muon_pipeline import SpikeSegment


def show_hover_map(
		x_axis: np.ndarray,
		spectra: np.ndarray,              # (H,W,N)
		score_map: np.ndarray,            # (H,W)
		candidate_mask: np.ndarray,       # (H,W)
		spikes_by_pixel: Dict[Tuple[int, int], List[SpikeSegment]],
		overlays: Dict[str, np.ndarray],  # erosion/dilation/opening/top_hat
		plot_raw: bool = True,
		plot_opening: bool = True,
		plot_erosion: bool = False,
		plot_dilation: bool = False,
		plot_top_hat: bool = True,
) -> None:
	H, W, N = spectra.shape

	fig, (ax_map, ax_spec) = plt.subplots(1, 2, figsize=(12, 5))
	fig.canvas.manager.set_window_title("Muon finder - hover viewer")

	im = ax_map.imshow(score_map, origin="upper", aspect="auto")
	ax_map.set_title("score map (hover)")
	ax_map.set_xlabel("x")
	ax_map.set_ylabel("y")
	fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

	# candidates' overlay (visually)
	ys, xs = np.where(candidate_mask)
	ax_map.scatter(xs, ys, s=8, marker="o", linewidths=0.5, facecolors="none")

	# marker for actual pixel
	marker = ax_map.scatter([0], [0], s=80, marker="s", facecolors="none", linewidths=2)

	# spectral lines
	(ln_raw,) = ax_spec.plot([], [], label="raw") if plot_raw else (None,)
	(ln_open,) = ax_spec.plot([], [], label="opening") if plot_opening else (None,)
	(ln_ero,) = ax_spec.plot([], [], label="erosion") if plot_erosion else None
	(ln_dil,) = ax_spec.plot([], [], label="dilation") if plot_dilation else None
	(ln_th,) = ax_spec.plot([], [], label="top_hat") if plot_top_hat else None

	spike_lines: List = []

	ax_spec.set_title("spectrum @ (y,x)")
	ax_spec.set_xlabel("Raman shift/cm$^{-1}$")
	ax_spec.set_ylabel("Intensity")
	ax_spec.legend(loc="best")

	frozen = {"state": False}  # right click = freeze/unfreeze

	def _update(y: int, x: int) -> None:
		y = int(np.clip(y, 0, H - 1))
		x = int(np.clip(x, 0, W - 1))

		marker.set_offsets([[x, y]])

		raw = spectra[y, x, :]
		if plot_raw and ln_raw is not None:
			ln_raw.set_data(x_axis, raw)

		if plot_opening and ln_open is not None:
			ln_open.set_data(x_axis, overlays["opening"][y, x, :])

		if plot_erosion and ln_ero is not None:
			ln_ero.set_data(x_axis, overlays["erosion"][y, x, :])

		if plot_dilation and ln_dil is not None:
			ln_dil.set_data(x_axis, overlays["dilation"][y, x, :])

		if plot_top_hat and ln_th is not None:
			ln_th.set_data(x_axis, overlays["top_hat"][y, x, :])

		# clear old spike lines
		for l in spike_lines:
			try:
				l.remove()
			except Exception:
				pass
		spike_lines.clear()

		# add spike markers
		segs = spikes_by_pixel.get((y, x), [])
		for s in segs:
			xx = x_axis[s.peak_index]
			l = ax_spec.axvline(xx, linestyle="--", linewidth=1)
			spike_lines.append(l)

		ax_spec.set_title(f"spectrum @ (y={y}, x={x}) | spikes={len(segs)}")
		ax_spec.relim()
		ax_spec.autoscale_view()

		fig.canvas.draw_idle()

	def on_move(event) -> None:
		if frozen["state"]:
			return
		if event.inaxes != ax_map:
			return
		if event.xdata is None or event.ydata is None:
			return
		x = int(round(event.xdata))
		y = int(round(event.ydata))
		_update(y, x)

	def on_click(event) -> None:
		if event.inaxes != ax_map:
			return
		if not event.button != 3:
			return
		frozen["state"] = not frozen["state"]

	fig.canvas.mpl_connect("motion_notify_event", on_move)
	fig.canvas.mpl_connect("button_press_event", on_click)

	# init
	_update(0, 0)
	plt.tight_layout()
	plt.show()