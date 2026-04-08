# manual_review.py
from __future__ import annotations

from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from muon_pipeline import SpikeSegment


def manual_confirm_signals(
		x_axis: np.ndarray,
		spectra: np.ndarray,  # (H,W,N)
		spikes_by_pixel: Dict[Tuple[int, int], List[SpikeSegment]],
		overlays: Optional[Dict[str, np.ndarray]] = None,  # can be None
) -> List[SpikeSegment]:
	"""
	Manual confirmation per spike segment (not per spectrum)

	Defaults to "accept all". User can reject individual spikes or whole spectra.

	Keys:
		- Enter / y : accept current spike and move forward
		- n         : reject current spike and move forward
		- a         : accept all spikes in this spectrum, jump to next spectrum
		- r         : reject all spikes in this spectrum, jump to next spectrum
		- q         : finish (close)
	"""
	# Matplotlib default keymaps clashes
	mpl.rcParams["keymap.save"] = []
	mpl.rcParams["keymap.quit"] = []

	# Flatten into an ordered list of pixels
	pix_keys = sorted(spikes_by_pixel.keys(), key=lambda t: (t[0],  t[1]))
	if not pix_keys:
		return []

	# Sort segments within each pixel by peak height (strongest first)
	for k in pix_keys:
		spikes_by_pixel[k] = sorted(spikes_by_pixel[k], key=lambda s: float(s.peak_height), reverse=True)

	# Default: accept all segments
	accepted: Set[SpikeSegment] = set()
	for k in pix_keys:
		for s in spikes_by_pixel[k]:
			accepted.add(s)

	fig, ax = plt.subplots(1, 1, figsize=(11, 4))
	fig.canvas.manager.set_window_title("Manual spike confirmation (per signal)")

	state = {"pi": 0, "si": 0, "done": False}

	def _current() -> Tuple[Tuple[int, int], SpikeSegment]:
		key = pix_keys[state["pi"]]
		segs = spikes_by_pixel[key]
		s = segs[state["si"]]
		return key, s

	def _advance_signal() -> None:
		"""Move to next signal; if end of current pixel, go to next pixel."""
		key = pix_keys[state["pi"]]
		segs = spikes_by_pixel[key]

		state["si"] += 1
		if state["si"] >= len(segs):
			state["pi"] += 1
			state["si"] = 0

		if state["pi"] >= len(pix_keys):
			state["done"] = True
			plt.close(fig)

	def _advance_pixel() -> None:
		"""Jump to next pixel."""
		state["pi"] += 1
		state["si"] = 0
		if state["pi"] >= len(pix_keys):
			state["done"] = True
			plt.close(fig)

	def _draw() -> None:
		ax.clear()

		key, s = _current()
		y, x = key

		raw = spectra[y, x, :]
		ax.plot(x_axis, raw, label="raw")

		if overlays is not None:
			if "opening" in overlays:
				ax.plot(x_axis, overlays["opening"][y, x, :], label="opening")
			if "top_hat" in overlays:
				ax.plot(x_axis, overlays["top_hat"][y, x, :], label="top_hat")

		# draw ALL segments in this pixel as thin anchor lines
		segs = spikes_by_pixel[key]
		for ss in segs:
			ax.axvline(x_axis[ss.start], linewidth=1, linestyle="--")
			ax.axvline(x_axis[ss.end], linewidth=1, linestyle="--")

		# highlight CURRENT segment
		# ax.axvline(x_axis[s.start], linewidth=2, linestyle="--")
		# ax.axvline(x_axis[s.end], linewidth=2, linestyle="--")
		ax.axvline(x_axis[s.peak_index], linewidth=2, linestyle="--", color="red")

		# status text
		is_acc = (s in accepted)
		ax.set_title(
			f"Pixel (y={y}, x={x})  |  Signal {state['si']+1}/{len(segs)}  |  "
			f"{'ACCEPTED' if is_acc else 'REJECTED'}  |  "
			f"peak_idx={s.peak_index}, anchors=({s.start},{s.end}), peak={s.peak_height:.0f}"
		)
		ax.legend(loc="best")
		fig.canvas.draw_idle()

	def _reject_all_in_pixel(key: Tuple[int, int]) -> None:
		for ss in spikes_by_pixel[key]:
			if ss in accepted:
				accepted.remove(ss)

	def _accept_all_in_pixel(key: Tuple[int, int]) -> None:
		for ss in spikes_by_pixel[key]:
			accepted.add(ss)

	def on_key(event) -> None:
		if state["done"]:
			return
		key = (event.key or "").lower()

		pix, s = _current()

		if key in ("enter", "y"):
			accepted.add(s)
			_advance_signal()
		elif key == "n":
			if s in accepted:
				accepted.remove(s)
			_advance_signal()
		elif key == "r":
			_reject_all_in_pixel(pix)
			_advance_pixel()
		elif key == "a":
			_accept_all_in_pixel(pix)
			_advance_pixel()
		elif key == "q":
			state["done"] = True
			plt.close(fig)
			return

		if not state["done"]:
			_draw()

	def on_close(_event) -> None:
		state["done"] = True

	fig.canvas.mpl_connect("key_press_event", on_key)
	fig.canvas.mpl_connect("close_event", on_close)

	_draw()
	plt.show()

	# Return accepted spikes as a list (sorted for determinism)
	out = sorted(list(accepted), key=lambda s: (s.y, s.x, s.start, s.end, s.peak_index))
	return out