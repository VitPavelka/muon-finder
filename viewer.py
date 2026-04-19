# viewer.py
from __future__ import annotations

from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import TextBox, Button, CheckButtons

from muon_pipeline import SpikeSegment


def show_hover_map(
		x_axis: np.ndarray,
		spectra: np.ndarray,              # (H,W,N)
		score_map: np.ndarray,            # (H,W)
		candidate_mask: np.ndarray,       # (H,W)
		spikes_by_pixel: Dict[Tuple[int, int], List[SpikeSegment]],
		overlays: Dict[str, np.ndarray],  # erosion/dilation/opening/top_hat
		source_coords_map: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,  # compact(y,x) -> source(y,x)
		plot_raw: bool = True,
		plot_opening: bool = False,
		plot_erosion: bool = False,
		plot_dilation: bool = False,
		plot_top_hat: bool = True,
		plot_corrected_spectra: bool = True,
		initial_checked: Optional[Dict[str, bool]] = None,
		corrected_spectra: Optional[np.ndarray] = None,
		map_central_mass: float = 0.95,
		highlight_detected_pixels: bool = True,
		hover_fps: float = 10.0,
) -> None:
	def _merge_duplicate_segments(segs: List[SpikeSegment]) -> List[SpikeSegment]:
		if not segs:
			return []
		sorted_segs = sorted(segs, key=lambda s: (int(s.start), int(s.end), int(s.peak_index)))
		out: List[SpikeSegment] = []
		for s in sorted_segs:
			if not out:
				out.append(s)
				continue
			last = out[-1]
			overlap_or_adjacent = max(int(last.start), int(s.start)) <= (min(int(last.end), int(s.end)) + 1)
			same_peak_family = abs(int(last.peak_index) - int(s.peak_index)) <= 2
			if overlap_or_adjacent and same_peak_family:
				new_start = min(int(last.start), int(s.start))
				new_end = max(int(last.end), int(s.end))
				if float(s.peak_height) >= float(last.peak_height):
					best_peak = int(s.peak_index)
					best_height = float(s.peak_height)
				else:
					best_peak = int(last.peak_index)
					best_height = float(last.peak_height)
				out[-1] = SpikeSegment(
					y=int(last.y),
					x=int(last.x),
					peak_index=best_peak,
					start=new_start,
					end=new_end,
					peak_height=best_height,
					area=float(last.area) + float(s.area)
				)
			else:
				out.append(s)
		return out

	merged_spikes_by_pixel: Dict[Tuple[int, int], List[SpikeSegment]] = {
		pix: _merge_duplicate_segments(segs)
		for pix, segs in spikes_by_pixel.items()
	}

	H, W, N = spectra.shape

	fig, (ax_map, ax_spec) = plt.subplots(1, 2, figsize=(13, 6))
	plt.subplots_adjust(bottom=0.30)
	fig.canvas.manager.set_window_title("Muon finder - hover viewer")

	v = score_map.astype(float)
	v = v[np.isfinite(v)]
	if v.size and 0.0 < map_central_mass < 1.0:
		tail = 0.5 * (1.0 - float(map_central_mass))
		vmin = float(np.quantile(v, tail))
		vmax = float(np.quantile(v, 1.0 - tail))
		if not np.isfinite(vmin) or not np.isfinite(vmax):
			vmin = None
			vmax = None
	else:
		vmin = None
		vmax = None

	im = ax_map.imshow(score_map, origin="upper", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
	ax_map.set_title("score map (hover)")
	ax_map.set_xlabel("x (pixel)")
	ax_map.set_ylabel("y (pixel)")
	cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
	cbar.set_label("z (score intensity)")

	# candidates' overlay (visually)
	# ys, xs = np.where(candidate_mask)
	# ax_map.scatter(xs, ys, s=8, marker="o", linewidths=0.5, facecolors="none")

	# # detected spikes overlay
	# if highlight_detected_pixels and merged_spikes_by_pixel:
	# 	sp_y = []
	# 	sp_x = []
	# 	for (py, px), segs in merged_spikes_by_pixel.items():
	# 		if not segs:
	# 			continue
	# 		sp_y.append(py)
	# 		sp_x.append(px)
	# 	if sp_x:
	# 		ax_map.scatter(
	# 			sp_x, sp_y,
	# 			s=42,
	# 			marker="s",
	# 			facecolors="none",
	# 			edgecolors="red",
	# 			linewidth=1.2
	# 		)

	# marker for actual pixel
	marker = ax_map.scatter([0], [0], s=80, marker="s", facecolors="none", edgecolors="white", linewidths=2)

	# stable line colors + checkbox-driven visibility
	line_colors = {
		"raw": "#1f77b4",
		"opening": "#9467bd",
		"erosion": "#7f7f7f",
		"dilation": "#8c564b",
		"top_hat": "#ff7f0e",
		"gradient": "#e377c2",
		"dilation_minus_opening": "#bcbd22",
		"corrected": "#2ca02c",
	}
	checked = {
		"raw": True,
		"opening": False,
		"erosion": False,
		"dilation": False,
		"top_hat": True,
		"gradient": True,
		"dilation_minus_opening": True,
		"corrected": False,
	}
	if initial_checked:
		for k, v in initial_checked.items():
			if k in checked:
				checked[k] = bool(v)

	# spectra lines
	(ln_raw,) = ax_spec.plot([], [], label="raw", color=line_colors['raw'])
	(ln_open,) = ax_spec.plot([], [], label="opening", color=line_colors['opening'])
	(ln_ero,) = ax_spec.plot([], [], label="erosion", color=line_colors['erosion'])
	(ln_dil,) = ax_spec.plot([], [], label="dilation", color=line_colors['dilation'])
	(ln_th,) = ax_spec.plot([], [], label="top_hat", color=line_colors['top_hat'])
	(ln_grad,) = ax_spec.plot([], [], label="gradient", color=line_colors['gradient'])
	(ln_dmo,) = ax_spec.plot([], [], label="dilation_minus_opening", color=line_colors['dilation_minus_opening'])
	(ln_corr,) = ax_spec.plot([], [], label="corrected", color=line_colors['corrected'])
	lines = {
		"raw": ln_raw,
		"opening": ln_open,
		"erosion": ln_ero,
		"dilation": ln_dil,
		"top_hat": ln_th,
		"gradient": ln_grad,
		"dilation_minus_opening": ln_dmo,
		"corrected": ln_corr,
	}

	spike_peak_lines: List = []
	spike_edge_lines: List = []
	spike_bands: List = []
	spike_overlay_checked = {
		"spike_peaks": True,
		"spike_edges": True,
		"spike_bands": True,
	}

	ax_spec.set_title("spectrum @ (y,x)")
	ax_spec.set_xlabel("Raman shift/cm$^{-1}$")
	ax_spec.set_ylabel("Intensity")
	ax_spec.legend(loc="upper right")

	frozen = {"state": False}  # right click = freeze/unfreeze
	current = {"y": 0, "x": 0}
	focus = {"which": "x", "replace_x": False, "replace_y": False}
	hover_state = {"last_t": 0.0, "last_xy": (-1, -1)}

	def _refresh_legend() -> None:
		handles = []
		labels = []
		for nm, ln in lines.items():
			if ln.get_visible():
				handles.append(ln)
				labels.append(nm)
		leg = ax_spec.get_legend()
		if leg is not None:
			leg.remove()
		if handles:
			ax_spec.legend(handles, labels, loc="best")

	def _set_focus(which: str) -> None:
		focus['which'] = which
		txt_x.set_active(which == "x")
		txt_y.set_active(which == "y")

		ax_txt_x.set_facecolor("#ffffff" if which == "x" else "#f0f0f0")
		ax_txt_y.set_facecolor("#ffffff" if which == "y" else "#f0f0f0")
		ax_btn_go.set_facecolor("#dfefff" if which == "go" else "#f0f0f0")
		fig.canvas.draw_idle()

	def _update(y: int, x: int) -> None:
		y = int(np.clip(y, 0, H - 1))
		x = int(np.clip(x, 0, W - 1))
		current["y"] = y
		current["x"] = x

		marker.set_offsets([[x, y]])

		raw = spectra[y, x, :]
		ln_raw.set_data(x_axis, raw)
		ln_open.set_data(x_axis, overlays['opening'][y, x, :])
		ln_ero.set_data(x_axis, overlays['erosion'][y, x, :])
		ln_dil.set_data(x_axis, overlays['dilation'][y, x, :])
		ln_th.set_data(x_axis, overlays['top_hat'][y, x, :])
		if "gradient" in overlays:
			ln_grad.set_data(x_axis, overlays["gradient"][y, x, :])
		else:
			ln_grad.set_data([], [])
		if "dilation_minus_opening" in overlays:
			ln_dmo.set_data(x_axis, overlays["dilation_minus_opening"][y, x, :])
		else:
			ln_dmo.set_data([], [])
		if corrected_spectra is not None:
			ln_corr.set_data(x_axis, corrected_spectra[y, x, :])
		else:
			ln_corr.set_data([], [])

		for nm, ln in lines.items():
			if nm == "corrected" and corrected_spectra is None:
				ln.set_visible(False)
			elif nm == "gradient" and "gradient" not in overlays:
				ln.set_visible(False)
			elif nm == "dilation_minus_opening" and "dilation_minus_opening" not in overlays:
				ln.set_visible(False)
			else:
				ln.set_visible(bool(checked[nm]))

		_refresh_legend()

		# clear old spike overlays
		for col in (spike_peak_lines, spike_edge_lines, spike_bands):
			for artist in col:
				try:
					artist.remove()
				except Exception:
					pass
			col.clear()

		# add spike markers (deduplicated by start/end)
		segs = merged_spikes_by_pixel.get((y,x), [])
		for s in segs:
			pi = int(np.clip(s.peak_index, 0, len(x_axis) - 1))
			si = int(np.clip(s.start, 0, len(x_axis) - 1))
			ei = int(np.clip(s.end, 0, len(x_axis) - 1))
			x_peak = x_axis[pi]
			x_start = x_axis[si]
			x_end = x_axis[ei]

			if spike_overlay_checked['spike_bands']:
				x0 = min(float(x_start), float(x_end))
				x1 = max(float(x_start), float(x_end))
				band = ax_spec.axvspan(x0, x1, color="green", alpha=0.12, zorder=0)
				spike_bands.append(band)

			if spike_overlay_checked['spike_edges']:
				le = ax_spec.axvline(x_start, linestyle="--", linewidth=1, color="green", alpha=0.9)
				re = ax_spec.axvline(x_end, linestyle="--", linewidth=1, color="green", alpha=0.9)
				spike_edge_lines.extend([le, re])

			if spike_overlay_checked['spike_peaks']:
				lp = ax_spec.axvline(x_peak, linestyle="--", linewidth=1, color="red", alpha=0.9)
				spike_peak_lines.append(lp)

		# title of the plot
		if source_coords_map is not None:
			src_y, src_x = source_coords_map.get((y, x), (y, x))
			ax_spec.set_title(
				f"spectrum @ compact(y={y}, x={x}) -> source (y={src_y}, x={src_x}) | spikes={len(segs)}"
			)
		else:
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

		if (x, y) == hover_state["last_xy"]:
			return
		now = time.perf_counter()
		if hover_fps > 0:
			min_dt = 1.0 / float(hover_fps)
			if (now - hover_state['last_t']) < min_dt:
				return
		hover_state['last_t'] = now
		hover_state['last_xy'] = (x, y)
		_update(y, x)

	def on_click(event) -> None:
		if event.inaxes == ax_txt_x:
			_set_focus("x")
			if getattr(event, "dblclick", False):
				focus['replace_x'] = True
			return
		if event.inaxes == ax_txt_y:
			_set_focus("y")
			if getattr(event, "dblclick", False):
				focus['replace_y'] = True
			return
		if event.inaxes == ax_btn_go:
			_set_focus("go")
			return

		if event.inaxes != ax_map:
			return
		if event.button != 3:
			return
		frozen["state"] = not frozen["state"]

	def _toggle_line(label: str) -> None:
		if label in checked:
			checked[label] = not checked[label]
		elif label in spike_overlay_checked:
			spike_overlay_checked[label] = not spike_overlay_checked[label]
		_update(current["y"], current["x"])

	ax_txt_y = fig.add_axes((0.12, 0.035, 0.08, 0.045))
	ax_txt_x = fig.add_axes((0.22, 0.035, 0.08, 0.045))
	ax_btn_go = fig.add_axes((0.32, 0.035, 0.08, 0.045))
	ax_chk = fig.add_axes((0.52, 0.02, 0.22, 0.24))
	txt_y = TextBox(ax_txt_y, "y", initial="0")
	txt_x = TextBox(ax_txt_x, "x", initial="0")
	btn_go = Button(ax_btn_go, "Go to (y,x)")

	chk = CheckButtons(
		ax_chk,
		labels=[
			"raw", "opening", "erosion", "dilation", "top_hat", "gradient", "dilation_minus_opening", "corrected",
			"spike_peaks", "spike_edges", "spike_bands"
		],
		actives=[
			checked["raw"], checked["opening"], checked["erosion"], checked["dilation"], checked["top_hat"],
			checked['gradient'], checked['dilation_minus_opening'], checked["corrected"],
			spike_overlay_checked['spike_peaks'], spike_overlay_checked['spike_edges'], spike_overlay_checked['spike_bands']

		],
	)
	chk.on_clicked(_toggle_line)
	ax_chk.set_title("signals")
	for lbl in chk.labels:
		lbl.set_fontsize(9)

	def _go_to_xy(_event=None) -> None:
		try:
			y = int(float(txt_y.text.strip()))
			x = int(float(txt_x.text.strip()))
		except Exception:
			return

		x = int(np.clip(x, 0, W - 1))
		y = int(np.clip(y, 0, H - 1))
		frozen["state"] = True
		txt_x.set_val(str(x))
		txt_y.set_val(str(y))
		_update(y, x)

	def on_key(event) -> None:
		key = (event.key or "").lower()
		if key == "tab":
			order = ["y", "x", "go"]
			i = order.index(focus['which']) if focus['which'] in order else 0
			_set_focus(order[(i + 1) % len(order)])
			return
		if key == "ctrl+a":
			if focus['which'] == "x":
				focus['replace_x'] = True
			elif focus['which'] == "y":
				focus['replace_y'] = True
			return
		if key in ("enter", "return") and focus['which'] in ("x", "y", "go"):
			_go_to_xy()
			return

		if focus['which'] == "x" and focus['replace_x'] and len(key) == 1 and key.isprintable():
			txt_x.set_val("")
			focus['replace_x'] = False
		if focus['which'] == "y" and focus['replace_y'] and len(key) == 1 and key.isprintable():
			txt_y.set_val("")
			focus['replace_y'] = False

	def _sync_inputs() -> None:
		txt_x.set_val(str(current["x"]))
		txt_y.set_val(str(current["y"]))

	btn_go.on_clicked(_go_to_xy)
	txt_x.on_submit(lambda _t: _go_to_xy())
	txt_y.on_submit(lambda _t: _go_to_xy())

	fig.canvas.mpl_connect("motion_notify_event", on_move)
	fig.canvas.mpl_connect("button_press_event", on_click)
	fig.canvas.mpl_connect("key_press_event", on_key)

	# init
	_set_focus("x")
	_update(0, 0)
	_sync_inputs()
	plt.show()
