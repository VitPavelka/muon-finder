# muon_finder.py
from __future__ import annotations

import numpy as np

from pathlib import Path

from wdf_io import load_dataset
from muon_pipeline import (
	compute_morph_overlays,
	score_map_from_top_hat,
	threshold_score_map,
	neighbour_filter_ratio,
	extract_spikes_for_candidates,
)
from manual_review import manual_confirm_signals
from despike import apply_despike
from results_io import save_result_npz, save_spikes_csv
from viewer import show_hover_map


# -------------------------
# CONFIG (temporarily here)
# -------------------------
WDF_PATH = Path(r"C:\Users\pavel\Desktop\AVCR\NewRaman\porizek\wdf\petalit_785nm_20xObj_800ctr_1s_100prc.wdf")

# Morphology: half-window w => structural element is (2*w + 1)
SE_SIZE = 5

# Score map from top-hat
SCORE_MODE = "max"  # "max" | "sum" | "l2"

# Thresholding candidates
THRESH_METHOD = "quantile"  # "quantile" | "mad"
THRESH_QUANTILE = 0.1
THRESH_K_MAD = 20.0         # thr = median + k*MAD
THRESH_MIN_ABS = None       # absolute threshold 200.0

# Extraction of spike segments in spectrum
SPIKE_MAX_WIDTH_PTS = 20   # max width of segment in spectral points
SPIKE_K_MAD_PIXEL = 8   # thr_pixel = median(top_hat) + k*MAD(top_hat) (on spectrum)
SPIKE_MIN_PEAK = 800     # can be >0

# Local filter on map
USE_NEIGHBOUR_FILTER = True
NEIGH_RADIUS = 5   # 1 => 3x3
NEIGH_RATIO_MIN = 1

# Manual conformation
MANUAL_CONFORMATION = False

# Despike
BASELINE_SE_SIZE = 11
EDGE_K_MAD = 2.0


# Saving
SAVE = False
SAVE_OVERLAYS = False  # save erosion/dilation/opening/top_hat (larger file)
OUT_NPZ = Path("muon_results.npz")
OUT_CSV = Path("muon_spikes.csv")

# Viewer
SHOW_VIEWER = True
PLOT_RAW = True
PLOT_OPENING = True
PLOT_EROSION = True
PLOT_DILATION = True
PLOT_TOPHAT = True
PLOT_CORRECTED = True


def main() -> None:
	# 0) load
	ds = load_dataset(WDF_PATH)
	print(f"[load] shape(H,W,N)={ds.spectra.shape}, axis={ds.x_axis.shape}, file={ds.path}")

	# 1) morphology
	overlays = compute_morph_overlays(ds.spectra, se_size=SE_SIZE)
	top_hat = overlays["top_hat"]

	# 2) score map + candidates
	score = score_map_from_top_hat(top_hat, mode=SCORE_MODE)
	thr = threshold_score_map(
		score,
		method=THRESH_METHOD,
		quantile=THRESH_QUANTILE,
		k_mad=THRESH_K_MAD,
		min_abs=THRESH_MIN_ABS,
	)
	cand = score >= thr

	# 2b) spike segments (narrow bands) for candidate pixels
	spikes, spikes_by_pixel = extract_spikes_for_candidates(
		x_axis=ds.x_axis,
		top_hat=top_hat,
		candidate_mask=cand,
		raw_spectra=ds.spectra,
		max_width_pts=SPIKE_MAX_WIDTH_PTS,
		k_mad_pixel=SPIKE_K_MAD_PIXEL,
		min_peak=SPIKE_MIN_PEAK,
		baseline_se_size=BASELINE_SE_SIZE,
		edge_k_mad=EDGE_K_MAD
	)
	print(f"[candidates] {int(cand.sum())} pixels above threshold (thr={thr:.6g})")
	print(f"[spikes] extracted segments: {len(spikes)}")

	# 3) comparison with neighbors
	pre_spikes = len(spikes)

	if USE_NEIGHBOUR_FILTER and spikes:
		spikes = neighbour_filter_ratio(
			top_hat=top_hat,
			spikes=spikes,
			radius=NEIGH_RADIUS,
			ratio_min=NEIGH_RATIO_MIN,
		)

		# rebuild spikes_by_pixel + rebuild cand from kept spikes
		spikes_by_pixel = {}
		cand2 = np.zeros_like(cand, dtype=bool)

		for s in spikes:
			spikes_by_pixel.setdefault((s.y, s.x), []).append(s)
			cand2[s.y, s.x] = True

		cand = cand2

		print(f"[neigh] spikes kept: {len(spikes)}/{pre_spikes}, pixels kept: {int(cand.sum())}")
	else:
		print(f"[neigh] skipped, spikes: {pre_spikes}")

	# 4) Manual conformation
	accepted_spikes = spikes
	if MANUAL_CONFORMATION and spikes_by_pixel:
		accepted_spikes = manual_confirm_signals(
			x_axis=ds.x_axis,
			spectra=ds.spectra,
			spikes_by_pixel=spikes_by_pixel,
			overlays=overlays,
		)

	# 5) Despike
	corrected_spectra = apply_despike(
		x_axis=ds.x_axis,
		spectra=ds.spectra,
		accepted_spikes=accepted_spikes,
	)

	# 6) save
	if SAVE:
		overlays_to_save = overlays if SAVE_OVERLAYS else None
		save_result_npz(
			out_path=OUT_NPZ,
			ds=ds,
			score_map=score,
			threshold=thr,
			candidate_mask=cand,
			spikes=spikes,
			overlays=overlays_to_save,
			corrected_spectra=corrected_spectra,
		)
		save_spikes_csv(OUT_CSV, spikes)
		print(f"[save] {OUT_NPZ} + {OUT_CSV}")

	# 7) viewer (hover)
	if SHOW_VIEWER:
		show_hover_map(
			x_axis=ds.x_axis,
			spectra=ds.spectra,
			score_map=score,
			candidate_mask=cand,
			spikes_by_pixel=spikes_by_pixel,
			overlays=overlays,
			plot_raw=PLOT_RAW,
			plot_opening=PLOT_OPENING,
			plot_erosion=PLOT_EROSION,
			plot_dilation=PLOT_DILATION,
			plot_top_hat=PLOT_TOPHAT,
			plot_corrected_spectra=PLOT_CORRECTED,
			corrected_spectra=corrected_spectra,
		)


if __name__ == "__main__":
	main()
