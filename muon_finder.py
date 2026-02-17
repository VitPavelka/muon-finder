# muon_finder.py
from __future__ import annotations

from pathlib import Path
from time import time

from wdf_io import load_wdf_map
from muon_pipeline import (
	compute_morph_overlays,
	score_map_from_top_hat,
	threshold_score_map,
	neighbour_filter_ratio,
	extract_spikes_for_candidates,
)
from results_io import save_result_npz, save_spikes_csv
from viewer import show_hover_map


# -------------------------
# CONFIG (temporarily here)
# -------------------------
WDF_PATH = Path("file.wdf")

# Morphology: half-window w => structural element is (2*w + 1)
SE_HALF_WINDOW = 3

# Score map from top-hat
SCORE_MODE = "max"  # "max" | "sum" | "l2"

# Thresholding candidates
THRESH_METHOD = "quantile"  # "quantile" | "mad"
THRESH_QUANTILE = 0.999
THRESH_K_MAD = 20.0         # thr = median + k*MAD
THRESH_MIN_ABS = None       # absolute threshold 200.0

# Simple local filter on map (score must be significantly higher than in the aroundness)
USE_NEIGHBOUR_FILTER = True
NEIGH_RADIUS = 1   # 1 => 3x3
NEIGH_RATIO_MIN = 3

# Extraction of spike segments in spectrum
SPIKE_MAX_WIDTH_PTS = 5   # max width of segment in spectral points
SPIKE_K_MAD_PIXEL = 8   # thr_pixel = median(top_hat) + k*MAD(top_hat) (on spectrum)
SPIKE_MIN_PEAK = 800      # can be >0

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


def main() -> None:
	# 0) load
	load_stime = time()
	ds = load_wdf_map(WDF_PATH)
	print(f"[load] shape(H,W,N)={ds.spectra.shape}, axis={ds.x_axis.shape}, file={ds.path}")
	print(f"{(load_stime - time()):.2f}s")

	# 1) morphology
	mstime = time()
	overlays = compute_morph_overlays(ds.spectra, half_window=SE_HALF_WINDOW)
	top_hat = overlays["top_hat"]
	print(f"morphology: {(mstime - time()):.2f}s")

	# 2) score map + candidates
	stime = time()
	score = score_map_from_top_hat(top_hat, mode=SCORE_MODE)
	thr = threshold_score_map(
		score,
		method=THRESH_METHOD,
		quantile=THRESH_QUANTILE,
		k_mad=THRESH_K_MAD,
		min_abs=THRESH_MIN_ABS,
	)
	cand = score >= thr

	if USE_NEIGHBOUR_FILTER:
		cand = neighbour_filter_ratio(score, cand, radius=NEIGH_RADIUS, ratio_min=NEIGH_RATIO_MIN)
	print(f"score map: {(stime - time()):.2f}s")

	# 2b) spike segments (narrow bands) for candidate pixels
	etime = time()
	spikes, spikes_by_pixel = extract_spikes_for_candidates(
		x_axis=ds.x_axis,
		top_hat=top_hat,
		candidate_mask=cand,
		max_width_pts=SPIKE_MAX_WIDTH_PTS,
		k_mad_pixel=SPIKE_K_MAD_PIXEL,
		min_peak=SPIKE_MIN_PEAK,
	)

	print(f"[candidates] {int(cand.sum())} pixels above threshold (thr={thr:.6g})")
	print(f"[spikes] extracted segments: {len(spikes)}")
	print(f"{(time()-etime):.2f}s")

	# 5) save
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
		)
		save_spikes_csv(OUT_CSV, spikes)
		print(f"[save] {OUT_NPZ} + {OUT_CSV}")

	# 3) viewer (hover)
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
		)


if __name__ == "__main__":
	main()
