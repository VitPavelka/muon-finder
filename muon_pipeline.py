# muon_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from morph1d import erosion_1d, dilation_1d, opening_1d, top_hat_1d


def compute_morph_overlays(spectra: np.ndarray, half_window: int) -> Dict[str, np.ndarray]:
	"""
	spectra: (H, W, N)
	returns dict: erosion, dilation, opening, top_hat
	"""
	eros = erosion_1d(spectra, half_window)
	dila = dilation_1d(spectra, half_window)
	opn = dilation_1d(eros, half_window)
	th = spectra - opn
	return {"erosion": eros, "dilation": dila, "opening": opn, "top_hat": th}


def score_map_from_top_hat(
		top_hat: np.ndarray,
		mode: str = "max"
) -> np.ndarray:
	"""
	top_hat: (H, W, N) → score_map: (H, W)
	"""
	if mode == "max":
		return top_hat.max(axis=-1)
	if mode == "sum":
		return top_hat.sum(axis=-1)
	if mode == "l2":
		return np.sqrt((top_hat * top_hat).sum(axis=-1))
	raise ValueError("`mode` must be: 'max' | 'sum' | 'l2'")


def threshold_score_map(
		score_map: np.ndarray,
		method: str = "quantile",
		quantile: float = 0.999,
		k_mad: float = 20.0,
		min_abs: Optional[float] = None,
) -> float:
	"""
	Returns a threshold.
	"""
	if min_abs is not None:
		return float(min_abs)

	v = score_map.astype(float).ravel()
	v = v[np.isfinite(v)]

	if v.size == 0:
		return float("inf")

	if method == "quantile":
		q = float(np.clip(quantile, 0.0, 1.0))
		return float(np.percentile(v, q))

	if method == "mad":
		med = float(np.median(v))
		mad = float(np.median(np.abs(v - med)))  # MAD without constant
		return med + k_mad * mad

	raise ValueError("`method` must be: 'quantile' | 'mad'")


def neighbour_filter_ratio(
		score_map: np.ndarray,
		candidate_mask: np.ndarray,
		radius: int = 1,
		ratio_min: float = 5.0,
		eps: float = 1e-12,
) -> np.ndarray:
	"""
	Candidate will pass if score / median in radius >= ratio_min.
	Median in radius is computed by median filter.
	"""
	if radius <= 0:
		return candidate_mask

	# SciPy
	try:
		from scipy.ndimage import median_filter  # type: ignore

		size = 2 * radius + 1
		med = median_filter(score_map, size=(size, size), mode="reflect")
		ratio = score_map / (med + eps)
		return candidate_mask & (ratio >= ratio_min)

	except ImportError:
		print("Consider installing `scipy`:\n"
		      "  pip install scipy")
		pass

	# Fallback
	out = candidate_mask.copy()
	H, W = score_map.shape
	ys, xs = np.where(candidate_mask)
	for y, x in zip(ys, xs):
		y0 = max(0, y - radius)
		y1 = min(H, y + radius + 1)
		x0 = max(0, x - radius)
		x1 = min(W, x + radius + 1)
		patch = score_map[y0:y1, x0:x1].ravel()
		if patch.size <= 1:
			continue
		med = np.median(patch)
		if score_map[y, x] / (med + eps) < ratio_min:
			out[y, x] = False
	return out


@dataclass(frozen=True)
class SpikeSegment:
	y: int
	x: int
	peak_index: int
	start: int
	end: int
	peak_height: float
	area: float


def extract_spikes_for_candidates(
		x_axis: np.ndarray,
		top_hat: np.ndarray,         # (H,W,N)
		candidate_mask: np.ndarray,  # (H,W)
		max_width_pts: int = 4,
		k_mad_pixel: float = 8.0,
		min_peak: float = 0.0,
) -> Tuple[List[SpikeSegment], Dict[Tuple[int, int], List[SpikeSegment]]]:
	"""
	For each candidate pixel finds narrow segments (spikes) in top_hat spectrum.
	Returns:
		- list of all the segments
		- dict mapping (x,y) → segments (for viewer)
	"""
	H, W, N = top_hat.shape
	if candidate_mask.shape != (H, W):
		raise ValueError(f"candidate_mask must have shape (H, W): {candidate_mask.shape} != ({H}, {W})")

	spikes: List[SpikeSegment] = []
	by_pix: Dict[Tuple[int, int], List[SpikeSegment]] = {}

	ys, xs = np.where(candidate_mask)
	for y, x in zip(ys, xs):
		th = top_hat[y, x, :].astype(float)

		med = float(np.median(th))
		mad = float(np.median(np.abs(th - med)))
		thr = med + k_mad_pixel * mad
		if thr < min_peak:
			thr = min_peak

		above = th > thr
		if not np.any(above):
			continue

		# segmentation of True segments
		idx = np.where(above)[0]
		# find segments' borders
		splits = np.where(np.diff(idx) > 1)[0] + 1
		groups = np.split(idx, splits)

		pix_segments: List[SpikeSegment] = []
		for g in groups:
			start = int(g[0])
			end = int(g[-1])
			width = end - start + 1
			if width > max_width_pts:
				continue

			seg = th[start:end + 1]
			peak_rel = int(np.argmax(seg))
			peak_index = start + peak_rel
			peak_height = float(seg[peak_rel])

			# integration over axis (trapezoid)
			area = float(np.trapezoid(seg, x_axis[start:end + 1]))

			s = SpikeSegment(
				y=int(y),
				x=int(x),
				peak_index=peak_index,
				start=start,
				end=end,
				peak_height=peak_height,
				area=area,
			)
			spikes.append(s)
			pix_segments.append(s)

		if pix_segments:
			by_pix[(int(y), int(x))] = pix_segments

	return spikes, by_pix
