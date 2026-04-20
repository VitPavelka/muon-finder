# muon_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from morph1d import erosion_1d, dilation_1d, opening_1d


def _split_group_into_peak_regions(indices: np.ndarray, signal: np.ndarray) -> List[Tuple[int, int, int]]:
	"""
	Split one contiguous supra-threshold group into one or more peak-centered regions.

	Returns list of (left_idx, right_idx, peak_idx), all inclusive and expressed in
	global spectral indices.
	"""
	if indices.size == 0:
		return []

	g = np.asarray(indices, dtype=int)
	vals = np.asarray(signal[g], dtype=float)

	# local maxima inside this contiguous group
	local_pos: List[int] = []
	for i in range(vals.size):
		lv = vals[i - 1] if i > 0 else -np.inf
		rv = vals[i + 1] if i < vals.size - 1 else -np.inf
		if vals[i] >= lv and vals[i] >= rv:
			local_pos.append(i)

	# fallback: at least one peak per group
	if not local_pos:
		p = int(g[int(np.argmax(vals))])
		return [(int(g[0]), int(g[-1]), p)]

	# compress plateau maxima: keep exactly one position per consecutive run
	peak_pos: List[int] = []
	local_sorted = sorted(set(local_pos))
	run: List[int] = [local_sorted[0]]
	for p in local_sorted[1:]:
		if p == run[-1] + 1:
			run.append(p)
			continue
		best = max(run, key=lambda idx: vals[idx])
		peak_pos.append(int(best))
		run = [p]
	best = max(run, key=lambda idx: vals[idx])
	peak_pos.append(int(best))
	peak_idx = [int(g[p]) for p in peak_pos]
	if len(peak_idx) == 1:
		return [(int(g[0]), int(g[-1]), peak_idx[0])]

	# split boundaries at valley (minimum) between neighboring peaks
	splits: List[int] = [int(g[0])]
	for a, b in zip(peak_pos[:-1], peak_pos[1:]):
		if b <= a + 1:
			splits.append(int(g[b]))
			continue
		mid_local = int(np.argmin(vals[a:b + 1]) + a)
		splits.append(int(g[mid_local]))
	splits.append(int(g[-1]))

	regions: List[Tuple[int, int, int]] = []
	for i, p in enumerate(peak_idx):
		left = splits[i]
		right = splits[i + 1]
		regions.append((int(left), int(right), int(p)))
	return regions


def compute_morph_overlays(spectra: np.ndarray, se_size: int) -> Dict[str, np.ndarray]:
	"""
	spectra: (H, W, N)
	returns dict: erosion, dilation, opening, top_hat, gradient
	"""
	eros = erosion_1d(spectra, se_size)
	dila = dilation_1d(spectra, se_size)
	opn = opening_1d(spectra, se_size)
	th = spectra - opn
	th[th < 0] = 0
	grad = dila - eros  # morphological gradient
	grad[grad < 0] = 0
	return {"erosion": eros, "dilation": dila, "opening": opn, "top_hat": th, "gradient": grad}


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
		return float(np.percentile(v, q * 100.0))

	if method == "mad":
		med = float(np.median(v))
		mad = float(np.median(np.abs(v - med)))  # MAD without constant
		return med + k_mad * mad

	raise ValueError("`method` must be: 'quantile' | 'mad'")


def neighbour_filter_ratio(
		top_hat: np.ndarray,  # (H,W,N)
		spikes: List[SpikeSegment],
		radius: int = 1,
		ratio_min: float = 3.0,
) -> list[SpikeSegment]:
	"""
	Reject a spike if any neighbor has comparable content in the same spectral interval.

	For each spike segment:
		1) normalize top_hat spectrum to [0..1] using its own max
		2) define a rectangle over [start:end] with height = max(top_hat_norm[start:end+1])
		3) compute area = width_pts * height
		4) compute the same area for every neighbor pixel in the same [start:end] interval
		5) if ANY neighbor_area >= center_area / ratio_min -> reject (not isolated)
	"""
	H, W, N = top_hat.shape
	kept: List[SpikeSegment] = []

	for s in spikes:
		y, x = int(s.y), int(s.x)
		a = int(s.start)
		b = int(s.end)
		if a < 0 or b >= N or a >= b:
			continue

		# Width in points (consistent across center and neighbors)
		width_pts = b - a  # or (b - a + 1) - doesn't matter much if consistent
		if width_pts <= 0:
			continue

		# Center normalization
		th_c = top_hat[y, x, :].astype(float)
		m_c = float(th_c.max())
		if m_c <= 0:
			continue
		th_c_norm = th_c / m_c

		# Rectangle height in the candidate interval
		h_c = float(np.max(th_c_norm[a:b+1]))
		center_area = width_pts * h_c
		if center_area <= 0:
			continue

		# Reject if ANY neighbor has too large area in the same interval
		reject = False
		y0 = max(0, y - radius); y1 = min(H, y + radius + 1)
		x0 = max(0, x - radius); x1 = min(W, x + radius + 1)

		for yy in range(y0, y1):
			for xx in range(x0, x1):
				if yy == y and xx == x:
					continue

				th_n = top_hat[yy, xx, :].astype(float)
				m_n = float(th_n.max())
				if m_n <= 0:
					continue
				th_n_norm = th_n / m_n

				h_n = float(np.max(th_n_norm[a:b+1]))
				neigh_area = width_pts * h_n

				# If neighbor is not at least ratio_min times smaller, reject
				if neigh_area * ratio_min >= center_area:
					reject = True
					break
			if reject:
				break

		if not reject:
			kept.append(s)

	return kept


@dataclass(frozen=True)
class SpikeSegment:
	y: int
	x: int
	peak_index: int
	start: int  # left anchor
	end: int    # right anchor
	peak_height: float
	area: float


def extract_spikes_for_candidates(
		x_axis: np.ndarray,
		top_hat: np.ndarray,          # (H,W,N)
		candidate_mask: np.ndarray,   # (H,W)
		raw_spectra: np.ndarray,      # (H,W,N)
		max_width_pts: int = 20,      # this is the *removal* width between anchors
		k_mad_pixel: float = 8.0,
		min_peak: float = 0.0,
		baseline_se_size: int = 11,  # larger than detection SE, odd recommended
		edge_k_mad: float = 2.0,
		pad_pts: int = 0,            # widen removal region (optional)
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
	pairs = list(zip(ys.tolist(), xs.tolist()))

	try:
		from tqdm import tqdm  # type: ignore
		it = tqdm(pairs, total=len(pairs), desc="Extracting spikes", unit="px")
	except ImportError:
		it = pairs

	for y, x in it:
		th = top_hat[y, x, :].astype(float)

		# high threshold for detection
		med = float(np.median(th))
		mad = float(np.median(np.abs(th - med)))

		thr_high = max(med + k_mad_pixel * mad, min_peak)

		idx_hi = np.where(th > thr_high)[0]
		if idx_hi.size == 0:
			continue

		# split into contiguous groups → each group is one spike candidate
		splits = np.where(np.diff(idx_hi) > 1)[0] + 1
		groups = np.split(idx_hi, splits)

		raw = raw_spectra[y, x, :].astype(float)

		# baseline via opening with a *larger* SE
		baseline = opening_1d(raw, se_size=baseline_se_size).astype(float)
		resid = raw - baseline

		res_med = float(np.median(resid))
		res_mad = float(np.median(np.abs(resid - res_med)))
		tol = res_med + edge_k_mad * res_mad

		pix_segments: List[SpikeSegment] = []

		for g in groups:
			regions = _split_group_into_peak_regions(g, th)
			for g0, g1, peak_index in regions:

				# enforce a tol smaller than the spike residual at the peak
				res_peak = float(resid[peak_index])
				tol_eff = min(tol, 0.5 * res_peak)

				# LEFT ANCHOR: search left starting from sub-group start
				left = g0
				while left > 0 and resid[left] > tol_eff:
					left -= 1

				# RIGHT ANCHOR: search right starting from sub-group end
				right = g1
				while right < (raw.size - 1) and resid[right] > tol_eff:
					right += 1

				# anchors must be strictly outside the spike core
				# if they collapse, force them to be at least neighbors if possible
				if left >= peak_index:
					left = peak_index - 1
				if right <= peak_index:
					right = peak_index + 1

				if left < 0 or right >= raw.size:
					continue
				if left >= right - 1:
					continue  # nothing to replace

				# optional widening (still keep anchors outside peak)
				left2 = max(0, left - pad_pts)
				right2 = min(raw.size - 1, right + pad_pts)

				# check max removal width (between anchors)
				removal_width = (right2 - left2 - 1)
				if removal_width > max_width_pts:
					continue

				# compute peak height in top-hat (diagnostics)
				peak_height = float(th[peak_index])

				# integrate spike "excess" (use abs for descending x_axis)
				seg = np.maximum(resid[left2+1:right2], 0.0)
				area = float(abs(np.trapezoid(seg, x_axis[left2+1:right2])))

				pix_segments.append(
					SpikeSegment(
						y=int(y), x=int(x),
						peak_index=peak_index,
						start=int(left2), end=int(right2),
						peak_height=peak_height,
						area=area,
					)
				)

		if pix_segments:
			spikes.extend(pix_segments)
			by_pix[(int(y), int(x))] = pix_segments

	return spikes, by_pix
