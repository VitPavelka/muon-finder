# debug_report.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Literal

import numpy as np

from muon_pipeline import SpikeSegment
from spike_merge import merge_spike_segments
from feature_window import (
	expand_interval_to_signal_foot,
	enforce_shared_boundaries_by_minima,
)


def build_debug_report(
		score_map: np.ndarray,
		candidate_mask: np.ndarray,
		spikes_by_pixel: Dict[Tuple[int, int], List[SpikeSegment]],
		threshold: float,
		target_coords: Optional[List[Tuple[int, int]]] = None,
		include_per_spectrum: bool = False,
		max_top_pixels: int = 25,
		raw_spectra: Optional[np.ndarray] = None,  # (H,W,N)
		overlays: Optional[Dict[str, np.ndarray]] = None,  # expects at least 'gradient' if available
		x_axis: Optional[np.ndarray] = None,
		feature_signal_source: str = "gradient",  # gradient | raw
		merge_duplicate_segments: bool = False,
		feature_expand_to_gradient_foot: bool = True,
		feature_foot_k_mad: float = 2.0,
		feature_foot_min_run: int = 2,
		feature_window_method: Literal["mad_run", "erosion_touch"] = "mad_run",
		feature_erosion_se_size: int = 5,
		boundary_minimum_source: Literal["raw", "gradient"] = "gradient",
) -> Dict[str, Any]:
	def _merge_duplicate_segments(segs: List[SpikeSegment]) -> List[SpikeSegment]:
		return merge_spike_segments(segs, merge_adjacent=True, peak_distance_max=None)

	if bool(merge_duplicate_segments):
		merged_spikes_by_pixel: Dict[Tuple[int, int], List[SpikeSegment]] = {
			pix: _merge_duplicate_segments(segs)
			for pix, segs in spikes_by_pixel.items()
		}
	else:
		merged_spikes_by_pixel: Dict[Tuple[int, int], List[SpikeSegment]] = {
			pix: list(segs)
			for pix, segs in spikes_by_pixel.items()
		}
	raw_n_spikes_total = int(sum(len(segs) for segs in spikes_by_pixel.values()))

	finite = score_map[np.isfinite(score_map)]
	if finite.size == 0:
		score_stats = {"min": None, "max": None, "median": None, "p95": None}
	else:
		score_stats = {
			"min": float(np.min(finite)),
			"max": float(np.max(finite)),
			"median": float(np.median(finite)),
			"p95": float(np.percentile(finite, 95.0)),
		}

	all_spikes = [s for segs in merged_spikes_by_pixel.values() for s in segs]
	pixel_rows = []
	for (y, x), segs in merged_spikes_by_pixel.items():
		if not segs:
			continue
		pixel_rows.append(
			{
				"y": int(y),
				"x": int(x),
				"n_spikes": int(len(segs)),
				"max_peak_height": float(max(s.peak_height for s in segs)),
				"sum_area": float(sum(s.area for s in segs)),
			}
		)
	pixel_rows.sort(key=lambda r: (r['n_spikes'], r['max_peak_height']), reverse=True)
	top_pixels = pixel_rows[: int(max_top_pixels)]

	report: Dict[str, Any] = {
		"shape_hw": [int(score_map.shape[0]), int(score_map.shape[1])],
		"threshold": float(threshold),
		"score_stats": score_stats,
		"n_candidates": int(np.count_nonzero(candidate_mask)),
		"n_pixels_with_spikes": int(sum(1 for segs in merged_spikes_by_pixel.values() if segs)),
		"n_spikes_total": int(len(all_spikes)),
		"n_spikes_total_raw": raw_n_spikes_total,
		"top_pixels": top_pixels,
	}

	def _muon_score(feat: Dict[str, Any]) -> Dict[str, Any]:
		# bounded transforms to avoid domination by any single large feature
		rise = float(feat.get("rise_slope", 0.0))
		fall = float(feat.get("fall_slope", 0.0))
		gabs = float(feat.get("gradient_max", 0.0))
		gz = float(feat.get("gradient_max_z", 0.0))
		pw = float(feat.get("plateau_width_90", 0.0))
		asym = abs(float(feat.get("edge_asymmetry", 1.0)))

		s_grad = float(np.tanh(gz / 6.0))
		s_grad_abs = float(np.tanh(gabs / 2000.0))
		s_rise_abs = float(np.tanh(max(rise, 0.0) / 1200.0))
		s_fall_abs = float(np.tanh(abs(min(fall, 0.0)) / 1200.0))
		s_plateau = float(np.exp(-((pw - 3.0) / 4.0) ** 2))
		s_asym = float(np.exp(-asym / 0.5))

		weights = {
			"s_rise_abs": 0.4,
			"s_fall_abs": 0.4,
			"s_grad_abs": 0.15,
			"s_grad": 0.03,
			"s_plateau": 0.01,
			"s_asym": 0.01
		}
		score = (
			weights['s_rise_abs'] * s_rise_abs
			+ weights['s_fall_abs'] * s_fall_abs
			+ weights['s_grad_abs'] * s_grad_abs
			+ weights['s_grad'] * s_grad
			+ weights['s_plateau'] * s_plateau
			+ weights['s_asym'] * s_asym
		)
		return {
			"muon_score": float(score),
			"muon_score_components": {
				"s_rise_abs": s_rise_abs,
				"s_fall_abs": s_fall_abs,
				"s_grad_abs": s_grad_abs,
				"s_grad": s_grad,
				"s_plateau": s_plateau,
				"s_asym": s_asym,
				"weights": weights,
			},
		}

	def _spike_features(
			y: int, x: int, s: SpikeSegment,
			window_override: Optional[Tuple[int, int]] = None,
	) -> Dict[str, Any]:
		out: Dict[str, Any] = {
			"width_pts": int(max(0, s.end - s.start - 1)),
			"peak_index": int(s.peak_index),
			"start": int(s.start),
			"end": int(s.end),
			"peak_height": float(s.peak_height),
			"area": float(s.area),
		}
		if raw_spectra is not None:
			out['raw_peak_value'] = float(raw_spectra[int(y), int(x), int(s.peak_index)])

		src = str(feature_signal_source).lower().strip()
		if src == "raw":
			if raw_spectra is None:
				return out
			sig_spec = raw_spectra[int(y), int(x), :].astype(float)
		else:
			# Primary features are computed from morphological gradient.
			if overlays is None or "gradient" not in overlays:
				return out
			sig_spec = overlays['gradient'][int(y), int(x), :].astype(float)
			src = "gradient"

		n = sig_spec.size
		a = int(np.clip(s.start, 0, n - 1))
		b = int(np.clip(s.end, 0, n - 1))
		p = int(np.clip(s.peak_index, 0, n - 1))
		if not (a < p < b):
			return out

		if window_override is None:
			a, b = expand_interval_to_signal_foot(
				sig=sig_spec,
				left=a,
				right=b,
				peak=p,
				enabled=bool(feature_expand_to_gradient_foot),
				k_mad=float(feature_foot_k_mad),
				min_run=int(feature_foot_min_run),
				method=feature_window_method,
				erosion_se_size=int(feature_erosion_se_size),
			)
		else:
			a = int(window_override[0])
			b = int(window_override[1])

		segment = sig_spec[a:b + 1]
		if segment.size < 3:
			return out

		d = np.diff(segment)
		rise = d[: max(1, p - a)]
		fall = d[max(1, p - a) :]
		rise_slope = float(np.max(rise)) if rise.size else 0.0
		fall_slope = float(np.min(fall)) if fall.size else 0.0
		rise_total_change = float(np.sum(np.maximum(rise, 0.0))) if rise.size else 0.0
		fall_total_change = float(np.sum(np.maximum(-fall, 0.0))) if fall.size else 0.0

		context = max(10, 3 * max(1, b - a + 1))
		l0 = max(0, a - context)
		r1 = min(n, b + context + 1)
		bg = np.concatenate([sig_spec[l0:a], sig_spec[b + 1:r1]])
		if bg.size < 5:
			bg = np.concatenate([sig_spec[:a], sig_spec[b + 1:]])
		if bg.size < 5:
			bg = sig_spec
		bg_med = float(np.median(bg))
		bg_mad = float(np.median(np.abs(bg - bg_med)))
		bg_mad = max(bg_mad, 1e-12)
		out['rise_slope'] = rise_slope
		out['fall_slope'] = fall_slope
		out['rise_total_change'] = rise_total_change
		out['fall_total_change'] = fall_total_change
		out['rise_slope_z'] = float(rise_slope / bg_mad)
		out['fall_slope_z'] = float(abs(fall_slope / bg_mad))
		out['rise_total_change_z'] = float(rise_total_change / bg_mad)
		out['fall_total_change_z'] = float(fall_total_change / bg_mad)
		area_z = float(out.get("area", 0.0) / (bg_mad * max(1.0, float(out.get("width_pts", 1)))))
		out['area_z'] = area_z
		out['noise_mad_signal'] = float(bg_mad)
		# Backward-compatible key kept for existing tooling
		out['noise_mad_gradient'] = float(bg_mad)

		peak_val = float(sig_spec[p])
		lvl = 0.9 * peak_val
		plateau = int(np.count_nonzero(segment >= lvl))
		out['plateau_width_90'] = plateau
		out['edge_asymmetry'] = float(
			(abs(rise_slope) - abs(fall_slope)) / (abs(rise_slope) + abs(fall_slope) + 1e-12)
		)
		out['edge_asymmetry_total_change'] = float(
			(rise_total_change - fall_total_change) / (rise_total_change + fall_total_change + 1e-12)
		)

		out['gradient_max'] = float(np.max(segment))
		out['gradient_max_z'] = float(np.max(segment) / bg_mad)
		# Additional morphology-like descriptors around the selected peak:
		# 1) local curvature at peak (2nd central difference)
		if 0 < p < (n - 1):
			curv = float(sig_spec[p - 1] - 2.0 * sig_spec[p] + sig_spec[p + 1])
		else:
			curv = 0.0
		out['peak_curvature'] = curv
		out['peak_curvature_z'] = float(curv / bg_mad)

		# 2) prominence on selected signal (vs local minima around the peak)
		left_min = float(np.min(sig_spec[a:p + 1])) if p >= a else float(sig_spec[p])
		right_min = float(np.min(sig_spec[p:b + 1])) if b >= p else float(sig_spec[p])
		prom = float(sig_spec[p] - max(left_min, right_min))
		out['prominence_local'] = prom
		out['prominence_local_z'] = float(prom / bg_mad)

		half = 0.5 * peak_val
		above_half = np.where(segment >= half)[0]
		if above_half.size:
			fwhm_pts = int(above_half[-1] - above_half[0] + 1)
		else:
			fwhm_pts = 0
		out['fwhm_pts'] = fwhm_pts
		if x_axis is not None and 0 <= p < x_axis.size and above_half.size:
			l_ix = int(a + above_half[0])
			r_ix = int(a + above_half[-1])
			out['fwhm_cm1'] = float(abs(float(x_axis[r_ix]) - float(x_axis[l_ix])))
		else:
			out['fwhm_cm1'] = float("nan")

		top_mask = segment >= lvl
		top_vals = segment[top_mask]
		out['top_points_n'] = int(top_vals.size)
		if top_vals.size >= 2:
			top_std = float(np.std(top_vals))
			top_med = float(np.median(top_vals))
			top_mad = float(np.median(np.abs(top_vals - top_med)))
		else:
			top_std = 0.0
			top_mad = 0.0
		out['top_fluct_std'] = top_std
		out['top_fluct_mad'] = top_mad
		out['top_fluck_rel_std'] = float(top_std / (abs(peak_val) + 1e-12))

		# 3) width shape across multiple levels + concentration
		def _width_at_level(frac: float) -> int:
			thr = float(frac * peak_val)
			idx = np.where(segment >= thr)[0]
			return int(idx[-1] + 1) if idx.size else 0

		w30 = int(_width_at_level(0.3))
		w50 = int(_width_at_level(0.5))
		w70 = int(_width_at_level(0.7))
		w90 = int(_width_at_level(0.9))
		out['width_30_pts'] = w30
		out['width_50_pts'] = w50
		out['width_70_pts'] = w70
		out['width_90_pts'] = w90
		out['width_ratio_90_50'] = float(w90 / max(1, w50))
		out['width_ratio_70_50'] = float(w70 / max(1, w50))

		# energy concentration near peak versus full feature interval
		r = max(1, int(round((b - a + 1) * 0.12)))
		lpk = max(a, p - r)
		rpk = min(b, p + r)
		e_full = float(np.sum(np.maximum(segment, 0.0)))
		e_core = float(np.sum(np.maximum(sig_spec[lpk:rpk + 1], 0.0)))
		out['energy_core_ratio'] = float(e_core / max(e_full, 1e-12))

		out['feature_source'] = src
		out['feature_window_start'] = int(a)
		out['feature_window_end'] = int(b)

		# SPIKE SCORES
		def _get_value(key: str, div: float):
			return float(np.tanh(float(out.get(key, 0.0)) / div))

		# Versioned spike score (v1) from empirically strongest features
		sr = _get_value('rise_slope_z', 6.0)
		sf = _get_value('fall_slope_z', 6.0)
		sg = _get_value('gradient_max_z', 6.0)
		sa = _get_value('area_z', 4.0)
		spike_score_v1 = float(
			0.36 * sr
			+ 0.36 * sf
			+ 0.18 * sg
			+ 0.10 * sa
		)
		out['spike_score_v1'] = spike_score_v1
		out['spike_score_v1_components'] = {
			'rise_slope_z_tanh': sr,
			'fall_slope_z_tanh': sf,
			'gradient_max_z_tanh': sg,
			'area_z_tanh': sa,
			'weights': {
				'rise_slope_z': 0.36,
				'fall_slope_z': 0.36,
				'gradient_max_z': 0.18,
				'area_z': 0.10,
			},
		}

		if x_axis is not None and 0 <= p < x_axis.size:
			out['peak_position_cm1'] = float(x_axis[p])
		out.update(_muon_score(out))

		return out

	if include_per_spectrum:
		if target_coords is not None:
			iter_coords = target_coords
		else:
			iter_coords = sorted(merged_spikes_by_pixel.keys(), key=lambda t: (t[0], t[1]))

		per_spec = []
		for y, x in iter_coords:
			segs = merged_spikes_by_pixel.get((int(y), int(x)), [])
			window_map: Dict[int, Tuple[int, int]] = {}
			if segs:
				peaks: List[int] = []
				lefts: List[int] = []
				rights: List[int] = []
				for s in segs:
					if str(feature_signal_source).lower().strip() == "raw" and raw_spectra is not None:
						src_sig = raw_spectra[int(y), int(x), :].astype(float)
					elif overlays is not None and "gradient" in overlays:
						src_sig = overlays['gradient'][int(y), int(x), :].astype(float)
					elif raw_spectra is not None:
						src_sig = raw_spectra[int(y), int(x), :].astype(float)
					else:
						continue
					a0, b0 = expand_interval_to_signal_foot(
						sig=src_sig,
						left=int(s.start),
						right=int(s.end),
						peak=int(s.peak_index),
						enabled=bool(feature_expand_to_gradient_foot),
						k_mad=float(feature_foot_k_mad),
						min_run=int(feature_foot_min_run),
						method=feature_window_method,
						erosion_se_size=int(feature_erosion_se_size)
					)
					peaks.append(int(s.peak_index))
					lefts.append(int(a0))
					rights.append(int(b0))

				bsrc = str(boundary_minimum_source).strip().lower()
				boundary_sig = None
				if bsrc == "gradient" and overlays is not None and "gradient" in overlays:
					boundary_sig = overlays['gradient'][int(y), int(x), :]
				elif raw_spectra is not None:
					boundary_sig = raw_spectra[int(y), int(x), :]
				elif overlays is not None and "gradient" in overlays:
					boundary_sig = overlays['gradient'][int(y), int(x), :]
				if boundary_sig is not None:
					lefts, rights = enforce_shared_boundaries_by_minima(
						peaks=peaks,
						lefts=lefts,
						rights=rights,
						signal=boundary_sig,
					)
				for idx, _s in enumerate(segs):
					window_map[idx] = (int(lefts[idx]), int(rights[idx]))
			per_spec.append(
				{
					"y": int(y),
					"x": int(x),
					"is_candidate": bool(candidate_mask[int(y), int(x)]),
					"score": float(score_map[int(y), int(x)]),
					"n_spikes": int(len(segs)),
					"spikes": [
						_spike_features(int(y), int(x), s, window_override=window_map.get(i))
						for i, s in enumerate(segs)
					],
				}
			)
		report['per_spectrum'] = per_spec

	return report


def save_debug_report_json(path: Path, report: Dict[str, Any]) -> None:
	path = Path(path)
	path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
