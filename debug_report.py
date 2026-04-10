# debug_report.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

from muon_pipeline import SpikeSegment


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
) -> Dict[str, Any]:
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

	all_spikes = [s for segs in spikes_by_pixel.values() for s in segs]
	pixel_rows = []
	for (y, x), segs in spikes_by_pixel.items():
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
		"n_pixels_with_spikes": int(sum(1 for segs in spikes_by_pixel.values() if segs)),
		"n_spikes_total": int(len(all_spikes)),
		"top_pixels": top_pixels,
	}

	def _merge_duplicate_segments(segs: List[SpikeSegment]) -> List[SpikeSegment]:
		"""
		Merge segments sharing identical (start, end) into one representative spike.
		Keeps highes peak_height as representative.
		"""
		if not segs:
			return []
		buckets: Dict[Tuple[int, int], List[SpikeSegment]] = {}
		for s in segs:
			buckets.setdefault((int(s.start), int(s.end)), []).append(s)
		out: List[SpikeSegment] = []
		for (_a, _b), group in buckets.items():
			best = max(group, key=lambda s: float(s.peak_height))
			out.append(best)
		out.sort(key=lambda s: (s.start, s.peak_index, s.end))
		return out

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
			+ weights['s_grad'] + s_grad
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

	def _spike_features(y: int, x: int, s: SpikeSegment) -> Dict[str, Any]:
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

		# Primary features are computed from morphological gradient.
		if overlays is None or "gradient" not in overlays:
			return out
		grad_spec = overlays['gradient'][int(y), int(x), :].astype(float)
		n = grad_spec.size
		a = int(np.clip(s.start, 0, n - 1))
		b = int(np.clip(s.end, 0, n - 1))
		p = int(np.clip(s.peak_index, 0, n - 1))
		if not (a < p < b):
			return out

		segment = grad_spec[a:b + 1]
		if segment.size < 3:
			return out

		d = np.diff(segment)
		rise = d[: max(1, p - a)]
		fall = d[max(1, p - a) :]
		rise_slope = float(np.max(rise)) if rise.size else 0.0
		fall_slope = float(np.min(fall)) if fall.size else 0.0

		context = max(10, 3 * max(1, b - a + 1))
		l0 = max(0, a - context)
		r1 = min(n, b + context + 1)
		bg = np.concatenate([grad_spec[l0:a], grad_spec[b + 1:r1]])
		if bg.size < 5:
			bg = np.concatenate([grad_spec[:a], grad_spec[b + 1:]])
		if bg.size < 5:
			bg = grad_spec
		bg_med = float(np.median(bg))
		bg_mad = float(np.median(np.abs(bg - bg_med)))
		bg_mad = max(bg_mad, 1e-12)
		out['rise_slope'] = rise_slope
		out['fall_slope'] = fall_slope
		out['rise_slope_z'] = float(rise_slope / bg_mad)
		out['fall_slope_z'] = float(abs(fall_slope / bg_mad))
		out['noise_mad_gradient'] = float(bg_mad)

		peak_val = float(grad_spec[p])
		lvl = 0.9 * peak_val
		plateau = int(np.count_nonzero(segment >= lvl))
		out['plateau_width_90'] = plateau
		out['edge_asymmetry'] = float(
			(abs(rise_slope) - abs(fall_slope)) / (abs(rise_slope) + abs(fall_slope) + 1e-12)
		)

		out['gradient_max'] = float(np.max(segment))
		out['gradient_max_z'] = float(np.max(segment) / bg_mad)
		out['feature_source'] = "gradient"
		if x_axis is not None and 0 <= p < x_axis.size:
			out['peak_position_cm1'] = float(x_axis[p])
		out.update(_muon_score(out))

		return out

	if include_per_spectrum:
		if target_coords is not None:
			iter_coords = target_coords
		else:
			iter_coords = sorted(spikes_by_pixel.keys(), key=lambda t: (t[0], t[1]))

		per_spec = []
		for y, x in iter_coords:
			segs = _merge_duplicate_segments(spikes_by_pixel.get((int(y), int(x)), []))
			per_spec.append(
				{
					"y": int(y),
					"x": int(x),
					"is_candidate": bool(candidate_mask[int(y), int(x)]),
					"score": float(score_map[int(y), int(x)]),
					"n_spikes": int(len(segs)),
					"spikes": [
						_spike_features(int(y), int(x), s)
						for s in segs
					],
				}
			)
		report['per_spectrum'] = per_spec

	return report


def save_debug_report_json(path: Path, report: Dict[str, Any]) -> None:
	path = Path(path)
	path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
