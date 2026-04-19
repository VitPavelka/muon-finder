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
		feature_signal_source: str = "gradient",  # gradient | raw
		merge_duplicate_segments: bool = False,
) -> Dict[str, Any]:
	def _merge_duplicate_segments(segs: List[SpikeSegment]) -> List[SpikeSegment]:
		"""
		Merge segments with overlapping intervals (not only exact same start/end).
		Keeps the strongest peak as representative and expands to union interval.
		"""
		if not segs:
			return []
		sorted_segs = sorted(segs, key=lambda s: (int(s.start), int(s.end), int(s.peak_index)))
		out: List[SpikeSegment] = []
		for s in sorted_segs:
			if not out:
				out.append(s)
				continue
			last = out[-1]
			overlap_or_adjacent = max(int(last.start), int(s.start)) <= min(int(last.end), int(s.end) + 1)
			# keep merge conservative by requiring close peaks
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
					area=float(last.area) + float(s.area),
				)
			else:
				out.append(s)
		return out

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

		segment = sig_spec[a:b + 1]
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
		out['rise_slope_z'] = float(rise_slope / bg_mad)
		out['fall_slope_z'] = float(abs(fall_slope / bg_mad))
		out['noise_mad_signal'] = float(bg_mad)
		# Backward-compatible key kept for existin tooling
		out['noise_mad_gradient'] = float(bg_mad)

		peak_val = float(sig_spec[p])
		lvl = 0.9 * peak_val
		plateau = int(np.count_nonzero(segment >= lvl))
		out['plateau_width_90'] = plateau
		out['edge_asymmetry'] = float(
			(abs(rise_slope) - abs(fall_slope)) / (abs(rise_slope) + abs(fall_slope) + 1e-12)
		)

		out['gradient_max'] = float(np.max(segment))
		out['gradient_max_z'] = float(np.max(segment) / bg_mad)
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

		out['feature_source'] = src
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
