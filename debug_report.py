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
		raw_spectra: Optional[np.ndarray] = None,   # (H,W,N)
		overlays: Optional[Dict[str, np.ndarray]] = None,  # expects at least 'gradient' if available
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
	pixel_rows.sort(key=lambda r: (r["n_spikes"], r["max_peak_height"]), reverse=True)
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
			out["raw_peak_value"] = float(raw_spectra[int(y), int(x), int(s.peak_index)])

		# Primary features are computed from morphological gradient.
		if overlays is None or "gradient" not in overlays:
			return out
		grad_spec = overlays["gradient"][int(y), int(x), :].astype(float)
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
		fall = d[max(1, p - a):]
		rise_slope = float(np.max(rise)) if rise.size else 0.0
		fall_slope = float(np.min(fall)) if fall.size else 0.0

		seg_med = float(np.median(segment))
		seg_mad = float(np.median(np.abs(segment - seg_med)))
		seg_mad = max(seg_mad, 1e-12)
		out["rise_slope"] = rise_slope
		out["fall_slope"] = fall_slope
		out["rise_slope_z"] = float(rise_slope / seg_mad)
		out["fall_slope_z"] = float(abs(fall_slope) / seg_mad)

		peak_val = float(grad_spec[p])
		lvl = 0.9 * peak_val
		plateau = int(np.count_nonzero(segment >= lvl))
		out["plateau_width_90"] = plateau
		out["edge_asymmetry"] = float(
			(abs(rise_slope) - abs(fall_slope)) / (abs(rise_slope) + abs(fall_slope) + 1e-12)
		)

		gmed = float(np.median(segment))
		gmad = float(np.median(np.abs(segment - gmed)))
		gmad = max(gmad, 1e-12)
		out["gradient_max"] = float(np.max(segment))
		out["gradient_max_z"] = float(np.max(segment) / gmad)
		out["feature_source"] = "gradient"

		return out

	if include_per_spectrum:
		if target_coords is not None:
			iter_coords = target_coords
		else:
			iter_coords = sorted(spikes_by_pixel.keys(), key=lambda t: (t[0], t[1]))

		per_spec = []
		for y, x in iter_coords:
			segs = spikes_by_pixel.get((int(y), int(x)), [])
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
		report["per_spectrum"] = per_spec

	return report


def save_debug_report_json(path: Path, report: Dict[str, Any]) -> None:
	path = Path(path)
	path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
