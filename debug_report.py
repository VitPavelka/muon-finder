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
						{
							"peak_index": int(s.peak_index),
							"start": int(s.start),
							"end": int(s.end),
							"peak_height": float(s.peak_height),
							"area": float(s.area),
						}
						for s in segs
					],
				}
			)
		report['per_spectrum'] = per_spec

	return report


def save_debug_report_json(path: Path, report: Dict[str, Any]) -> None:
	path = Path(path)
	path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
