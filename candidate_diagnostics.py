from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from morph1d import dilation_1d, erosion_1d
from muon_pipeline import SpikeSegment


@dataclass
class SmallMorphologyBundle:
	erosion: np.ndarray
	dilation: np.ndarray
	morph_range: np.ndarray
	erosion_contacts: np.ndarray
	dilation_contacts: np.ndarray


def candidate_segment_key(seg: SpikeSegment) -> Tuple[int, int, int, int, int]:
	return (int(seg.y), int(seg.x), int(seg.peak_index), int(seg.start), int(seg.end))


def candidate_segment_id(seg: SpikeSegment) -> str:
	y, x, peak, start, end = candidate_segment_key(seg)
	return f"primary:{y}:{x}:{peak}:{start}:{end}"


def get_or_compute_small_morphology(
		*,
		cache: MutableMapping[Tuple[int, int], SmallMorphologyBundle],
		raw_spectra: np.ndarray,
		y: int,
		x: int,
		window_size: int = 3,
) -> SmallMorphologyBundle:
	key = (int(y), int(x))
	if key in cache:
		return cache[key]
	raw = np.asarray(raw_spectra[int(y), int(x), :], dtype=float)
	raw2 = raw.reshape(1, 1, -1)
	ero = erosion_1d(raw2, int(window_size)).reshape(-1).astype(float)
	dil = dilation_1d(raw2, int(window_size)).reshape(-1).astype(float)
	morph_range = (dil - ero).astype(float)
	eq = raw == ero
	if not np.any(eq):
		eq = np.isclose(raw, ero)
	dq = raw == dil
	if not np.any(dq):
		dq = np.isclose(raw, dil)
	bundle = SmallMorphologyBundle(
		erosion=ero,
		dilation=dil,
		morph_range=morph_range,
		erosion_contacts=np.flatnonzero(eq).astype(int),
		dilation_contacts=np.flatnonzero(dq).astype(int),
	)
	cache[key] = bundle
	return bundle


def _indices_to_spans(indices: np.ndarray) -> List[List[int]]:
	idx = np.asarray(indices, dtype=int)
	if idx.size == 0:
		return []
	spans: List[List[int]] = []
	start = int(idx[0])
	prev = int(idx[0])
	for value in idx[1:]:
		v = int(value)
		if v == prev + 1:
			prev = v
			continue
		spans.append([start, prev])
		start = v
		prev = v
	spans.append([start, prev])
	return spans


def _robust_noise_reference_from_morph_range(morph_range: np.ndarray) -> Tuple[float, np.ndarray, str]:
	x = np.asarray(morph_range, dtype=float)
	valid = np.flatnonzero(np.isfinite(x)).astype(int)
	if valid.size < 9:
		return float("nan"), np.asarray([], dtype=int), "insufficient"
	vals = x[valid]
	q1, q3 = np.percentile(vals, [25.0, 75.0])
	iqr = float(q3 - q1)
	if not np.isfinite(iqr):
		iqr = 0.0
	upper = float(q3 + 1.5 * iqr) if iqr > 1e-12 else float(np.percentile(vals, 80.0))
	lower = float(max(0.0, q1 - 1.5 * iqr))
	keep_mask = (vals >= lower) & (vals <= upper)
	kept = valid[keep_mask]
	if kept.size < 9:
		p80 = float(np.percentile(vals, 80.0))
		keep_mask = vals <= p80
		kept = valid[keep_mask]
	if kept.size < 9:
		return float("nan"), np.asarray([], dtype=int), "insufficient"
	noise_height = float(np.median(x[kept]))
	if not np.isfinite(noise_height) or noise_height <= 0.0:
		return float("nan"), np.asarray([], dtype=int), "insufficient"
	return noise_height, kept, "ok"


def _nearest_contact_feet(contacts: np.ndarray, apex: int) -> Tuple[Optional[int], Optional[int]]:
	idx = np.asarray(contacts, dtype=int)
	if idx.size == 0:
		return None, None
	left = idx[idx < int(apex)]
	right = idx[idx > int(apex)]
	left_foot = int(left[-1]) if left.size else None
	right_foot = int(right[0]) if right.size else None
	return left_foot, right_foot


def _height_above_chord(raw: np.ndarray, apex: int, left_foot: int, right_foot: int) -> Tuple[float, float]:
	if right_foot <= left_foot:
		return float("nan"), float("nan")
	t = float((int(apex) - int(left_foot)) / max(int(right_foot) - int(left_foot), 1))
	chord_y = float((1.0 - t) * float(raw[int(left_foot)]) + t * float(raw[int(right_foot)]))
	return float(raw[int(apex)] - chord_y), float(chord_y)


def evaluate_candidate_noise_prefilter(
		*,
		y: int,
		x: int,
		segs: Sequence[SpikeSegment],
		raw_signal: np.ndarray,
		small_morphology: SmallMorphologyBundle,
		enabled: bool,
		mode: str,
		height_factor: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
	mode_name = str(mode).strip().lower()
	if mode_name not in {"morph_range_chord"}:
		mode_name = "morph_range_chord"
	raw = np.asarray(raw_signal, dtype=float)
	rows: List[Dict[str, Any]] = []
	noise_height, kept_idx, ref_status = _robust_noise_reference_from_morph_range(small_morphology.morph_range)
	noise_spans = _indices_to_spans(kept_idx)
	if not segs:
		return rows, {
			"y": int(y),
			"x": int(x),
			"n_candidates_before_noise_prefilter": 0,
			"n_candidates_after_noise_prefilter": 0,
			"n_candidates_rejected_by_noise_prefilter": 0,
			"noise_prefilter_mode_used": mode_name,
			"noise_height_morph_range": float(noise_height) if np.isfinite(noise_height) else np.nan,
			"noise_reference_n_points": int(kept_idx.size),
			"noise_reference_status": str(ref_status),
			"noise_reference_method": "morph_range_w3",
			"noise_reference_spans": noise_spans,
		}
	height_thr = float(max(height_factor, 0.0)) * float(noise_height) if np.isfinite(noise_height) else float("nan")
	for seg in segs:
		apex = int(seg.peak_index)
		left_foot, right_foot = _nearest_contact_feet(small_morphology.erosion_contacts, apex)
		status = "kept"
		reason = "prefilter_disabled"
		height_above_chord = float("nan")
		chord_y_at_apex = float("nan")
		height_ratio = float("nan")
		if not bool(enabled):
			status = "kept"
			reason = "prefilter_disabled"
		elif ref_status != "ok" or not np.isfinite(noise_height):
			status = "not_evaluated"
			reason = "noise_reference_insufficient"
		elif left_foot is None or right_foot is None:
			status = "not_evaluated"
			reason = "missing_erosion_contact_feet"
		else:
			height_above_chord, chord_y_at_apex = _height_above_chord(raw, apex, int(left_foot), int(right_foot))
			if not np.isfinite(height_above_chord):
				status = "not_evaluated"
				reason = "invalid_height_above_chord"
			else:
				if np.isfinite(noise_height) and noise_height > 0.0:
					height_ratio = float(height_above_chord / noise_height)
				if np.isfinite(height_thr) and height_thr >= 0.0:
					if height_above_chord < float(height_thr):
						status = "rejected_noise"
						reason = "height_below_morph_range_threshold"
					else:
						status = "kept"
						reason = "height_above_morph_range_threshold"
				else:
					status = "not_evaluated"
					reason = "invalid_height_threshold"
		rows.append(
			{
				"candidate_id": candidate_segment_id(seg),
				"y": int(seg.y),
				"x": int(seg.x),
				"peak_index": int(seg.peak_index),
				"start": int(seg.start),
				"end": int(seg.end),
				"peak_height": float(seg.peak_height),
				"area": float(seg.area),
				"candidate_noise_prefilter_status": str(status),
				"candidate_noise_prefilter_mode": mode_name,
				"candidate_noise_prefilter_reason": str(reason),
				"candidate_noise_chord_y_at_apex": float(chord_y_at_apex) if np.isfinite(chord_y_at_apex) else np.nan,
				"candidate_noise_height_above_chord": float(height_above_chord) if np.isfinite(height_above_chord) else np.nan,
				"candidate_noise_height_threshold": float(height_thr) if np.isfinite(height_thr) else np.nan,
				"candidate_noise_height_factor": float(height_factor),
				"candidate_noise_height_ratio": float(height_ratio) if np.isfinite(height_ratio) else np.nan,
				"candidate_noise_left_foot": np.nan if left_foot is None else int(left_foot),
				"candidate_noise_right_foot": np.nan if right_foot is None else int(right_foot),
				"candidate_noise_apex": int(apex),
				"candidate_noise_estimate_used": float(noise_height) if np.isfinite(noise_height) else np.nan,
				"noise_height_morph_range": float(noise_height) if np.isfinite(noise_height) else np.nan,
				"noise_reference_n_points": int(kept_idx.size),
				"noise_reference_status": str(ref_status),
				"noise_reference_method": "morph_range_w3",
				"noise_reference_spans": noise_spans,
			}
		)
	rejected_n = int(sum(1 for row in rows if str(row.get("candidate_noise_prefilter_status")) == "rejected_noise"))
	kept_n = int(sum(1 for row in rows if str(row.get("candidate_noise_prefilter_status")) == "kept"))
	return rows, {
		"y": int(y),
		"x": int(x),
		"n_candidates_before_noise_prefilter": int(len(rows)),
		"n_candidates_after_noise_prefilter": int(kept_n),
		"n_candidates_rejected_by_noise_prefilter": int(rejected_n),
		"noise_prefilter_mode_used": mode_name,
		"noise_height_morph_range": float(noise_height) if np.isfinite(noise_height) else np.nan,
		"noise_reference_n_points": int(kept_idx.size),
		"noise_reference_status": str(ref_status),
		"noise_reference_method": "morph_range_w3",
		"noise_reference_spans": noise_spans,
	}


def apply_global_metric_ranks(rows: Iterable[Mapping[str, Any]]) -> None:
	row_list = [row for row in rows if isinstance(row, Mapping)]

	def _assign(metric_key: str, out_key: str, *, spike_oriented_negative: bool = False) -> None:
		vals: List[Tuple[int, float]] = []
		for idx, row in enumerate(row_list):
			try:
				v = float(row.get(metric_key, np.nan))
			except Exception:
				v = np.nan
			if np.isfinite(v):
				vals.append((idx, -v if spike_oriented_negative else v))
		if not vals:
			return
		order = sorted(vals, key=lambda item: item[1])
		den = max(len(order) - 1, 1)
		for rank_idx, (row_idx, _) in enumerate(order):
			row = row_list[row_idx]
			if isinstance(row, dict):
				row[out_key] = float(rank_idx / den)

	for row in row_list:
		if isinstance(row, dict):
			row["ss1_global_rank"] = np.nan
			row["pce_global_rank"] = np.nan
			row["edge_global_rank"] = np.nan
			row["edge_global_spike_rank"] = np.nan
	_assign("spike_score_v1", "ss1_global_rank")
	_assign("pce_negpref_t098_evidence_signed", "pce_global_rank")
	_assign("recdw_sum_0_90_raman_veto_evidence_signed", "edge_global_rank")
	_assign("recdw_sum_0_90_raman_veto_evidence_signed", "edge_global_spike_rank", spike_oriented_negative=True)
