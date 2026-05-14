from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .candidates import CandidateSegment
from .morphology import dilation_1d, erosion_1d
from .utils import to_contiguous_spans


@dataclass
class SmallMorphologyBundle:
    erosion: np.ndarray
    dilation: np.ndarray
    morph_range: np.ndarray
    erosion_contacts: np.ndarray
    dilation_contacts: np.ndarray


def get_or_compute_small_morphology(
    cache: dict[tuple[int, int], SmallMorphologyBundle],
    raw_spectra: np.ndarray,
    y: int,
    x: int,
    window_size: int = 3,
) -> SmallMorphologyBundle:
    key = (int(y), int(x))
    if key in cache:
        return cache[key]
    raw = np.asarray(raw_spectra[y, x, :], dtype=float)
    raw2 = raw.reshape(1, 1, -1)
    erosion = np.asarray(erosion_1d(raw2, int(window_size)).reshape(-1), dtype=float)
    dilation = np.asarray(dilation_1d(raw2, int(window_size)).reshape(-1), dtype=float)
    morph_range = np.asarray(dilation - erosion, dtype=float)
    eq = raw == erosion
    if not np.any(eq):
        eq = np.isclose(raw, erosion)
    dq = raw == dilation
    if not np.any(dq):
        dq = np.isclose(raw, dilation)
    bundle = SmallMorphologyBundle(
        erosion=erosion,
        dilation=dilation,
        morph_range=morph_range,
        erosion_contacts=np.flatnonzero(eq).astype(int),
        dilation_contacts=np.flatnonzero(dq).astype(int),
    )
    cache[key] = bundle
    return bundle


def estimate_noise_from_morph_range(morph_range: np.ndarray) -> tuple[float, np.ndarray, str]:
    x = np.asarray(morph_range, dtype=float)
    valid = np.flatnonzero(np.isfinite(x)).astype(int)
    if valid.size < 9:
        return float("nan"), np.asarray([], dtype=int), "insufficient"
    vals = x[valid]
    q1, q3 = np.percentile(vals, [25.0, 75.0])
    iqr = float(q3 - q1)
    upper = float(q3 + 1.5 * iqr) if iqr > 1e-12 else float(np.percentile(vals, 80.0))
    lower = float(max(0.0, q1 - 1.5 * iqr))
    keep = valid[(vals >= lower) & (vals <= upper)]
    if keep.size < 9:
        p80 = float(np.percentile(vals, 80.0))
        keep = valid[vals <= p80]
    if keep.size < 9:
        return float("nan"), np.asarray([], dtype=int), "insufficient"
    noise_height = float(np.median(x[keep]))
    if not np.isfinite(noise_height) or noise_height <= 0.0:
        return float("nan"), np.asarray([], dtype=int), "insufficient"
    return noise_height, keep, "ok"


def _nearest_contact_feet(contacts: np.ndarray, apex: int) -> tuple[int | None, int | None]:
    idx = np.asarray(contacts, dtype=int)
    if idx.size == 0:
        return None, None
    left = idx[idx < int(apex)]
    right = idx[idx > int(apex)]
    left_foot = int(left[-1]) if left.size else None
    right_foot = int(right[0]) if right.size else None
    return left_foot, right_foot


def _height_above_chord(raw: np.ndarray, apex: int, left_foot: int, right_foot: int) -> tuple[float, float]:
    if right_foot <= left_foot:
        return float("nan"), float("nan")
    t = float((int(apex) - int(left_foot)) / max(int(right_foot) - int(left_foot), 1))
    chord_y = float((1.0 - t) * float(raw[int(left_foot)]) + t * float(raw[int(right_foot)]))
    return float(raw[int(apex)] - chord_y), float(chord_y)


def evaluate_candidate_noise_prefilter(
    *,
    segs: list[CandidateSegment],
    raw_signal: np.ndarray,
    small_morphology: SmallMorphologyBundle,
    enabled: bool,
    mode: str,
    height_factor: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = np.asarray(raw_signal, dtype=float)
    rows: list[dict[str, Any]] = []
    noise_height, keep_idx, ref_status = estimate_noise_from_morph_range(small_morphology.morph_range)
    height_threshold = float(max(0.0, float(height_factor)) * noise_height) if np.isfinite(noise_height) else float("nan")
    spans = [[int(a), int(b)] for a, b in to_contiguous_spans([int(v) for v in keep_idx.tolist()])]
    for seg in segs:
        apex = int(seg.peak_index)
        left_foot, right_foot = _nearest_contact_feet(small_morphology.erosion_contacts, apex)
        status = "kept"
        reason = "prefilter_disabled"
        height_above_chord = float("nan")
        chord_y_at_apex = float("nan")
        height_ratio = float("nan")
        if not enabled:
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
                height_ratio = float(height_above_chord / noise_height) if noise_height > 0.0 else float("nan")
                if height_above_chord < height_threshold:
                    status = "rejected_noise"
                    reason = "height_below_morph_range_threshold"
                else:
                    status = "kept"
                    reason = "height_above_morph_range_threshold"
        rows.append(
            {
                "candidate_id": seg.candidate_id,
                "y": int(seg.y),
                "x": int(seg.x),
                "peak_index": int(seg.peak_index),
                "start": int(seg.start),
                "end": int(seg.end),
                "candidate_noise_prefilter_status": str(status),
                "candidate_noise_prefilter_mode": str(mode),
                "candidate_noise_prefilter_reason": str(reason),
                "candidate_noise_chord_y_at_apex": float(chord_y_at_apex) if np.isfinite(chord_y_at_apex) else np.nan,
                "candidate_noise_height_above_chord": float(height_above_chord) if np.isfinite(height_above_chord) else np.nan,
                "candidate_noise_height_threshold": float(height_threshold) if np.isfinite(height_threshold) else np.nan,
                "candidate_noise_height_factor": float(height_factor),
                "candidate_noise_height_ratio": float(height_ratio) if np.isfinite(height_ratio) else np.nan,
                "candidate_noise_left_foot": np.nan if left_foot is None else int(left_foot),
                "candidate_noise_right_foot": np.nan if right_foot is None else int(right_foot),
                "candidate_noise_apex": int(apex),
                "candidate_noise_estimate_used": float(noise_height) if np.isfinite(noise_height) else np.nan,
                "noise_height_morph_range": float(noise_height) if np.isfinite(noise_height) else np.nan,
                "noise_reference_n_points": int(keep_idx.size),
                "noise_reference_status": str(ref_status),
                "noise_reference_method": "morph_range_w3",
                "noise_reference_spans": spans,
            }
        )
    kept_n = sum(1 for row in rows if row["candidate_noise_prefilter_status"] == "kept")
    rejected_n = sum(1 for row in rows if row["candidate_noise_prefilter_status"] == "rejected_noise")
    summary = {
        "n_candidates_before_noise_prefilter": int(len(segs)),
        "n_candidates_after_noise_prefilter": int(kept_n),
        "n_candidates_rejected_by_noise_prefilter": int(rejected_n),
        "noise_prefilter_mode_used": str(mode),
        "noise_height_morph_range": float(noise_height) if np.isfinite(noise_height) else np.nan,
        "noise_reference_n_points": int(keep_idx.size),
        "noise_reference_status": str(ref_status),
        "noise_reference_method": "morph_range_w3",
        "noise_reference_spans": spans,
    }
    return rows, summary


def apply_global_metric_ranks(rows: list[dict[str, Any]]) -> None:
    def _assign(metric_key: str, out_key: str, *, spike_oriented_negative: bool = False) -> None:
        vals: list[tuple[int, float]] = []
        for idx, row in enumerate(rows):
            try:
                value = float(row.get(metric_key, np.nan))
            except Exception:
                value = np.nan
            if np.isfinite(value):
                vals.append((idx, -value if spike_oriented_negative else value))
        if not vals:
            return
        order = sorted(vals, key=lambda item: item[1])
        denom = max(len(order) - 1, 1)
        for rank_idx, (row_idx, _) in enumerate(order):
            rows[row_idx][out_key] = float(rank_idx / denom)

    _assign("spike_score_v1", "ss1_global_rank")
    _assign("pce_negpref_t098_evidence_signed", "pce_global_rank")
    _assign("recdw_sum_0_90_raman_veto_evidence_signed", "edge_global_rank")
    _assign("recdw_sum_0_90_raman_veto_evidence_signed", "edge_global_spike_rank", spike_oriented_negative=True)
