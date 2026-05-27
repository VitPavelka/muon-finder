from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .cache import load_viewer_cache
from .cap_metrics import join_extra_feature_rows, load_extra_feature_rows
from .data_model import CandidateSegment, CorrectionResult, DespikeChord, DespikeStage
from .experimental_common import select_experimental_noise
from .experimental_edge_variants import compute_experimental_edge_variants
from .experimental_residual_pce import compute_experimental_residual_features
from .metrics import MetricComputationContext, compute_raw_edge_metric, compute_ss1_pce_features, robust_center_scale, sigmoid_support
from .morphology import dilation_1d, erosion_1d
from .ss6_decision import compute_ss6_row, ss6_defaults


@dataclass(frozen=True)
class DespikeArtifacts:
    final_corrected: np.ndarray
    chords: list[DespikeChord]
    debug_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class LocalGeometry:
    detected_peak_index: int
    fixed_context_left: int
    fixed_context_right: int
    cell_left: int
    cell_right: int
    left_anchor: int
    right_anchor: int
    has_left_erosion_neighbor: bool
    has_right_erosion_neighbor: bool
    used_context_boundary_anchor: bool
    contains_dilation_contact: bool
    nearest_dilation_contact: int | None


def build_placeholder_correction(raw_spectra: np.ndarray) -> CorrectionResult:
    return CorrectionResult(
        corrected_spectra=np.asarray(raw_spectra, dtype=np.float32).copy(),
        stages=[DespikeStage(stage_id="stage:0", name="preview", description="Raw passthrough placeholder")],
        chords=[],
    )


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _to_float_array(signal: np.ndarray) -> np.ndarray:
    return np.asarray(signal, dtype=float).reshape(-1)


def _compute_contacts(signal: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = _to_float_array(signal)
    arr = x.reshape(1, 1, -1)
    erosion = np.asarray(erosion_1d(arr, int(window)).reshape(-1), dtype=float)
    dilation = np.asarray(dilation_1d(arr, int(window)).reshape(-1), dtype=float)
    erosion_mask = np.isclose(x, erosion) if not np.any(x == erosion) else (x == erosion)
    dilation_mask = np.isclose(x, dilation) if not np.any(x == dilation) else (x == dilation)
    return erosion, dilation, np.flatnonzero(erosion_mask).astype(int), np.flatnonzero(dilation_mask).astype(int)


def _morph_gradient_1d(signal: np.ndarray, window: int) -> np.ndarray:
    x = _to_float_array(signal).reshape(1, 1, -1)
    dil = np.asarray(dilation_1d(x, int(window)).reshape(-1), dtype=float)
    ero = np.asarray(erosion_1d(x, int(window)).reshape(-1), dtype=float)
    return np.asarray(dil - ero, dtype=float)


def get_contact_context_bounds(row: dict[str, Any], n_points: int, pad: int = 0) -> tuple[int, int]:
    start = int(row.get("start_index", row.get("start", 0)))
    end = int(row.get("end_index", row.get("end", 0)))
    left = max(0, min(start, end) - int(pad))
    right = min(int(n_points) - 1, max(start, end) + int(pad))
    return int(left), int(right)


def _local_noise(signal: np.ndarray, peak_idx: int, pad: int = 6) -> tuple[float, str, bool]:
    row = {"peak_index": int(peak_idx), "start": max(0, int(peak_idx) - int(pad)), "end": min(len(signal) - 1, int(peak_idx) + int(pad))}
    info = select_experimental_noise(np.asarray(signal, dtype=float), row, "morph_range")
    return float(info["exp_noise_value"]), str(info["exp_noise_source"]), bool(info["exp_noise_fallback_used"])


def _chord_values(left_idx: int, right_idx: int, left_val: float, right_val: float) -> np.ndarray:
    if right_idx <= left_idx:
        return np.asarray([float(left_val)], dtype=float)
    xs = np.arange(int(left_idx), int(right_idx) + 1, dtype=float)
    return np.interp(xs, [float(left_idx), float(right_idx)], [float(left_val), float(right_val)]).astype(float)


def _candidate_id(row: dict[str, Any]) -> str:
    candidate_id = str(row.get("candidate_id", "")).strip()
    if candidate_id:
        return candidate_id
    return f"candidate:{int(row.get('y', -1))}:{int(row.get('x', -1))}:{int(row.get('peak_index', -1))}"


def _local_dilation_candidates(dilation_contacts: np.ndarray, left: int, right: int) -> list[int]:
    return [int(idx) for idx in dilation_contacts.tolist() if int(left) <= int(idx) <= int(right)]


def _peak_shift(x_axis: np.ndarray, peak_idx: int) -> float:
    if 0 <= int(peak_idx) < int(len(x_axis)):
        return float(x_axis[int(peak_idx)])
    return float("nan")


def _nearest_erosion_neighbors(
    erosion_contacts: np.ndarray,
    peak_idx: int,
    n_points: int,
) -> tuple[int, int, bool, bool, bool]:
    peak = int(peak_idx)
    contacts = [int(idx) for idx in erosion_contacts.tolist()]
    left_candidates = [idx for idx in contacts if idx < peak]
    right_candidates = [idx for idx in contacts if idx > peak]
    has_left = bool(left_candidates)
    has_right = bool(right_candidates)
    left_anchor = max(left_candidates) if has_left else 0
    right_anchor = min(right_candidates) if has_right else int(n_points) - 1
    used_boundary = not (has_left and has_right)
    return int(left_anchor), int(right_anchor), bool(has_left), bool(has_right), bool(used_boundary)


def _geometry_for_peak(
    signal: np.ndarray,
    erosion_contacts: np.ndarray,
    dilation_contacts: np.ndarray,
    peak_idx: int,
    context_left: int,
    context_right: int,
) -> LocalGeometry:
    d_contacts = [int(idx) for idx in dilation_contacts.tolist() if int(context_left) <= int(idx) <= int(context_right)]
    left_anchor, right_anchor, has_left, has_right, used_boundary = _nearest_erosion_neighbors(
        erosion_contacts=erosion_contacts,
        peak_idx=int(peak_idx),
        n_points=int(len(signal)),
    )
    if right_anchor <= left_anchor:
        left_anchor = max(0, int(peak_idx) - 1)
        right_anchor = min(int(context_right), int(peak_idx) + 1)
        used_boundary = True
    nearest_dilation = min(d_contacts, key=lambda idx: abs(idx - int(peak_idx))) if d_contacts else None
    return LocalGeometry(
        detected_peak_index=int(peak_idx),
        fixed_context_left=int(context_left),
        fixed_context_right=int(context_right),
        cell_left=int(left_anchor),
        cell_right=int(right_anchor),
        left_anchor=int(left_anchor),
        right_anchor=int(right_anchor),
        has_left_erosion_neighbor=bool(has_left),
        has_right_erosion_neighbor=bool(has_right),
        used_context_boundary_anchor=bool(used_boundary),
        contains_dilation_contact=bool(d_contacts),
        nearest_dilation_contact=(None if nearest_dilation is None else int(nearest_dilation)),
    )


def _height_above_chord(signal: np.ndarray, peak_idx: int, left_anchor: int, right_anchor: int) -> tuple[float, float]:
    peak = int(peak_idx)
    chord = _chord_values(int(left_anchor), int(right_anchor), float(signal[int(left_anchor)]), float(signal[int(right_anchor)]))
    if int(right_anchor) <= int(left_anchor):
        return float("nan"), float("nan")
    chord_at_peak = float(np.interp(float(peak), [float(left_anchor), float(right_anchor)], [float(signal[int(left_anchor)]), float(signal[int(right_anchor)])]))
    return float(signal[peak] - chord_at_peak), chord_at_peak


def _serialize_contact_list(indices: list[int]) -> str:
    return json.dumps([int(v) for v in indices], ensure_ascii=False)


def _context_contact_lists(
    erosion_contacts: np.ndarray,
    dilation_contacts: np.ndarray,
    context_left: int,
    context_right: int,
) -> tuple[list[int], list[int]]:
    eros = [int(idx) for idx in erosion_contacts.tolist() if int(context_left) <= int(idx) <= int(context_right)]
    dils = [int(idx) for idx in dilation_contacts.tolist() if int(context_left) <= int(idx) <= int(context_right)]
    return eros, dils


def _mask_overlap_fraction(mask: np.ndarray, left: int, right: int) -> tuple[float, bool, bool]:
    li = int(left)
    ri = int(right)
    if ri < li or li < 0 or ri >= int(mask.size):
        return 0.0, False, False
    seg = np.asarray(mask[li : ri + 1], dtype=bool)
    if seg.size == 0:
        return 0.0, False, False
    frac = float(np.mean(seg.astype(float)))
    return frac, bool(frac > 0.0), bool(np.all(seg))


def _signed_edge_evidence_from_width_sum(value: float, center: float, scale: float, ctx: MetricComputationContext) -> float:
    if not (np.isfinite(value) and np.isfinite(center) and np.isfinite(scale) and float(scale) > 1e-12):
        return float("nan")
    z = float((float(value) - float(center)) / float(scale))
    support = sigmoid_support(z, float(ctx.recdw_support_z_scale), float(ctx.recdw_z_clip))
    return float(2.0 * support - 1.0)


def _plans_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return not (int(a["right_anchor"]) <= int(b["left_anchor"]) or int(b["right_anchor"]) <= int(a["left_anchor"]))


def _resolve_stage_plans(plans: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not plans:
        return [], []
    ordered = sorted(
        plans,
        key=lambda item: (
            -_safe_float(item.get("score", np.nan)),
            int(item.get("detected_peak_index", -1)),
            int(item.get("original_peak_index", -1)),
        ),
    )
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for plan in ordered:
        if any(_plans_overlap(plan, other) for other in kept):
            out = dict(plan)
            out["correction_applied"] = 0
            out["status"] = "skipped_overlap_duplicate"
            out["skipped_reason"] = "overlap_duplicate"
            skipped.append(out)
            continue
        kept.append(plan)
    kept.sort(key=lambda item: (int(item.get("detected_peak_index", -1)), int(item.get("original_peak_index", -1))))
    return kept, skipped


def _edge_signed_with_global_norm(signal: np.ndarray, seg: CandidateSegment, ctx: MetricComputationContext, global_center: float, global_scale: float, candidate_noise_estimate: float | None) -> tuple[dict[str, Any], float]:
    features = compute_raw_edge_metric(
        raw_signal=np.asarray(signal, dtype=float),
        seg=seg,
        candidate_noise_estimate=candidate_noise_estimate,
        ctx=ctx,
    )
    raw_value = _safe_float(features.get("recdw_sum_0_90"))
    if np.isfinite(raw_value) and np.isfinite(global_center) and np.isfinite(global_scale) and global_scale > 1e-12:
        z = float((raw_value - global_center) / global_scale)
        support = float(sigmoid_support(z, float(ctx.recdw_support_z_scale), float(ctx.recdw_z_clip)))
        features["recdw_sum_0_90_z"] = float(z)
        features["recdw_sum_0_90_support01"] = float(support)
        features["recdw_sum_0_90_raman_veto_evidence_signed"] = float(2.0 * support - 1.0)
    return features, _safe_float(features.get("recdw_sum_0_90_raman_veto_evidence_signed"))


def _build_local_ss6_row(
    *,
    signal: np.ndarray,
    gradient_signal: np.ndarray,
    base_row: dict[str, Any],
    peak_idx: int,
    geom: LocalGeometry,
    noise_value: float,
    ss6_cfg: dict[str, Any],
    metric_ctx: MetricComputationContext,
    global_edge_center: float,
    global_edge_scale: float,
    parent_noise_source: str = "",
    ss1_override: float | None = None,
    pce_override: float | None = None,
) -> dict[str, Any]:
    seg = CandidateSegment(
        y=int(base_row.get("y", -1)),
        x=int(base_row.get("x", -1)),
        peak_index=int(peak_idx),
        start=int(geom.cell_left),
        end=int(geom.cell_right),
        peak_height=float(signal[int(peak_idx)]),
        area=float(max(0.0, signal[int(peak_idx)])),
    )
    row = dict(base_row)
    row["peak_index"] = int(peak_idx)
    row["start"] = int(geom.cell_left)
    row["end"] = int(geom.cell_right)
    row["candidate_id"] = _candidate_id(row)
    row.update(
        compute_ss1_pce_features(
            raw_signal=np.asarray(signal, dtype=float),
            gradient_signal=np.asarray(gradient_signal, dtype=float),
            seg=seg,
            feature_signal_source="gradient",
            bg_noise_override=(float(noise_value) if np.isfinite(noise_value) and float(noise_value) > 0.0 else None),
        )
    )
    row["ss1_pce_noise_used"] = float(noise_value) if np.isfinite(noise_value) else np.nan
    row["ss1_pce_noise_source"] = str(parent_noise_source or "parent_noise_override")
    row["ss1_pce_noise_override_used"] = 1 if np.isfinite(noise_value) and float(noise_value) > 0.0 else 0
    row["pce_t098_chosen_value"] = _safe_float(((row.get("pce_t98_debug", {}) if isinstance(row.get("pce_t98_debug", {}), dict) else {}).get("chosen_value")))
    bg = _safe_float(row.get("bg_mad"))
    chosen_raw = _safe_float(row.get("pce_t098_chosen_value"))
    row["pce_t098_chosen_value_z"] = float(chosen_raw / bg) if np.isfinite(chosen_raw) and np.isfinite(bg) and bg > 0.0 else float("nan")
    row["pce_t098_evidence_signed"] = row.get("pce_negpref_t098_evidence_signed", np.nan)

    edge_features, edge_signed = _edge_signed_with_global_norm(
        signal=np.asarray(signal, dtype=float),
        seg=seg,
        ctx=metric_ctx,
        global_center=float(global_edge_center),
        global_scale=float(global_edge_scale),
        candidate_noise_estimate=(float(noise_value) if np.isfinite(noise_value) else None),
    )
    row.update(edge_features)
    row["candidate_noise_prefilter_status"] = "kept"
    row["candidate_noise_estimate_used"] = float(noise_value) if np.isfinite(noise_value) else np.nan
    row["noise_height_morph_range"] = float(noise_value) if np.isfinite(noise_value) else np.nan

    eel_features = compute_experimental_edge_variants(np.asarray(signal, dtype=float), row, float(noise_value), {"edge_variants": {"enabled": True}})
    resid_features = compute_experimental_residual_features(np.asarray(signal, dtype=float), row, float(noise_value), {"residual_pce": {"enabled": True, "median_window": 3, "near_apex_radius_pts": 2}, "residual_threshold": {"enabled": True, "median_window": 3, "threshold_noise_factor": 3.0}, "noise_threshold_factor": 3.0})
    row.update(eel_features)
    row.update(resid_features)
    legacy_width = _safe_float(row.get("exp_edge_legacy_width_sum_0_90"))
    legacy_signed = _signed_edge_evidence_from_width_sum(legacy_width, float(global_edge_center), float(global_edge_scale), metric_ctx)
    if np.isfinite(legacy_signed):
        row["exp_edge_legacy_evidence_signed_modernnorm"] = float(legacy_signed)
        row["exp_edge_legacy_like"] = float(legacy_signed)
        row["exp_edge_legacy_like_evidence_signed"] = float(legacy_signed)
    else:
        row["exp_edge_legacy_evidence_signed_modernnorm"] = row.get("exp_edge_legacy_evidence_signed_modernnorm", np.nan)
    row["exp_resid3_height_noise_z"] = row.get("exp_resid3_height_noise_z", np.nan)
    if ss1_override is not None and np.isfinite(float(ss1_override)):
        row["spike_score_v1"] = float(ss1_override)
    if pce_override is not None and np.isfinite(float(pce_override)):
        row["pce_local_audit"] = _safe_float(row.get("pce_negpref_t098_evidence_signed"))
        row["pce_negpref_t098_evidence_signed"] = float(pce_override)
        row["pce_t098_evidence_signed"] = float(pce_override)
        row["pce"] = float(pce_override)
    ss6_row = compute_ss6_row(row, ss6_cfg)
    row.update(ss6_row)
    row["ss6_local_accept"] = int(ss6_row.get("ss6_accept", 0))
    row["ss6_local_branch"] = str(ss6_row.get("ss6_branch", ""))
    row["ss6_local_pce"] = ss6_row.get("ss6_pce", np.nan)
    return row


def _compute_context_metrics(
    *,
    signal: np.ndarray,
    gradient_signal: np.ndarray,
    base_row: dict[str, Any],
    context_left: int,
    context_right: int,
    peak_idx: int,
    noise_value: float,
    noise_source: str,
) -> dict[str, Any]:
    seg = CandidateSegment(
        y=int(base_row.get("y", -1)),
        x=int(base_row.get("x", -1)),
        peak_index=int(peak_idx),
        start=int(context_left),
        end=int(context_right),
        peak_height=float(signal[int(peak_idx)]),
        area=float(max(0.0, signal[int(peak_idx)])),
    )
    features = compute_ss1_pce_features(
        raw_signal=np.asarray(signal, dtype=float),
        gradient_signal=np.asarray(gradient_signal, dtype=float),
        seg=seg,
        feature_signal_source="gradient",
        bg_noise_override=(float(noise_value) if np.isfinite(noise_value) and float(noise_value) > 0.0 else None),
    )
    return {
        "context_ss1": _safe_float(features.get("spike_score_v1")),
        "context_pce": _safe_float(features.get("pce_negpref_t098_evidence_signed")),
        "context_pce_status": ("ok" if np.isfinite(_safe_float(features.get("pce_negpref_t098_evidence_signed"))) else "fallback_to_local_pce"),
        "context_pce_left_index": int(context_left),
        "context_pce_right_index": int(context_right),
        "context_pce_peak_index": int(peak_idx),
        "context_pce_noise_value": float(noise_value) if np.isfinite(noise_value) else np.nan,
        "context_pce_noise_source": str(noise_source),
        "context_pce_debug": (features.get("pce_t98_debug", {}) if isinstance(features.get("pce_t98_debug", {}), dict) else {}),
        "context_ss1_noise_value": float(noise_value) if np.isfinite(noise_value) else np.nan,
        "context_ss1_noise_source": str(noise_source),
    }


def _local_missing_metrics(local_row: dict[str, Any], ss6_cfg: dict[str, Any]) -> str:
    metric_names = dict(ss6_cfg.get("metric_names", {}))
    missing: list[str] = []
    for key in ("ss1", "pce", "edge", "eel", "resid"):
        col = str(metric_names.get(key, "")).strip()
        if not col:
            continue
        if not np.isfinite(_safe_float(local_row.get(col))):
            missing.append(key)
    return ",".join(missing)


def compute_despike_from_cache_and_ss6(
    *,
    cache_path: Path,
    ss6_path: Path,
    corrected_path: Path | None,
    debug_path: Path | None,
    attempts_path: Path | None,
    summary_path: Path | None,
    morph_window: int,
    despike_context_window_pad: int,
    noise_height_factor: float,
    max_iterations: int,
    ss1_context_threshold: float = 0.95,
    pce_context_enabled: bool = False,
    ss6_config: dict[str, Any] | None = None,
    metric_context: MetricComputationContext | None = None,
    progress_iter: Callable[[list[tuple[tuple[int, int], list[dict[str, Any]]]]], Any] | None = None,
    timings_out: dict[str, float] | None = None,
    config_path: Path | str | None = None,
) -> DespikeArtifacts:
    t0 = time.perf_counter()
    cache = load_viewer_cache(cache_path)
    if timings_out is not None:
        timings_out["load cache"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rows = [dict(row) for row in cache.get("candidate_records", [])]
    ss6_rows, _ = load_extra_feature_rows(ss6_path)
    if timings_out is not None:
        timings_out["load ss6 decisions"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    join_info = join_extra_feature_rows(rows, ss6_rows)
    accepted_rows = [row for row in rows if int(_safe_float(row.get("ss6_accept", 0))) == 1]
    accepted_by_pixel: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in accepted_rows:
        accepted_by_pixel.setdefault((int(row["y"]), int(row["x"])), []).append(row)
    for pixel_rows in accepted_by_pixel.values():
        pixel_rows.sort(key=lambda row: int(row.get("peak_index", -1)))
    if timings_out is not None:
        timings_out["group accepted candidates"] = time.perf_counter() - t0

    raw_spectra = np.asarray(cache["spectra"], dtype=np.float32)
    corrected = np.asarray(raw_spectra, dtype=np.float32).copy()
    chords: list[DespikeChord] = []
    debug_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    per_spectrum_correction_counts: dict[str, int] = {}

    metric_ctx = metric_context if metric_context is not None else MetricComputationContext()
    edge_raw_values = np.asarray(
        [_safe_float(row.get("recdw_sum_0_90")) for row in rows if np.isfinite(_safe_float(row.get("recdw_sum_0_90")))],
        dtype=float,
    )
    global_edge_center, global_edge_scale = robust_center_scale(edge_raw_values)
    ss6_cfg = ss6_defaults(ss6_config or {})
    morph_window_used = int(morph_window)
    local_context_ss1_threshold = float(ss1_context_threshold)
    pce_context_enabled = bool(pce_context_enabled)

    summary_counts = {
        "accepted_ss6_parent_candidates": int(len(accepted_rows)),
        "parent_corrected": 0,
        "parent_skipped_below_noise_height": 0,
        "parent_skipped_no_erosion_neighbors": 0,
        "local_candidates_from_dilation_contacts": 0,
        "local_candidates_passed_noise_height": 0,
        "local_candidates_rejected_by_noise_height": 0,
        "local_contexts_with_context_ss1": 0,
        "local_candidates_rejected_by_context_ss1_low": 0,
        "local_candidates_sent_to_ss6": 0,
        "local_candidates_accepted_by_ss6": 0,
        "local_candidates_corrected": 0,
        "local_context_pce_computed": 0,
        "local_context_pce_missing": 0,
        "local_candidates_using_context_pce": 0,
        "local_candidates_fallback_to_local_pce": 0,
        "total_corrections_applied": 0,
        "spectra_with_corrections": 0,
        "technical_failures": 0,
        "max_pass_index_used": 0,
        "skipped_overlap_duplicate": 0,
        "mask_cleanup_candidates_tested": 0,
        "mask_cleanup_corrected": 0,
        "mask_cleanup_rejected_below_noise_height": 0,
        "local_candidates_with_parent_noise": 0,
        "local_candidates_with_missing_parent_noise": 0,
        "noise_ratio_values": [],
        "chord_crossing_adjusted_count": 0,
        "replacement_invariant_violations": 0,
    }

    def _apply_plans(stage_index: int, y: int, x: int, plans: list[dict[str, Any]], corrected_mask: np.ndarray) -> int:
        applied = 0
        for attempt_index, plan in enumerate(plans, start=1):
            li = int(plan["left_anchor"])
            ri = int(plan["right_anchor"])
            current_signal = np.asarray(corrected[y, x, :], dtype=float)
            if ri - li < 2:
                out = dict(plan)
                out["stage_index"] = int(stage_index)
                out["attempt_index_within_stage"] = int(attempt_index)
                out["correction_applied"] = 0
                out["status"] = "skipped_unsafe_geometry"
                out["skipped_reason"] = "unsafe_geometry"
                attempt_rows.append(out)
                continue
            chord = _chord_values(li, ri, float(current_signal[li]), float(current_signal[ri]))
            before_interior = np.asarray(current_signal[li + 1 : ri], dtype=float)
            chord_interior = np.asarray(chord[1:-1], dtype=float)
            local_range = float(np.nanmax(current_signal[li : ri + 1]) - np.nanmin(current_signal[li : ri + 1])) if ri > li else 0.0
            eps = 1e-12 * max(1.0, local_range)
            overshoot = np.asarray(chord_interior - before_interior, dtype=float) if before_interior.size else np.asarray([], dtype=float)
            max_overshoot = float(np.max(overshoot)) if overshoot.size else 0.0
            crossing_detected = bool(max_overshoot > eps)
            after_interior = np.minimum(chord_interior, before_interior)
            support_adjusted = bool(crossing_detected)
            if crossing_detected:
                summary_counts["chord_crossing_adjusted_count"] += 1
            if before_interior.size == 0 or np.allclose(before_interior, after_interior, atol=1e-9, rtol=0.0):
                out = dict(plan)
                out["stage_index"] = int(stage_index)
                out["attempt_index_within_stage"] = int(attempt_index)
                out["correction_applied"] = 0
                out["status"] = "stopped_no_meaningful_change"
                out["skipped_reason"] = "no_meaningful_change"
                out["raw_peak_value_before"] = float(current_signal[int(plan["detected_peak_index"])])
                out["corrected_peak_value_after"] = float(current_signal[int(plan["detected_peak_index"])])
                out["correction_height"] = 0.0
                out["chord_crossing_detected"] = int(crossing_detected)
                out["chord_crossing_max"] = float(max_overshoot)
                out["chord_support_adjustment_applied"] = int(support_adjusted)
                out["replacement_increases_signal_max"] = float(np.max(after_interior - before_interior)) if before_interior.size else 0.0
                attempt_rows.append(out)
                continue
            raw_peak_value_before = float(current_signal[int(plan["detected_peak_index"])])
            corrected[y, x, li + 1 : ri] = after_interior.astype(np.float32)
            corrected_peak_value_after = float(corrected[y, x, int(plan["detected_peak_index"])])
            correction_height = float(raw_peak_value_before - corrected_peak_value_after)
            replacement_increase = float(np.max(np.asarray(after_interior - before_interior, dtype=float))) if before_interior.size else 0.0
            if replacement_increase > eps:
                summary_counts["replacement_invariant_violations"] += 1
            applied += 1
            summary_counts["total_corrections_applied"] += 1
            corrected_mask[li : ri + 1] = True
            chords.append(
                DespikeChord(
                    chord_id=f"despike:{y}:{x}:{stage_index}:{attempt_index}:{li}:{ri}",
                    stage_id=f"stage:{stage_index}",
                    y=int(y),
                    x=int(x),
                    left=int(li),
                    right=int(ri),
                    method="morph_contact_cells",
                    y_left=float(current_signal[li]),
                    y_right=float(current_signal[ri]),
                )
            )
            out = dict(plan)
            out["stage_index"] = int(stage_index)
            out["attempt_index_within_stage"] = int(attempt_index)
            out["correction_applied"] = 1
            out["raw_peak_value_before"] = raw_peak_value_before
            out["corrected_peak_value_after"] = corrected_peak_value_after
            out["correction_height"] = correction_height
            out["chord_crossing_detected"] = int(crossing_detected)
            out["chord_crossing_max"] = float(max_overshoot)
            out["chord_support_adjustment_applied"] = int(support_adjusted)
            out["replacement_increases_signal_max"] = float(replacement_increase)
            out["skipped_reason"] = ""
            attempt_rows.append(out)
        return applied

    spectrum_items = list(accepted_by_pixel.items())
    iterator = progress_iter(spectrum_items) if progress_iter is not None else spectrum_items
    t0 = time.perf_counter()
    for (y, x), pixel_rows in iterator:
        current_signal = np.asarray(corrected[y, x, :], dtype=float)
        contexts: list[dict[str, Any]] = []
        spectrum_corrections = 0
        corrected_mask = np.zeros(len(current_signal), dtype=bool)

        for row in pixel_rows:
            peak_idx = int(row.get("peak_index", -1))
            context_left, context_right = get_contact_context_bounds(row, len(current_signal), pad=int(despike_context_window_pad))
            parent_noise_value = _safe_float(row.get("candidate_noise_estimate_used", row.get("noise_height_morph_range", np.nan)))
            parent_noise_source = "candidate_noise_estimate_used" if np.isfinite(_safe_float(row.get("candidate_noise_estimate_used", np.nan))) else "noise_height_morph_range"
            parent_noise_fallback_used = False
            if not np.isfinite(parent_noise_value) or parent_noise_value <= 0.0:
                parent_noise_value, parent_noise_source, parent_noise_fallback_used = _local_noise(current_signal, peak_idx, pad=6)
            contexts.append(
                {
                    "row": row,
                    "source_y": int(row.get("source_y", y)),
                    "source_x": int(row.get("source_x", x)),
                    "compact_y": int(y),
                    "compact_x": int(x),
                    "candidate_id": _candidate_id(row),
                    "original_peak_index": int(peak_idx),
                    "original_ss6_branch": str(row.get("ss6_branch", "")),
                    "original_ss6_reason": str(row.get("ss6_reason", "")),
                    "context_left": int(context_left),
                    "context_right": int(context_right),
                    "parent_noise_value": float(parent_noise_value) if np.isfinite(parent_noise_value) else np.nan,
                    "parent_noise_source": str(parent_noise_source),
                    "noise_fallback_used": bool(parent_noise_fallback_used),
                }
            )

        erosion, dilation, erosion_contacts, dilation_contacts = _compute_contacts(current_signal, morph_window_used)
        parent_plans: list[dict[str, Any]] = []
        for context in contexts:
            peak_idx = int(context["original_peak_index"])
            context_eros, context_dils = _context_contact_lists(
                erosion_contacts,
                dilation_contacts,
                int(context["context_left"]),
                int(context["context_right"]),
            )
            context_eros_json = _serialize_contact_list(context_eros)
            context_dils_json = _serialize_contact_list(context_dils)
            if not (0 <= peak_idx < len(current_signal)):
                summary_counts["technical_failures"] += 1
                summary_counts["parent_skipped_no_erosion_neighbors"] += 1
                debug_rows.append(
                    {
                        "source_y": context["source_y"],
                        "source_x": context["source_x"],
                        "compact_y": context["compact_y"],
                        "compact_x": context["compact_x"],
                        "original_peak_index": peak_idx,
                        "candidate_id": context["candidate_id"],
                        "ss6_branch": context["original_ss6_branch"],
                        "fixed_context_left": context["context_left"],
                        "fixed_context_right": context["context_right"],
                        "morph_window": int(morph_window_used),
                        "corrections_applied": 0,
                        "stopped_reason": "technical_failure",
                        "max_iterations_reached": 0,
                        "technical_failure": 1,
                        "technical_failure_reason": "invalid_peak_index",
                    }
                )
                continue
            noise_value = float(context["parent_noise_value"])
            geom = _geometry_for_peak(
                current_signal,
                erosion_contacts,
                dilation_contacts,
                peak_idx,
                int(context["context_left"]),
                int(context["context_right"]),
            )
            if geom.right_anchor <= geom.left_anchor:
                summary_counts["parent_skipped_no_erosion_neighbors"] += 1
                attempt_rows.append(
                    {
                        "source_y": context["source_y"],
                        "source_x": context["source_x"],
                        "compact_y": context["compact_y"],
                        "compact_x": context["compact_x"],
                        "stage_index": 0,
                        "attempt_index_within_stage": "",
                        "attempt_type": "parent",
                        "original_peak_index": peak_idx,
                        "detected_peak_index": peak_idx,
                        "candidate_id": context["candidate_id"],
                        "original_ss6_branch": context["original_ss6_branch"],
                        "original_ss6_reason": context["original_ss6_reason"],
                        "local_ss6_branch": "",
                        "local_ss6_accept": "",
                        "context_left": context["context_left"],
                        "context_right": context["context_right"],
                        "tested_left": "",
                        "tested_right": "",
                        "left_erosion_contact": "",
                        "right_erosion_contact": "",
                        "dilation_contact": peak_idx,
                        "morph_window_used": int(morph_window_used),
                        "context_erosion_contacts": context_eros_json,
                        "context_dilation_contacts": context_dils_json,
                        "chord_crossing_detected": 0,
                        "chord_crossing_max": 0.0,
                        "chord_support_adjustment_applied": 0,
                        "replacement_increases_signal_max": 0.0,
                        "correction_applied": 0,
                        "status": "skipped_parent_no_erosion_neighbors",
                        "skipped_reason": "no_erosion_neighbors",
                        "ss1": _safe_float(context["row"].get("ss6_ss1")),
                        "pce": _safe_float(context["row"].get("ss6_pce")),
                        "edge": _safe_float(context["row"].get("ss6_edge")),
                        "eel": _safe_float(context["row"].get("ss6_eel")),
                        "resid": _safe_float(context["row"].get("ss6_resid")),
                    }
                )
                debug_rows.append(
                    {
                        "source_y": context["source_y"],
                        "source_x": context["source_x"],
                        "compact_y": context["compact_y"],
                        "compact_x": context["compact_x"],
                        "original_peak_index": peak_idx,
                        "candidate_id": context["candidate_id"],
                        "ss6_branch": context["original_ss6_branch"],
                        "fixed_context_left": context["context_left"],
                        "fixed_context_right": context["context_right"],
                        "morph_window": int(morph_window_used),
                        "corrections_applied": 0,
                        "stopped_reason": "skipped_parent_no_erosion_neighbors",
                        "max_iterations_reached": 0,
                        "technical_failure": 0,
                        "technical_failure_reason": "",
                    }
                )
                continue
            if not np.isfinite(noise_value) or noise_value <= 0.0:
                summary_counts["technical_failures"] += 1
                continue
            height_above_chord, _ = _height_above_chord(current_signal, peak_idx, geom.left_anchor, geom.right_anchor)
            height_z = float(height_above_chord / noise_value) if np.isfinite(height_above_chord) else float("nan")
            if not np.isfinite(height_z) or height_z < float(noise_height_factor):
                summary_counts["parent_skipped_below_noise_height"] += 1
                attempt_rows.append(
                    {
                        "source_y": context["source_y"],
                        "source_x": context["source_x"],
                        "compact_y": context["compact_y"],
                        "compact_x": context["compact_x"],
                        "stage_index": 0,
                        "attempt_index_within_stage": "",
                        "attempt_type": "parent",
                        "original_peak_index": peak_idx,
                        "detected_peak_index": peak_idx,
                        "candidate_id": context["candidate_id"],
                        "original_ss6_branch": context["original_ss6_branch"],
                        "original_ss6_reason": context["original_ss6_reason"],
                        "local_ss6_branch": "",
                        "local_ss6_accept": "",
                        "context_left": context["context_left"],
                        "context_right": context["context_right"],
                        "tested_left": int(geom.cell_left),
                        "tested_right": int(geom.cell_right),
                        "left_erosion_contact": int(geom.left_anchor),
                        "right_erosion_contact": int(geom.right_anchor),
                        "dilation_contact": peak_idx,
                        "morph_window_used": int(morph_window_used),
                        "context_erosion_contacts": context_eros_json,
                        "context_dilation_contacts": context_dils_json,
                        "height_above_chord": float(height_above_chord),
                        "height_above_chord_noise_z": float(height_z),
                        "noise_value": float(noise_value),
                        "chord_crossing_detected": 0,
                        "chord_crossing_max": 0.0,
                        "chord_support_adjustment_applied": 0,
                        "replacement_increases_signal_max": 0.0,
                        "correction_applied": 0,
                        "status": "skipped_parent_below_noise_height",
                        "skipped_reason": "below_noise_height",
                        "ss1": _safe_float(context["row"].get("ss6_ss1")),
                        "pce": _safe_float(context["row"].get("ss6_pce")),
                        "edge": _safe_float(context["row"].get("ss6_edge")),
                        "eel": _safe_float(context["row"].get("ss6_eel")),
                        "resid": _safe_float(context["row"].get("ss6_resid")),
                    }
                )
                debug_rows.append(
                    {
                        "source_y": context["source_y"],
                        "source_x": context["source_x"],
                        "compact_y": context["compact_y"],
                        "compact_x": context["compact_x"],
                        "original_peak_index": peak_idx,
                        "candidate_id": context["candidate_id"],
                        "ss6_branch": context["original_ss6_branch"],
                        "fixed_context_left": context["context_left"],
                        "fixed_context_right": context["context_right"],
                        "morph_window": int(morph_window_used),
                        "corrections_applied": 0,
                        "stopped_reason": "skipped_parent_below_noise_height",
                        "max_iterations_reached": 0,
                        "technical_failure": 0,
                        "technical_failure_reason": "",
                    }
                )
                continue
            parent_plans.append(
                {
                    "source_y": context["source_y"],
                    "source_x": context["source_x"],
                    "compact_y": context["compact_y"],
                    "compact_x": context["compact_x"],
                    "attempt_type": "parent",
                    "original_peak_index": peak_idx,
                    "detected_peak_index": peak_idx,
                    "candidate_id": context["candidate_id"],
                    "original_ss6_branch": context["original_ss6_branch"],
                    "original_ss6_reason": context["original_ss6_reason"],
                    "local_ss6_branch": "",
                    "local_ss6_accept": "",
                    "context_left": context["context_left"],
                    "context_right": context["context_right"],
                    "tested_left": int(geom.cell_left),
                    "tested_right": int(geom.cell_right),
                    "left_erosion_contact": int(geom.left_anchor),
                    "right_erosion_contact": int(geom.right_anchor),
                    "dilation_contact": peak_idx,
                    "morph_window_used": int(morph_window_used),
                    "context_erosion_contacts": context_eros_json,
                    "context_dilation_contacts": context_dils_json,
                    "left_anchor": int(geom.left_anchor),
                    "right_anchor": int(geom.right_anchor),
                    "cell_left": int(geom.cell_left),
                    "cell_right": int(geom.cell_right),
                    "used_context_boundary_anchor": int(geom.used_context_boundary_anchor),
                    "height_above_chord": float(height_above_chord),
                    "height_above_chord_noise_z": float(height_z),
                    "noise_value": float(noise_value),
                    "status": "corrected_parent",
                    "score": float(height_z),
                    "ss1": _safe_float(context["row"].get("ss6_ss1")),
                    "pce": _safe_float(context["row"].get("ss6_pce")),
                    "edge": _safe_float(context["row"].get("ss6_edge")),
                    "eel": _safe_float(context["row"].get("ss6_eel")),
                    "resid": _safe_float(context["row"].get("ss6_resid")),
                }
            )

        kept_parent, skipped_parent = _resolve_stage_plans(parent_plans)
        for item in skipped_parent:
            summary_counts["skipped_overlap_duplicate"] += 1
            item["stage_index"] = 0
            item["attempt_index_within_stage"] = ""
            attempt_rows.append(item)
        spectrum_corrections += _apply_plans(0, int(y), int(x), kept_parent, corrected_mask)
        summary_counts["parent_corrected"] += len(kept_parent)

        max_pass_index = 0
        for pass_index in range(1, int(max_iterations) + 1):
            current_signal = np.asarray(corrected[y, x, :], dtype=float)
            gradient_signal = _morph_gradient_1d(current_signal, morph_window_used)
            _erosion, _dilation, erosion_contacts, dilation_contacts = _compute_contacts(current_signal, morph_window_used)
            local_plans: list[dict[str, Any]] = []
            for context in contexts:
                context_eros, context_dils = _context_contact_lists(
                    erosion_contacts,
                    dilation_contacts,
                    int(context["context_left"]),
                    int(context["context_right"]),
                )
                context_eros_json = _serialize_contact_list(context_eros)
                context_dils_json = _serialize_contact_list(context_dils)
                passed_noise_candidates: list[dict[str, Any]] = []
                for dilation_peak in _local_dilation_candidates(dilation_contacts, int(context["context_left"]), int(context["context_right"])):
                    summary_counts["local_candidates_from_dilation_contacts"] += 1
                    geom = _geometry_for_peak(
                        current_signal,
                        erosion_contacts,
                        dilation_contacts,
                        int(dilation_peak),
                        int(context["context_left"]),
                        int(context["context_right"]),
                    )
                    if geom.right_anchor <= geom.left_anchor:
                        continue
                    noise_value = float(context["parent_noise_value"])
                    local_noise_would_have_been, previous_local_noise_source, previous_local_noise_fallback = _local_noise(current_signal, int(dilation_peak), pad=6)
                    if np.isfinite(noise_value) and noise_value > 0.0:
                        summary_counts["local_candidates_with_parent_noise"] += 1
                        if np.isfinite(local_noise_would_have_been) and local_noise_would_have_been > 0.0:
                            summary_counts["noise_ratio_values"].append(float(local_noise_would_have_been / noise_value))
                    else:
                        summary_counts["local_candidates_with_missing_parent_noise"] += 1
                        noise_value = float(local_noise_would_have_been)
                    if not np.isfinite(noise_value) or noise_value <= 0.0:
                        summary_counts["technical_failures"] += 1
                        continue
                    height_above_chord, _ = _height_above_chord(current_signal, int(dilation_peak), geom.left_anchor, geom.right_anchor)
                    height_z = float(height_above_chord / noise_value) if np.isfinite(height_above_chord) else float("nan")
                    if not np.isfinite(height_z) or height_z < float(noise_height_factor):
                        summary_counts["local_candidates_rejected_by_noise_height"] += 1
                        attempt_rows.append(
                            {
                                "source_y": context["source_y"],
                                "source_x": context["source_x"],
                                "compact_y": context["compact_y"],
                                "compact_x": context["compact_x"],
                                "stage_index": int(pass_index),
                                "attempt_index_within_stage": "",
                                "attempt_type": "iterative_local_ss6",
                                "original_peak_index": context["original_peak_index"],
                                "detected_peak_index": int(dilation_peak),
                                "candidate_id": context["candidate_id"],
                                "original_ss6_branch": context["original_ss6_branch"],
                                "original_ss6_reason": context["original_ss6_reason"],
                                "local_ss6_branch": "",
                                "local_ss6_accept": "",
                                "context_left": context["context_left"],
                                "context_right": context["context_right"],
                                "tested_left": int(geom.cell_left),
                                "tested_right": int(geom.cell_right),
                                "left_erosion_contact": int(geom.left_anchor),
                                "right_erosion_contact": int(geom.right_anchor),
                                "dilation_contact": int(dilation_peak),
                                "morph_window_used": int(morph_window_used),
                                "context_erosion_contacts": context_eros_json,
                                "context_dilation_contacts": context_dils_json,
                                "height_above_chord": float(height_above_chord),
                                "height_above_chord_noise_z": float(height_z),
                                "noise_value": float(noise_value),
                                "parent_noise_value": float(context["parent_noise_value"]),
                                "parent_noise_source": context["parent_noise_source"],
                                "previous_local_noise_value_if_available": float(local_noise_would_have_been) if np.isfinite(local_noise_would_have_been) else np.nan,
                                "local_noise_would_have_been": float(local_noise_would_have_been) if np.isfinite(local_noise_would_have_been) else np.nan,
                                "noise_ratio_local_to_parent": (float(local_noise_would_have_been / context["parent_noise_value"]) if np.isfinite(local_noise_would_have_been) and np.isfinite(context["parent_noise_value"]) and float(context["parent_noise_value"]) > 0.0 else np.nan),
                                "noise_fallback_used": int(bool(context["noise_fallback_used"]) or bool(previous_local_noise_fallback)),
                                "ss1_pce_noise_used": "",
                                "ss1_pce_noise_source": "",
                                "ss1_pce_noise_override_used": 0,
                                "chord_crossing_detected": 0,
                                "chord_crossing_max": 0.0,
                                "chord_support_adjustment_applied": 0,
                                "replacement_increases_signal_max": 0.0,
                                "correction_applied": 0,
                                "status": "rejected_by_despike_noise_height",
                                "skipped_reason": "noise_height",
                            }
                        )
                        continue
                    summary_counts["local_candidates_passed_noise_height"] += 1
                    passed_noise_candidates.append(
                        {
                            "dilation_peak": int(dilation_peak),
                            "geom": geom,
                            "height_above_chord": float(height_above_chord),
                            "height_above_chord_noise_z": float(height_z),
                            "noise_value": float(noise_value),
                            "local_noise_would_have_been": float(local_noise_would_have_been) if np.isfinite(local_noise_would_have_been) else np.nan,
                            "previous_local_noise_fallback": bool(previous_local_noise_fallback),
                        }
                    )
                if not passed_noise_candidates:
                    continue
                strongest_candidate = max(
                    passed_noise_candidates,
                    key=lambda item: (_safe_float(item.get("height_above_chord_noise_z")), _safe_float(item.get("height_above_chord"))),
                )
                context_metrics = _compute_context_metrics(
                    signal=current_signal,
                    gradient_signal=gradient_signal,
                    base_row=context["row"],
                    context_left=int(context["context_left"]),
                    context_right=int(context["context_right"]),
                    peak_idx=int(strongest_candidate["dilation_peak"]),
                    noise_value=float(context["parent_noise_value"]),
                    noise_source=str(context["parent_noise_source"]),
                )
                summary_counts["local_contexts_with_context_ss1"] += 1
                context_ss1 = _safe_float(context_metrics.get("context_ss1"))
                context_pce = _safe_float(context_metrics.get("context_pce"))
                context_pce_status = ("disabled" if not pce_context_enabled else str(context_metrics.get("context_pce_status", "fallback_to_local_pce")))
                if pce_context_enabled and np.isfinite(context_pce):
                    summary_counts["local_context_pce_computed"] += 1
                elif pce_context_enabled:
                    summary_counts["local_context_pce_missing"] += 1
                context_ss1_fields = {
                    "context_ss1": context_ss1,
                    "context_ss1_peak_index": int(strongest_candidate["dilation_peak"]),
                    "context_ss1_peak_x": _peak_shift(np.asarray(cache["x_axis"], dtype=float), int(strongest_candidate["dilation_peak"])),
                    "context_ss1_noise_value": _safe_float(context_metrics.get("context_ss1_noise_value")),
                    "context_ss1_noise_source": str(context_metrics.get("context_ss1_noise_source", "")),
                    "context_ss1_candidate_count": int(len(passed_noise_candidates)),
                    "context_ss1_threshold": float(local_context_ss1_threshold),
                    "local_ss1_gate_used": float(local_context_ss1_threshold),
                    "context_pce": context_pce,
                    "context_pce_peak_index": int(context_metrics.get("context_pce_peak_index", strongest_candidate["dilation_peak"])),
                    "context_pce_peak_x": _peak_shift(np.asarray(cache["x_axis"], dtype=float), int(context_metrics.get("context_pce_peak_index", strongest_candidate["dilation_peak"]))),
                    "context_pce_left_index": int(context_metrics.get("context_pce_left_index", context["context_left"])),
                    "context_pce_right_index": int(context_metrics.get("context_pce_right_index", context["context_right"])),
                    "context_pce_noise_value": _safe_float(context_metrics.get("context_pce_noise_value")),
                    "context_pce_noise_source": str(context_metrics.get("context_pce_noise_source", "")),
                    "context_pce_candidate_count": int(len(passed_noise_candidates)),
                    "context_pce_status": context_pce_status,
                    "context_pce_enabled": int(bool(pce_context_enabled)),
                    "context_pce_debug": context_metrics.get("context_pce_debug", {}),
                }
                local_ss6_cfg = dict(ss6_cfg)
                local_ss6_cfg["ss1_gate"] = float(local_context_ss1_threshold)
                if not np.isfinite(context_ss1) or context_ss1 < float(local_context_ss1_threshold):
                    summary_counts["local_candidates_rejected_by_context_ss1_low"] += int(len(passed_noise_candidates))
                    for candidate in passed_noise_candidates:
                        geom = candidate["geom"]
                        local_noise_would_have_been = _safe_float(candidate.get("local_noise_would_have_been"))
                        previous_local_noise_fallback = bool(candidate.get("previous_local_noise_fallback"))
                        attempt_rows.append(
                            {
                                "source_y": context["source_y"],
                                "source_x": context["source_x"],
                                "compact_y": context["compact_y"],
                                "compact_x": context["compact_x"],
                                "stage_index": int(pass_index),
                                "attempt_index_within_stage": "",
                                "attempt_type": "iterative_local_ss6",
                                "original_peak_index": context["original_peak_index"],
                                "detected_peak_index": int(candidate["dilation_peak"]),
                                "candidate_id": context["candidate_id"],
                                "original_ss6_branch": context["original_ss6_branch"],
                                "original_ss6_reason": context["original_ss6_reason"],
                                "local_ss6_branch": "",
                                "local_ss6_accept": 0,
                                "context_left": context["context_left"],
                                "context_right": context["context_right"],
                                "tested_left": int(geom.cell_left),
                                "tested_right": int(geom.cell_right),
                                "left_erosion_contact": int(geom.left_anchor),
                                "right_erosion_contact": int(geom.right_anchor),
                                "dilation_contact": int(candidate["dilation_peak"]),
                                "morph_window_used": int(morph_window_used),
                                "context_erosion_contacts": context_eros_json,
                                "context_dilation_contacts": context_dils_json,
                                "height_above_chord": float(candidate["height_above_chord"]),
                                "height_above_chord_noise_z": float(candidate["height_above_chord_noise_z"]),
                                "noise_value": float(candidate["noise_value"]),
                                "parent_noise_value": float(context["parent_noise_value"]),
                                "parent_noise_source": context["parent_noise_source"],
                                "previous_local_noise_value_if_available": local_noise_would_have_been,
                                "local_noise_would_have_been": local_noise_would_have_been,
                                "noise_ratio_local_to_parent": (float(local_noise_would_have_been / context["parent_noise_value"]) if np.isfinite(local_noise_would_have_been) and np.isfinite(context["parent_noise_value"]) and float(context["parent_noise_value"]) > 0.0 else np.nan),
                                "noise_fallback_used": int(bool(context["noise_fallback_used"]) or bool(previous_local_noise_fallback)),
                                "ss1_pce_noise_used": float(context["parent_noise_value"]) if np.isfinite(float(context["parent_noise_value"])) else np.nan,
                                "ss1_pce_noise_source": str(context["parent_noise_source"]),
                                "ss1_pce_noise_override_used": 1 if np.isfinite(float(context["parent_noise_value"])) and float(context["parent_noise_value"]) > 0.0 else 0,
                                "chord_crossing_detected": 0,
                                "chord_crossing_max": 0.0,
                                "chord_support_adjustment_applied": 0,
                                "replacement_increases_signal_max": 0.0,
                                "correction_applied": 0,
                                "status": "context_ss1_low",
                                "skipped_reason": "context_ss1_low",
                                "ss1": context_ss1,
                                **context_ss1_fields,
                            }
                        )
                    continue
                for candidate in passed_noise_candidates:
                    summary_counts["local_candidates_sent_to_ss6"] += 1
                    dilation_peak = int(candidate["dilation_peak"])
                    geom = candidate["geom"]
                    height_above_chord = float(candidate["height_above_chord"])
                    height_z = float(candidate["height_above_chord_noise_z"])
                    noise_value = float(candidate["noise_value"])
                    local_noise_would_have_been = _safe_float(candidate.get("local_noise_would_have_been"))
                    previous_local_noise_fallback = bool(candidate.get("previous_local_noise_fallback"))
                    use_context_pce = bool(pce_context_enabled and np.isfinite(context_pce))
                    pce_decision_source = "context" if use_context_pce else "local"
                    if use_context_pce:
                        summary_counts["local_candidates_using_context_pce"] += 1
                    elif pce_context_enabled:
                        summary_counts["local_candidates_fallback_to_local_pce"] += 1
                    local_row = _build_local_ss6_row(
                        signal=current_signal,
                        gradient_signal=gradient_signal,
                        base_row=context["row"],
                        peak_idx=int(dilation_peak),
                        geom=geom,
                        noise_value=float(noise_value),
                        ss6_cfg=local_ss6_cfg,
                        metric_ctx=metric_ctx,
                        global_edge_center=float(global_edge_center),
                        global_edge_scale=float(global_edge_scale),
                        parent_noise_source=str(context["parent_noise_source"]),
                        ss1_override=context_ss1,
                        pce_override=(context_pce if use_context_pce else None),
                    )
                    local_accept = int(_safe_float(local_row.get("ss6_accept", 0)))
                    local_branch = str(local_row.get("ss6_branch", "")).strip()
                    if local_accept != 1:
                        missing_metrics = _local_missing_metrics(local_row, ss6_cfg)
                        attempt_rows.append(
                            {
                                "source_y": context["source_y"],
                                "source_x": context["source_x"],
                                "compact_y": context["compact_y"],
                                "compact_x": context["compact_x"],
                                "stage_index": int(pass_index),
                                "attempt_index_within_stage": "",
                                "attempt_type": "iterative_local_ss6",
                                "original_peak_index": context["original_peak_index"],
                                "detected_peak_index": int(dilation_peak),
                                "candidate_id": context["candidate_id"],
                                "original_ss6_branch": context["original_ss6_branch"],
                                "original_ss6_reason": context["original_ss6_reason"],
                                "local_ss6_branch": local_branch,
                                "local_ss6_accept": local_accept,
                                "context_left": context["context_left"],
                                "context_right": context["context_right"],
                                "tested_left": int(geom.cell_left),
                                "tested_right": int(geom.cell_right),
                                "left_erosion_contact": int(geom.left_anchor),
                                "right_erosion_contact": int(geom.right_anchor),
                                "dilation_contact": int(dilation_peak),
                                "morph_window_used": int(morph_window_used),
                                "context_erosion_contacts": context_eros_json,
                                "context_dilation_contacts": context_dils_json,
                                "height_above_chord": float(height_above_chord),
                                "height_above_chord_noise_z": float(height_z),
                                "noise_value": float(noise_value),
                                "parent_noise_value": float(context["parent_noise_value"]),
                                "parent_noise_source": context["parent_noise_source"],
                                "previous_local_noise_value_if_available": float(local_noise_would_have_been) if np.isfinite(local_noise_would_have_been) else np.nan,
                                "local_noise_would_have_been": float(local_noise_would_have_been) if np.isfinite(local_noise_would_have_been) else np.nan,
                                "noise_ratio_local_to_parent": (float(local_noise_would_have_been / context["parent_noise_value"]) if np.isfinite(local_noise_would_have_been) and np.isfinite(context["parent_noise_value"]) and float(context["parent_noise_value"]) > 0.0 else np.nan),
                                "noise_fallback_used": int(bool(context["noise_fallback_used"]) or bool(previous_local_noise_fallback)),
                                "ss1_pce_noise_used": _safe_float(local_row.get("ss1_pce_noise_used")),
                                "ss1_pce_noise_source": str(local_row.get("ss1_pce_noise_source", "")),
                                "ss1_pce_noise_override_used": int(_safe_float(local_row.get("ss1_pce_noise_override_used", 0))),
                                "chord_crossing_detected": 0,
                                "chord_crossing_max": 0.0,
                                "chord_support_adjustment_applied": 0,
                                "replacement_increases_signal_max": 0.0,
                                "correction_applied": 0,
                                "status": ("local_missing_metric" if local_branch == "missing_metric" else "local_ss6_rejected"),
                                "skipped_reason": (missing_metrics if local_branch == "missing_metric" else local_branch),
                                "ss1": context_ss1,
                                "pce": _safe_float(local_row.get("ss6_pce")),
                                "edge": _safe_float(local_row.get("ss6_edge")),
                                "eel": _safe_float(local_row.get("ss6_eel")),
                                "resid": _safe_float(local_row.get("ss6_resid")),
                                "pce_local_audit": _safe_float(local_row.get("pce_local_audit", local_row.get("pce_negpref_t098_evidence_signed"))),
                                "pce_decision_source": pce_decision_source,
                                "pce_t98_debug_local": (local_row.get("pce_t98_debug", {}) if isinstance(local_row.get("pce_t98_debug", {}), dict) else {}),
                                **context_ss1_fields,
                            }
                        )
                        continue
                    summary_counts["local_candidates_accepted_by_ss6"] += 1
                    local_plans.append(
                        {
                            "source_y": context["source_y"],
                            "source_x": context["source_x"],
                            "compact_y": context["compact_y"],
                            "compact_x": context["compact_x"],
                            "attempt_type": "iterative_local_ss6",
                            "original_peak_index": context["original_peak_index"],
                            "detected_peak_index": int(dilation_peak),
                            "candidate_id": context["candidate_id"],
                            "original_ss6_branch": context["original_ss6_branch"],
                            "original_ss6_reason": context["original_ss6_reason"],
                            "local_ss6_branch": local_branch,
                            "local_ss6_accept": local_accept,
                            "context_left": context["context_left"],
                            "context_right": context["context_right"],
                            "tested_left": int(geom.cell_left),
                            "tested_right": int(geom.cell_right),
                            "left_erosion_contact": int(geom.left_anchor),
                            "right_erosion_contact": int(geom.right_anchor),
                            "dilation_contact": int(dilation_peak),
                            "morph_window_used": int(morph_window_used),
                            "context_erosion_contacts": context_eros_json,
                            "context_dilation_contacts": context_dils_json,
                            "left_anchor": int(geom.left_anchor),
                            "right_anchor": int(geom.right_anchor),
                            "cell_left": int(geom.cell_left),
                            "cell_right": int(geom.cell_right),
                            "used_context_boundary_anchor": int(geom.used_context_boundary_anchor),
                            "height_above_chord": float(height_above_chord),
                            "height_above_chord_noise_z": float(height_z),
                            "noise_value": float(noise_value),
                            "parent_noise_value": float(context["parent_noise_value"]),
                            "parent_noise_source": context["parent_noise_source"],
                            "previous_local_noise_value_if_available": float(local_noise_would_have_been) if np.isfinite(local_noise_would_have_been) else np.nan,
                            "local_noise_would_have_been": float(local_noise_would_have_been) if np.isfinite(local_noise_would_have_been) else np.nan,
                            "noise_ratio_local_to_parent": (float(local_noise_would_have_been / context["parent_noise_value"]) if np.isfinite(local_noise_would_have_been) and np.isfinite(context["parent_noise_value"]) and float(context["parent_noise_value"]) > 0.0 else np.nan),
                            "noise_fallback_used": int(bool(context["noise_fallback_used"]) or bool(previous_local_noise_fallback)),
                            "ss1_pce_noise_used": _safe_float(local_row.get("ss1_pce_noise_used")),
                            "ss1_pce_noise_source": str(local_row.get("ss1_pce_noise_source", "")),
                            "ss1_pce_noise_override_used": int(_safe_float(local_row.get("ss1_pce_noise_override_used", 0))),
                            "status": "corrected_iterative_local_ss6",
                            "score": float(height_z),
                            "ss1": context_ss1,
                            "pce": _safe_float(local_row.get("ss6_pce")),
                            "edge": _safe_float(local_row.get("ss6_edge")),
                            "eel": _safe_float(local_row.get("ss6_eel")),
                            "resid": _safe_float(local_row.get("ss6_resid")),
                            "pce_local_audit": _safe_float(local_row.get("pce_local_audit", local_row.get("pce_negpref_t098_evidence_signed"))),
                            "pce_decision_source": pce_decision_source,
                            "pce_t98_debug_local": (local_row.get("pce_t98_debug", {}) if isinstance(local_row.get("pce_t98_debug", {}), dict) else {}),
                            **context_ss1_fields,
                        }
                    )

            if not local_plans:
                break
            kept_local, skipped_local = _resolve_stage_plans(local_plans)
            for item in skipped_local:
                summary_counts["skipped_overlap_duplicate"] += 1
                item["stage_index"] = int(pass_index)
                item["attempt_index_within_stage"] = ""
                attempt_rows.append(item)
            applied_local = _apply_plans(int(pass_index), int(y), int(x), kept_local, corrected_mask)
            spectrum_corrections += int(applied_local)
            summary_counts["local_candidates_corrected"] += int(applied_local)
            if applied_local > 0:
                max_pass_index = int(pass_index)

        current_signal = np.asarray(corrected[y, x, :], dtype=float)
        _erosion, _dilation, erosion_contacts, dilation_contacts = _compute_contacts(current_signal, morph_window_used)
        mask_plans: list[dict[str, Any]] = []
        for context in contexts:
            context_eros, context_dils = _context_contact_lists(
                erosion_contacts,
                dilation_contacts,
                int(context["context_left"]),
                int(context["context_right"]),
            )
            context_eros_json = _serialize_contact_list(context_eros)
            context_dils_json = _serialize_contact_list(context_dils)
            for dilation_peak in _local_dilation_candidates(dilation_contacts, int(context["context_left"]), int(context["context_right"])):
                geom = _geometry_for_peak(
                    current_signal,
                    erosion_contacts,
                    dilation_contacts,
                    int(dilation_peak),
                    int(context["context_left"]),
                    int(context["context_right"]),
                )
                if geom.right_anchor <= geom.left_anchor:
                    continue
                overlap_fraction, overlaps_mask, fully_inside_mask = _mask_overlap_fraction(corrected_mask, int(geom.left_anchor), int(geom.right_anchor))
                if overlap_fraction < 0.8:
                    continue
                summary_counts["mask_cleanup_candidates_tested"] += 1
                noise_value = float(context["parent_noise_value"])
                if not np.isfinite(noise_value) or noise_value <= 0.0:
                    summary_counts["local_candidates_with_missing_parent_noise"] += 1
                    continue
                height_above_chord, _ = _height_above_chord(current_signal, int(dilation_peak), geom.left_anchor, geom.right_anchor)
                height_z = float(height_above_chord / noise_value) if np.isfinite(height_above_chord) else float("nan")
                if not np.isfinite(height_z) or height_z < float(noise_height_factor):
                    summary_counts["mask_cleanup_rejected_below_noise_height"] += 1
                    attempt_rows.append(
                        {
                            "source_y": context["source_y"],
                            "source_x": context["source_x"],
                            "compact_y": context["compact_y"],
                            "compact_x": context["compact_x"],
                            "stage_index": int(max_pass_index + 1),
                            "attempt_index_within_stage": "",
                            "attempt_type": "mask_cleanup",
                            "original_peak_index": context["original_peak_index"],
                            "detected_peak_index": int(dilation_peak),
                            "candidate_id": context["candidate_id"],
                            "original_ss6_branch": context["original_ss6_branch"],
                            "original_ss6_reason": context["original_ss6_reason"],
                            "local_ss6_branch": "",
                            "local_ss6_accept": "",
                            "context_left": context["context_left"],
                            "context_right": context["context_right"],
                            "tested_left": int(geom.cell_left),
                            "tested_right": int(geom.cell_right),
                            "left_erosion_contact": int(geom.left_anchor),
                            "right_erosion_contact": int(geom.right_anchor),
                            "dilation_contact": int(dilation_peak),
                            "morph_window_used": int(morph_window_used),
                            "context_erosion_contacts": context_eros_json,
                            "context_dilation_contacts": context_dils_json,
                            "height_above_chord": float(height_above_chord),
                            "height_above_chord_noise_z": float(height_z),
                            "noise_value": float(noise_value),
                            "parent_noise_value": float(context["parent_noise_value"]),
                            "parent_noise_source": context["parent_noise_source"],
                            "corrected_mask_overlap_fraction": float(overlap_fraction),
                            "overlaps_corrected_mask": int(overlaps_mask),
                            "fully_inside_corrected_mask": int(fully_inside_mask),
                            "mask_cleanup_candidate": 1,
                            "mask_cleanup_status": "rejected_mask_residual_below_noise_height",
                            "chord_crossing_detected": 0,
                            "chord_crossing_max": 0.0,
                            "chord_support_adjustment_applied": 0,
                            "replacement_increases_signal_max": 0.0,
                            "correction_applied": 0,
                            "status": "rejected_mask_residual_below_noise_height",
                            "skipped_reason": "below_noise_height",
                        }
                    )
                    continue
                mask_plans.append(
                    {
                        "source_y": context["source_y"],
                        "source_x": context["source_x"],
                        "compact_y": context["compact_y"],
                        "compact_x": context["compact_x"],
                        "attempt_type": "mask_cleanup",
                        "original_peak_index": context["original_peak_index"],
                        "detected_peak_index": int(dilation_peak),
                        "candidate_id": context["candidate_id"],
                        "original_ss6_branch": context["original_ss6_branch"],
                        "original_ss6_reason": context["original_ss6_reason"],
                        "local_ss6_branch": "",
                        "local_ss6_accept": "",
                        "context_left": context["context_left"],
                        "context_right": context["context_right"],
                        "tested_left": int(geom.cell_left),
                        "tested_right": int(geom.cell_right),
                        "left_erosion_contact": int(geom.left_anchor),
                        "right_erosion_contact": int(geom.right_anchor),
                        "dilation_contact": int(dilation_peak),
                        "morph_window_used": int(morph_window_used),
                        "context_erosion_contacts": context_eros_json,
                        "context_dilation_contacts": context_dils_json,
                        "left_anchor": int(geom.left_anchor),
                        "right_anchor": int(geom.right_anchor),
                        "cell_left": int(geom.cell_left),
                        "cell_right": int(geom.cell_right),
                        "used_context_boundary_anchor": int(geom.used_context_boundary_anchor),
                        "height_above_chord": float(height_above_chord),
                        "height_above_chord_noise_z": float(height_z),
                        "noise_value": float(noise_value),
                        "parent_noise_value": float(context["parent_noise_value"]),
                        "parent_noise_source": context["parent_noise_source"],
                        "corrected_mask_overlap_fraction": float(overlap_fraction),
                        "overlaps_corrected_mask": int(overlaps_mask),
                        "fully_inside_corrected_mask": int(fully_inside_mask),
                        "mask_cleanup_candidate": 1,
                        "mask_cleanup_status": "corrected_mask_residual_cleanup",
                        "status": "corrected_mask_residual_cleanup",
                        "score": float(height_z),
                    }
                )
        if mask_plans:
            kept_mask, skipped_mask = _resolve_stage_plans(mask_plans)
            for item in skipped_mask:
                summary_counts["skipped_overlap_duplicate"] += 1
                item["stage_index"] = int(max_pass_index + 1)
                item["attempt_index_within_stage"] = ""
                attempt_rows.append(item)
            applied_mask = _apply_plans(int(max_pass_index + 1), int(y), int(x), kept_mask, corrected_mask)
            spectrum_corrections += int(applied_mask)
            summary_counts["mask_cleanup_corrected"] += int(applied_mask)
            if applied_mask > 0:
                max_pass_index = int(max_pass_index + 1)

        summary_counts["max_pass_index_used"] = max(int(summary_counts["max_pass_index_used"]), int(max_pass_index))
        per_spectrum_correction_counts[f"{int(y)}:{int(x)}"] = int(spectrum_corrections)
        if spectrum_corrections > 0:
            summary_counts["spectra_with_corrections"] += 1

        for context in contexts:
            applied_for_candidate = sum(
                1
                for row in attempt_rows
                if str(row.get("candidate_id", "")) == str(context["candidate_id"])
                and int(_safe_float(row.get("correction_applied", 0))) == 1
            )
            stopped_reason = "corrected" if applied_for_candidate > 0 else "skipped_parent_no_erosion_neighbors"
            candidate_rows = [row for row in attempt_rows if str(row.get("candidate_id", "")) == str(context["candidate_id"])]
            if candidate_rows:
                stopped_reason = str(candidate_rows[-1].get("status", stopped_reason))
            debug_rows.append(
                {
                    "source_y": context["source_y"],
                    "source_x": context["source_x"],
                    "compact_y": context["compact_y"],
                    "compact_x": context["compact_x"],
                    "original_peak_index": context["original_peak_index"],
                    "candidate_id": context["candidate_id"],
                    "ss6_branch": context["original_ss6_branch"],
                    "fixed_context_left": context["context_left"],
                    "fixed_context_right": context["context_right"],
                    "morph_window": int(morph_window_used),
                    "corrections_applied": int(applied_for_candidate),
                    "stopped_reason": stopped_reason,
                    "max_iterations_reached": int(max_pass_index >= int(max_iterations)),
                    "technical_failure": 0,
                    "technical_failure_reason": "",
                }
            )

    if timings_out is not None:
        timings_out["despike correction"] = time.perf_counter() - t0

    summary = {
        "accepted_ss6_parent_candidates": int(summary_counts["accepted_ss6_parent_candidates"]),
        "spectra_with_accepted_spikes": int(len(accepted_by_pixel)),
        "parent_corrected": int(summary_counts["parent_corrected"]),
        "parent_skipped_below_noise_height": int(summary_counts["parent_skipped_below_noise_height"]),
        "parent_skipped_no_erosion_neighbors": int(summary_counts["parent_skipped_no_erosion_neighbors"]),
        "local_candidates_from_dilation_contacts": int(summary_counts["local_candidates_from_dilation_contacts"]),
        "local_candidates_passed_noise_height": int(summary_counts["local_candidates_passed_noise_height"]),
        "local_candidates_rejected_by_noise_height": int(summary_counts["local_candidates_rejected_by_noise_height"]),
        "local_contexts_with_context_ss1": int(summary_counts["local_contexts_with_context_ss1"]),
        "local_candidates_rejected_by_context_ss1_low": int(summary_counts["local_candidates_rejected_by_context_ss1_low"]),
        "local_candidates_rejected_by_context_ss1": int(summary_counts["local_candidates_rejected_by_context_ss1_low"]),
        "local_candidates_sent_to_ss6": int(summary_counts["local_candidates_sent_to_ss6"]),
        "local_candidates_accepted_by_ss6": int(summary_counts["local_candidates_accepted_by_ss6"]),
        "local_candidates_corrected": int(summary_counts["local_candidates_corrected"]),
        "pce_context_enabled": bool(pce_context_enabled),
        "local_context_pce_computed": int(summary_counts["local_context_pce_computed"]),
        "local_context_pce_missing": int(summary_counts["local_context_pce_missing"]),
        "local_candidates_using_context_pce": int(summary_counts["local_candidates_using_context_pce"]),
        "local_candidates_fallback_to_local_pce": int(summary_counts["local_candidates_fallback_to_local_pce"]),
        "mask_cleanup_candidates_tested": int(summary_counts["mask_cleanup_candidates_tested"]),
        "mask_cleanup_corrected": int(summary_counts["mask_cleanup_corrected"]),
        "mask_cleanup_rejected_below_noise_height": int(summary_counts["mask_cleanup_rejected_below_noise_height"]),
        "ss1_context_threshold": float(local_context_ss1_threshold),
        "local_candidates_with_parent_noise": int(summary_counts["local_candidates_with_parent_noise"]),
        "local_candidates_with_missing_parent_noise": int(summary_counts["local_candidates_with_missing_parent_noise"]),
        "median_noise_ratio_local_to_parent_if_available": (float(np.median(np.asarray(summary_counts["noise_ratio_values"], dtype=float))) if summary_counts["noise_ratio_values"] else float("nan")),
        "max_noise_ratio_local_to_parent_if_available": (float(np.max(np.asarray(summary_counts["noise_ratio_values"], dtype=float))) if summary_counts["noise_ratio_values"] else float("nan")),
        "total_corrections_applied": int(summary_counts["total_corrections_applied"]),
        "spectra_with_corrections": int(summary_counts["spectra_with_corrections"]),
        "technical_failures": int(summary_counts["technical_failures"]),
        "max_pass_index_used": int(summary_counts["max_pass_index_used"]),
        "chord_crossing_adjusted_count": int(summary_counts["chord_crossing_adjusted_count"]),
        "replacement_invariant_violations": int(summary_counts["replacement_invariant_violations"]),
        "output_paths": {
            "corrected_path": str(corrected_path) if corrected_path is not None else "",
            "debug_path": str(debug_path) if debug_path is not None else "",
            "attempts_path": str(attempts_path) if attempts_path is not None else "",
            "summary_path": str(summary_path) if summary_path is not None else "",
        },
        "viewer_cache_path": str(cache_path),
        "viewer_cache_shape": [int(v) for v in np.asarray(cache["spectra"]).shape],
        "corrected_shape": [int(v) for v in np.asarray(corrected).shape],
        "config_path": (str(Path(config_path)) if config_path is not None else ""),
        "viewer_cache_identity": dict(cache.get("metadata", {})).get("viewer_cache_identity", {}),
        "join_info": join_info,
    }

    t0 = time.perf_counter()
    if corrected_path is not None:
        corrected_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "corrected_spectra": np.asarray(corrected, dtype=np.float32),
            "x_axis": np.asarray(cache["x_axis"], dtype=float),
            "coord_map_json": np.array([json.dumps(cache.get("coord_map", []), ensure_ascii=False)], dtype=object),
            "despike_chords_json": np.array([json.dumps([chord.__dict__ for chord in chords], ensure_ascii=False)], dtype=object),
            "despike_debug_rows_json": np.array([json.dumps(debug_rows, ensure_ascii=False)], dtype=object),
            "despike_attempt_rows_json": np.array([json.dumps(attempt_rows, ensure_ascii=False)], dtype=object),
            "per_spectrum_correction_counts_json": np.array([json.dumps(per_spectrum_correction_counts, ensure_ascii=False)], dtype=object),
            "metadata_json": np.array([json.dumps(summary, ensure_ascii=False)], dtype=object),
        }
        np.savez_compressed(corrected_path, **payload)
    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "source_y", "source_x", "compact_y", "compact_x", "original_peak_index", "candidate_id", "ss6_branch",
            "fixed_context_left", "fixed_context_right", "morph_window", "corrections_applied", "stopped_reason",
            "max_iterations_reached", "technical_failure", "technical_failure_reason",
        ]
        with debug_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in debug_rows:
                writer.writerow(row)
    if attempts_path is not None:
        attempts_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "source_y", "source_x", "compact_y", "compact_x",
            "stage_index", "attempt_index_within_stage", "attempt_type",
            "original_peak_index", "detected_peak_index", "candidate_id",
            "original_ss6_branch", "original_ss6_reason", "local_ss6_branch", "local_ss6_accept",
            "context_left", "context_right", "tested_left", "tested_right",
            "left_erosion_contact", "right_erosion_contact", "dilation_contact", "morph_window_used",
            "context_erosion_contacts", "context_dilation_contacts",
            "left_anchor", "right_anchor", "cell_left", "cell_right", "used_context_boundary_anchor",
            "raw_peak_value_before", "corrected_peak_value_after", "correction_height",
            "height_above_chord", "height_above_chord_noise_z", "noise_value",
            "parent_noise_value", "parent_noise_source", "previous_local_noise_value_if_available", "local_noise_would_have_been", "noise_ratio_local_to_parent", "noise_fallback_used",
            "ss1_pce_noise_used", "ss1_pce_noise_source", "ss1_pce_noise_override_used",
            "context_ss1", "context_ss1_peak_index", "context_ss1_peak_x", "context_ss1_noise_value", "context_ss1_noise_source", "context_ss1_candidate_count", "context_ss1_threshold", "local_ss1_gate_used",
            "context_pce", "context_pce_peak_index", "context_pce_peak_x", "context_pce_left_index", "context_pce_right_index", "context_pce_noise_value", "context_pce_noise_source", "context_pce_candidate_count", "context_pce_status", "context_pce_enabled", "pce_local_audit", "pce_decision_source",
            "corrected_mask_overlap_fraction", "overlaps_corrected_mask", "fully_inside_corrected_mask", "mask_cleanup_candidate", "mask_cleanup_status",
            "chord_crossing_detected", "chord_crossing_max", "chord_support_adjustment_applied", "replacement_increases_signal_max",
            "ss1", "pce", "edge", "eel", "resid",
            "correction_applied", "status", "skipped_reason",
        ]
        with attempts_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in attempt_rows:
                writer.writerow(row)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if timings_out is not None:
        timings_out["write outputs"] = time.perf_counter() - t0

    return DespikeArtifacts(
        final_corrected=np.asarray(corrected, dtype=np.float32),
        chords=chords,
        debug_rows=debug_rows,
        summary=summary,
    )

def load_despike_bundle(path: Path | str) -> dict[str, Any]:
    data = np.load(Path(path), allow_pickle=True)
    return {
        "corrected_spectra": np.asarray(data["corrected_spectra"]),
        "x_axis": np.asarray(data["x_axis"]) if "x_axis" in data.files else None,
        "coord_map": json.loads(str(np.asarray(data["coord_map_json"]).reshape(-1)[0])) if "coord_map_json" in data.files else [],
        "despike_chords": json.loads(str(np.asarray(data["despike_chords_json"]).reshape(-1)[0])) if "despike_chords_json" in data.files else [],
        "despike_debug_rows": json.loads(str(np.asarray(data["despike_debug_rows_json"]).reshape(-1)[0])) if "despike_debug_rows_json" in data.files else [],
        "despike_attempt_rows": json.loads(str(np.asarray(data["despike_attempt_rows_json"]).reshape(-1)[0])) if "despike_attempt_rows_json" in data.files else [],
        "per_spectrum_correction_counts": json.loads(str(np.asarray(data["per_spectrum_correction_counts_json"]).reshape(-1)[0])) if "per_spectrum_correction_counts_json" in data.files else {},
        "metadata": json.loads(str(np.asarray(data["metadata_json"]).reshape(-1)[0])) if "metadata_json" in data.files else {},
    }
