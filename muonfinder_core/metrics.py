from __future__ import annotations

"""
Clean core wrapper around the currently active primary metric formulas.

This module intentionally keeps the exact active metric formulas by calling the
legacy feature functions from the repository root. The dependency is narrow and
documented: only the active background MAD, curvature/PCE, and edge/RVE metric
implementations are used here. The surrounding pipeline, cache, and viewer are
fully core-native.
"""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .data_model import BackgroundNoiseEstimate, CandidateSegment

from .legacy_formula_adapter import (
    compute_edge_width_metrics,
    compute_peak_curvature_features,
    compute_spike_score_v2_features,
    estimate_background_mad,
)


@dataclass(frozen=True)
class MetricComputationContext:
    feature_signal_source: str = "gradient"
    edge_context_pad_pts: int = 20
    edge_context_min_pad_pts: int = 10
    edge_context_max_pad_pts: int = 80
    edge_dense_min_snr: float = 3.0
    edge_robust_reference_enabled: bool = True
    edge_noise_guard_enabled: bool = True
    edge_noise_guard_factor: float = 3.0
    edge_use_enhanced_spike_mapping: bool = True
    edge_mapping_enable_merge_guard: bool = True
    edge_mapping_noise_guard_enabled: bool = False
    edge_mapping_levels_desc: Sequence[int] = (95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5)
    edge_mapping_refine_step_percent: int = 1
    edge_mapping_min_level_percent: int = 1
    edge_mapping_require_closed_interval: bool = True
    edge_mapping_use_apex_component: bool = True
    edge_mapping_max_width_jump_factor: float = 2.5
    edge_mapping_max_width_jump_points: float = 8.0
    edge_mapping_fallback_to_old: bool = False
    recdw_z_clip: float = 6.0
    recdw_support_z_scale: float = 1.0


def robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    center = float(np.median(x))
    mad = float(np.median(np.abs(x - center)))
    scale = float(1.4826 * mad)
    if np.isfinite(scale) and scale > 1e-12:
        return center, scale
    q25, q75 = np.percentile(x, [25.0, 75.0])
    scale = float((float(q75) - float(q25)) / 1.349)
    if np.isfinite(scale) and scale > 1e-12:
        return center, scale
    scale = float(np.std(x))
    return (center, scale) if np.isfinite(scale) and scale > 1e-12 else (center, float("nan"))


def sigmoid_support(z: float, scale: float, clip: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    z_clip = float(np.clip(float(z), -abs(float(clip)), abs(float(clip))))
    return float(1.0 / (1.0 + np.exp(-z_clip / max(float(scale), 1e-12))))


def estimate_bg_noise(signal: np.ndarray, left: int, right: int) -> BackgroundNoiseEstimate:
    return BackgroundNoiseEstimate(
        method="bg_mad",
        value=float(estimate_background_mad(np.asarray(signal, dtype=float), int(left), int(right))),
    )


def estimate_edge_context_bg_mad(signal: np.ndarray, left: int, right: int) -> BackgroundNoiseEstimate:
    x = np.asarray(signal, dtype=float)
    n = int(x.size)
    a = int(np.clip(left, 0, max(0, n - 1)))
    b = int(np.clip(right, 0, max(0, n - 1)))
    if b < a:
        a, b = b, a
    feature_context = max(10, 3 * max(1, b - a + 1))
    l0 = max(0, a - feature_context)
    r1 = min(n, b + feature_context + 1)
    bg = np.concatenate([x[l0:a], x[b + 1 : r1]])
    if bg.size < 5:
        bg = np.concatenate([x[:a], x[b + 1 :]])
    if bg.size < 5:
        bg = x
    bg_med = float(np.nanmedian(bg)) if bg.size else 0.0
    bg_mad = max(float(np.nanmedian(np.abs(bg - bg_med))) if bg.size else 0.0, 1e-12)
    return BackgroundNoiseEstimate(method="edge_context_bg_mad", value=float(bg_mad))


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _normalize_edge_debug(debug: dict[str, Any], width_sum: float) -> dict[str, Any]:
    out = dict(debug)
    reason = str(out.get("reason", "")).strip().lower()
    guard_passed = bool(reason == "ok")
    out["edge_guard_passed"] = guard_passed
    out["edge_guard_reason"] = reason or "unknown"
    before_guard = _float_or_nan(out.get("dense_width_0_90"))
    if not np.isfinite(before_guard):
        before_guard = float(width_sum)
    out["edge_value_before_guard"] = before_guard
    out["edge_value_after_guard"] = float(width_sum) if guard_passed else float("nan")
    out["edge_selected_context"] = {
        "measurement_left": int(out.get("measurement_left", -1)),
        "measurement_right": int(out.get("measurement_right", -1)),
    }
    selected_levels: list[dict[str, float | int]] = []
    for level in range(0, 91, 5):
        if str(out.get(f"width_reason_{level}", "")).strip().lower() != "ok":
            continue
        left_cross = _float_or_nan(out.get(f"left_cross_{level}"))
        right_cross = _float_or_nan(out.get(f"right_cross_{level}"))
        level_y = _float_or_nan(out.get(f"level_{level}"))
        if not (np.isfinite(left_cross) and np.isfinite(right_cross) and np.isfinite(level_y)):
            continue
        selected_levels.append(
            {
                "percent": int(level),
                "left_cross": float(left_cross),
                "right_cross": float(right_cross),
                "level_value": float(level_y),
            }
        )
    out["edge_selected_levels"] = selected_levels
    out["edge_selected_support_points"] = [
        {
            "percent": int(item["percent"]),
            "left_cross": float(item["left_cross"]),
            "right_cross": float(item["right_cross"]),
            "level_value": float(item["level_value"]),
        }
        for item in selected_levels
    ]
    return out


def compute_ss1_pce_features(
    *,
    raw_signal: np.ndarray,
    gradient_signal: np.ndarray | None,
    seg: CandidateSegment,
    feature_signal_source: str,
) -> dict[str, Any]:
    src = str(feature_signal_source).strip().lower()
    signal = np.asarray(raw_signal, dtype=float) if src == "raw" or gradient_signal is None else np.asarray(gradient_signal, dtype=float)
    n = int(signal.size)
    a = int(np.clip(seg.start, 0, n - 1))
    b = int(np.clip(seg.end, 0, n - 1))
    p = int(np.clip(seg.peak_index, 0, n - 1))
    if b < a:
        a, b = b, a
    out: dict[str, Any] = {
        "spike_score_v1": float("nan"),
        "pce_negpref_t098_evidence_signed": float("nan"),
    }
    if not (a < p < b):
        return out
    segment = np.asarray(signal[a : b + 1], dtype=float)
    if segment.size < 3:
        return out
    d = np.diff(segment)
    rise = d[: max(1, p - a)]
    fall = d[max(1, p - a) :]
    rise_slope = float(np.nanmax(rise)) if rise.size else 0.0
    fall_slope = float(np.nanmin(fall)) if fall.size else 0.0
    bg = estimate_bg_noise(signal, a, b)
    bg_mad = max(float(bg.value), 1e-12)
    out["bg_mad"] = float(bg_mad)
    out["rise_slope_z"] = float(rise_slope / bg_mad)
    out["fall_slope_z"] = float(abs(fall_slope) / bg_mad)
    sr = float(np.tanh(out["rise_slope_z"] / 6.0))
    sf = float(np.tanh(out["fall_slope_z"] / 6.0))
    out["spike_score_v1"] = float(0.5 * sr + 0.5 * sf)
    peak_rel = int(np.clip(p - a, 0, segment.size - 1))
    out.update(compute_peak_curvature_features(segment, bg_mad, peak_rel=peak_rel))
    out.update(compute_spike_score_v2_features(out))
    return out


def compute_raw_edge_metric(
    *,
    raw_signal: np.ndarray,
    seg: CandidateSegment,
    candidate_noise_estimate: float | None,
    ctx: MetricComputationContext,
) -> dict[str, Any]:
    raw = np.asarray(raw_signal, dtype=float)
    n = int(raw.size)
    edge_pad = max(int(ctx.edge_context_pad_pts), int(ctx.edge_context_min_pad_pts))
    edge_pad = min(edge_pad, int(ctx.edge_context_max_pad_pts))
    left = max(0, int(seg.start) - edge_pad)
    right = min(n - 1, int(seg.end) + edge_pad)
    bg = estimate_edge_context_bg_mad(raw, int(seg.start), int(seg.end))
    edge_metrics = compute_edge_width_metrics(
        raw,
        detection_left=int(left),
        detection_right=int(right),
        prefix="raw_edge_ctx",
        apex_idx=int(seg.peak_index),
        bg_mad=float(bg.value),
        include_low_root_metrics=True,
        low_root_noise_k_mad=float(ctx.edge_dense_min_snr),
        use_enhanced_spike_mapping=bool(ctx.edge_use_enhanced_spike_mapping),
        mapping_levels_desc=tuple(int(v) for v in ctx.edge_mapping_levels_desc),
        mapping_refine_step_percent=int(ctx.edge_mapping_refine_step_percent),
        mapping_enable_merge_guard=bool(ctx.edge_mapping_enable_merge_guard),
        mapping_noise_guard_enabled=bool(ctx.edge_mapping_noise_guard_enabled),
        mapping_min_level_percent=int(ctx.edge_mapping_min_level_percent),
        mapping_require_closed_interval=bool(ctx.edge_mapping_require_closed_interval),
        mapping_use_apex_component=bool(ctx.edge_mapping_use_apex_component),
        mapping_max_width_jump_factor=float(ctx.edge_mapping_max_width_jump_factor),
        mapping_max_width_jump_points=float(ctx.edge_mapping_max_width_jump_points),
        mapping_fallback_to_old=bool(ctx.edge_mapping_fallback_to_old),
        robust_reference_enabled=bool(ctx.edge_robust_reference_enabled),
        robust_reference_noise=(float(candidate_noise_estimate) if candidate_noise_estimate is not None else float(bg.value)),
        edge_noise_guard_enabled=bool(ctx.edge_noise_guard_enabled),
        edge_noise_guard_factor=float(ctx.edge_noise_guard_factor),
        edge_noise_guard_value=(float(candidate_noise_estimate) if candidate_noise_estimate is not None else None),
    )
    value = edge_metrics.get("raw_edge_ctx_dense_width_sum_0_90", np.nan)
    debug = edge_metrics.get("raw_edge_ctx_debug", {})
    raw_value = float(value) if value is not None else np.nan
    debug_dict = debug if isinstance(debug, dict) else {}
    dense_debug = debug_dict.get("dense_width_0_90", {}) if isinstance(debug_dict, dict) else {}
    edge_noise_ratio = float(dense_debug.get("root_snr", np.nan)) if isinstance(dense_debug, dict) else float("nan")
    edge_guard_enabled = bool(ctx.edge_noise_guard_enabled)
    edge_guard_factor = float(ctx.edge_noise_guard_factor)
    edge_guard_passed = bool((not edge_guard_enabled) or (np.isfinite(edge_noise_ratio) and edge_noise_ratio >= edge_guard_factor))
    if not np.isfinite(raw_value):
        edge_guard_reason = "edge_value_missing"
    elif not edge_guard_enabled:
        edge_guard_reason = "guard_disabled"
    elif not np.isfinite(edge_noise_ratio):
        edge_guard_reason = "edge_noise_ratio_missing"
    elif edge_guard_passed:
        edge_guard_reason = "guard_passed"
    else:
        edge_guard_reason = "below_edge_noise_guard"
    clean_value = float(raw_value) if edge_guard_passed else float("nan")
    clean_debug = _normalize_edge_debug(debug_dict, raw_value)
    clean_debug["edge_noise_ratio"] = float(edge_noise_ratio) if np.isfinite(edge_noise_ratio) else float("nan")
    clean_debug["edge_noise_guard_factor"] = float(edge_guard_factor)
    clean_debug["edge_guard_passed"] = bool(edge_guard_passed)
    clean_debug["edge_guard_reason"] = str(edge_guard_reason)
    clean_debug["edge_value_before_guard"] = float(raw_value) if np.isfinite(raw_value) else float("nan")
    clean_debug["edge_value_after_guard"] = float(clean_value) if np.isfinite(clean_value) else float("nan")
    return {
        "recdw_sum_0_90": clean_value,
        "edge_debug": clean_debug,
        "edge_noise_ratio": clean_debug["edge_noise_ratio"],
        "edge_noise_guard_factor": clean_debug["edge_noise_guard_factor"],
        "edge_guard_passed": clean_debug["edge_guard_passed"],
        "edge_guard_reason": clean_debug["edge_guard_reason"],
        "edge_value_before_guard": clean_debug["edge_value_before_guard"],
        "edge_value_after_guard": clean_debug["edge_value_after_guard"],
    }


def finalize_edge_evidence(rows: list[dict[str, Any]], ctx: MetricComputationContext) -> None:
    vals = np.asarray([float(row.get("recdw_sum_0_90", np.nan)) for row in rows], dtype=float)
    center, scale = robust_center_scale(vals)
    for row in rows:
        value = float(row.get("recdw_sum_0_90", np.nan))
        if np.isfinite(value) and np.isfinite(center) and np.isfinite(scale) and scale > 1e-12:
            z = float((value - center) / scale)
            support = sigmoid_support(z, float(ctx.recdw_support_z_scale), float(ctx.recdw_z_clip))
            row["recdw_sum_0_90_z"] = float(z)
            row["recdw_sum_0_90_support01"] = float(support)
            row["recdw_sum_0_90_raman_veto_evidence_signed"] = float(2.0 * support - 1.0)
        else:
            row["recdw_sum_0_90_z"] = float("nan")
            row["recdw_sum_0_90_support01"] = float("nan")
            row["recdw_sum_0_90_raman_veto_evidence_signed"] = float("nan")
