from __future__ import annotations

"""
Clean core metric computation.

Primary curvature/PCE and background MAD still use a narrow legacy adapter.
EDGE/RVE helper logic is core-native.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data_model import BackgroundNoiseEstimate, CandidateSegment
from .edge import EDGE_ALL_LEVELS_ASC, EDGE_DENSE_LEVELS_ASC, compute_edge_metric
from .legacy_formula_adapter import (
    CURVATURE_NEGPREF_LOCAL_RADIUS,
    compute_curvature_negpref_diagnostics,
    compute_peak_curvature_features,
    estimate_background_mad,
)


PCE_SUPPORT_FULL = -1992.0
PCE_SUPPORT_ZERO = -273.0
PCE_VETO_ZERO = -273.0
PCE_VETO_FULL = 503.0


@dataclass(frozen=True)
class MetricComputationContext:
    feature_signal_source: str = "gradient"
    noise_source: str = "morph_range"
    noise_height_factor: float = 3.0
    edge_foot_method: str = "noise_quantized_component"
    edge_pre_level_step_noise: float = 1.0
    edge_pre_transient_tolerance_levels: int = 2
    edge_pre_min_stable_levels: int = 2
    edge_neighbor_structure_factor: float = 3.0
    edge_dense_context_min_pad_pts: int = 10
    edge_dense_context_pad_pts: int = 20
    edge_dense_context_max_pad_pts: int = 120
    edge_context_expand_step_pts: int = 10
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


def _ramp_up(x: float, low: float, high: float) -> float:
    if not np.isfinite(x):
        return 0.0
    if high <= low:
        return 1.0 if x >= high else 0.0
    return float(np.clip((float(x) - float(low)) / (float(high) - float(low)), 0.0, 1.0))


def _ramp_down(x: float, low: float, high: float) -> float:
    return float(1.0 - _ramp_up(x, low, high))


def _signed_evidence(support01: float, veto01: float = 0.0) -> float:
    support = float(np.clip(support01, 0.0, 1.0))
    veto = float(np.clip(veto01, 0.0, 1.0))
    return float(np.clip(support - veto, -1.0, 1.0))


def estimate_bg_noise(signal: np.ndarray, left: int, right: int) -> BackgroundNoiseEstimate:
    return BackgroundNoiseEstimate(
        method="bg_mad",
        value=float(estimate_background_mad(np.asarray(signal, dtype=float), int(left), int(right))),
    )


def build_pce_t98_debug(signal_segment: np.ndarray, *, peak_rel: int) -> dict[str, Any]:
    seg = np.asarray(signal_segment, dtype=float)
    if seg.size < 3:
        return {}
    diag = compute_curvature_negpref_diagnostics(
        seg,
        peak_rel=int(peak_rel),
        tolerance=0.98,
        local=False,
        local_radius=int(CURVATURE_NEGPREF_LOCAL_RADIUS),
    )
    local_diag = compute_curvature_negpref_diagnostics(
        seg,
        peak_rel=int(peak_rel),
        tolerance=0.98,
        local=True,
        local_radius=int(CURVATURE_NEGPREF_LOCAL_RADIUS),
    )
    d2 = np.asarray(diag.get("d2", []), dtype=float)
    if d2.size == 0:
        return {}
    return {
        "curve_x_rel": [int(v) for v in range(1, 1 + int(d2.size))],
        "curve_y": [float(v) for v in d2.tolist()],
        "apex_idx_rel": int(1 + int(diag.get("apex_d2_idx", 0))),
        "chosen_idx_rel": int(1 + int(diag.get("chosen_idx", 0))),
        "base_idx_rel": int(1 + int(diag.get("base_idx", 0))),
        "negative_idx_rel": (None if diag.get("negative_idx") is None else int(1 + int(diag["negative_idx"]))),
        "local_left_idx_rel": int(1 + int(local_diag.get("local_left_idx", 0))),
        "local_right_idx_rel": int(1 + int(local_diag.get("local_right_idx", 0))),
        "local_chosen_idx_rel": int(1 + int(local_diag.get("chosen_idx", 0))),
        "chosen_value": float(diag.get("chosen_value", 0.0)),
        "base_value": float(diag.get("base_value", 0.0)),
        "negative_value": float(diag.get("negative_value", 0.0)),
        "local_chosen_value": float(local_diag.get("chosen_value", 0.0)),
        "tolerance": 0.98,
        "label": "PCE t98",
    }


def compute_ss1_pce_features(
    *,
    raw_signal: np.ndarray,
    gradient_signal: np.ndarray | None,
    seg: CandidateSegment,
    feature_signal_source: str,
    bg_noise_override: float | None = None,
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
        "pce_t098_chosen_value": float("nan"),
        "pce_t098_chosen_value_z": float("nan"),
        "pce_t098_evidence_signed": float("nan"),
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
    bg_override = float(bg_noise_override) if bg_noise_override is not None else float("nan")
    if np.isfinite(bg_override) and bg_override > 0.0:
        bg_mad = max(float(bg_override), 1e-12)
        out["bg_noise_override_used"] = 1.0
        out["bg_noise_override_value"] = float(bg_mad)
    else:
        bg = estimate_bg_noise(signal, a, b)
        bg_mad = max(float(bg.value), 1e-12)
        out["bg_noise_override_used"] = 0.0
        out["bg_noise_override_value"] = float("nan")
    out["bg_mad"] = float(bg_mad)
    out["rise_slope_z"] = float(rise_slope / bg_mad)
    out["fall_slope_z"] = float(abs(fall_slope) / bg_mad)
    sr = float(np.tanh(out["rise_slope_z"] / 6.0))
    sf = float(np.tanh(out["fall_slope_z"] / 6.0))
    out["spike_score_v1"] = float(0.5 * sr + 0.5 * sf)
    peak_rel = int(np.clip(p - a, 0, segment.size - 1))
    out.update(compute_peak_curvature_features(segment, bg_mad, peak_rel=peak_rel))
    pce_raw = float(out.get("peak_curvature_extreme_negpref_t098", np.nan))
    if np.isfinite(pce_raw):
        pce_support01 = _ramp_down(pce_raw, PCE_SUPPORT_FULL, PCE_SUPPORT_ZERO)
        pce_veto01 = _ramp_up(pce_raw, PCE_VETO_ZERO, PCE_VETO_FULL)
        out["pce_negpref_t098_support01"] = float(pce_support01)
        out["pce_negpref_t098_veto01"] = float(pce_veto01)
        out["pce_negpref_t098_evidence_signed"] = float(_signed_evidence(pce_support01, pce_veto01))
    debug = build_pce_t98_debug(segment, peak_rel=peak_rel)
    chosen_value = float(debug.get("chosen_value", np.nan)) if isinstance(debug, dict) else float("nan")
    out["pce_t098_chosen_value"] = float(chosen_value)
    if np.isfinite(chosen_value) and np.isfinite(bg_mad) and bg_mad > 0.0:
        out["pce_t098_chosen_value_z"] = float(chosen_value / bg_mad)
    out["pce_t098_evidence_signed"] = float(out.get("pce_negpref_t098_evidence_signed", np.nan))
    if np.isfinite(float(out.get("pce_negpref_t098_evidence_signed", np.nan))):
        out["pce"] = float(out["pce_negpref_t098_evidence_signed"])
    out["pce_t98_debug"] = debug
    return out


def compute_raw_edge_metric(
    *,
    raw_signal: np.ndarray,
    seg: CandidateSegment,
    candidate_noise_estimate: float | None,
    ctx: MetricComputationContext,
) -> dict[str, Any]:
    if str(ctx.edge_foot_method).strip().lower() != "noise_quantized_component":
        raise ValueError(f"Unsupported edge_foot_method: {ctx.edge_foot_method}")
    return compute_edge_metric(
        np.asarray(raw_signal, dtype=float),
        candidate_left=int(seg.start),
        candidate_right=int(seg.end),
        apex_idx=int(seg.peak_index),
        candidate_noise_value=(None if candidate_noise_estimate is None else float(candidate_noise_estimate)),
        noise_source=str(ctx.noise_source),
        noise_height_factor=float(ctx.noise_height_factor),
        edge_pre_level_step_noise=float(ctx.edge_pre_level_step_noise),
        edge_pre_transient_tolerance_levels=int(ctx.edge_pre_transient_tolerance_levels),
        edge_pre_min_stable_levels=int(ctx.edge_pre_min_stable_levels),
        edge_neighbor_structure_factor=float(ctx.edge_neighbor_structure_factor),
        edge_dense_context_min_pad_pts=int(ctx.edge_dense_context_min_pad_pts),
        edge_dense_context_pad_pts=int(ctx.edge_dense_context_pad_pts),
        edge_dense_context_max_pad_pts=int(ctx.edge_dense_context_max_pad_pts),
        edge_context_expand_step_pts=int(ctx.edge_context_expand_step_pts),
    )


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
