from __future__ import annotations

"""
Clean core metric computation.

Primary curvature/PCE and background MAD still use a narrow legacy adapter.
EDGE/RVE helper logic is core-native.
"""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .data_model import BackgroundNoiseEstimate, CandidateSegment
from .edge import compute_edge_width_metrics, edge_component_width_at_level

from .legacy_formula_adapter import (
    CURVATURE_NEGPREF_LOCAL_RADIUS,
    compute_curvature_negpref_diagnostics,
    compute_peak_curvature_features,
    estimate_background_mad,
)


EDGE_FOOT_LEVEL = 0
EDGE_DENSE_LEVELS_ASC = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90)
EDGE_ALL_LEVELS_ASC = (EDGE_FOOT_LEVEL, *EDGE_DENSE_LEVELS_ASC)
EDGE_MAPPING_LEVELS_DESC = tuple(reversed(EDGE_DENSE_LEVELS_ASC))
PCE_SUPPORT_FULL = -1992.0
PCE_SUPPORT_ZERO = -273.0
PCE_VETO_ZERO = -273.0
PCE_VETO_FULL = 503.0


@dataclass(frozen=True)
class MetricComputationContext:
    feature_signal_source: str = "gradient"
    edge_context_pad_pts: int = 20
    edge_context_min_pad_pts: int = 10
    edge_context_max_pad_pts: int = 80
    noise_height_factor: float = 5.0
    edge_robust_reference_enabled: bool = True
    edge_use_enhanced_spike_mapping: bool = True
    edge_mapping_enable_merge_guard: bool = True
    edge_mapping_noise_guard_enabled: bool = False
    edge_mapping_levels_desc: Sequence[int] = EDGE_MAPPING_LEVELS_DESC
    edge_mapping_refine_step_percent: int = 5
    edge_mapping_min_level_percent: int = 5
    edge_mapping_require_closed_interval: bool = True
    edge_mapping_use_apex_component: bool = True
    edge_mapping_max_width_jump_factor: float = 2.5
    edge_mapping_max_width_jump_points: float = 8.0
    edge_mapping_fallback_to_old: bool = False
    noise_aware_foot_search_enabled: bool = False
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
    out["edge_context_left"] = int(out.get("measurement_left", -1))
    out["edge_context_right"] = int(out.get("measurement_right", -1))
    selected_levels: list[dict[str, float | int]] = []
    dense_debug = out.get("dense_width_0_90", {})
    dense_widths = dense_debug.get("widths", {}) if isinstance(dense_debug, dict) else {}
    if isinstance(dense_widths, dict):
        for level in EDGE_ALL_LEVELS_ASC:
            item = dense_widths.get(str(level), {})
            if not isinstance(item, dict):
                continue
            if str(item.get("reason", "")).strip().lower() != "ok":
                continue
            left_cross = _float_or_nan(item.get("left_cross"))
            right_cross = _float_or_nan(item.get("right_cross"))
            level_y = _float_or_nan(item.get("level"))
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
    if not selected_levels:
        for level in EDGE_ALL_LEVELS_ASC:
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
    edge_mapping = out.get("edge_mapping", {})
    if isinstance(edge_mapping, dict):
        try:
            out["edge_active_left"] = int(out["measurement_left"]) + int(edge_mapping.get("active_left", 0))
            out["edge_active_right"] = int(out["measurement_left"]) + int(edge_mapping.get("active_right", 0))
        except Exception:
            pass
    return out


def _dense_width_sum_from_debug(debug_dict: dict[str, Any]) -> float:
    if not isinstance(debug_dict, dict):
        return float("nan")
    dense_debug = debug_dict.get("dense_width_0_90", {})
    if not isinstance(dense_debug, dict):
        return float("nan")
    widths = dense_debug.get("widths", {})
    if not isinstance(widths, dict):
        return float("nan")
    vals: list[float] = []
    for level in EDGE_DENSE_LEVELS_ASC:
        item = widths.get(str(level), {})
        if not isinstance(item, dict):
            return float("nan")
        width = _float_or_nan(item.get("width"))
        if not np.isfinite(width):
            return float("nan")
        vals.append(float(width))
    return float(np.sum(np.asarray(vals, dtype=float))) if vals else float("nan")


def _direct_foot_from_debug(
    seg: np.ndarray,
    segment_left_index: int,
    debug_dict: dict[str, Any],
) -> tuple[int | None, float, tuple[int, int] | None]:
    local_zero_interval = debug_dict.get("edge_mapping", {}).get("local_zero_interval") if isinstance(debug_dict.get("edge_mapping", {}), dict) else None
    if isinstance(local_zero_interval, (list, tuple)) and len(local_zero_interval) == 2:
        lo = int(np.clip(int(local_zero_interval[0]), 0, seg.size - 1))
        hi = int(np.clip(int(local_zero_interval[1]), 0, seg.size - 1))
        if hi < lo:
            lo, hi = hi, lo
        interval = np.asarray(seg[lo : hi + 1], dtype=float)
        if interval.size:
            rel = int(np.nanargmin(interval))
            idx = int(lo + rel)
            return int(segment_left_index + idx), float(seg[idx]), (lo, hi)
    level0 = (((debug_dict.get("dense_width_0_90", {}) if isinstance(debug_dict, dict) else {}) or {}).get("widths", {}) or {}).get("0", {})
    if isinstance(level0, dict):
        left_cross = _float_or_nan(level0.get("left_cross"))
        right_cross = _float_or_nan(level0.get("right_cross"))
        if np.isfinite(left_cross) and np.isfinite(right_cross):
            foot_abs = int(round(min(left_cross, right_cross)))
            foot_local = int(np.clip(foot_abs - int(segment_left_index), 0, seg.size - 1))
            return int(segment_left_index + foot_local), float(seg[foot_local]), (foot_local, foot_local)
    return None, float("nan"), None


def _local_minima_indices(y: np.ndarray, start_idx: int, end_idx: int) -> list[int]:
    lo = int(max(1, start_idx))
    hi = int(min(int(y.size) - 2, end_idx))
    out: list[int] = []
    for i in range(lo, hi + 1):
        if float(y[i]) <= float(y[i - 1]) and float(y[i]) <= float(y[i + 1]):
            out.append(int(i))
    return out


def _build_edge_levels_from_foot(
    seg: np.ndarray,
    *,
    segment_left_index: int,
    apex_local: int,
    foot_local: int,
) -> tuple[float, list[dict[str, float | int]], int | None]:
    seg = np.asarray(seg, dtype=float)
    apex_value = float(seg[int(apex_local)])
    foot_value = float(seg[int(foot_local)])
    amp = float(apex_value - foot_value)
    if not np.isfinite(amp) or amp <= 1e-12:
        return float("nan"), [], None
    prev_interval = None
    widths: list[float] = []
    level_rows: list[dict[str, float | int]] = [
        {
            "percent": 0,
            "left_cross": float(segment_left_index + int(foot_local)),
            "right_cross": float(segment_left_index + int(foot_local)),
            "level_value": float(foot_value),
        }
    ]
    for pct in EDGE_DENSE_LEVELS_ASC:
        level_value = float(foot_value + (float(pct) / 100.0) * amp)
        width, left_cross, right_cross, _used_interp, _reason, interval, _crossing_count = edge_component_width_at_level(
            seg,
            float(level_value),
            apex_local=int(apex_local),
            reference_interval=prev_interval,
            bounds=None,
            require_apex=True,
        )
        if interval is not None:
            prev_interval = interval
        if not np.isfinite(width) or not np.isfinite(left_cross) or not np.isfinite(right_cross):
            last_touch: int | None = None
            if level_rows:
                prev_level = level_rows[-1]
                touch_cross = prev_level.get("left_cross") if foot_local <= apex_local else prev_level.get("right_cross")
                touch_cross_f = _float_or_nan(touch_cross)
                if np.isfinite(touch_cross_f):
                    last_touch = int(np.clip(round(touch_cross_f) - int(segment_left_index), 0, seg.size - 1))
            return float("nan"), [], last_touch
        widths.append(float(width))
        level_rows.append(
            {
                "percent": int(pct),
                "left_cross": float(segment_left_index + float(left_cross)),
                "right_cross": float(segment_left_index + float(right_cross)),
                "level_value": float(level_value),
            }
        )
    return float(np.sum(np.asarray(widths, dtype=float))), level_rows, None


def _ensure_level0_row(
    levels: list[dict[str, float | int]],
    *,
    foot_index: int | None,
    foot_value: float,
) -> list[dict[str, float | int]]:
    if foot_index is None or not np.isfinite(foot_value):
        return levels
    if any(int(item.get("percent", -1)) == 0 for item in levels if isinstance(item, dict)):
        return levels
    return [
        {
            "percent": 0,
            "left_cross": float(foot_index),
            "right_cross": float(foot_index),
            "level_value": float(foot_value),
        },
        *levels,
    ]


def _merge_rejected_foots(*groups: Any) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int]] = set()
    for group in groups:
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index", -1))
            except Exception:
                idx = -1
            reason = str(item.get("reason", ""))
            r_val = _float_or_nan(item.get("r"))
            r_bucket = int(round(r_val * 1000.0)) if np.isfinite(r_val) else -10**9
            key = (idx, reason, r_bucket)
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(item))
    return merged


def _search_lower_edge_foot(
    *,
    raw_segment: np.ndarray,
    segment_left_index: int,
    apex_local: int,
    direct_foot_index: int | None,
    direct_foot_value: float,
    noise_value: float,
    noise_height_factor: float,
) -> tuple[float, dict[str, Any]]:
    seg = np.asarray(raw_segment, dtype=float)
    if seg.ndim != 1 or seg.size < 3 or direct_foot_index is None or not np.isfinite(direct_foot_value):
        return float("nan"), {}
    direct_local = int(np.clip(int(direct_foot_index) - int(segment_left_index), 0, seg.size - 1))
    apex_local = int(np.clip(apex_local, 0, seg.size - 1))
    apex_value = float(seg[apex_local])
    noise_range = float(max(0.0, float(noise_height_factor)) * max(float(noise_value), 0.0))
    rejected_foots: list[dict[str, Any]] = []
    if direct_local <= apex_local:
        scan = range(max(0, direct_local - 1), -1, -1)
    else:
        scan = range(min(int(seg.size) - 1, direct_local + 1), int(seg.size))
    candidate_indices: list[int] = []
    lowest_seen = float(direct_foot_value)
    for idx in scan:
        val = float(seg[int(idx)])
        is_local_min = 0 < idx < int(seg.size) - 1 and float(seg[idx]) <= float(seg[idx - 1]) and float(seg[idx]) <= float(seg[idx + 1])
        is_new_lower = val < lowest_seen - 1e-12
        if is_local_min or is_new_lower:
            candidate_indices.append(int(idx))
            lowest_seen = min(lowest_seen, val)
    for foot_local in candidate_indices:
        foot_value = float(seg[int(foot_local)])
        foot_height = float(apex_value - foot_value)
        ratio = float(foot_height / max(float(noise_value), 1e-12)) if np.isfinite(foot_height) and np.isfinite(noise_value) and float(noise_value) > 0.0 else float("nan")
        if not (np.isfinite(foot_height) and foot_height > noise_range):
            rejected_foots.append(
                {
                    "index": int(segment_left_index + int(foot_local)),
                    "value": float(foot_value),
                    "height": float(foot_height) if np.isfinite(foot_height) else float("nan"),
                    "r": float(ratio) if np.isfinite(ratio) else float("nan"),
                    "reason": "within_noise_range",
                }
            )
            continue
        width_sum, level_rows, last_touch = _build_edge_levels_from_foot(
            seg,
            segment_left_index=int(segment_left_index),
            apex_local=int(apex_local),
            foot_local=int(foot_local),
        )
        if np.isfinite(width_sum) and level_rows:
            return width_sum, {
                "edge_foot_search_triggered": True,
                "edge_foot_search_status": "searched",
                "edge_direct_foot_index": int(direct_foot_index),
                "edge_selected_foot_index": int(segment_left_index + int(foot_local)),
                "edge_noise_range": float(noise_range),
                "edge_foot_height": float(foot_height),
                "edge_selected_levels": level_rows,
                "edge_rejected_foots": rejected_foots,
            }
        if last_touch is not None and int(last_touch) != int(foot_local):
            touch_value = float(seg[int(last_touch)])
            touch_height = float(apex_value - touch_value)
            touch_ratio = float(touch_height / max(float(noise_value), 1e-12)) if np.isfinite(noise_value) and float(noise_value) > 0.0 else float("nan")
            if np.isfinite(touch_height) and touch_height > noise_range:
                touch_sum, touch_levels, _ = _build_edge_levels_from_foot(
                    seg,
                    segment_left_index=int(segment_left_index),
                    apex_local=int(apex_local),
                    foot_local=int(last_touch),
                )
                if np.isfinite(touch_sum) and touch_levels:
                    return touch_sum, {
                        "edge_foot_search_triggered": True,
                        "edge_foot_search_status": "searched",
                        "edge_direct_foot_index": int(direct_foot_index),
                        "edge_selected_foot_index": int(segment_left_index + int(last_touch)),
                        "edge_noise_range": float(noise_range),
                        "edge_foot_height": float(touch_height),
                        "edge_selected_levels": touch_levels,
                        "edge_rejected_foots": rejected_foots,
                    }
            rejected_foots.append(
                {
                    "index": int(segment_left_index + int(last_touch)),
                    "value": float(touch_value),
                    "height": float(touch_height) if np.isfinite(touch_height) else float("nan"),
                    "r": float(touch_ratio) if np.isfinite(touch_ratio) else float("nan"),
                    "reason": "level_geometry_failed",
                }
            )
    return float("nan"), {"edge_rejected_foots": rejected_foots}


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
    pce_raw = _float_or_nan(out.get("peak_curvature_extreme_negpref_t098"))
    if np.isfinite(pce_raw):
        pce_support01 = _ramp_down(pce_raw, PCE_SUPPORT_FULL, PCE_SUPPORT_ZERO)
        pce_veto01 = _ramp_up(pce_raw, PCE_VETO_ZERO, PCE_VETO_FULL)
        out["pce_negpref_t098_support01"] = float(pce_support01)
        out["pce_negpref_t098_veto01"] = float(pce_veto01)
        out["pce_negpref_t098_evidence_signed"] = float(_signed_evidence(pce_support01, pce_veto01))
    if np.isfinite(_float_or_nan(out.get("pce_negpref_t098_evidence_signed"))):
        out["pce"] = float(out["pce_negpref_t098_evidence_signed"])
    out["pce_t98_debug"] = build_pce_t98_debug(segment, peak_rel=peak_rel)
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
    base_pad = max(int(ctx.edge_context_pad_pts), int(ctx.edge_context_min_pad_pts))
    max_pad = max(base_pad, int(ctx.edge_context_max_pad_pts))
    expand_step = max(1, int(ctx.edge_context_min_pad_pts))

    def _edge_call(
        *,
        left: int,
        right: int,
        bg_value: float,
        use_enhanced_spike_mapping: bool,
        mapping_fallback_to_old: bool,
    ) -> dict[str, Any]:
        return compute_edge_width_metrics(
            raw,
            detection_left=int(left),
            detection_right=int(right),
            prefix="raw_edge_ctx",
            apex_idx=int(seg.peak_index),
            bg_mad=float(bg_value),
            include_low_root_metrics=True,
            low_root_noise_k_mad=0.0,
            use_enhanced_spike_mapping=bool(use_enhanced_spike_mapping),
            mapping_levels_desc=tuple(int(v) for v in ctx.edge_mapping_levels_desc),
            mapping_refine_step_percent=int(ctx.edge_mapping_refine_step_percent),
            mapping_enable_merge_guard=bool(ctx.edge_mapping_enable_merge_guard),
            mapping_noise_guard_enabled=bool(ctx.edge_mapping_noise_guard_enabled),
            mapping_min_level_percent=int(ctx.edge_mapping_min_level_percent),
            mapping_require_closed_interval=bool(ctx.edge_mapping_require_closed_interval),
            mapping_use_apex_component=bool(ctx.edge_mapping_use_apex_component),
            mapping_max_width_jump_factor=float(ctx.edge_mapping_max_width_jump_factor),
            mapping_max_width_jump_points=float(ctx.edge_mapping_max_width_jump_points),
            mapping_fallback_to_old=bool(mapping_fallback_to_old),
            robust_reference_enabled=bool(ctx.edge_robust_reference_enabled),
            robust_reference_noise=(float(candidate_noise_estimate) if candidate_noise_estimate is not None else float(bg_value)),
        )

    def _compute_for_pad(pad: int) -> tuple[float, dict[str, Any], bool, str]:
        left = max(0, int(seg.start) - int(pad))
        right = min(n - 1, int(seg.end) + int(pad))
        bg = estimate_edge_context_bg_mad(raw, int(left), int(right))
        edge_noise_value = float(candidate_noise_estimate) if candidate_noise_estimate is not None and np.isfinite(float(candidate_noise_estimate)) else float(bg.value)
        edge_noise_source = "morph_range" if candidate_noise_estimate is not None and np.isfinite(float(candidate_noise_estimate)) else "bg_mad"
        edge_metrics = _edge_call(
            left=int(left),
            right=int(right),
            bg_value=float(bg.value),
            use_enhanced_spike_mapping=bool(ctx.edge_use_enhanced_spike_mapping),
            mapping_fallback_to_old=bool(ctx.edge_mapping_fallback_to_old),
        )
        value = edge_metrics.get("raw_edge_ctx_dense_width_sum_0_90", np.nan)
        debug = edge_metrics.get("raw_edge_ctx_debug", {})
        raw_value = float(value) if value is not None else np.nan
        debug_dict = debug if isinstance(debug, dict) else {}
        if not np.isfinite(raw_value):
            edge_metrics = _edge_call(
                left=int(left),
                right=int(right),
                bg_value=float(bg.value),
                use_enhanced_spike_mapping=False,
                mapping_fallback_to_old=True,
            )
            value = edge_metrics.get("raw_edge_ctx_dense_width_sum_0_90", np.nan)
            debug = edge_metrics.get("raw_edge_ctx_debug", {})
            raw_value = float(value) if value is not None else np.nan
            debug_dict = debug if isinstance(debug, dict) else {}
        if not np.isfinite(raw_value):
            reconstructed = _dense_width_sum_from_debug(debug_dict)
            if np.isfinite(reconstructed):
                raw_value = float(reconstructed)
        dense_debug = debug_dict.get("dense_width_0_90", {}) if isinstance(debug_dict, dict) else {}
        clean_value = float(raw_value) if np.isfinite(raw_value) else float("nan")
        clean_debug = _normalize_edge_debug(debug_dict, raw_value)
        direct_foot_index, direct_foot_value, _direct_interval = _direct_foot_from_debug(
            np.asarray(raw[left : right + 1], dtype=float),
            int(left),
            debug_dict,
        )
        direct_height = float(raw[int(seg.peak_index)] - direct_foot_value) if np.isfinite(direct_foot_value) else float("nan")
        noise_range = float(max(0.0, float(ctx.noise_height_factor)) * max(float(edge_noise_value), 0.0))
        direct_is_strong = bool(np.isfinite(direct_height) and direct_height > noise_range)
        edge_noise_ratio = float(dense_debug.get("root_snr", np.nan)) if isinstance(dense_debug, dict) else float("nan")
        clean_debug["edge_foot_search_triggered"] = False
        clean_debug["edge_foot_search_status"] = "direct"
        clean_debug["edge_direct_foot_index"] = (None if direct_foot_index is None else int(direct_foot_index))
        clean_debug["edge_selected_foot_index"] = (None if direct_foot_index is None else int(direct_foot_index))
        clean_debug["edge_noise_source"] = str(edge_noise_source)
        clean_debug["edge_noise_value"] = float(edge_noise_value)
        clean_debug["edge_noise_range"] = float(noise_range)
        clean_debug["edge_foot_height"] = float(direct_height) if np.isfinite(direct_height) else float("nan")
        clean_debug["edge_context_left"] = int(left)
        clean_debug["edge_context_right"] = int(right)
        clean_debug["edge_context_pad_used"] = int(pad)
        clean_debug["edge_rejected_foots"] = []
        if bool(ctx.noise_aware_foot_search_enabled) and ((np.isfinite(clean_value) and not direct_is_strong) or (not np.isfinite(clean_value))):
            direct_ratio = float(direct_height / max(float(edge_noise_value), 1e-12)) if np.isfinite(direct_height) and float(edge_noise_value) > 0.0 else float("nan")
            if direct_foot_index is not None and np.isfinite(direct_foot_value):
                clean_debug["edge_rejected_foots"] = [
                    {
                        "index": int(direct_foot_index),
                        "value": float(direct_foot_value),
                        "height": float(direct_height) if np.isfinite(direct_height) else float("nan"),
                        "r": float(direct_ratio) if np.isfinite(direct_ratio) else float("nan"),
                        "reason": "direct_foot_within_noise_range",
                    }
                ]
            noise_value, noise_debug = _search_lower_edge_foot(
                raw_segment=np.asarray(raw[left : right + 1], dtype=float),
                segment_left_index=int(left),
                apex_local=int(np.clip(int(seg.peak_index) - int(left), 0, max(0, right - left))),
                noise_value=float(edge_noise_value),
                noise_height_factor=float(ctx.noise_height_factor),
                direct_foot_index=direct_foot_index,
                direct_foot_value=direct_foot_value,
            )
            if np.isfinite(noise_value):
                clean_value = float(noise_value)
                merged_debug = dict(clean_debug)
                merged_debug.update(noise_debug)
                merged_debug["edge_rejected_foots"] = _merge_rejected_foots(
                    clean_debug.get("edge_rejected_foots", []),
                    noise_debug.get("edge_rejected_foots", []),
                )
                clean_debug = merged_debug
            elif np.isfinite(clean_value):
                clean_debug["edge_foot_search_triggered"] = True
                clean_debug["edge_foot_search_status"] = "unresolved"
                clean_debug["edge_rejected_foots"] = _merge_rejected_foots(
                    clean_debug.get("edge_rejected_foots", []),
                    noise_debug.get("edge_rejected_foots", []),
                )
        selected_foot_index = clean_debug.get("edge_selected_foot_index")
        selected_foot_value = float("nan")
        if selected_foot_index is not None:
            try:
                sel_idx = int(selected_foot_index)
                if 0 <= sel_idx < n:
                    selected_foot_value = float(raw[sel_idx])
            except Exception:
                pass
        if not np.isfinite(selected_foot_value):
            selected_foot_value = direct_foot_value
        clean_debug["edge_selected_levels"] = _ensure_level0_row(
            list(clean_debug.get("edge_selected_levels", [])),
            foot_index=(None if selected_foot_index is None else int(selected_foot_index)),
            foot_value=float(selected_foot_value),
        )
        foot_at_boundary = False
        if selected_foot_index is not None:
            sel_idx = int(selected_foot_index)
            foot_at_boundary = bool(abs(sel_idx - int(left)) <= 1 or abs(sel_idx - int(right)) <= 1)
            active_left = clean_debug.get("edge_active_left")
            active_right = clean_debug.get("edge_active_right")
            if active_left is not None:
                foot_at_boundary = foot_at_boundary or abs(sel_idx - int(active_left)) <= 1
            if active_right is not None:
                foot_at_boundary = foot_at_boundary or abs(sel_idx - int(active_right)) <= 1
        clean_debug["edge_noise_ratio"] = float(edge_noise_ratio) if np.isfinite(edge_noise_ratio) else float("nan")
        expand_reason = ""
        need_expand = False
        if foot_at_boundary:
            need_expand = True
            expand_reason = "foot_at_boundary"
        elif clean_debug.get("edge_selected_foot_index") is None:
            need_expand = True
            expand_reason = "missing_selected_foot"
        elif str(clean_debug.get("edge_foot_search_status", "")) == "unresolved":
            need_expand = True
            expand_reason = "unresolved_search"
        return clean_value, clean_debug, need_expand, expand_reason

    pad = int(max(int(ctx.edge_context_min_pad_pts), int(ctx.edge_context_pad_pts)))
    expand_count = 0
    context_expanded = False
    clean_value, clean_debug, need_expand, expand_reason = _compute_for_pad(pad)
    rejected_history = list(clean_debug.get("edge_rejected_foots", []))
    while need_expand and pad < max_pad:
        context_expanded = True
        expand_count += 1
        pad = min(max_pad, pad + expand_step)
        clean_value, clean_debug, need_expand, expand_reason = _compute_for_pad(pad)
        rejected_history = _merge_rejected_foots(rejected_history, clean_debug.get("edge_rejected_foots", []))
    clean_debug["edge_rejected_foots"] = _merge_rejected_foots(rejected_history, clean_debug.get("edge_rejected_foots", []))
    clean_debug["edge_context_expanded"] = bool(context_expanded)
    clean_debug["edge_context_expand_count"] = int(expand_count)
    clean_debug["edge_foot_at_context_boundary"] = bool(need_expand)
    clean_debug["edge_context_limited"] = bool(need_expand and pad >= max_pad)
    clean_debug["edge_expand_reason"] = str(expand_reason)
    if need_expand and pad >= max_pad:
        clean_debug["edge_foot_search_status"] = "context_limited"
    return {
        "recdw_sum_0_90": clean_value,
        "edge_debug": clean_debug,
        "edge_noise_ratio": clean_debug["edge_noise_ratio"],
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
