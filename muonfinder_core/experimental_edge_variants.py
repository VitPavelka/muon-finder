from __future__ import annotations

from typing import Any

import numpy as np

from .edge import (
    EDGE_ALL_LEVELS_ASC,
    EDGE_DENSE_LEVELS_ASC,
    _component_crossings,
    _components_from_mask,
    _find_apex_component,
)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _edge_width_sum_from_levels(
    signal: np.ndarray,
    levels: list[float],
    apex_idx: int,
    context_left: int,
    context_right: int,
    *,
    include_zero_width_at_first_level: bool = False,
) -> tuple[float, int]:
    if not levels:
        return float("nan"), 0
    seg = np.asarray(signal[int(context_left) : int(context_right) + 1], dtype=float)
    apex_local = int(np.clip(int(apex_idx) - int(context_left), 0, seg.size - 1))
    widths: list[float] = [0.0] if include_zero_width_at_first_level else []
    for idx, level in enumerate(levels):
        if include_zero_width_at_first_level and idx == 0:
            continue
        intervals = _components_from_mask(seg >= float(level))
        _idx, interval = _find_apex_component(intervals, apex_local)
        if interval is None:
            return float("nan"), 0
        width, _left_cross, _right_cross = _component_crossings(seg, float(level), interval)
        widths.append(float(width))
    if not widths:
        return float("nan"), 0
    return float(np.sum(np.asarray(widths, dtype=float))), int(len(widths))


def _edge_recip_mean_from_levels(
    signal: np.ndarray,
    levels: list[float],
    apex_idx: int,
    context_left: int,
    context_right: int,
    *,
    skip_first_level: bool = False,
) -> tuple[float, int]:
    if not levels:
        return float("nan"), 0
    seg = np.asarray(signal[int(context_left) : int(context_right) + 1], dtype=float)
    apex_local = int(np.clip(int(apex_idx) - int(context_left), 0, seg.size - 1))
    rec_widths: list[float] = []
    for idx, level in enumerate(levels):
        if skip_first_level and idx == 0:
            continue
        intervals = _components_from_mask(seg >= float(level))
        _idx, interval = _find_apex_component(intervals, apex_local)
        if interval is None:
            continue
        width, _left_cross, _right_cross = _component_crossings(seg, float(level), interval)
        if np.isfinite(width) and width > 1e-12:
            rec_widths.append(float(1.0 / width))
    if not rec_widths:
        return float("nan"), 0
    return float(np.mean(np.asarray(rec_widths, dtype=float))), int(len(rec_widths))


def _legacy_local_foot(signal: np.ndarray, apex_idx: int, context_left: int, context_right: int) -> tuple[int, int, float, str]:
    y = np.asarray(signal, dtype=float)
    apex = int(np.clip(int(apex_idx), 0, y.size - 1))
    left = int(max(0, context_left))
    right = int(min(y.size - 1, context_right))
    left_foot = apex
    for idx in range(apex - 1, left, -1):
        if float(y[idx - 1]) > float(y[idx]) <= float(y[idx + 1]):
            left_foot = int(idx)
            break
        if float(y[idx - 1]) > float(y[idx]):
            left_foot = int(idx)
            break
    right_foot = apex
    for idx in range(apex + 1, right):
        if float(y[idx + 1]) > float(y[idx]) <= float(y[idx - 1]):
            right_foot = int(idx)
            break
        if float(y[idx + 1]) > float(y[idx]):
            right_foot = int(idx)
            break
    if left_foot == apex and apex > left:
        left_foot = int(np.argmin(y[left : apex + 1]) + left)
    if right_foot == apex and apex < right:
        right_foot = int(np.argmin(y[apex : right + 1]) + apex)
    status = "ok"
    if left_foot <= left or right_foot >= right:
        status = "context_limited"
    foot_value = float(max(y[left_foot], y[right_foot]))
    return int(left_foot), int(right_foot), float(foot_value), status


def compute_experimental_edge_variants(signal: np.ndarray, row: dict[str, Any], noise_value: float, config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "exp_edge_variants_status": "ok",
        "exp_edge_percent_0_90_width_sum": np.nan,
        "exp_edge_percent_5_90_width_sum": np.nan,
        "exp_edge_percent_0_90_recip_mean": np.nan,
        "exp_edge_percent_5_90_recip_mean": np.nan,
        "exp_edge_percent_0_90_n_levels": 0,
        "exp_edge_percent_5_90_n_levels": 0,
        "exp_edge_percent_0_90_evidence_signed_selfnorm": np.nan,
        "exp_edge_percent_0_90_evidence_signed_modernnorm": np.nan,
        "exp_edge_percent_5_90_evidence_signed_selfnorm": np.nan,
        "exp_edge_percent_5_90_evidence_signed_modernnorm": np.nan,
        "exp_edge_noise_from_0_width_sum": np.nan,
        "exp_edge_noise_from_1_width_sum": np.nan,
        "exp_edge_noise_n_levels_from_0": 0,
        "exp_edge_noise_n_levels_from_1": 0,
        "exp_edge_noise_from_0_evidence_signed_selfnorm": np.nan,
        "exp_edge_noise_from_1_evidence_signed_selfnorm": np.nan,
        "exp_edge_noise_from_0_evidence_signed_modernnorm": np.nan,
        "exp_edge_noise_from_1_evidence_signed_modernnorm": np.nan,
        "exp_edge_legacy_width_sum_0_90": np.nan,
        "exp_edge_legacy_width_sum_5_90": np.nan,
        "exp_edge_legacy_evidence_signed_selfnorm": np.nan,
        "exp_edge_legacy_evidence_signed_modernnorm": np.nan,
        "exp_edge_legacy_like_evidence_signed": np.nan,
        "exp_edge_legacy_like": np.nan,
        "exp_edge_legacy_n_levels": 0,
        "exp_edge_legacy_foot_left": np.nan,
        "exp_edge_legacy_foot_right": np.nan,
        "exp_edge_legacy_foot_value": np.nan,
        "exp_edge_legacy_status": "",
    }
    debug = row.get("edge_debug", {})
    if not isinstance(debug, dict):
        debug = {}
    context_left = int(debug.get("edge_context_left", row.get("start_index", row.get("start", 0))))
    context_right = int(debug.get("edge_context_right", row.get("end_index", row.get("end", max(0, len(signal) - 1)))))
    apex_idx = int(debug.get("edge_apex_index", row.get("peak_index", -1)))
    apex_value = _safe_float(debug.get("edge_apex_value", signal[apex_idx] if 0 <= apex_idx < len(signal) else np.nan))
    foot_value = _safe_float(debug.get("edge_selected_foot_value", row.get("edge_selected_foot_value", row.get("edge_foot_value", row.get("edge_base_value")))))
    if not np.isfinite(foot_value):
        out["exp_edge_variants_status"] = "missing_edge_foot"
        return out
    prominence = float(apex_value - foot_value)
    if not np.isfinite(prominence) or prominence <= 1e-12:
        out["exp_edge_variants_status"] = "invalid_prominence"
        return out

    percent_levels = [float(item.get("level_value")) for item in debug.get("edge_selected_levels", []) if isinstance(item, dict)]
    percent_levels_5_90 = [
        float(item.get("level_value"))
        for item in debug.get("edge_selected_levels", [])
        if isinstance(item, dict) and int(item.get("percent", -1)) in EDGE_DENSE_LEVELS_ASC
    ]
    scalar, n_levels = _edge_width_sum_from_levels(
        np.asarray(signal, dtype=float),
        percent_levels,
        apex_idx,
        context_left,
        context_right,
        include_zero_width_at_first_level=True,
    )
    recip_scalar, recip_n = _edge_recip_mean_from_levels(
        np.asarray(signal, dtype=float),
        percent_levels,
        apex_idx,
        context_left,
        context_right,
        skip_first_level=True,
    )
    out["exp_edge_percent_0_90_width_sum"] = float(scalar) if np.isfinite(scalar) else np.nan
    out["exp_edge_percent_0_90_recip_mean"] = float(recip_scalar) if np.isfinite(recip_scalar) else np.nan
    out["exp_edge_percent_0_90_n_levels"] = int(n_levels)
    scalar5, n5 = _edge_width_sum_from_levels(
        np.asarray(signal, dtype=float),
        percent_levels_5_90,
        apex_idx,
        context_left,
        context_right,
        include_zero_width_at_first_level=False,
    )
    recip_scalar5, recip_n5 = _edge_recip_mean_from_levels(
        np.asarray(signal, dtype=float),
        percent_levels_5_90,
        apex_idx,
        context_left,
        context_right,
        skip_first_level=False,
    )
    out["exp_edge_percent_5_90_width_sum"] = float(scalar5) if np.isfinite(scalar5) else np.nan
    out["exp_edge_percent_5_90_recip_mean"] = float(recip_scalar5) if np.isfinite(recip_scalar5) else np.nan
    out["exp_edge_percent_5_90_n_levels"] = int(n5)

    edge_cfg = dict(config.get("edge_variants", {}))
    step = float(max(edge_cfg.get("noise_level_step_factor", 1.0), 1e-12)) * float(noise_value)
    start_factor = float(edge_cfg.get("noise_level_start_factor", 0.0))
    max_levels = int(max(1, edge_cfg.get("max_noise_levels", 50)))
    if np.isfinite(noise_value) and noise_value > 0.0:
        levels_from_0: list[float] = []
        levels_from_1: list[float] = []
        for k in range(max_levels):
            level = float(foot_value + (start_factor + float(k)) * step)
            if level >= apex_value:
                break
            levels_from_0.append(level)
            if k >= 1:
                levels_from_1.append(level)
        scalar0, n0 = _edge_width_sum_from_levels(np.asarray(signal, dtype=float), levels_from_0, apex_idx, context_left, context_right)
        scalar1, n1 = _edge_width_sum_from_levels(np.asarray(signal, dtype=float), levels_from_1, apex_idx, context_left, context_right)
        out["exp_edge_noise_from_0_width_sum"] = float(scalar0) if np.isfinite(scalar0) else np.nan
        out["exp_edge_noise_from_1_width_sum"] = float(scalar1) if np.isfinite(scalar1) else np.nan
        out["exp_edge_noise_n_levels_from_0"] = int(n0)
        out["exp_edge_noise_n_levels_from_1"] = int(n1)

    left_foot, right_foot, legacy_foot, legacy_status = _legacy_local_foot(np.asarray(signal, dtype=float), apex_idx, context_left, context_right)
    out["exp_edge_legacy_foot_left"] = float(left_foot)
    out["exp_edge_legacy_foot_right"] = float(right_foot)
    out["exp_edge_legacy_foot_value"] = float(legacy_foot)
    out["exp_edge_legacy_status"] = str(legacy_status)
    legacy_levels = [float(legacy_foot + (percent / 100.0) * (apex_value - legacy_foot)) for percent in EDGE_ALL_LEVELS_ASC if apex_value > legacy_foot]
    legacy_levels_5_90 = [float(legacy_foot + (percent / 100.0) * (apex_value - legacy_foot)) for percent in EDGE_DENSE_LEVELS_ASC if apex_value > legacy_foot]
    legacy_scalar, legacy_n = _edge_width_sum_from_levels(
        np.asarray(signal, dtype=float),
        legacy_levels,
        apex_idx,
        context_left,
        context_right,
        include_zero_width_at_first_level=True,
    )
    legacy_scalar5, _legacy_n5 = _edge_width_sum_from_levels(
        np.asarray(signal, dtype=float),
        legacy_levels_5_90,
        apex_idx,
        context_left,
        context_right,
        include_zero_width_at_first_level=False,
    )
    out["exp_edge_legacy_width_sum_0_90"] = float(legacy_scalar) if np.isfinite(legacy_scalar) else np.nan
    out["exp_edge_legacy_width_sum_5_90"] = float(legacy_scalar5) if np.isfinite(legacy_scalar5) else np.nan
    out["exp_edge_legacy_n_levels"] = int(legacy_n)
    return out
