from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


EDGE_FOOT_LEVEL = 0
EDGE_DENSE_LEVELS_ASC = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90)
EDGE_ALL_LEVELS_ASC = (EDGE_FOOT_LEVEL, *EDGE_DENSE_LEVELS_ASC)


@dataclass(frozen=True)
class EdgeFootSelection:
    foot_value: float
    foot_index: int
    foot_left: float
    foot_right: float
    status: str
    pre_levels: tuple[dict[str, Any], ...]


def _components_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    comps: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, value in enumerate(mask.astype(bool)):
        if value and not in_run:
            start = i
            in_run = True
        elif (not value) and in_run:
            comps.append((start, i - 1))
            in_run = False
    if in_run:
        comps.append((start, int(mask.size) - 1))
    return comps


def _component_crossings(signal: np.ndarray, level: float, interval: tuple[int, int]) -> tuple[float, float, float]:
    y = np.asarray(signal, dtype=float)
    left_idx = int(interval[0])
    right_idx = int(interval[1])
    left_cross = float(left_idx)
    right_cross = float(right_idx)
    if left_idx > 0:
        y0 = float(y[left_idx - 1])
        y1 = float(y[left_idx])
        if y1 != y0:
            left_cross = float((left_idx - 1) + (float(level) - y0) / (y1 - y0))
    if right_idx < int(y.size) - 1:
        y0 = float(y[right_idx])
        y1 = float(y[right_idx + 1])
        if y1 != y0:
            right_cross = float(right_idx + (float(level) - y0) / (y1 - y0))
    return float(max(0.0, right_cross - left_cross)), float(left_cross), float(right_cross)


def _find_apex_component(intervals: list[tuple[int, int]], apex_local: int) -> tuple[int, tuple[int, int] | None]:
    for idx, interval in enumerate(intervals):
        if int(interval[0]) <= int(apex_local) <= int(interval[1]):
            return int(idx), interval
    return -1, None


def _interval_min_index(signal: np.ndarray, interval: tuple[int, int]) -> int:
    y = np.asarray(signal, dtype=float)
    left = int(interval[0])
    right = int(interval[1])
    return int(left + int(np.argmin(y[left : right + 1])))


def _neighbor_info(
    signal: np.ndarray,
    *,
    apex_local: int,
    apex_component_idx: int,
    intervals: list[tuple[int, int]],
    noise_value: float,
) -> dict[str, Any] | None:
    y = np.asarray(signal, dtype=float)
    candidates: list[dict[str, Any]] = []
    for idx in (apex_component_idx - 1, apex_component_idx + 1):
        if not (0 <= idx < len(intervals)):
            continue
        left, right = intervals[idx]
        peak_local = int(left + int(np.argmax(y[left : right + 1])))
        saddle_lo = min(int(apex_local), peak_local)
        saddle_hi = max(int(apex_local), peak_local)
        saddle_local = int(saddle_lo + int(np.argmin(y[saddle_lo : saddle_hi + 1])))
        peak_value = float(y[peak_local])
        saddle_value = float(y[saddle_local])
        neighbor_height = float(max(0.0, peak_value - saddle_value))
        candidates.append(
            {
                "side": "left" if peak_local < int(apex_local) else "right",
                "peak_local": int(peak_local),
                "peak_value": float(peak_value),
                "saddle_local": int(saddle_local),
                "saddle_value": float(saddle_value),
                "height": float(neighbor_height),
                "r": float(neighbor_height / max(float(noise_value), 1e-12)),
                "distance": abs(int(apex_local) - peak_local),
            }
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: (float(item["r"]), -int(item["distance"])))


def _public_pre_level(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "level_index": int(row["level_index"]),
        "level_value": float(row["level_value"]),
        "apex_left": float(row["apex_left"]) if np.isfinite(float(row["apex_left"])) else np.nan,
        "apex_right": float(row["apex_right"]) if np.isfinite(float(row["apex_right"])) else np.nan,
        "apex_width": float(row["apex_width"]) if np.isfinite(float(row["apex_width"])) else np.nan,
        "n_components": int(row["n_components"]),
        "neighbor_detected": bool(row["neighbor_detected"]),
        "neighbor_height": float(row["neighbor_height"]) if np.isfinite(float(row["neighbor_height"])) else np.nan,
        "neighbor_r": float(row["neighbor_r"]) if np.isfinite(float(row["neighbor_r"])) else np.nan,
        "component_status": str(row["component_status"]),
        "selected_for_foot": bool(row["selected_for_foot"]),
    }


def select_edge_foot_noise_quantized(
    signal: np.ndarray,
    *,
    apex_idx: int,
    candidate_left: int,
    candidate_right: int,
    context_left: int,
    context_right: int,
    noise_value: float,
    noise_height_factor: float,
    neighbor_structure_factor: float,
    level_step_noise: float,
    transient_tolerance_levels: int,
    min_stable_levels: int,
) -> EdgeFootSelection | None:
    y = np.asarray(signal, dtype=float)
    left = int(max(0, context_left))
    right = int(min(int(y.size) - 1, context_right))
    if y.ndim != 1 or y.size == 0 or right < left or not np.all(np.isfinite(y[left : right + 1])):
        return None
    seg = np.asarray(y[left : right + 1], dtype=float)
    apex_local = int(np.clip(int(apex_idx) - left, 0, seg.size - 1))
    apex_value = float(seg[apex_local])
    noise = float(noise_value)
    if not np.isfinite(apex_value) or not np.isfinite(noise) or noise <= 0.0:
        return None
    step = float(max(float(level_step_noise) * noise, 1e-12))
    noise_range = float(max(0.0, float(noise_height_factor)) * noise)
    min_stable = int(max(1, min_stable_levels))
    transient_tol = int(max(0, transient_tolerance_levels))
    context_min = float(np.min(seg))
    max_levels = int(max(4, min(2000, np.ceil((apex_value - context_min) / step) + min_stable + transient_tol + 4)))
    rows: list[dict[str, Any]] = []
    last_valid: dict[str, Any] | None = None
    neighbor_runs = {"left": 0, "right": 0}
    transient_runs = {"left": 0, "right": 0}
    for level_index in range(1, max_levels + 1):
        level_value = float(apex_value - level_index * step)
        if level_value < context_min:
            break
        intervals = _components_from_mask(seg >= level_value)
        apex_component_idx, apex_interval = _find_apex_component(intervals, apex_local)
        row: dict[str, Any] = {
            "level_index": int(level_index),
            "level_value": float(level_value),
            "apex_left": np.nan,
            "apex_right": np.nan,
            "apex_width": np.nan,
            "n_components": int(len(intervals)),
            "neighbor_detected": False,
            "neighbor_height": np.nan,
            "neighbor_r": np.nan,
            "component_status": "lost",
            "selected_for_foot": False,
        }
        if apex_interval is None:
            rows.append(row)
            break
        apex_width, apex_left, apex_right = _component_crossings(seg, level_value, apex_interval)
        row.update(
            {
                "apex_left": float(left + apex_left),
                "apex_right": float(left + apex_right),
                "apex_width": float(apex_width),
            }
        )
        foot_local = _interval_min_index(seg, apex_interval)
        last_valid = {
            "foot_index": int(left + foot_local),
            "foot_value": float(seg[foot_local]),
            "foot_left": float(left + foot_local),
            "foot_right": float(left + foot_local),
            "row_index": int(len(rows)),
            "status": "ok",
        }
        if int(apex_interval[0]) <= 0 or int(apex_interval[1]) >= int(seg.size) - 1:
            row["component_status"] = "context_boundary"
            rows.append(row)
            last_valid["status"] = "context_limited"
            break
        neighbor = _neighbor_info(seg, apex_local=apex_local, apex_component_idx=apex_component_idx, intervals=intervals, noise_value=noise)
        if neighbor is None:
            neighbor_runs["left"] = 0
            neighbor_runs["right"] = 0
            transient_runs["left"] = 0
            transient_runs["right"] = 0
            row["component_status"] = "ok"
            rows.append(row)
            continue
        side = str(neighbor["side"])
        row["neighbor_detected"] = True
        row["neighbor_height"] = float(neighbor["height"])
        row["neighbor_r"] = float(neighbor["r"])
        transient_runs[side] = int(transient_runs[side]) + 1
        transient_runs["left" if side == "right" else "right"] = 0
        if float(neighbor["r"]) > float(neighbor_structure_factor):
            neighbor_runs[side] = int(neighbor_runs[side]) + 1
        else:
            neighbor_runs[side] = 0
        neighbor_runs["left" if side == "right" else "right"] = 0
        if float(neighbor["r"]) > float(neighbor_structure_factor) and int(neighbor_runs[side]) >= min_stable:
            saddle_local = int(neighbor["saddle_local"])
            row["component_status"] = "neighbor_structure"
            rows.append(row)
            rows[-1]["selected_for_foot"] = True
            return EdgeFootSelection(
                foot_value=float(seg[saddle_local]),
                foot_index=int(left + saddle_local),
                foot_left=float(left + saddle_local),
                foot_right=float(left + saddle_local),
                status="neighbor_structure_stop",
                pre_levels=tuple(_public_pre_level(item) for item in rows),
            )
        if float(neighbor["height"]) <= noise_range or int(transient_runs[side]) <= transient_tol:
            row["component_status"] = "transient"
        else:
            row["component_status"] = "ok"
        rows.append(row)
    if last_valid is None:
        return None
    if rows and str(rows[-1].get("component_status", "")) == "lost":
        last_valid["status"] = "last_valid_touch"
    rows[int(last_valid["row_index"])]["selected_for_foot"] = True
    return EdgeFootSelection(
        foot_value=float(last_valid["foot_value"]),
        foot_index=int(last_valid["foot_index"]),
        foot_left=float(last_valid["foot_left"]),
        foot_right=float(last_valid["foot_right"]),
        status=str(last_valid["status"]),
        pre_levels=tuple(_public_pre_level(item) for item in rows),
    )


def _estimate_edge_context_bg_mad(
    signal: np.ndarray,
    *,
    candidate_left: int,
    candidate_right: int,
    context_left: int,
    context_right: int,
) -> float:
    y = np.asarray(signal, dtype=float)
    bg = np.concatenate([y[context_left:candidate_left], y[candidate_right + 1 : context_right + 1]])
    if bg.size < 5:
        bg = np.asarray(y[context_left : context_right + 1], dtype=float)
    bg_med = float(np.nanmedian(bg)) if bg.size else 0.0
    return float(max(np.nanmedian(np.abs(bg - bg_med)) if bg.size else 0.0, 1e-12))


def _compute_final_levels(
    signal: np.ndarray,
    *,
    apex_idx: int,
    context_left: int,
    context_right: int,
    foot_value: float,
    foot_index: int,
) -> tuple[float, list[dict[str, Any]]]:
    y = np.asarray(signal, dtype=float)
    left = int(context_left)
    right = int(context_right)
    seg = np.asarray(y[left : right + 1], dtype=float)
    apex_local = int(np.clip(int(apex_idx) - left, 0, seg.size - 1))
    apex_value = float(seg[apex_local])
    if not (np.isfinite(foot_value) and np.isfinite(apex_value) and apex_value > foot_value):
        return float("nan"), []
    foot_cross = float(np.clip(int(foot_index), left, right))
    rows: list[dict[str, Any]] = [
        {
            "percent": 0,
            "level_value": float(foot_value),
            "left_cross": float(foot_cross),
            "right_cross": float(foot_cross),
        }
    ]
    widths = [0.0]
    for percent in EDGE_DENSE_LEVELS_ASC:
        level_value = float(foot_value + (float(percent) / 100.0) * (apex_value - foot_value))
        intervals = _components_from_mask(seg >= level_value)
        _idx, interval = _find_apex_component(intervals, apex_local)
        if interval is None:
            return float("nan"), []
        width, left_cross, right_cross = _component_crossings(seg, level_value, interval)
        rows.append(
            {
                "percent": int(percent),
                "level_value": float(level_value),
                "left_cross": float(left + left_cross),
                "right_cross": float(left + right_cross),
            }
        )
        widths.append(float(width))
    return float(np.sum(np.asarray(widths, dtype=float))), rows


def compute_edge_metric(
    signal: np.ndarray,
    *,
    candidate_left: int,
    candidate_right: int,
    apex_idx: int,
    candidate_noise_value: float | None,
    noise_source: str,
    noise_height_factor: float,
    edge_pre_level_step_noise: float,
    edge_pre_transient_tolerance_levels: int,
    edge_pre_min_stable_levels: int,
    edge_neighbor_structure_factor: float,
    edge_dense_context_min_pad_pts: int,
    edge_dense_context_pad_pts: int,
    edge_dense_context_max_pad_pts: int,
    edge_context_expand_step_pts: int,
) -> dict[str, Any]:
    y = np.asarray(signal, dtype=float)
    n = int(y.size)
    pad = int(max(edge_dense_context_min_pad_pts, edge_dense_context_pad_pts))
    max_pad = int(max(pad, edge_dense_context_max_pad_pts))
    expand_step = int(max(1, edge_context_expand_step_pts))
    expand_count = 0
    selected_value = float("nan")
    selected_debug: dict[str, Any] = {}
    while True:
        context_left = int(max(0, int(candidate_left) - pad))
        context_right = int(min(n - 1, int(candidate_right) + pad))
        bg_mad = _estimate_edge_context_bg_mad(
            y,
            candidate_left=int(candidate_left),
            candidate_right=int(candidate_right),
            context_left=int(context_left),
            context_right=int(context_right),
        )
        requested_source = str(noise_source).strip().lower()
        fallback_used = bool(requested_source == "morph_range" and not (candidate_noise_value is not None and np.isfinite(float(candidate_noise_value))))
        if requested_source == "bg_mad":
            edge_noise_source = "bg_mad"
            edge_noise_value = float(bg_mad)
        elif candidate_noise_value is not None and np.isfinite(float(candidate_noise_value)):
            edge_noise_source = "morph_range"
            edge_noise_value = float(candidate_noise_value)
        else:
            edge_noise_source = "bg_mad"
            edge_noise_value = float(bg_mad)
        selection = select_edge_foot_noise_quantized(
            y,
            apex_idx=int(apex_idx),
            candidate_left=int(candidate_left),
            candidate_right=int(candidate_right),
            context_left=int(context_left),
            context_right=int(context_right),
            noise_value=float(edge_noise_value),
            noise_height_factor=float(noise_height_factor),
            neighbor_structure_factor=float(edge_neighbor_structure_factor),
            level_step_noise=float(edge_pre_level_step_noise),
            transient_tolerance_levels=int(edge_pre_transient_tolerance_levels),
            min_stable_levels=int(edge_pre_min_stable_levels),
        )
        selected_debug = {
            "edge_algorithm": "noise_quantized_component",
            "edge_noise_source": str(edge_noise_source),
            "edge_noise_value": float(edge_noise_value) if np.isfinite(edge_noise_value) else np.nan,
            "edge_noise_height_factor": float(noise_height_factor),
            "edge_neighbor_structure_factor": float(edge_neighbor_structure_factor),
            "edge_noise_range": float(max(0.0, float(noise_height_factor)) * max(float(edge_noise_value), 0.0)) if np.isfinite(edge_noise_value) else np.nan,
            "edge_noise_source_requested": str(requested_source),
            "edge_noise_fallback_used": bool(fallback_used),
            "edge_context_left": int(context_left),
            "edge_context_right": int(context_right),
            "edge_context_pad_used": int(pad),
            "edge_context_expanded": bool(expand_count > 0),
            "edge_context_expand_count": int(expand_count),
            "edge_context_limited": False,
            "edge_apex_index": int(apex_idx),
            "edge_apex_value": float(y[int(apex_idx)]) if 0 <= int(apex_idx) < n and np.isfinite(y[int(apex_idx)]) else np.nan,
            "edge_selected_foot_value": np.nan,
            "edge_selected_foot_left": np.nan,
            "edge_selected_foot_right": np.nan,
            "edge_selected_foot_index": None,
            "edge_selected_foot_status": "last_valid_touch",
            "edge_pre_levels": [],
            "edge_selected_levels": [],
        }
        if selection is not None:
            near_boundary = bool(
                abs(float(selection.foot_left) - float(context_left)) <= 1.0
                or abs(float(selection.foot_right) - float(context_right)) <= 1.0
                or str(selection.status) == "context_limited"
            )
            if near_boundary and pad < max_pad:
                pad = min(max_pad, pad + expand_step)
                expand_count += 1
                continue
            selected_value, selected_levels = _compute_final_levels(
                y,
                apex_idx=int(apex_idx),
                context_left=int(context_left),
                context_right=int(context_right),
                foot_value=float(selection.foot_value),
                foot_index=int(selection.foot_index),
            )
            selected_debug.update(
                {
                    "edge_context_expanded": bool(expand_count > 0),
                    "edge_context_expand_count": int(expand_count),
                    "edge_context_limited": bool(near_boundary and pad >= max_pad),
                    "edge_selected_foot_value": float(selection.foot_value),
                    "edge_selected_foot_left": float(selection.foot_left),
                    "edge_selected_foot_right": float(selection.foot_right),
                    "edge_selected_foot_index": int(selection.foot_index),
                    "edge_selected_foot_status": "context_limited" if near_boundary and pad >= max_pad else str(selection.status),
                    "edge_pre_levels": list(selection.pre_levels),
                    "edge_selected_levels": selected_levels,
                }
            )
            if np.isfinite(selected_value) and selected_levels:
                break
        if pad >= max_pad:
            selected_debug["edge_context_limited"] = True
            selected_debug["edge_context_expanded"] = bool(expand_count > 0)
            selected_debug["edge_context_expand_count"] = int(expand_count)
            break
        pad = min(max_pad, pad + expand_step)
        expand_count += 1
    apex_value = float(y[int(apex_idx)]) if 0 <= int(apex_idx) < n and np.isfinite(y[int(apex_idx)]) else float("nan")
    foot_value = float(selected_debug.get("edge_selected_foot_value", np.nan))
    noise_value = float(selected_debug.get("edge_noise_value", np.nan))
    noise_ratio = float((apex_value - foot_value) / noise_value) if np.isfinite(apex_value) and np.isfinite(foot_value) and np.isfinite(noise_value) and noise_value > 0.0 else float("nan")
    return {
        "recdw_sum_0_90": float(selected_value) if np.isfinite(selected_value) else float("nan"),
        "edge_debug": selected_debug,
        "edge_noise_ratio": float(noise_ratio) if np.isfinite(noise_ratio) else float("nan"),
    }


__all__ = [
    "EDGE_ALL_LEVELS_ASC",
    "EDGE_DENSE_LEVELS_ASC",
    "EDGE_FOOT_LEVEL",
    "EdgeFootSelection",
    "compute_edge_metric",
    "select_edge_foot_noise_quantized",
]
