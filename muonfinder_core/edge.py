from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _components_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    comps: List[Tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(mask.astype(bool)):
        if v and not in_run:
            start = i
            in_run = True
        elif (not v) and in_run:
            comps.append((start, i - 1))
            in_run = False
    if in_run:
        comps.append((start, int(mask.size) - 1))
    return comps


def width_between_outer_level_crossings(seg: np.ndarray, level: float) -> Tuple[float, float, float, bool, str]:
    y = np.asarray(seg, dtype=float)
    if y.ndim != 1 or y.size == 0:
        return np.nan, np.nan, np.nan, False, "empty_segment"
    if not np.all(np.isfinite(y)):
        return np.nan, np.nan, np.nan, False, "nonfinite_segment"
    above = y >= float(level)
    idx = np.flatnonzero(above)
    if idx.size == 0:
        return np.nan, np.nan, np.nan, False, "level_not_reached"
    left_i = int(idx[0])
    right_i = int(idx[-1])
    left_cross = float(left_i)
    right_cross = float(right_i)
    used_interp = False
    if left_i > 0:
        y0 = float(y[left_i - 1])
        y1 = float(y[left_i])
        if y1 != y0:
            left_cross = float((left_i - 1) + (float(level) - y0) / (y1 - y0))
            used_interp = True
    if right_i < y.size - 1:
        y0 = float(y[right_i])
        y1 = float(y[right_i + 1])
        if y1 != y0:
            right_cross = float(right_i + (float(level) - y0) / (y1 - y0))
            used_interp = True
    width = float(max(0.0, right_cross - left_cross))
    return width, float(left_cross), float(right_cross), bool(used_interp), "ok"


def _above_level_intervals(seg: np.ndarray, level: float) -> List[Tuple[int, int]]:
    y = np.asarray(seg, dtype=float)
    if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
        return []
    return _components_from_mask(y >= float(level))


def _select_tracked_interval(
    intervals: Sequence[Tuple[int, int]],
    *,
    apex_local: int,
    prev_interval: Optional[Tuple[int, int]] = None,
    allow_fallback: bool = True,
) -> Optional[Tuple[int, int]]:
    if not intervals:
        return None
    items = [(int(a), int(b)) for a, b in intervals]
    if prev_interval is None:
        containing = [(a, b) for a, b in items if a <= apex_local <= b]
        if containing:
            return min(containing, key=lambda t: (t[1] - t[0], abs(0.5 * (t[0] + t[1]) - apex_local)))
        if not bool(allow_fallback):
            return None
        return min(items, key=lambda t: abs(0.5 * (t[0] + t[1]) - apex_local))
    pl, pr = int(prev_interval[0]), int(prev_interval[1])
    containing_prev = [(a, b) for a, b in items if a <= pl and b >= pr]
    if containing_prev:
        return min(containing_prev, key=lambda t: (t[1] - t[0], abs(0.5 * (t[0] + t[1]) - apex_local)))
    if not bool(allow_fallback):
        return None

    def _overlap(t: Tuple[int, int]) -> int:
        return max(0, min(t[1], pr) - max(t[0], pl) + 1)

    best_overlap = max(_overlap(t) for t in items)
    if best_overlap > 0:
        candidates = [t for t in items if _overlap(t) == best_overlap]
        return min(candidates, key=lambda t: (t[1] - t[0], abs(0.5 * (t[0] + t[1]) - apex_local)))
    return min(items, key=lambda t: abs(0.5 * (t[0] + t[1]) - apex_local))


def _crossings_for_interval(
    seg: np.ndarray,
    level: float,
    interval: Tuple[int, int],
    *,
    bounds: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float, float, bool, str]:
    y = np.asarray(seg, dtype=float)
    if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
        return np.nan, np.nan, np.nan, False, "invalid_segment"
    bound_l = 0
    bound_r = y.size - 1
    if bounds is not None:
        bound_l = int(np.clip(int(bounds[0]), 0, y.size - 1))
        bound_r = int(np.clip(int(bounds[1]), 0, y.size - 1))
        if bound_r < bound_l:
            bound_l, bound_r = bound_r, bound_l
    l = int(np.clip(int(interval[0]), 0, y.size - 1))
    r = int(np.clip(int(interval[1]), 0, y.size - 1))
    if r < l:
        l, r = r, l
    l = max(l, bound_l)
    r = min(r, bound_r)
    if r < l:
        return np.nan, np.nan, np.nan, False, "outside_bounds"
    used_interp = False
    if l > bound_l:
        y0 = float(y[l - 1])
        y1 = float(y[l])
        if y1 != y0:
            left_cross = float((l - 1) + (float(level) - y0) / (y1 - y0))
            used_interp = True
        else:
            left_cross = float(l)
    else:
        left_cross = float(bound_l)
    if r < bound_r:
        y0 = float(y[r])
        y1 = float(y[r + 1])
        if y1 != y0:
            right_cross = float(r + (float(level) - y0) / (y1 - y0))
            used_interp = True
        else:
            right_cross = float(r)
    else:
        right_cross = float(bound_r)
    if not (np.isfinite(left_cross) and np.isfinite(right_cross)):
        return np.nan, np.nan, np.nan, bool(used_interp), "invalid_crossing"
    width = float(max(0.0, right_cross - left_cross))
    return width, float(left_cross), float(right_cross), bool(used_interp), "ok"


def _above_level_intervals_bounded(seg: np.ndarray, level: float, bounds: Tuple[int, int]) -> List[Tuple[int, int]]:
    y = np.asarray(seg, dtype=float)
    if y.ndim != 1 or y.size == 0:
        return []
    left = int(np.clip(int(bounds[0]), 0, y.size - 1))
    right = int(np.clip(int(bounds[1]), 0, y.size - 1))
    if right < left:
        left, right = right, left
    intervals = _above_level_intervals(y[left:right + 1], level)
    return [(int(a + left), int(b + left)) for a, b in intervals]


def edge_component_width_at_level(
    seg: np.ndarray,
    level: float,
    *,
    apex_local: int,
    reference_interval: Optional[Tuple[int, int]],
    bounds: Optional[Tuple[int, int]] = None,
    require_apex: bool = True,
) -> Tuple[float, float, float, bool, str, Optional[Tuple[int, int]], int]:
    y = np.asarray(seg, dtype=float)
    if y.ndim != 1 or y.size == 0:
        return np.nan, np.nan, np.nan, False, "invalid_segment", None, 0
    if bounds is None:
        bounds_used = (0, y.size - 1)
    else:
        bounds_used = (
            int(np.clip(int(bounds[0]), 0, y.size - 1)),
            int(np.clip(int(bounds[1]), 0, y.size - 1)),
        )
        if bounds_used[1] < bounds_used[0]:
            bounds_used = (bounds_used[1], bounds_used[0])
    intervals = _above_level_intervals_bounded(y, level, bounds_used)
    ref = None if bool(require_apex) else reference_interval
    interval = _select_tracked_interval(
        intervals,
        apex_local=int(apex_local),
        prev_interval=ref,
        allow_fallback=False,
    )
    if interval is None:
        return np.nan, np.nan, np.nan, False, "apex_component_not_reached", None, int(len(intervals))
    width, left_cross, right_cross, used_interp, reason = _crossings_for_interval(y, level, interval, bounds=bounds_used)
    return width, float(left_cross), float(right_cross), bool(used_interp), reason, interval, int(len(intervals))


def _compute_enhanced_edge_mapping(
    seg: np.ndarray,
    *,
    apex_local: int,
    bg_mad: float | None,
    levels_desc: Sequence[int],
    refine_step_percent: int,
    min_level_percent: int,
    require_closed_interval: bool,
    use_apex_component: bool,
    enable_merge_guard: bool,
    max_width_jump_factor: float,
    max_width_jump_points: float,
    noise_k_mad: float,
    noise_guard_enabled: bool = False,
) -> Dict[str, object]:
    y = np.asarray(seg, dtype=float)
    diag: Dict[str, object] = {
        "edge_mapping_enabled": True,
        "tested_levels": [],
        "valid_levels": [],
        "invalid_levels": [],
        "selected_interval_per_level": {},
        "crossing_count_per_level": {},
        "reason_invalid_per_level": {},
        "local_zero_level_percent_original": np.nan,
        "local_zero_level_value": np.nan,
        "local_top_value": np.nan,
        "merge_guard_triggered": False,
        "merge_clip_applied": False,
        "active_left": 0,
        "active_right": np.nan,
        "noise_guard_triggered": False,
        "fallback_used": False,
        "reason": "ok",
    }
    if y.ndim != 1 or y.size < 2 or not np.all(np.isfinite(y)):
        diag["reason"] = "invalid_segment"
        return diag
    ap = int(np.clip(apex_local, 0, y.size - 1))
    baseline0 = float(np.min(y))
    y_apex = float(y[ap])
    amp0 = float(y_apex - baseline0)
    diag.update({"baseline0": baseline0, "y_apex": y_apex, "amp0": amp0})
    if not np.isfinite(amp0) or amp0 <= 1e-12:
        diag["reason"] = "flat_or_negative_apex"
        return diag
    raw_levels = sorted({int(v) for v in levels_desc if int(v) >= int(min_level_percent)}, reverse=True)
    if not raw_levels:
        diag["reason"] = "no_mapping_levels"
        return diag
    if int(min(raw_levels)) > int(min_level_percent):
        raw_levels.append(int(min_level_percent))
        raw_levels = sorted(set(raw_levels), reverse=True)
    prev_interval: Optional[Tuple[int, int]] = None
    prev_width: Optional[float] = None
    last_valid: Optional[Dict[str, object]] = None
    active_left = 0
    active_right = int(y.size - 1)
    diag["active_right"] = int(active_right)

    def _clip_for_merge(prev: Tuple[int, int], curr: Tuple[int, int]) -> bool:
        nonlocal active_left, active_right
        prev_l, prev_r = int(prev[0]), int(prev[1])
        curr_l, curr_r = int(curr[0]), int(curr[1])
        changed = False
        left_growth = max(0, prev_l - curr_l)
        right_growth = max(0, curr_r - prev_r)
        if left_growth > 0 and left_growth >= right_growth:
            lo = max(active_left, curr_l)
            hi = min(active_right, prev_l)
            if hi >= lo:
                valley = int(lo + int(np.argmin(y[lo:hi + 1])))
                new_left = int(min(max(valley, active_left), prev_l))
                if new_left > active_left:
                    active_left = new_left
                    changed = True
                    diag["merge_clip_left"] = int(active_left)
        if right_growth > 0 and right_growth >= left_growth:
            lo = max(active_left, prev_r)
            hi = min(active_right, curr_r)
            if hi >= lo:
                valley = int(lo + int(np.argmin(y[lo:hi + 1])))
                new_right = int(max(min(valley, active_right), prev_r))
                if new_right < active_right:
                    active_right = new_right
                    changed = True
                    diag["merge_clip_right"] = int(active_right)
        if changed:
            diag["merge_clip_applied"] = True
            diag["active_left"] = int(active_left)
            diag["active_right"] = int(active_right)
        return bool(changed)

    def _clip_at_neighbor_crossings(level_percent: int, level_value: float, selected_interval: Tuple[int, int], intervals: Sequence[Tuple[int, int]]) -> None:
        nonlocal active_left, active_right
        sel_l, sel_r = int(selected_interval[0]), int(selected_interval[1])
        left_neighbors = [(int(a), int(b)) for a, b in intervals if int(b) < sel_l]
        right_neighbors = [(int(a), int(b)) for a, b in intervals if int(a) > sel_r]
        diag["neighbor_clip_per_level"] = diag.get("neighbor_clip_per_level", {})
        level_clip: Dict[str, object] = {}
        if left_neighbors:
            neighbor = max(left_neighbors, key=lambda t: int(t[1]))
            _, _, right_cross, _, _ = _crossings_for_interval(y, float(level_value), neighbor, bounds=(active_left, active_right))
            if np.isfinite(right_cross):
                new_left = int(np.clip(int(math.ceil(float(right_cross))), active_left, active_right))
                if new_left > active_left and new_left <= ap:
                    active_left = new_left
                    level_clip["left"] = float(right_cross)
                    level_clip["left_index"] = int(active_left)
        if right_neighbors:
            neighbor = min(right_neighbors, key=lambda t: int(t[0]))
            _, left_cross, _, _, _ = _crossings_for_interval(y, float(level_value), neighbor, bounds=(active_left, active_right))
            if np.isfinite(left_cross):
                new_right = int(np.clip(int(math.floor(float(left_cross))), active_left, active_right))
                if new_right < active_right and new_right >= ap:
                    active_right = new_right
                    level_clip["right"] = float(left_cross)
                    level_clip["right_index"] = int(active_right)
        if level_clip:
            diag["merge_clip_applied"] = True
            diag["active_left"] = int(active_left)
            diag["active_right"] = int(active_right)
            diag["neighbor_clip_per_level"][str(int(level_percent))] = level_clip

    def _test_level(level_percent: int) -> Tuple[bool, str, Optional[Tuple[int, int]], float, float]:
        level_value = float(baseline0 + (float(level_percent) / 100.0) * amp0)
        intervals = _above_level_intervals_bounded(y, level_value, (active_left, active_right))
        ref = None if (prev_interval is None or not bool(use_apex_component)) else prev_interval
        interval = _select_tracked_interval(intervals, apex_local=ap, prev_interval=ref, allow_fallback=False)
        diag["tested_levels"].append(int(level_percent))
        diag["crossing_count_per_level"][str(int(level_percent))] = int(2 * len(intervals))
        if interval is None:
            diag["invalid_levels"].append(int(level_percent))
            diag["reason_invalid_per_level"][str(int(level_percent))] = "no_component"
            return False, "no_component", None, level_value, np.nan
        l, r = int(interval[0]), int(interval[1])
        width, left_cross, right_cross, _used_interp, width_reason = _crossings_for_interval(y, level_value, interval, bounds=(active_left, active_right))
        diag["selected_interval_per_level"][str(int(level_percent))] = [int(l), int(r)]
        diag["selected_crossing_per_level"] = diag.get("selected_crossing_per_level", {})
        diag["selected_crossing_per_level"][str(int(level_percent))] = [float(left_cross), float(right_cross)]
        if not np.isfinite(width):
            diag["invalid_levels"].append(int(level_percent))
            diag["reason_invalid_per_level"][str(int(level_percent))] = str(width_reason)
            return False, str(width_reason), interval, level_value, np.nan
        if bool(require_closed_interval) and (l <= active_left or r >= active_right):
            reason = "open_interval_at_active_boundary"
            diag["invalid_levels"].append(int(level_percent))
            diag["reason_invalid_per_level"][str(int(level_percent))] = reason
            return False, reason, interval, level_value, width
        _clip_at_neighbor_crossings(int(level_percent), float(level_value), interval, intervals)
        if bool(enable_merge_guard) and prev_width is not None and np.isfinite(prev_width) and float(prev_width) >= 1.0:
            if width > float(prev_width) * float(max_width_jump_factor) or (width - float(prev_width)) > float(max_width_jump_points):
                diag["merge_guard_triggered"] = True
                if prev_interval is not None and _clip_for_merge(prev_interval, interval):
                    return _test_level(int(level_percent))
                diag["reason_invalid_per_level"][str(int(level_percent))] = "merge_guard_unclipped_but_accepted"
        diag["valid_levels"].append(int(level_percent))
        return True, "ok", interval, level_value, width

    for pct in raw_levels:
        ok, _reason, interval, level_value, width = _test_level(int(pct))
        if ok and interval is not None:
            prev_interval = interval
            prev_width = float(width)
            last_valid = {"pct": int(pct), "value": float(level_value), "interval": interval, "width": float(width)}
            continue
        if last_valid is not None:
            upper = int(last_valid["pct"])
            lower = int(pct)
            step = max(1, int(refine_step_percent))
            for rpct in range(upper - step, lower - 1, -step):
                if rpct < int(min_level_percent):
                    continue
                ok_r, _reason_r, interval_r, value_r, width_r = _test_level(int(rpct))
                if ok_r and interval_r is not None:
                    prev_interval = interval_r
                    prev_width = float(width_r)
                    last_valid = {"pct": int(rpct), "value": float(value_r), "interval": interval_r, "width": float(width_r)}
                else:
                    break
        break

    if last_valid is None:
        diag["reason"] = "no_valid_closed_interval"
        return diag
    local_zero = float(last_valid["value"])
    if bool(noise_guard_enabled) and bg_mad is not None:
        noise_floor = float(baseline0 + max(0.0, float(noise_k_mad)) * max(float(bg_mad), 0.0))
        diag["noise_floor"] = float(noise_floor)
        if np.isfinite(noise_floor) and local_zero < noise_floor < y_apex:
            local_zero = float(noise_floor)
            diag["noise_guard_triggered"] = True
    diag["local_zero_level_percent_original"] = int(last_valid["pct"])
    diag["local_zero_level_value"] = float(local_zero)
    diag["local_top_value"] = float(y_apex)
    diag["local_zero_interval"] = [int(last_valid["interval"][0]), int(last_valid["interval"][1])]
    return diag


def compute_edge_width_metrics(
    signal: np.ndarray,
    *,
    detection_left: int,
    detection_right: int,
    prefix: str,
    apex_idx: int | None = None,
    bg_mad: float | None = None,
    min_width_delta: float = 0.5,
    include_low_root_metrics: bool = False,
    low_root_noise_k_mad: float = 1.0,
    use_enhanced_spike_mapping: bool = False,
    mapping_levels_desc: Sequence[int] = (90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5),
    mapping_refine_step_percent: int = 5,
    mapping_min_level_percent: int = 5,
    mapping_require_closed_interval: bool = True,
    mapping_use_apex_component: bool = True,
    mapping_enable_merge_guard: bool = True,
    mapping_max_width_jump_factor: float = 2.5,
    mapping_max_width_jump_points: float = 8.0,
    mapping_fallback_to_old: bool = False,
    mapping_noise_guard_enabled: bool = False,
    robust_reference_enabled: bool = False,
    robust_reference_noise: float | None = None,
) -> Dict[str, object]:
    x = np.asarray(signal, dtype=float)
    out: Dict[str, object] = {
        f"{prefix}_dense_width_sum_0_90": np.nan,
        f"{prefix}_dense_width_valid_n_0_90": 0.0,
        f"{prefix}_dense_width_missing_n_0_90": 19.0,
        f"{prefix}_dense_width_complete_0_90": 0.0,
    }
    if x.ndim != 1 or x.size == 0:
        out[f"{prefix}_debug"] = {"reason": "empty_signal"}
        return out
    n = int(x.size)
    left = int(np.clip(detection_left, 0, n - 1))
    right = int(np.clip(detection_right, 0, n - 1))
    if right < left:
        left, right = right, left
    seg = np.asarray(x[left:right + 1], dtype=float)
    if seg.size == 0 or not np.all(np.isfinite(seg)):
        out[f"{prefix}_debug"] = {"reason": "invalid_segment", "measurement_left": int(left), "measurement_right": int(right)}
        return out
    y_min = float(np.min(seg))
    y_max = float(np.max(seg))
    dseg = np.diff(seg)
    if dseg.size:
        dmed = float(np.median(dseg))
        diff_noise_mad = float(np.median(np.abs(dseg - dmed)) / max(np.sqrt(2.0), 1e-12))
    else:
        diff_noise_mad = np.nan
    if bg_mad is None:
        edge_noise_mad = diff_noise_mad
    else:
        candidates_noise = [float(bg_mad)]
        if np.isfinite(diff_noise_mad) and diff_noise_mad > 0.0:
            candidates_noise.append(float(diff_noise_mad))
        edge_noise_mad = float(min(candidates_noise))
    apex_local = int(np.argmax(seg)) if apex_idx is None else int(np.clip(int(apex_idx) - int(left), 0, seg.size - 1))
    reference_original = float(y_max)
    reference_noise = float(max(robust_reference_noise if robust_reference_noise is not None else edge_noise_mad, 0.0)) if np.isfinite(robust_reference_noise if robust_reference_noise is not None else edge_noise_mad) else float("nan")
    reference_robust = float(reference_original)
    reference_delta = 0.0
    reference_adjusted = False
    reference_reason = "disabled"
    if bool(robust_reference_enabled) and seg.size >= 3 and np.isfinite(reference_noise) and reference_noise > 0.0:
        nb_l = max(0, apex_local - 2)
        nb_r = min(seg.size, apex_local + 3)
        neighbor_vals = np.asarray([seg[i] for i in range(nb_l, nb_r) if i != apex_local], dtype=float)
        local_top_support = float(np.max(neighbor_vals)) if neighbor_vals.size else float(reference_original)
        protrusion = float(max(reference_original - local_top_support, 0.0))
        protrusion_cap = float(1.25 * reference_noise)
        if protrusion > 0.0 and protrusion <= protrusion_cap:
            reference_delta = float(min(protrusion, reference_noise))
            reference_robust = float(reference_original - reference_delta)
            reference_adjusted = bool(reference_delta > 0.0)
            reference_reason = "small_noise_scale_apex_protrusion"
        else:
            reference_reason = "apex_protrusion_not_noise_scale"
    amp = float(max(reference_robust - y_min, 0.0))
    debug: Dict[str, object] = {
        "measurement_left": int(left),
        "measurement_right": int(right),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "amp": float(amp),
        "apex_local": int(apex_local),
        "apex": int(left + apex_local),
        "interpolation_used": False,
        "reason": "ok",
        "min_width_delta": float(min_width_delta),
        "input_bg_mad": np.nan if bg_mad is None else float(bg_mad),
        "diff_noise_mad": float(diff_noise_mad) if np.isfinite(diff_noise_mad) else np.nan,
        "edge_noise_mad": float(edge_noise_mad) if np.isfinite(edge_noise_mad) else np.nan,
        "edge_robust_reference_enabled": bool(robust_reference_enabled),
        "edge_reference_original": float(reference_original),
        "edge_reference_robust": float(reference_robust),
        "edge_reference_delta": float(reference_delta),
        "edge_reference_noise_used": float(reference_noise) if np.isfinite(reference_noise) else np.nan,
        "edge_reference_adjusted": bool(reference_adjusted),
        "edge_reference_reason": str(reference_reason),
    }
    if not np.isfinite(amp) or amp <= 1e-12:
        debug["reason"] = "flat_segment"
        out[f"{prefix}_debug"] = debug
        return out
    mapping_diag: Dict[str, object] = {"edge_mapping_enabled": bool(use_enhanced_spike_mapping)}
    mapping_reference_interval: Optional[Tuple[int, int]] = None
    mapping_active_bounds: Optional[Tuple[int, int]] = None
    if bool(use_enhanced_spike_mapping):
        mapping_diag = _compute_enhanced_edge_mapping(
            seg,
            apex_local=int(apex_local),
            bg_mad=bg_mad,
            levels_desc=mapping_levels_desc,
            refine_step_percent=int(mapping_refine_step_percent),
            min_level_percent=int(mapping_min_level_percent),
            require_closed_interval=bool(mapping_require_closed_interval),
            use_apex_component=bool(mapping_use_apex_component),
            enable_merge_guard=bool(mapping_enable_merge_guard),
            max_width_jump_factor=float(mapping_max_width_jump_factor),
            max_width_jump_points=float(mapping_max_width_jump_points),
            noise_k_mad=float(low_root_noise_k_mad),
            noise_guard_enabled=bool(mapping_noise_guard_enabled),
        )
        mapping_reference_interval_raw = mapping_diag.get("local_zero_interval")
        if isinstance(mapping_reference_interval_raw, (list, tuple)) and len(mapping_reference_interval_raw) == 2:
            mapping_reference_interval = (int(mapping_reference_interval_raw[0]), int(mapping_reference_interval_raw[1]))
            mapping_active_bounds = (
                int(mapping_diag.get("active_left", 0)),
                int(mapping_diag.get("active_right", seg.size - 1)),
            )
        elif not bool(mapping_fallback_to_old):
            debug["edge_mapping"] = mapping_diag
            debug["reason"] = str(mapping_diag.get("reason", "mapping_failed"))
            out[f"{prefix}_debug"] = debug
            return out
    debug["edge_mapping"] = mapping_diag
    widths: Dict[str, float] = {}
    levels: Dict[str, float] = {}
    level_fracs = {"0": 0.0, "5": 0.05, "10": 0.10, "15": 0.15, "20": 0.20, "25": 0.25, "30": 0.30, "35": 0.35, "40": 0.40, "45": 0.45, "50": 0.50, "55": 0.55, "60": 0.60, "65": 0.65, "70": 0.70, "75": 0.75, "80": 0.80, "85": 0.85, "90": 0.90}
    noise_level = (
        float(y_min + 0.05 * amp)
        if not np.isfinite(edge_noise_mad)
        else float(y_min + max(0.0, float(low_root_noise_k_mad)) * max(float(edge_noise_mad), 0.0))
    )
    debug["low_root_noise_level"] = float(noise_level)
    for tag, frac in level_fracs.items():
        if bool(use_enhanced_spike_mapping) and mapping_reference_interval is not None:
            local_zero = float(mapping_diag.get("local_zero_level_value", np.nan))
            local_top = float(mapping_diag.get("local_top_value", np.nan))
            level = float(local_zero + frac * (local_top - local_zero))
        else:
            level = float(y_min + frac * amp)
        levels[tag] = float(level)
        if bool(use_enhanced_spike_mapping) and mapping_reference_interval is not None:
            width, left_cross, right_cross, used_interp, reason, selected_interval, crossing_count = edge_component_width_at_level(
                seg,
                level,
                apex_local=int(apex_local),
                reference_interval=mapping_reference_interval,
                bounds=mapping_active_bounds,
            )
            debug[f"selected_interval_{tag}"] = None if selected_interval is None else [int(selected_interval[0]), int(selected_interval[1])]
            debug[f"crossing_count_{tag}"] = int(crossing_count)
        else:
            width, left_cross, right_cross, used_interp, reason = width_between_outer_level_crossings(seg, level)
        debug[f"level_{tag}"] = float(level)
        debug[f"left_cross_{tag}"] = (np.nan if not np.isfinite(left_cross) else float(left + left_cross))
        debug[f"right_cross_{tag}"] = (np.nan if not np.isfinite(right_cross) else float(left + right_cross))
        debug[f"width_reason_{tag}"] = str(reason)
        debug["interpolation_used"] = bool(debug["interpolation_used"] or used_interp)
        widths[tag] = float(width)
        out[f"{prefix}_width_at_{tag}"] = float(width)
    dense_tags = tuple(str(v) for v in range(0, 95, 5))
    dense_widths: List[float] = []
    dense_levels: List[int] = []
    dense_debug: Dict[str, object] = {"levels": [int(v) for v in range(0, 95, 5)], "widths": {}, "ratios": {}, "skipped": {}}
    if bool(use_enhanced_spike_mapping) and mapping_reference_interval is not None:
        local_zero = float(mapping_diag.get("local_zero_level_value", np.nan))
        local_top = float(mapping_diag.get("local_top_value", np.nan))
        root_amp = float(local_top - local_zero) if np.isfinite(local_zero) and np.isfinite(local_top) else np.nan
    else:
        local_zero = float(y_min)
        local_top = float(y_max)
        root_amp = float(amp)
    noise_amp = (
        float(0.05 * max(root_amp, 0.0))
        if not np.isfinite(edge_noise_mad)
        else float(max(0.0, float(low_root_noise_k_mad)) * 1.4826 * max(float(edge_noise_mad), 0.0))
    )
    root95_amp = float(0.95 * root_amp) if np.isfinite(root_amp) else np.nan
    dense_debug["local_zero_level_value"] = float(local_zero) if np.isfinite(local_zero) else np.nan
    dense_debug["local_top_value"] = float(local_top) if np.isfinite(local_top) else np.nan
    dense_debug["root_amp"] = float(root_amp) if np.isfinite(root_amp) else np.nan
    dense_debug["root95_amp"] = float(root95_amp) if np.isfinite(root95_amp) else np.nan
    dense_debug["noise_amp"] = float(noise_amp) if np.isfinite(noise_amp) else np.nan
    dense_debug["input_bg_mad"] = np.nan if bg_mad is None else float(bg_mad)
    dense_debug["diff_noise_mad"] = float(diff_noise_mad) if np.isfinite(diff_noise_mad) else np.nan
    dense_debug["edge_noise_mad"] = float(edge_noise_mad) if np.isfinite(edge_noise_mad) else np.nan
    dense_debug["root_snr"] = float(root95_amp / max(noise_amp, 1e-12)) if np.isfinite(root95_amp) and np.isfinite(noise_amp) else np.nan
    if (not np.isfinite(root95_amp)) or root95_amp <= max(float(noise_amp), 1e-12):
        dense_debug["reason"] = "root_amp_below_noise"
        debug["dense_width_0_90"] = dense_debug
        out[f"{prefix}_debug"] = debug
        return out
    for tag in dense_tags:
        width = float(widths.get(tag, np.nan))
        left_cross = float(debug.get(f"left_cross_{tag}", np.nan))
        right_cross = float(debug.get(f"right_cross_{tag}", np.nan))
        level = float(levels.get(tag, np.nan))
        reason = str(debug.get(f"width_reason_{tag}", "ok"))
        dense_debug["widths"][tag] = {
            "level": float(level) if np.isfinite(level) else np.nan,
            "width": float(width) if np.isfinite(width) else np.nan,
            "left_cross": float(left_cross) if np.isfinite(left_cross) else np.nan,
            "right_cross": float(right_cross) if np.isfinite(right_cross) else np.nan,
            "reason": str(reason),
        }
        if not np.isfinite(width):
            dense_debug["skipped"][tag] = str(reason)
            continue
        dense_widths.append(float(width))
        dense_levels.append(int(tag))
    out[f"{prefix}_dense_width_valid_n_0_90"] = float(len(dense_widths))
    out[f"{prefix}_dense_width_missing_n_0_90"] = float(max(0, len(dense_tags) - len(dense_widths)))
    if not dense_widths:
        dense_debug["reason"] = "no_valid_widths"
        debug["dense_width_0_90"] = dense_debug
        out[f"{prefix}_debug"] = debug
        return out
    if len(dense_widths) != len(dense_tags):
        dense_debug["reason"] = "incomplete_dense_levels"
        debug["dense_width_0_90"] = dense_debug
        out[f"{prefix}_debug"] = debug
        return out
    out[f"{prefix}_dense_width_complete_0_90"] = 1.0
    out[f"{prefix}_dense_width_missing_n_0_90"] = 0.0
    w = np.asarray(dense_widths, dtype=float)
    out[f"{prefix}_dense_width_sum_0_90"] = float(np.sum(w))
    dense_debug["reason"] = "ok"
    debug["dense_width_0_90"] = dense_debug
    out[f"{prefix}_debug"] = debug
    return out
