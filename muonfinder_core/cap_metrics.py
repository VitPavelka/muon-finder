from __future__ import annotations

"""Experimental CAP v3 features based on EDGE-foot prominence fractions.

CAP v3 keeps the CAP v2 core fraction:
what fraction of the EDGE-based candidate prominence is removed by a median
filter at a given scale.

It then describes the whole early-to-mid median-scale curve instead of relying
on one window or a single ratio.

Spike-like cap:
higher small-window removal, higher early-to-mid curve level, limited later
growth.

Sharp Raman-like structure:
lower early removal, lower early-to-mid curve level, more gradual growth.

This is exploratory and does not modify final pipeline decisions.
"""

import csv
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


CAP_JOIN_COLUMNS = (
    "source_y",
    "source_x",
    "compact_y",
    "compact_x",
    "peak_index",
    "candidate_id",
    "start_index",
    "end_index",
)

NOISE_REJECTED_VALUES = {"rejected_noise", "rejected", "noise_rejected"}
NOISE_STATUS_FIELDS = (
    "candidate_noise_prefilter_status",
    "noise_prefilter_status",
    "candidate_noise_status",
)
DEFAULT_MEDIAN_WINDOWS = [3, 5, 7, 9, 13, 17, 21, 31]
DEFAULT_CURVE_WINDOWS = [3, 5, 7, 9, 13, 17]
DEFAULT_EARLY_WINDOWS = [3, 5, 7]
DEFAULT_MID_WINDOWS = [9, 13, 17]
DEFAULT_LATE_WINDOWS = [17, 21, 31]
DEFAULT_LATE_REFERENCE_WINDOW = 31


def cap_defaults(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(config or {})
    return {
        "features_path": str(cfg.get("features_path", "outputs_core/cap_features.csv")),
        "summary_path": str(cfg.get("summary_path", "outputs_core/cap_features_summary.json")),
        "candidate_scope": str(cfg.get("candidate_scope", "noise-kept")),
        "signal_source": str(cfg.get("signal_source", "raw")),
        "median_windows": [int(v) for v in cfg.get("median_windows", DEFAULT_MEDIAN_WINDOWS)],
        "curve_windows": [int(v) for v in cfg.get("curve_windows", DEFAULT_CURVE_WINDOWS)],
        "early_windows": [int(v) for v in cfg.get("early_windows", DEFAULT_EARLY_WINDOWS)],
        "mid_windows": [int(v) for v in cfg.get("mid_windows", DEFAULT_MID_WINDOWS)],
        "late_windows": [int(v) for v in cfg.get("late_windows", DEFAULT_LATE_WINDOWS)],
        "late_reference_window": int(cfg.get("late_reference_window", DEFAULT_LATE_REFERENCE_WINDOW)),
        "min_prominence": float(cfg.get("min_prominence", 1e-12)),
        "eps": float(cfg.get("eps", 1e-12)),
    }


def resolve_extra_features_path(cfg: Any, override: Path | None = None) -> Path | None:
    if override is not None:
        return Path(override)
    cap_cfg = getattr(cfg, "cap", {}) if cfg is not None else {}
    path = str((cap_cfg or {}).get("features_path", "")).strip()
    return Path(path) if path else None


def load_extra_feature_rows(path: Path | str) -> tuple[list[dict[str, Any]], list[str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    columns = sorted({key for row in rows for key in row.keys() if key})
    return rows, columns


def join_extra_feature_rows(rows: list[dict[str, Any]], extra_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not extra_rows:
        return {"loaded_rows": 0, "matched_rows": 0, "unmatched_rows": 0, "extra_columns": []}
    extra_cols = sorted({key for row in extra_rows for key in row.keys() if key and key not in CAP_JOIN_COLUMNS})
    by_candidate: dict[str, int] = {}
    by_peak: dict[tuple[int, int, int], int] = {}
    for idx, row in enumerate(extra_rows):
        candidate_id = str(row.get("candidate_id", "")).strip()
        if candidate_id:
            by_candidate[candidate_id] = idx
        try:
            key = (int(row.get("source_y", -1)), int(row.get("source_x", -1)), int(row.get("peak_index", -1)))
        except Exception:
            continue
        by_peak[key] = idx
    used: set[int] = set()
    matched = 0
    for row in rows:
        idx: int | None = None
        candidate_id = str(row.get("candidate_id", "")).strip()
        if candidate_id and candidate_id in by_candidate:
            idx = by_candidate[candidate_id]
        else:
            key = (
                int(row.get("source_y", row.get("y", -1))),
                int(row.get("source_x", row.get("x", -1))),
                int(row.get("peak_index", -1)),
            )
            if key in by_peak:
                idx = by_peak[key]
        if idx is None:
            continue
        used.add(idx)
        matched += 1
        extra = extra_rows[idx]
        for col in extra_cols:
            row[col] = extra.get(col)
    return {
        "loaded_rows": int(len(extra_rows)),
        "matched_rows": int(matched),
        "unmatched_rows": int(len(extra_rows) - len(used)),
        "extra_columns": extra_cols,
    }


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else float("nan")


def _median_filter_1d(signal: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if w <= 1 or x.size <= 1:
        return np.asarray(x, dtype=float)
    pad = w // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.median(sliding_window_view(padded, w), axis=-1)


def _clean_windows(values: list[int], default: list[int]) -> list[int]:
    out = sorted({max(1, int(v) + (0 if int(v) % 2 == 1 else 1)) for v in values})
    return out or [int(v) for v in default]


def _window_label(windows: list[int]) -> str:
    return f"w{int(windows[0])}_to_w{int(windows[-1])}"


def _signal_from_cache(cache: dict[str, Any], row: dict[str, Any], signal_source: str) -> np.ndarray:
    y = int(row.get("y", -1))
    x = int(row.get("x", -1))
    source = str(signal_source).strip().lower()
    if source == "corrected":
        return np.asarray(cache["corrected_spectra"][y, x, :], dtype=float)
    return np.asarray(cache["spectra"][y, x, :], dtype=float)


def _noise_prefilter_status(row: dict[str, Any]) -> tuple[str | None, str | None]:
    for field in NOISE_STATUS_FIELDS:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        return field, str(value).strip().lower()
    return None, None


def is_noise_rejected(row: dict[str, Any]) -> tuple[bool | None, str | None, str | None]:
    field, status = _noise_prefilter_status(row)
    if field is None:
        return None, None, None
    return bool(status in NOISE_REJECTED_VALUES), field, status


def filter_cap_candidate_rows(rows: list[dict[str, Any]], candidate_scope: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scope = str(candidate_scope).strip().lower()
    total = int(len(rows))
    rejected = 0
    missing_status = 0
    used_rows: list[dict[str, Any]] = []
    for row in rows:
        is_rejected, _field, _status = is_noise_rejected(row)
        if is_rejected is None:
            missing_status += 1
        elif is_rejected:
            rejected += 1
        if scope == "all":
            used_rows.append(dict(row))
            continue
        if is_rejected is True:
            continue
        used_rows.append(dict(row))
    return used_rows, {
        "candidate_scope": scope,
        "total_loaded_candidates": total,
        "candidates_rejected_by_noise_filter": int(rejected),
        "candidates_missing_noise_status": int(missing_status),
        "candidates_used": int(len(used_rows)),
    }


def _edge_foot_value(row: dict[str, Any]) -> tuple[str | None, float]:
    debug = row.get("edge_debug", {})
    if isinstance(debug, dict):
        foot = _safe_float(debug.get("edge_selected_foot_value"))
        if np.isfinite(foot):
            return "edge_selected_foot_value", foot
    for key in ("edge_selected_foot_value", "edge_foot_value", "edge_base_value"):
        foot = _safe_float(row.get(key))
        if np.isfinite(foot):
            return key, foot
    return None, float("nan")


def _finite_values(frac_by_window: dict[int, float], windows: list[int], clipped: bool = False) -> np.ndarray:
    values = [_clip01(frac_by_window[w]) if clipped else frac_by_window[w] for w in windows]
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _mean_from_windows(frac_by_window: dict[int, float], windows: list[int], clipped: bool = False) -> float:
    values = _finite_values(frac_by_window, windows, clipped=clipped)
    return float(np.mean(values)) if values.size else float("nan")


def _curve_stats(frac_by_window: dict[int, float], windows: list[int], clipped: bool = False) -> dict[str, float]:
    values = _finite_values(frac_by_window, windows, clipped=clipped)
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan"), "min": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def _curve_auc(frac_by_window: dict[int, float], windows: list[int], clipped: bool = False, log_scale: bool = False) -> float:
    xs = np.log(np.asarray(windows, dtype=float)) if log_scale else np.asarray(windows, dtype=float)
    ys = np.asarray([_clip01(frac_by_window[w]) if clipped else frac_by_window[w] for w in windows], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if np.sum(mask) < 2:
        return float("nan")
    x_use = xs[mask]
    y_use = ys[mask]
    span = float(x_use[-1] - x_use[0])
    if not np.isfinite(span) or abs(span) <= 1e-18:
        return float("nan")
    return float(np.trapezoid(y_use, x_use) / span)


def _curve_fit(frac_by_window: dict[int, float], windows: list[int], clipped: bool = False) -> dict[str, float]:
    xs = np.log(np.asarray(windows, dtype=float))
    ys = np.asarray([_clip01(frac_by_window[w]) if clipped else frac_by_window[w] for w in windows], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if np.sum(mask) < 2:
        return {"intercept": float("nan"), "slope": float("nan"), "r2": float("nan")}
    x_use = xs[mask]
    y_use = ys[mask]
    slope, intercept = np.polyfit(x_use, y_use, 1)
    fitted = intercept + slope * x_use
    ss_res = float(np.sum((y_use - fitted) ** 2))
    ss_tot = float(np.sum((y_use - np.mean(y_use)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-18 else float("nan")
    return {"intercept": float(intercept), "slope": float(slope), "r2": r2}


def _window_value(frac_by_window: dict[int, float], window: int, clipped: bool = False) -> float:
    value = float(frac_by_window.get(int(window), np.nan))
    return _clip01(value) if clipped else value


def _difference(frac_by_window: dict[int, float], left: int, right: int, clipped: bool = False) -> float:
    a = _window_value(frac_by_window, left, clipped=clipped)
    b = _window_value(frac_by_window, right, clipped=clipped)
    return float(b - a) if np.isfinite(a) and np.isfinite(b) else float("nan")


def _relative_growth(frac_by_window: dict[int, float], left: int, right: int, eps: float, clipped: bool = False) -> float:
    growth = _difference(frac_by_window, left, right, clipped=clipped)
    denom = abs(_window_value(frac_by_window, right, clipped=clipped))
    if not np.isfinite(growth) or not np.isfinite(denom):
        return float("nan")
    return float(growth / (denom + eps))


def _step_like(mean_clip: float, growth_rel_clip: float) -> float:
    if not np.isfinite(mean_clip) or not np.isfinite(growth_rel_clip):
        return float("nan")
    return float(mean_clip * (1.0 - _clip01(growth_rel_clip)))


def compute_cap_feature_row(cache: dict[str, Any], row: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    cfg = cap_defaults(options)
    median_windows = _clean_windows([int(v) for v in cfg["median_windows"]], DEFAULT_MEDIAN_WINDOWS)
    curve_windows = _clean_windows([int(v) for v in cfg["curve_windows"]], DEFAULT_CURVE_WINDOWS)
    early_windows = _clean_windows([int(v) for v in cfg["early_windows"]], DEFAULT_EARLY_WINDOWS)
    mid_windows = _clean_windows([int(v) for v in cfg["mid_windows"]], DEFAULT_MID_WINDOWS)
    late_windows = _clean_windows([int(v) for v in cfg["late_windows"]], DEFAULT_LATE_WINDOWS)
    late_ref = int(_clean_windows([int(cfg["late_reference_window"])], [DEFAULT_LATE_REFERENCE_WINDOW])[0])
    required = sorted(
        set(median_windows + DEFAULT_MEDIAN_WINDOWS + curve_windows + early_windows + mid_windows + late_windows + [late_ref])
    )
    median_windows = required
    curve_label = _window_label(curve_windows)
    signal = _signal_from_cache(cache, row, str(cfg["signal_source"]))
    peak_index = int(row.get("peak_index", -1))
    base_source, edge_foot = _edge_foot_value(row)
    out: dict[str, Any] = {
        "source_y": int(row.get("source_y", row.get("y", -1))),
        "source_x": int(row.get("source_x", row.get("x", -1))),
        "compact_y": int(row.get("y", -1)),
        "compact_x": int(row.get("x", -1)),
        "peak_index": int(peak_index),
        "candidate_id": row.get("candidate_id", ""),
        "start_index": row.get("start_index", row.get("start", "")),
        "end_index": row.get("end_index", row.get("end", "")),
        "cap_v2_status": "ok",
        "cap_base_source": str(base_source) if base_source is not None else "",
        "cap_raw_apex": np.nan,
        "cap_edge_foot_value": float(edge_foot) if np.isfinite(edge_foot) else np.nan,
        "cap_prominence": np.nan,
    }
    for w in median_windows:
        out[f"cap_median_w{w}"] = np.nan
        out[f"cap_removed_w{w}"] = np.nan
        out[f"cap_frac_w{w}"] = np.nan
        out[f"cap_frac_w{w}_clip01"] = np.nan
    scalar_features = (
        f"cap_curve_mean_{curve_label}",
        f"cap_curve_median_{curve_label}",
        f"cap_curve_max_{curve_label}",
        f"cap_curve_min_{curve_label}",
        f"cap_curve_mean_{curve_label}_clip01",
        f"cap_curve_median_{curve_label}_clip01",
        f"cap_curve_max_{curve_label}_clip01",
        f"cap_curve_min_{curve_label}_clip01",
        f"cap_curve_auc_{curve_label}",
        f"cap_curve_auc_logw_{curve_label}",
        f"cap_curve_auc_{curve_label}_clip01",
        f"cap_curve_auc_logw_{curve_label}_clip01",
        f"cap_curve_intercept_logw_{curve_label}",
        f"cap_curve_slope_logw_{curve_label}",
        f"cap_curve_r2_logw_{curve_label}",
        f"cap_curve_intercept_logw_{curve_label}_clip01",
        f"cap_curve_slope_logw_{curve_label}_clip01",
        f"cap_curve_r2_logw_{curve_label}_clip01",
        f"cap_curve_early_growth_w{early_windows[0]}_to_w{mid_windows[0]}",
        f"cap_curve_early_growth_w{early_windows[0]}_to_w{mid_windows[0]}_clip01",
        f"cap_curve_mid_growth_w{mid_windows[0]}_to_w{mid_windows[-1]}",
        f"cap_curve_mid_growth_w{mid_windows[0]}_to_w{mid_windows[-1]}_clip01",
        f"cap_curve_late_growth_w{late_windows[0]}_to_w{late_windows[-1]}",
        f"cap_curve_late_growth_w{late_windows[0]}_to_w{late_windows[-1]}_clip01",
        f"cap_curve_total_growth_w{early_windows[0]}_to_w{late_ref}",
        f"cap_curve_total_growth_w{early_windows[0]}_to_w{late_ref}_clip01",
        f"cap_curve_mid_growth_rel_w{mid_windows[0]}_to_w{mid_windows[-1]}",
        f"cap_curve_mid_growth_rel_w{mid_windows[0]}_to_w{mid_windows[-1]}_clip01",
        f"cap_curve_late_growth_rel_w{late_windows[0]}_to_w{late_windows[-1]}",
        f"cap_curve_late_growth_rel_w{late_windows[0]}_to_w{late_windows[-1]}_clip01",
        f"cap_curve_frontload_{_window_label(curve_windows)}",
        f"cap_curve_frontload_{_window_label(curve_windows)}_clip01",
        f"cap_curve_frontload_w{early_windows[0]}_to_w{late_ref}",
        f"cap_curve_frontload_w{early_windows[0]}_to_w{late_ref}_clip01",
        "cap_curve_step_like_v1",
        "cap_curve_step_like_v2",
    )
    for name in scalar_features:
        out[name] = np.nan
    for left, right in zip(median_windows[:-1], median_windows[1:]):
        out[f"cap_curve_d_w{left}_w{right}"] = np.nan
        out[f"cap_curve_d_w{left}_w{right}_clip01"] = np.nan
    if signal.ndim != 1 or signal.size == 0 or not (0 <= peak_index < signal.size):
        out["cap_v2_status"] = "invalid_signal"
        return out
    raw_apex = _safe_float(signal[peak_index])
    out["cap_raw_apex"] = float(raw_apex) if np.isfinite(raw_apex) else np.nan
    if not np.isfinite(edge_foot):
        out["cap_v2_status"] = "missing_edge_foot"
        return out
    prominence = float(raw_apex - edge_foot)
    out["cap_prominence"] = float(prominence) if np.isfinite(prominence) else np.nan
    if not np.isfinite(prominence) or prominence <= float(cfg["min_prominence"]):
        out["cap_v2_status"] = "invalid_prominence"
        return out

    frac_by_window: dict[int, float] = {}
    for w in median_windows:
        median_value = _safe_float(_median_filter_1d(signal, w)[peak_index])
        removed = float(raw_apex - median_value) if np.isfinite(median_value) else float("nan")
        frac = float(removed / prominence) if np.isfinite(removed) else float("nan")
        out[f"cap_median_w{w}"] = float(median_value) if np.isfinite(median_value) else np.nan
        out[f"cap_removed_w{w}"] = float(removed) if np.isfinite(removed) else np.nan
        out[f"cap_frac_w{w}"] = float(frac) if np.isfinite(frac) else np.nan
        out[f"cap_frac_w{w}_clip01"] = _clip01(frac)
        frac_by_window[w] = frac

    stats_raw = _curve_stats(frac_by_window, curve_windows, clipped=False)
    stats_clip = _curve_stats(frac_by_window, curve_windows, clipped=True)
    out[f"cap_curve_mean_{curve_label}"] = stats_raw["mean"]
    out[f"cap_curve_median_{curve_label}"] = stats_raw["median"]
    out[f"cap_curve_max_{curve_label}"] = stats_raw["max"]
    out[f"cap_curve_min_{curve_label}"] = stats_raw["min"]
    out[f"cap_curve_mean_{curve_label}_clip01"] = stats_clip["mean"]
    out[f"cap_curve_median_{curve_label}_clip01"] = stats_clip["median"]
    out[f"cap_curve_max_{curve_label}_clip01"] = stats_clip["max"]
    out[f"cap_curve_min_{curve_label}_clip01"] = stats_clip["min"]
    out[f"cap_curve_auc_{curve_label}"] = _curve_auc(frac_by_window, curve_windows, clipped=False, log_scale=False)
    out[f"cap_curve_auc_logw_{curve_label}"] = _curve_auc(frac_by_window, curve_windows, clipped=False, log_scale=True)
    out[f"cap_curve_auc_{curve_label}_clip01"] = _curve_auc(frac_by_window, curve_windows, clipped=True, log_scale=False)
    out[f"cap_curve_auc_logw_{curve_label}_clip01"] = _curve_auc(frac_by_window, curve_windows, clipped=True, log_scale=True)

    fit_raw = _curve_fit(frac_by_window, curve_windows, clipped=False)
    fit_clip = _curve_fit(frac_by_window, curve_windows, clipped=True)
    out[f"cap_curve_intercept_logw_{curve_label}"] = fit_raw["intercept"]
    out[f"cap_curve_slope_logw_{curve_label}"] = fit_raw["slope"]
    out[f"cap_curve_r2_logw_{curve_label}"] = fit_raw["r2"]
    out[f"cap_curve_intercept_logw_{curve_label}_clip01"] = fit_clip["intercept"]
    out[f"cap_curve_slope_logw_{curve_label}_clip01"] = fit_clip["slope"]
    out[f"cap_curve_r2_logw_{curve_label}_clip01"] = fit_clip["r2"]

    for left, right in zip(median_windows[:-1], median_windows[1:]):
        out[f"cap_curve_d_w{left}_w{right}"] = _difference(frac_by_window, left, right, clipped=False)
        out[f"cap_curve_d_w{left}_w{right}_clip01"] = _difference(frac_by_window, left, right, clipped=True)

    eps = float(max(cfg["eps"], 1e-18))
    early_start = early_windows[0]
    mid_start = mid_windows[0]
    mid_end = mid_windows[-1]
    late_start = late_windows[0]
    late_end = late_windows[-1]
    out[f"cap_curve_early_growth_w{early_start}_to_w{mid_start}"] = _difference(frac_by_window, early_start, mid_start, clipped=False)
    out[f"cap_curve_early_growth_w{early_start}_to_w{mid_start}_clip01"] = _difference(frac_by_window, early_start, mid_start, clipped=True)
    out[f"cap_curve_mid_growth_w{mid_start}_to_w{mid_end}"] = _difference(frac_by_window, mid_start, mid_end, clipped=False)
    out[f"cap_curve_mid_growth_w{mid_start}_to_w{mid_end}_clip01"] = _difference(frac_by_window, mid_start, mid_end, clipped=True)
    out[f"cap_curve_late_growth_w{late_start}_to_w{late_end}"] = _difference(frac_by_window, late_start, late_end, clipped=False)
    out[f"cap_curve_late_growth_w{late_start}_to_w{late_end}_clip01"] = _difference(frac_by_window, late_start, late_end, clipped=True)
    out[f"cap_curve_total_growth_w{early_start}_to_w{late_ref}"] = _difference(frac_by_window, early_start, late_ref, clipped=False)
    out[f"cap_curve_total_growth_w{early_start}_to_w{late_ref}_clip01"] = _difference(frac_by_window, early_start, late_ref, clipped=True)
    out[f"cap_curve_mid_growth_rel_w{mid_start}_to_w{mid_end}"] = _relative_growth(frac_by_window, mid_start, mid_end, eps, clipped=False)
    out[f"cap_curve_mid_growth_rel_w{mid_start}_to_w{mid_end}_clip01"] = _relative_growth(frac_by_window, mid_start, mid_end, eps, clipped=True)
    out[f"cap_curve_late_growth_rel_w{late_start}_to_w{late_end}"] = _relative_growth(frac_by_window, late_start, late_end, eps, clipped=False)
    out[f"cap_curve_late_growth_rel_w{late_start}_to_w{late_end}_clip01"] = _relative_growth(frac_by_window, late_start, late_end, eps, clipped=True)

    curve_front_raw_num = _mean_from_windows(frac_by_window, early_windows, clipped=False)
    curve_front_raw_den = _mean_from_windows(frac_by_window, mid_windows, clipped=False)
    curve_front_clip_num = _mean_from_windows(frac_by_window, early_windows, clipped=True)
    curve_front_clip_den = _mean_from_windows(frac_by_window, mid_windows, clipped=True)
    if np.isfinite(curve_front_raw_num) and np.isfinite(curve_front_raw_den):
        out[f"cap_curve_frontload_{_window_label(curve_windows)}"] = float(curve_front_raw_num / (curve_front_raw_den + eps))
    if np.isfinite(curve_front_clip_num) and np.isfinite(curve_front_clip_den):
        out[f"cap_curve_frontload_{_window_label(curve_windows)}_clip01"] = float(curve_front_clip_num / (curve_front_clip_den + eps))

    front31_raw_num = _mean_from_windows(frac_by_window, [3, 5, 7, 9], clipped=False)
    front31_raw_den = _mean_from_windows(frac_by_window, [13, 17, 21, late_ref], clipped=False)
    front31_clip_num = _mean_from_windows(frac_by_window, [3, 5, 7, 9], clipped=True)
    front31_clip_den = _mean_from_windows(frac_by_window, [13, 17, 21, late_ref], clipped=True)
    if np.isfinite(front31_raw_num) and np.isfinite(front31_raw_den):
        out[f"cap_curve_frontload_w{early_start}_to_w{late_ref}"] = float(front31_raw_num / (front31_raw_den + eps))
    if np.isfinite(front31_clip_num) and np.isfinite(front31_clip_den):
        out[f"cap_curve_frontload_w{early_start}_to_w{late_ref}_clip01"] = float(front31_clip_num / (front31_clip_den + eps))

    out["cap_curve_step_like_v1"] = _step_like(
        out[f"cap_curve_mean_{curve_label}_clip01"],
        out[f"cap_curve_mid_growth_rel_w{mid_start}_to_w{mid_end}_clip01"],
    )
    out["cap_curve_step_like_v2"] = _step_like(
        out[f"cap_curve_mean_{curve_label}_clip01"],
        out[f"cap_curve_late_growth_rel_w{late_start}_to_w{late_end}_clip01"],
    )
    return out


def compute_cap_feature_rows(cache: dict[str, Any], rows: list[dict[str, Any]], options: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = cap_defaults(options)
    return [compute_cap_feature_row(cache, dict(row), cfg) for row in rows]


def summarize_cap_rows(rows: list[dict[str, Any]], stats: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    cfg = cap_defaults(options)
    feature_names = sorted({key for row in rows for key in row.keys() if key.startswith("cap_")})
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("cap_v2_status", ""))
        status_counts[status] = int(status_counts.get(status, 0) + 1)
    return {
        "total_loaded_candidates": int(stats.get("total_loaded_candidates", 0)),
        "candidates_skipped_by_noise_filter": int(stats.get("total_loaded_candidates", 0)) - int(stats.get("candidates_used", 0)),
        "candidates_used": int(stats.get("candidates_used", 0)),
        "valid_cap_rows": int(status_counts.get("ok", 0)),
        "missing_edge_foot_count": int(status_counts.get("missing_edge_foot", 0)),
        "invalid_prominence_count": int(status_counts.get("invalid_prominence", 0)),
        "candidates_missing_noise_status": int(stats.get("candidates_missing_noise_status", 0)),
        "candidate_scope": str(stats.get("candidate_scope", "")),
        "median_windows": [int(v) for v in _clean_windows([int(v) for v in cfg["median_windows"]], DEFAULT_MEDIAN_WINDOWS)],
        "curve_windows": [int(v) for v in _clean_windows([int(v) for v in cfg["curve_windows"]], DEFAULT_CURVE_WINDOWS)],
        "feature_names": feature_names,
    }


def write_cap_csv(path: Path | str, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CAP_JOIN_COLUMNS)
    fieldnames.extend(sorted({key for row in rows for key in row.keys() if key not in fieldnames}))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
