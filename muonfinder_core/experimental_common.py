from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .legacy_formula_adapter import compute_peak_curvature_features, estimate_background_mad


EXPERIMENTAL_JOIN_COLUMNS = (
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

PCE_SUPPORT_FULL = -1992.0
PCE_SUPPORT_ZERO = -273.0
PCE_VETO_ZERO = -273.0
PCE_VETO_FULL = 503.0

DEFAULT_EXPERIMENTAL_FEATURES: dict[str, Any] = {
    "enabled": False,
    "features_path": "outputs_core/experimental_features.csv",
    "summary_path": "outputs_core/experimental_features_summary.json",
    "candidate_scope": "noise-kept",
    "noise_source": "morph_range",
    "noise_threshold_factor": 3.0,
    "raw_pce": {
        "enabled": False,
        "smoothing_modes": ["none", "savgol5", "savgol7"],
        "near_apex_radius_pts": 2,
    },
    "residual_pce": {
        "enabled": False,
        "median_window": 3,
        "near_apex_radius_pts": 2,
    },
    "residual_threshold": {
        "enabled": True,
        "median_window": 3,
        "threshold_noise_factor": 3.0,
    },
    "edge_variants": {
        "enabled": True,
        "noise_level_step_factor": 1.0,
        "noise_level_start_factor": 0.0,
        "max_noise_levels": 50,
    },
    "viewer_columns": [
        "exp_resid3_height_noise_z",
        "exp_edge_legacy_evidence_signed_modernnorm",
    ],
    "viewer_label_aliases": {
        "exp_edge_legacy_evidence_signed_modernnorm": "eel",
        "exp_resid3_height_noise_z": "erhnz",
    },
}


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def experimental_defaults(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return _deep_merge(DEFAULT_EXPERIMENTAL_FEATURES, dict(config or {}))


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _ramp_up(x: float, low: float, high: float) -> float:
    if not np.isfinite(x):
        return 0.0
    if high <= low:
        return 1.0 if x >= high else 0.0
    return float(np.clip((float(x) - float(low)) / (float(high) - float(low)), 0.0, 1.0))


def _ramp_down(x: float, low: float, high: float) -> float:
    return float(1.0 - _ramp_up(x, low, high))


def pce_evidence_from_signal(segment: np.ndarray, peak_rel: int, noise_value: float) -> float:
    if segment.size < 3 or not np.isfinite(noise_value) or noise_value <= 0.0:
        return float("nan")
    features = compute_peak_curvature_features(np.asarray(segment, dtype=float), float(noise_value), peak_rel=int(peak_rel))
    pce_raw = _safe_float(features.get("peak_curvature_extreme_negpref_t098"))
    if not np.isfinite(pce_raw):
        return float("nan")
    support = _ramp_down(pce_raw, PCE_SUPPORT_FULL, PCE_SUPPORT_ZERO)
    veto = _ramp_up(pce_raw, PCE_VETO_ZERO, PCE_VETO_FULL)
    return float(np.clip(support - veto, -1.0, 1.0))


def median_filter_1d(signal: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if w <= 1 or x.size <= 1:
        return np.asarray(x, dtype=float)
    pad = w // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.median(sliding_window_view(padded, w), axis=-1)


def savgol_like_filter(signal: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    key = str(mode).strip().lower()
    if key in {"", "none"}:
        return np.asarray(x, dtype=float)
    kernels = {
        "savgol5": np.asarray([-3.0, 12.0, 17.0, 12.0, -3.0], dtype=float) / 35.0,
        "savgol7": np.asarray([-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0], dtype=float) / 21.0,
    }
    kernel = kernels.get(key)
    if kernel is None:
        return np.asarray(x, dtype=float)
    pad = kernel.size // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def local_second_derivative_min(signal: np.ndarray, peak_index: int, radius: int) -> float:
    x = np.asarray(signal, dtype=float)
    peak = int(peak_index)
    if x.size < 3 or not (0 <= peak < x.size):
        return float("nan")
    left = max(1, peak - int(max(1, radius)))
    right = min(int(x.size) - 2, peak + int(max(1, radius)))
    if right < left:
        return float("nan")
    d2 = x[:-2] - 2.0 * x[1:-1] + x[2:]
    return float(np.min(d2[left - 1 : right]))


def build_join_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_y": int(row.get("source_y", row.get("y", -1))),
        "source_x": int(row.get("source_x", row.get("x", -1))),
        "compact_y": int(row.get("y", -1)),
        "compact_x": int(row.get("x", -1)),
        "peak_index": int(row.get("peak_index", -1)),
        "candidate_id": row.get("candidate_id", ""),
        "start_index": row.get("start_index", row.get("start", "")),
        "end_index": row.get("end_index", row.get("end", "")),
    }


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


def filter_candidate_rows(rows: list[dict[str, Any]], candidate_scope: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        elif is_rejected is not True:
            used_rows.append(dict(row))
    return used_rows, {
        "candidate_scope": scope,
        "total_loaded_candidates": total,
        "candidates_rejected_by_noise_filter": int(rejected),
        "candidates_missing_noise_status": int(missing_status),
        "candidates_used": int(len(used_rows)),
    }


def select_experimental_noise(signal: np.ndarray, row: dict[str, Any], requested_source: str) -> dict[str, Any]:
    request = str(requested_source).strip().lower()
    morph_noise = _safe_float(row.get("candidate_noise_estimate_used", row.get("noise_height_morph_range")))
    start = int(row.get("start_index", row.get("start", 0)))
    end = int(row.get("end_index", row.get("end", max(0, len(signal) - 1))))
    bg_mad = _safe_float(row.get("bg_mad"))
    if not np.isfinite(bg_mad):
        bg_mad = float(max(estimate_background_mad(np.asarray(signal, dtype=float), int(start), int(end)), 1e-12))
    fallback = False
    if request == "bg_mad":
        source = "bg_mad"
        value = bg_mad
    elif np.isfinite(morph_noise) and morph_noise > 0.0:
        source = "morph_range"
        value = morph_noise
    else:
        source = "bg_mad"
        value = bg_mad
        fallback = True
    return {
        "exp_noise_source": str(source),
        "exp_noise_value": float(value) if np.isfinite(value) and value > 0.0 else np.nan,
        "exp_noise_fallback_used": bool(fallback),
    }


def signal_segment(signal: np.ndarray, row: dict[str, Any]) -> tuple[np.ndarray, int]:
    start = int(row.get("start_index", row.get("start", 0)))
    end = int(row.get("end_index", row.get("end", 0)))
    peak = int(row.get("peak_index", -1))
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size == 0 or not (0 <= peak < x.size):
        return np.asarray([], dtype=float), -1
    lo = int(max(0, min(start, end)))
    hi = int(min(x.size - 1, max(start, end)))
    if hi < lo or not (lo <= peak <= hi):
        return np.asarray([], dtype=float), -1
    return np.asarray(x[lo : hi + 1], dtype=float), int(peak - lo)


def write_feature_csv(path: Path | str, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EXPERIMENTAL_JOIN_COLUMNS)
    fieldnames.extend(sorted({key for row in rows for key in row.keys() if key not in fieldnames}))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
