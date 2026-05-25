from __future__ import annotations

from typing import Any

import numpy as np

from .experimental_common import (
    local_second_derivative_min,
    pce_evidence_from_signal,
    savgol_like_filter,
    signal_segment,
)


def compute_experimental_raw_pce(signal: np.ndarray, row: dict[str, Any], noise_value: float, config: dict[str, Any]) -> dict[str, Any]:
    modes = [str(mode) for mode in config.get("smoothing_modes", ["none", "savgol5", "savgol7"])]
    radius = int(max(1, config.get("near_apex_radius_pts", 2)))
    out: dict[str, Any] = {"exp_raw_pce_status": "ok"}
    segment, peak_rel = signal_segment(np.asarray(signal, dtype=float), row)
    peak_index = int(row.get("peak_index", -1))
    if segment.size < 3 or peak_rel < 0 or not (0 <= peak_index < len(signal)):
        out["exp_raw_pce_status"] = "missing_raw_signal"
        for mode in modes:
            suffix = str(mode).strip().lower()
            out[f"exp_raw_pce_{suffix}"] = np.nan
            out[f"exp_raw_curv_min_{suffix}"] = np.nan
            out[f"exp_raw_curv_neg_z_{suffix}"] = np.nan
        return out
    raw = np.asarray(signal, dtype=float)
    for mode in modes:
        suffix = str(mode).strip().lower()
        smooth_full = savgol_like_filter(raw, suffix)
        smooth_segment, smooth_peak_rel = signal_segment(smooth_full, row)
        curv_min = local_second_derivative_min(smooth_full, peak_index, radius)
        neg_z = float(max(-curv_min, 0.0) / noise_value) if np.isfinite(curv_min) and np.isfinite(noise_value) and noise_value > 0.0 else np.nan
        out[f"exp_raw_curv_min_{suffix}"] = float(curv_min) if np.isfinite(curv_min) else np.nan
        out[f"exp_raw_curv_neg_z_{suffix}"] = float(neg_z) if np.isfinite(neg_z) else np.nan
        out[f"exp_raw_pce_{suffix}"] = float(
            pce_evidence_from_signal(smooth_segment, smooth_peak_rel, noise_value)
        ) if smooth_segment.size >= 3 else np.nan
    return out
