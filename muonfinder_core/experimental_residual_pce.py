from __future__ import annotations

from typing import Any

import numpy as np

from .experimental_common import local_second_derivative_min, median_filter_1d, pce_evidence_from_signal, signal_segment


def compute_experimental_residual_features(signal: np.ndarray, row: dict[str, Any], noise_value: float, config: dict[str, Any]) -> dict[str, Any]:
    residual_cfg = dict(config.get("residual_pce", {}))
    threshold_cfg = dict(config.get("residual_threshold", {}))
    median_window = int(residual_cfg.get("median_window", threshold_cfg.get("median_window", 3)))
    radius = int(max(1, residual_cfg.get("near_apex_radius_pts", 2)))
    threshold_factor = float(threshold_cfg.get("threshold_noise_factor", config.get("noise_threshold_factor", 3.0)))
    out: dict[str, Any] = {
        "exp_resid3_status": "ok",
        "exp_resid3_pce": np.nan,
        "exp_resid3_curv_min": np.nan,
        "exp_resid3_curv_neg_z": np.nan,
        "exp_resid3_apex_value": np.nan,
        "exp_resid3_abs_apex_noise_z": np.nan,
        "exp_resid3_height_noise_z": np.nan,
        "exp_resid3_above_3noise": np.nan,
        "exp_resid3_local_max_noise_z": np.nan,
    }
    raw = np.asarray(signal, dtype=float)
    peak_index = int(row.get("peak_index", -1))
    if raw.ndim != 1 or raw.size < 3 or not (0 <= peak_index < raw.size):
        out["exp_resid3_status"] = "missing_raw_signal"
        return out
    residual = raw - median_filter_1d(raw, median_window)
    segment, peak_rel = signal_segment(residual, row)
    if segment.size < 3 or peak_rel < 0:
        out["exp_resid3_status"] = "missing_raw_signal"
        return out
    apex_value = float(residual[peak_index])
    curv_min = local_second_derivative_min(residual, peak_index, radius)
    height_z = float(apex_value / noise_value) if np.isfinite(apex_value) and np.isfinite(noise_value) and noise_value > 0.0 else np.nan
    local_lo = max(0, peak_index - radius)
    local_hi = min(raw.size - 1, peak_index + radius)
    local_max = float(np.max(residual[local_lo : local_hi + 1]))
    local_max_z = float(local_max / noise_value) if np.isfinite(local_max) and np.isfinite(noise_value) and noise_value > 0.0 else np.nan
    curv_neg_z = float(max(-curv_min, 0.0) / noise_value) if np.isfinite(curv_min) and np.isfinite(noise_value) and noise_value > 0.0 else np.nan
    out["exp_resid3_curv_min"] = float(curv_min) if np.isfinite(curv_min) else np.nan
    out["exp_resid3_curv_neg_z"] = float(curv_neg_z) if np.isfinite(curv_neg_z) else np.nan
    out["exp_resid3_apex_value"] = float(apex_value) if np.isfinite(apex_value) else np.nan
    out["exp_resid3_abs_apex_noise_z"] = float(height_z) if np.isfinite(height_z) else np.nan
    out["exp_resid3_height_noise_z"] = float(height_z) if np.isfinite(height_z) else np.nan
    out["exp_resid3_local_max_noise_z"] = float(local_max_z) if np.isfinite(local_max_z) else np.nan
    out["exp_resid3_above_3noise"] = float(1.0 if np.isfinite(apex_value) and np.isfinite(noise_value) and apex_value > threshold_factor * noise_value else 0.0)
    out["exp_resid3_pce"] = float(pce_evidence_from_signal(segment, peak_rel, noise_value))
    return out
