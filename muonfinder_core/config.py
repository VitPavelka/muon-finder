from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json


@dataclass
class CoreConfig:
    data: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, Any] = field(default_factory=dict)
    morphology: dict[str, Any] = field(default_factory=dict)
    candidates: dict[str, Any] = field(default_factory=dict)
    background: dict[str, Any] = field(default_factory=dict)
    noise: dict[str, Any] = field(default_factory=dict)
    cap: dict[str, Any] = field(default_factory=dict)
    experimental_features: dict[str, Any] = field(default_factory=dict)
    ss6: dict[str, Any] = field(default_factory=dict)
    despike: dict[str, Any] = field(default_factory=dict)
    ss4: dict[str, Any] = field(default_factory=dict)
    ss5: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    viewer: dict[str, Any] = field(default_factory=dict)
    decision_profile: str = "ss4"


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "input_path": "data/petalit_785nm_20xObj_800ctr_1s_100prc.wdf",
        "coords_csv": "100coords.csv",
        "labels_csv": "labels.csv",
        "output_dir": "outputs_core",
        "viewer_cache_path": "outputs_core/viewer_cache.npz",
        "light_debug_path": "outputs_core/debug_light.json",
    },
    "data": {
        "input_format": "auto",
        "use_compact_coords_view": True,
    },
    "morphology": {
        "tophat_window": 3,
        "noise_window": 3,
        "morphology_windows": [3, 5, 7],
        "baseline_window": 5,
        "feature_erosion_window": 20,
    },
    "candidates": {
        "score_mode": "max",
        "threshold_method": "quantile",
        "threshold_quantile": 0.1,
        "threshold_k_mad": 20.0,
        "threshold_min_abs": None,
        "k_mad_pixel": 8.0,
        "min_peak": 80.0,
        "max_width_pts": 24,
        "pad_pts": 0,
        "edge_k_mad": 2.0,
        "merge_duplicate_segments": True,
        "feature_expand_to_gradient_foot": True,
        "feature_foot_k_mad": 2.0,
        "feature_foot_min_run": 2,
        "feature_window_method": "mad_run",
    },
    "background": {
        "method": "bg_mad",
    },
    "noise": {
        "noise_source": "morph_range",
        "noise_height_factor": 3.0,
        "candidate_noise_prefilter_enabled": True,
        "candidate_noise_prefilter_mode": "morph_range_chord",
        "edge_foot_method": "noise_quantized_component",
        "edge_pre_level_step_noise": 1.0,
        "edge_pre_transient_tolerance_levels": 2,
        "edge_pre_min_stable_levels": 2,
        "edge_neighbor_structure_factor": 3.0,
        "edge_dense_context_min_pad_pts": 10,
        "edge_dense_context_pad_pts": 20,
        "edge_dense_context_max_pad_pts": 120,
        "edge_context_expand_step_pts": 10,
    },
    "cap": {
        "features_path": "outputs_core/cap_features.csv",
        "summary_path": "outputs_core/cap_features_summary.json",
        "candidate_scope": "noise-kept",
        "signal_source": "raw",
        "median_windows": [3, 5, 7, 9, 13, 17, 21, 31],
        "curve_windows": [3, 5, 7, 9, 13, 17],
        "early_windows": [3, 5, 7],
        "mid_windows": [9, 13, 17],
        "late_windows": [17, 21, 31],
        "late_reference_window": 31,
        "min_prominence": 1e-12,
        "eps": 1e-12,
    },
    "experimental_features": {
        "enabled": False,
        "features_path": "outputs_core/experimental_features.csv",
        "summary_path": "outputs_core/experimental_features_summary.json",
        "auto_recompute": True,
        "candidate_scope": "noise-kept",
        "noise_source": "morph_range",
        "noise_threshold_factor": 3.0,
        "raw_pce": {
            "enabled": True,
            "smoothing_modes": ["none", "savgol5", "savgol7"],
            "near_apex_radius_pts": 2,
        },
        "residual_pce": {
            "enabled": True,
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
            "exp_raw_pce_none",
            "exp_raw_pce_savgol5",
            "exp_resid3_pce",
            "exp_resid3_height_noise_z",
            "d3rawM",
            "d3rawS",
            "d3gradM",
            "d3gradS",
            "exp_resid3_above_3noise",
            "exp_edge_percent_0_90",
            "exp_edge_noise_from_0",
            "exp_edge_noise_from_1",
            "exp_edge_legacy_like",
        ],
    },
    "ss6": {
        "enabled": False,
        "decisions_path": "outputs_core/ss6_decisions.csv",
        "summary_path": "outputs_core/ss6_decisions_summary.json",
        "auto_load_in_viewer": True,
        "ss1_gate": 0.95,
        "pce_strong_min": 0.75,
        "pce_dead_max": -0.999,
        "pce_gray_min": 0.10,
        "pce_gray_max": 0.75,
        "edge_spike_max": -0.50,
        "eel_spike_max": -0.50,
        "edge_eel_delta_min": 0.50,
        "pce_dead_edge_spike_max": -0.45,
        "pce_gray_delta_eel_max": -0.40,
        "pce_gray_delta_min": 0.50,
        "pce_gray_delta_resid_min": 8.5,
        "pce_gray_high_pce_min": 0.60,
        "pce_gray_high_pce_eel_max": -0.60,
        "pce_gray_high_pce_resid_min": 6.0,
        "pce_gray_high_resid_min": 12.0,
        "pce_gray_soft_edge_max": -0.45,
        "pce_gray_soft_eel_max": -0.45,
        "low_pce_double_edge_edge_max": -0.60,
        "low_pce_double_edge_eel_max": -0.60,
        "low_pce_double_edge_resid_min": 5.5,
        "resid_rescue_min": 8.5,
        "resid_strong_min": 20.0,
        "require_noise_kept": True,
        "save_histograms": True,
        "histograms_dir": "outputs_core/ss6_histograms",
        "histogram_bins": 40,
        "metric_names": {
            "ss1": "spike_score_v1",
            "pce": "pce_negpref_t098_evidence_signed",
            "edge": "recdw_sum_0_90_raman_veto_evidence_signed",
            "eel": "exp_edge_legacy_evidence_signed_modernnorm",
            "resid": "exp_resid3_height_noise_z",
        },
        "viewer_columns": [],
        "viewer_label_aliases": {
            "ss6_accept": "s6",
            "ss6_branch": "s6b",
            "ss6_edge_eel_delta": "eed",
            "ss6_resid": "r3",
            "ss6_eel": "eel",
            "ss6_edge": "edge",
        },
        "write_pce_metric_audit": False,
    },
    "despike": {
        "enabled": False,
        "source": "ss6",
        "corrected_path": "outputs_core/despike_corrected.npz",
        "debug_path": "outputs_core/despike_debug.csv",
        "attempts_path": "outputs_core/despike_attempts.csv",
        "summary_path": "outputs_core/despike_summary.json",
        "morph_window": 3,
        "despike_context_window_pad": 0,
        "noise_height_factor": 3.0,
        "max_iterations": 1000,
    },
    "ss4": {
        "ss_blue_max": 0.95,
        "ss_red_min": 0.9999,
        "pce_red_min": 0.4,
        "edge_red_max": -0.3,
        "missing_policy": "review",
        "pce_dead_zone_enabled": False,
        "pce_dead_zone_low": -0.8,
        "pce_dead_zone_high": -0.2,
    },
    "ss5": {
        "ss1_threshold": 0.95,
        "pce_spike_min": 0.8,
        "edge_spike_max": -0.4,
    },
    "outputs": {
        "save_viewer_cache": True,
        "save_light_debug": True,
        "save_full_debug": False,
    },
    "viewer": {
        "open_after_pipeline": False,
        "show_candidate_status_summary_box": False,
        "map_color_percentiles": [5, 95],
        "show_map_colorbar": False,
    },
    "decision_profile": "ss4",
}


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _resolve_path_values(cfg: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    out = dict(cfg)
    paths = dict(out.get("paths", {}))
    cap = dict(out.get("cap", {}))
    experimental = dict(out.get("experimental_features", {}))
    ss6 = dict(out.get("ss6", {}))
    despike = dict(out.get("despike", {}))
    repo_root = base_dir.parent
    for key, value in list(paths.items()):
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            continue
        path = Path(value)
        if path.is_absolute():
            continue
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            paths[key] = str(candidate)
            continue
        repo_candidate = (repo_root / path).resolve()
        paths[key] = str(repo_candidate if repo_candidate.exists() else candidate)
    for key in ("features_path", "summary_path"):
        value = cap.get(key)
        if value in (None, "") or not isinstance(value, str):
            continue
        path = Path(value)
        if path.is_absolute():
            continue
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            cap[key] = str(candidate)
            continue
        repo_candidate = (repo_root / path).resolve()
        cap[key] = str(repo_candidate if repo_candidate.exists() else candidate)
    for key in ("features_path", "summary_path"):
        value = experimental.get(key)
        if value in (None, "") or not isinstance(value, str):
            continue
        path = Path(value)
        if path.is_absolute():
            continue
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            experimental[key] = str(candidate)
            continue
        repo_candidate = (repo_root / path).resolve()
        experimental[key] = str(repo_candidate if repo_candidate.exists() else candidate)
    for key in ("decisions_path", "summary_path"):
        value = ss6.get(key)
        if value in (None, "") or not isinstance(value, str):
            continue
        path = Path(value)
        if path.is_absolute():
            continue
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            ss6[key] = str(candidate)
            continue
        repo_candidate = (repo_root / path).resolve()
        ss6[key] = str(repo_candidate if repo_candidate.exists() else candidate)
    for key in ("corrected_path", "debug_path", "attempts_path", "summary_path"):
        value = despike.get(key)
        if value in (None, "") or not isinstance(value, str):
            continue
        path = Path(value)
        if path.is_absolute():
            continue
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            despike[key] = str(candidate)
            continue
        repo_candidate = (repo_root / path).resolve()
        despike[key] = str(repo_candidate if repo_candidate.exists() else candidate)
    out["paths"] = paths
    out["cap"] = cap
    out["experimental_features"] = experimental
    out["ss6"] = ss6
    out["despike"] = despike
    return out


def load_config(path: Path | str) -> CoreConfig:
    cfg_path = Path(path)
    user_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    merged = _resolve_path_values(_deep_merge(DEFAULT_CONFIG, user_cfg), cfg_path.resolve().parent)
    return CoreConfig(
        data=dict(merged.get("data", {})),
        paths=dict(merged.get("paths", {})),
        morphology=dict(merged.get("morphology", {})),
        candidates=dict(merged.get("candidates", {})),
        background=dict(merged.get("background", {})),
        noise=dict(merged.get("noise", {})),
        cap=dict(merged.get("cap", {})),
        experimental_features=dict(merged.get("experimental_features", {})),
        ss6=dict(merged.get("ss6", {})),
        despike=dict(merged.get("despike", {})),
        ss4=dict(merged.get("ss4", {})),
        ss5=dict(merged.get("ss5", {})),
        outputs=dict(merged.get("outputs", {})),
        viewer=dict(merged.get("viewer", {})),
        decision_profile=str(merged.get("decision_profile", "ss4")).strip().lower(),
    )
