# muon_finder.py
from __future__ import annotations

import argparse
import csv
import time
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

from candidate_labels import load_binary_labels
from wdf_io import load_dataset
from muon_pipeline import (
	compute_morph_overlays,
	score_map_from_top_hat,
	threshold_score_map,
	extract_spikes_for_candidates,
	neighbour_filter_ratio,
	SpikeSegment
)
from despike import apply_despike
from despike_contact_analysis import (
	analyze_erosion_dilation_contact_cells,
	apply_contact_cell_despike_chords,
	build_despike_contact_debug_payload,
	save_despike_contact_debug_json,
)
from viewer import show_hover_map
from results_io import save_result_npz, save_spikes_csv, save_viewer_cache
from debug_report import build_debug_report, save_debug_report_json
from preprocess import resample_axis_and_spectra
from feature_discrimination import compute_edge_width_metrics, compute_peak_curvature_features, compute_spike_score_v2_features
from muon_decision import (
	annotate_feature_dict_with_spike_score_v4,
	annotate_feature_dict_with_spike_score_v5,
	classify_segment_with_muon_rule_v3,
)
from primary_candidate_preparation import prepare_primary_ss4_segments
from candidate_diagnostics import (
	SmallMorphologyBundle,
	apply_global_metric_ranks,
	candidate_segment_id,
	candidate_segment_key,
	evaluate_candidate_noise_prefilter,
	get_or_compute_small_morphology,
)

DEFAULT_CONFIG: Dict[str, Any] = {
	"input_path": "",
	"se_size": 5,
	"score_mode": "max",
	"threshold_method": "quantile",
	"threshold_quantile": 0.1,
	"threshold_k_mad": 20.0,
	"threshold_min_abs": None,
	"k_mad_pixel": 8.0,
	"min_peak": 0.0,
	"baseline_se_size": 11,
	"edge_k_mad": 2.0,
	"max_width_pts": 20,
	"pad_pts": 0,
	"neighbor_filter_enabled": False,
	"neighbor_radius": 1,
	"neighbor_ratio_min": 3.0,
	"coords_csv": None,
	"use_compact_coords_view": True,
	"despike_enabled": True,
	"resample_enabled": False,
	"resample_factor": 2,
	"save_npz_path": None,
	"save_spikes_csv_path": None,
	"save_corrected_in_npz": True,
	"save_overlays_in_npz": True,
	"debug_report_path": None,
	"debug_report_enabled": False,
	"debug_feature_report_enabled": False,
	"debug_feature_report_path": "outputs/debug_feature_report.json",
	"labels_csv": None,
	"debug_include_per_spectrum": True,
	"debug_progress": True,
	"debug_progress_print_every": 1000,
	"debug_top_pixels": 25,
	"debug_feature_signal_source": "gradient",  # gradient | raw
	"merge_duplicate_segments": False,
	"feature_expand_to_gradient_foot": True,
	"feature_foot_k_mad": 2.0,
	"feature_foot_min_run": 2,
	"feature_window_method": "mad_run",  # mad_run | erosion_touch
	"feature_erosion_se_size": 5,
	"candidate_noise_prefilter_enabled": True,
	"candidate_noise_prefilter_mode": "morph_range_chord",
	"candidate_noise_prefilter_sensitivity": 1.0,
	"candidate_noise_height_factor": 3.0,
	"boundary_minimum_source": "gradient",
	"gws_split_overlapping_contexts": False,
	"gws_split_source": "gradient",
	"gws_split_smooth_pts": 3,
	"gws_split_valley_alpha": 0.75,
	"gws_split_min_distance_from_apex": 1,
	"gws_split_min_context_width": 3,
	"gws_split_debug": True,
	"gws_source": "morph_gradient",
	"gws_source_modes": [
		"morph_gradient",
		"morph_gradient_med3",
		"morph_gradient_med5",
		"morph_gradient_mean3",
		"morph_gradient_mean5",
	],
	"gws_include_scale_zero": False,
	"gws_measure_region": "mask",
	"gws_threshold_region": "spike_edges",
	"edge_dense_enabled": True,
	"edge_dense_levels": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
	"edge_dense_min_snr": 1.0,
	"edge_dense_context_pad_pts": 20,
	"edge_dense_context_min_pad_pts": 10,
	"edge_dense_context_max_pad_pts": 80,
	"edge_use_enhanced_spike_mapping": False,
	"edge_enhanced_in_debug_report": True,
	"edge_mapping_levels_desc": [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5],
	"edge_mapping_refine_step_percent": 1,
	"edge_mapping_min_level_percent": 1,
	"edge_mapping_require_closed_interval": True,
	"edge_mapping_use_apex_component": True,
	"edge_mapping_enable_merge_guard": True,
	"edge_mapping_max_width_jump_factor": 2.5,
	"edge_mapping_max_width_jump_points": 8,
	"edge_mapping_fallback_to_old": False,
	"edge_mapping_noise_guard_enabled": False,
	"edge_robust_reference_enabled": True,
	"recdw_evidence_enabled": True,
	"recdw_evidence_metrics": ["recdw_sum_0_90"],
	"recdw_z_clip": 6.0,
	"recdw_support_z_scale": 1.0,
	"rucdw_enabled": True,
	"rucdw_context_pad_pts": 20,
	"rucdw_context_max_pad_pts": 80,
	"rucdw_levels": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
	"rucdw_min_snr": 1.0,
	"rucdw_noise_fallback_rel_amp": 0.05,
	"rucdw_anchor_mode": "max_in_candidate",
	"rucdw_baseline_mode": "context_low_percentile",
	"rucdw_baseline_percentile": 5.0,
	"rucdw_z_clip": 6.0,
	"rucdw_support_z_scale": 1.0,
	"ss4_enabled": True,
	"ss4_fast_selection": True,
	"ss4_fast_recdw_population": "needed",  # needed | all
	"ss4_histogram_enabled": False,
	"ss4_histogram_dir": None,
	"ss4_histogram_prefix": "ss4_primary",
	"ss4_ss_blue_max": 0.95,
	"ss4_ss_red_min": 0.9999,
	"ss4_pce_red_min": 0.4,
	"ss4_rve_feature": "recdw_sum_0_90_raman_veto_evidence_signed",
	"ss4_pce_dead_zone_enabled": True,
	"ss4_pce_dead_zone_low": -0.8,
	"ss4_pce_dead_zone_high": -0.2,
	"ss4_rve_red_max": -0.1,
	"ss4_missing_policy": "review",
	"ss5_ss1_threshold": 0.95,
	"ss5_pce_spike_min": 0.8,
	"ss5_edge_spike_max": -0.4,
	"spike_score_v4_enabled": True,
	"spike_score_v4_ss_blue_max": 0.95,
	"spike_score_v4_ss_red_min": 0.9999,
	"spike_score_v4_pce_red_min": 0.4,
	"spike_score_v4_edge_feature": "recdw_sum_0_90_raman_veto_evidence_signed",
	"spike_score_v4_edge_red_min": -0.1,
	"spike_score_v4_missing_policy": "review",
	"ball_context_pad_pts": 20,
	"ball_context_min_pad_pts": 10,
	"ball_context_max_pad_pts": 80,
	"ball_stop_k_mad": 1.0,
	"ball_stop_rel_amp": 0.05,
	"ball_prevent_crossing_neighbor_peak": True,
	"exp_context_pad_pts": 20,
	"exp_foot_low_rel": 0.05,
	"exp_foot_high_rel": 0.45,
	"exp_foot_noise_k_mad": 1.0,
	"exp_min_points": 3,
	"exp_prevent_apex_region": True,
	"decision_profile": "muon_rule_v3",
	"despike_decision_action": "auto_only",  # auto_only | auto_and_maybe | none
	"despike_decision_feature": "ss4",
	"despike_contact_candidates_enabled": True,
	"despike_contact_analysis_enabled": True,
	"despike_contact_context_pad_pts": 4,
	"despike_contact_use_existing_morphology": True,
	"despike_contact_strict_equal": True,
	"despike_sensitivity": 1.0,
	"secondary_noise_thr": 0.05,
	"secondary_uncertain_thr": 0.5,
	"secondary_edge_rescue_ss_min": 0.85,
	"despike_debug_json_path": "despike_debug.json",
	"despike_debug_lite_enabled": True,
	"despike_debug_lite_path": "outputs/despike_debug_lite.json",
	"despike_debug_full_enabled": False,
	"despike_debug_full_path": "outputs/despike_debug_full.json",
	"debug_store_arrays": False,
	"save_viewer_cache": True,
	"viewer_cache_path": "outputs/viewer_cache.npz",
	"threshold_calibration_enabled": True,
	"threshold_calibration_dir": "outputs/threshold_calibration",
	"threshold_calibration_make_plots": True,
	"threshold_calibration_suggest_thresholds": True,
}


def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
	cfg = dict(DEFAULT_CONFIG)
	if config_path is None:
		return cfg
	path = Path(config_path)
	user_cfg = json.loads(path.read_text(encoding="utf-8"))
	cfg.update(user_cfg)
	if "debug_feature_report_enabled" not in user_cfg and "debug_report_enabled" in user_cfg:
		cfg["debug_feature_report_enabled"] = bool(user_cfg["debug_report_enabled"])
	if "debug_feature_report_path" not in user_cfg and "debug_report_path" in user_cfg:
		cfg["debug_feature_report_path"] = user_cfg["debug_report_path"]
	return cfg


def load_target_coords_csv(path: Path, shape_hw: Tuple[int, int]) -> List[Tuple[int, int]]:
	h, w = shape_hw
	raw = np.genfromtxt(path, delimiter=";", names=True, dtype=None, encoding="utf-8")
	if raw.size == 0:
		return []

	cols = {c.lower(): c for c in raw.dtype.names or []}
	y_col = cols.get("y")
	x_col = cols.get("x")
	if y_col is None or x_col is None:
		raise ValueError("coords CSV must contain headers 'y' and 'x'.")

	coords: List[Tuple[int, int]] = []
	seen = set()
	for yv, xv in zip(np.atleast_1d(raw[y_col]), np.atleast_1d(raw[x_col])):
		y = int(yv)
		x = int(xv)
		if y < 0 or y >= h or x < 0 or x >= w:
			continue
		key = (y, x)
		if key in seen:
			continue
		seen.add(key)
		coords.append(key)
	return coords


def load_labels_csv(path: Path) -> Dict[Tuple[int, int, int], bool]:
	return {key: bool(value) for key, value in load_binary_labels(Path(path)).items()}


def build_compact_subset(
		x_axis: np.ndarray,
		raw_spectra: np.ndarray,
		score_map: np.ndarray,
		candidate_mask: np.ndarray,
		overlays: Dict[str, np.ndarray],
		spikes_by_pixel: Dict[Tuple[int, int], List[SpikeSegment]],
		coords: List[Tuple[int, int]],
		corrected_spectra: Optional[np.ndarray] = None,
) -> Tuple[
	np.ndarray, np.ndarray, np.ndarray,
	Dict[str, np.ndarray], Dict[Tuple[int, int], List[SpikeSegment]],
	Dict[Tuple[int, int], Tuple[int, int]], Optional[np.ndarray]
]:
	n = len(coords)
	if n == 0:
		raise ValueError("No coordinates for compact subset.")

	grid_w = int(math.ceil(math.sqrt(n)))
	grid_h = int(math.ceil(n / grid_w))
	spec_n = x_axis.size

	raw_compact = np.zeros((grid_h, grid_w, spec_n), dtype=raw_spectra.dtype)
	corr_compact = None
	if corrected_spectra is not None:
		corr_compact = np.zeros((grid_h, grid_w, spec_n), dtype=corrected_spectra.dtype)
	score_compact = np.zeros((grid_h, grid_w), dtype=score_map.dtype)
	cand_compact = np.zeros((grid_h, grid_w), dtype=bool)
	overlay_compact: Dict[str, np.ndarray] = {
		k: np.zeros((grid_h, grid_w, spec_n), dtype=v.dtype) for k, v in overlays.items()
	}
	spikes_compact: Dict[Tuple[int, int], List[SpikeSegment]] = {}
	coord_map: Dict[Tuple[int, int], Tuple[int, int]] = {}

	for i, (src_y, src_x) in enumerate(coords):
		yy = i // grid_w
		xx = i % grid_w
		coord_map[(yy, xx)] = (src_y, src_x)

		raw_compact[yy, xx, :] = raw_spectra[src_y, src_x, :]
		if corr_compact is not None:
			corr_compact[yy, xx, :] = corrected_spectra[src_y, src_x, :]
		score_compact[yy, xx] = score_map[src_y, src_x]
		cand_compact[yy, xx] = bool(candidate_mask[src_y, src_x])
		for k in overlay_compact:
			overlay_compact[k][yy, xx, :] = overlays[k][src_y, src_x, :]

		src_spikes = spikes_by_pixel.get((src_y, src_x), [])
		if src_spikes:
			spikes_compact[(yy, xx)] = [
				SpikeSegment(
					y=yy,
					x=xx,
					peak_index=s.peak_index,
					start=s.start,
					end=s.end,
					peak_height=s.peak_height,
					area=s.area,
				)
				for s in src_spikes
			]

	return raw_compact, score_compact, cand_compact, overlay_compact, spikes_compact, coord_map, corr_compact


def _ensure_parent_dir(path_like: Optional[Any]) -> Optional[Path]:
	if path_like in (None, ""):
		return None
	path = Path(str(path_like))
	path.parent.mkdir(parents=True, exist_ok=True)
	return path


def _invert_coord_map(coord_map: Optional[Dict[Tuple[int, int], Tuple[int, int]]]) -> Dict[Tuple[int, int], Tuple[int, int]]:
	out: Dict[Tuple[int, int], Tuple[int, int]] = {}
	for compact, source in (coord_map or {}).items():
		out[(int(source[0]), int(source[1]))] = (int(compact[0]), int(compact[1]))
	return out


def _flatten_ss4_candidate_rows(metrics_by_pixel: Dict[Tuple[int, int], List[Dict[str, object]]]) -> List[Dict[str, object]]:
	rows: List[Dict[str, object]] = []
	seen: set[Tuple[int, int, int, int, int]] = set()
	for (y, x), items in metrics_by_pixel.items():
		for item in items:
			row = dict(item)
			row["y"] = int(row.get("y", y))
			row["x"] = int(row.get("x", x))
			row["peak_index"] = int(row.get("peak_index", -1))
			row["start"] = int(row.get("start", -1))
			row["end"] = int(row.get("end", -1))
			row["candidate_id"] = str(row.get("candidate_id", f"{row['y']}:{row['x']}:{row['peak_index']}:{row['start']}:{row['end']}"))
			key = (row["y"], row["x"], row["peak_index"], row["start"], row["end"])
			if key in seen:
				continue
			seen.add(key)
			rows.append(row)
	return rows


def _metric_float(row: Dict[str, object], key: str) -> float:
	try:
		return float(row.get(key, np.nan))
	except Exception:
		return float("nan")


def _compute_ss4_decision_for_row(
		row: Dict[str, object],
		*,
		ss_blue_max: float,
		ss_red_min: float,
		pce_red_min: float,
		rve_red_max: float,
		missing_policy: str,
) -> Dict[str, Any]:
	features = {
		"spike_score_v1": _metric_float(row, "spike_score_v1"),
		"pce_negpref_t098_evidence_signed": _metric_float(row, "pce_negpref_t098_evidence_signed"),
		"recdw_sum_0_90_raman_veto_evidence_signed": _metric_float(row, "recdw_sum_0_90_raman_veto_evidence_signed"),
	}
	return annotate_feature_dict_with_spike_score_v4(
		features,
		edge_feature="recdw_sum_0_90_raman_veto_evidence_signed",
		ss_blue_max=float(ss_blue_max),
		ss_red_min=float(ss_red_min),
		pce_red_min=float(pce_red_min),
		edge_red_min=float(rve_red_max),
		missing_policy=str(missing_policy),
	)


def _compute_ss5_decision_for_row(
		row: Dict[str, object],
		*,
		ss1_threshold: float,
		pce_spike_min: float,
		edge_spike_max: float,
) -> Dict[str, Any]:
	features = {
		"spike_score_v1": _metric_float(row, "spike_score_v1"),
		"pce_negpref_t098_evidence_signed": _metric_float(row, "pce_negpref_t098_evidence_signed"),
		"recdw_sum_0_90_raman_veto_evidence_signed": _metric_float(row, "recdw_sum_0_90_raman_veto_evidence_signed"),
	}
	return annotate_feature_dict_with_spike_score_v5(
		features,
		edge_feature="recdw_sum_0_90_raman_veto_evidence_signed",
		ss1_threshold=float(ss1_threshold),
		pce_spike_min=float(pce_spike_min),
		edge_spike_max=float(edge_spike_max),
	)


def _summarize_ss4_ss5_overlap(rows_iter: Any) -> Dict[str, int]:
	rows = [row for row in rows_iter]
	ss4_spikes = 0
	ss5_spikes = 0
	both = 0
	ss4_only = 0
	ss5_only = 0
	neither = 0
	for row in rows:
		try:
			ss4_is = str(row.get("primary_ss4_decision", row.get("ss4_decision", ""))).strip().lower() == "spike"
		except Exception:
			ss4_is = False
		try:
			ss5_is = str(row.get("primary_ss5_decision", row.get("ss5_decision", ""))).strip().lower() == "spike"
		except Exception:
			ss5_is = False
		ss4_spikes += int(ss4_is)
		ss5_spikes += int(ss5_is)
		if ss4_is and ss5_is:
			both += 1
		elif ss4_is:
			ss4_only += 1
		elif ss5_is:
			ss5_only += 1
		else:
			neither += 1
	return {
		"ss4_spikes": int(ss4_spikes),
		"ss5_spikes": int(ss5_spikes),
		"accepted_by_both": int(both),
		"accepted_by_ss4_only": int(ss4_only),
		"accepted_by_ss5_only": int(ss5_only),
		"rejected_by_both": int(neither),
	}


def _suggest_threshold_from_histogram(values: np.ndarray, current: float) -> float:
	x = np.asarray(values, dtype=float)
	x = x[np.isfinite(x)]
	if x.size < 32 or not np.isfinite(current):
		return float(current)
	vmin = float(np.min(x))
	vmax = float(np.max(x))
	if vmax <= vmin + 1e-12:
		return float(current)
	bins = min(180, max(48, int(np.sqrt(x.size) * 3)))
	counts, edges = np.histogram(x, bins=bins, range=(vmin, vmax))
	if counts.size < 5:
		return float(current)
	kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
	smooth = np.convolve(counts.astype(float), kernel / np.sum(kernel), mode="same")
	centers = 0.5 * (edges[:-1] + edges[1:])
	local_min = []
	for i in range(1, smooth.size - 1):
		if smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1]:
			local_min.append(i)
	if not local_min:
		return float(current)
	best_i = min(local_min, key=lambda i: abs(float(centers[i]) - float(current)))
	return float(centers[best_i])


def _write_simple_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow({k: row.get(k, "") for k in fieldnames})


def _run_threshold_calibration(
		*,
		rows: List[Dict[str, object]],
		cfg: Dict[str, Any],
		base_dir: Path,
) -> None:
	if not bool(cfg.get("threshold_calibration_enabled", True)):
		print("Threshold calibration disabled.")
		return
	if not rows:
		print("Threshold calibration skipped: no SS4 candidate rows.")
		return
	base_dir.mkdir(parents=True, exist_ok=True)
	make_plots = bool(cfg.get("threshold_calibration_make_plots", True))
	suggest = bool(cfg.get("threshold_calibration_suggest_thresholds", True))
	ss_blue = float(cfg.get("ss4_ss_blue_max", cfg.get("spike_score_v4_ss_blue_max", 0.95)))
	ss_red = float(cfg.get("ss4_ss_red_min", cfg.get("spike_score_v4_ss_red_min", 0.9999)))
	pce_red = float(cfg.get("ss4_pce_red_min", cfg.get("spike_score_v4_pce_red_min", 0.4)))
	rve_red = float(cfg.get("ss4_rve_red_max", cfg.get("spike_score_v4_edge_red_min", -0.1)))
	missing_policy = str(cfg.get("ss4_missing_policy", cfg.get("spike_score_v4_missing_policy", "review")))

	metric_specs = [
		("spike_score_v1", "spike_score_v1", [ss_blue, ss_red]),
		("pce_negpref_t098_evidence_signed", "pce", [pce_red]),
		("recdw_sum_0_90_raman_veto_evidence_signed", "edge_rve", [rve_red]),
	]
	suggestions: Dict[str, float] = {
		"ss1_blue": ss_blue,
		"ss1_red": ss_red,
		"pce": pce_red,
		"edge_rve": rve_red,
	}

	try:
		import matplotlib.pyplot as plt
	except Exception:
		plt = None  # type: ignore[assignment]
		make_plots = False

	for metric_key, metric_slug, thresholds in metric_specs:
		values = np.asarray([_metric_float(r, metric_key) for r in rows], dtype=float)
		values = values[np.isfinite(values)]
		if values.size == 0:
			continue
		if metric_slug == "pce" and suggest:
			suggestions["pce"] = _suggest_threshold_from_histogram(values, pce_red)
		elif metric_slug == "edge_rve" and suggest:
			suggestions["edge_rve"] = _suggest_threshold_from_histogram(values, rve_red)
		elif metric_slug == "spike_score_v1" and suggest:
			suggestions["ss1_blue"] = _suggest_threshold_from_histogram(values, ss_blue)
			suggestions["ss1_red"] = _suggest_threshold_from_histogram(values, ss_red)
		counts, edges = np.histogram(values, bins=min(180, max(48, int(np.sqrt(values.size) * 3))))
		_write_simple_csv(
			base_dir / f"threshold_calibration_{metric_slug}_hist.csv",
			["bin_left", "bin_right", "count"],
			[
				{"bin_left": float(edges[i]), "bin_right": float(edges[i + 1]), "count": int(counts[i])}
				for i in range(len(counts))
			],
		)
		if make_plots and plt is not None:
			fig, ax = plt.subplots(figsize=(7.5, 4.5))
			ax.hist(values, bins=min(180, max(48, int(np.sqrt(values.size) * 3))), color="#4c78a8", alpha=0.84)
			for thr in thresholds:
				ax.axvline(float(thr), color="#d62728", linestyle="--", linewidth=1.4)
			if metric_slug == "pce":
				ax.axvline(float(suggestions["pce"]), color="#2ca02c", linestyle=":", linewidth=1.5)
			elif metric_slug == "edge_rve":
				ax.axvline(float(suggestions["edge_rve"]), color="#2ca02c", linestyle=":", linewidth=1.5)
			elif metric_slug == "spike_score_v1":
				ax.axvline(float(suggestions["ss1_blue"]), color="#2ca02c", linestyle=":", linewidth=1.2)
				ax.axvline(float(suggestions["ss1_red"]), color="#2ca02c", linestyle=":", linewidth=1.2)
			ax.set_title(f"Threshold calibration: {metric_slug}")
			ax.set_xlabel(metric_key)
			ax.set_ylabel("count")
			ax.grid(alpha=0.18)
			fig.tight_layout()
			fig.savefig(base_dir / f"threshold_calibration_{metric_slug}.png", dpi=150)
			plt.close(fig)

	def _scenario_decisions(name: str, *, ss_blue_thr: float, ss_red_thr: float, pce_thr: float, rve_thr: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
		changed_rows: List[Dict[str, Any]] = []
		new_spikes = 0
		old_spikes = 0
		spike_to_non = 0
		non_to_spike = 0
		review_changes = 0
		pce_rescued_by_edge = 0
		edge_affected = 0
		for row in rows:
			old_ss4 = _metric_float(row, "ss4")
			old_decision = str(row.get("ss4_decision", "review"))
			new_info = _compute_ss4_decision_for_row(
				row,
				ss_blue_max=ss_blue_thr,
				ss_red_min=ss_red_thr,
				pce_red_min=pce_thr,
				rve_red_max=rve_thr,
				missing_policy=missing_policy,
			)
			new_ss4 = _metric_float(new_info, "ss4")
			new_decision = str(new_info.get("ss4_decision", "review"))
			if np.isfinite(old_ss4) and old_ss4 >= 0.5:
				old_spikes += 1
			if np.isfinite(new_ss4) and new_ss4 >= 0.5:
				new_spikes += 1
			if old_decision != new_decision or (np.isfinite(old_ss4) != np.isfinite(new_ss4)) or (np.isfinite(old_ss4) and np.isfinite(new_ss4) and (old_ss4 >= 0.5) != (new_ss4 >= 0.5)):
				if old_decision == "spike" and new_decision != "spike":
					spike_to_non += 1
				elif old_decision != "spike" and new_decision == "spike":
					non_to_spike += 1
				elif old_decision != new_decision:
					review_changes += 1
				if old_decision == "spike" and new_decision == "spike" and _metric_float(row, "pce_negpref_t098_evidence_signed") < pce_thr and _metric_float(row, "recdw_sum_0_90_raman_veto_evidence_signed") <= rve_thr:
					pce_rescued_by_edge += 1
				if _metric_float(row, "recdw_sum_0_90_raman_veto_evidence_signed") != _metric_float(row, "recdw_sum_0_90_raman_veto_evidence_signed"):
					pass
				changed_rows.append(
					{
						"scenario": name,
						"source_y": int(row["y"]),
						"source_x": int(row["x"]),
						"peak_index": int(row["peak_index"]),
						"spike_score_v1": _metric_float(row, "spike_score_v1"),
						"pce_negpref_t098_evidence_signed": _metric_float(row, "pce_negpref_t098_evidence_signed"),
						"recdw_sum_0_90_raman_veto_evidence_signed": _metric_float(row, "recdw_sum_0_90_raman_veto_evidence_signed"),
						"old_ss4": old_ss4,
						"old_decision": old_decision,
						"new_ss4": new_ss4,
						"new_decision": new_decision,
						"old_reason": row.get("ss4_reason"),
						"new_reason": new_info.get("ss4_reason"),
					}
				)
			if pce_thr != pce_red and old_decision != new_decision:
				if _metric_float(row, "pce_negpref_t098_evidence_signed") >= min(pce_thr, pce_red) and _metric_float(row, "pce_negpref_t098_evidence_signed") < max(pce_thr, pce_red):
					if new_decision == "spike" and _metric_float(row, "recdw_sum_0_90_raman_veto_evidence_signed") <= rve_thr:
						pce_rescued_by_edge += 1
			if rve_thr != rve_red and old_decision != new_decision:
				edge_affected += 1
		return (
			{
				"scenario": name,
				"current_spike_count": int(old_spikes),
				"suggested_spike_count": int(new_spikes),
				"spike_to_non_spike": int(spike_to_non),
				"non_spike_to_spike": int(non_to_spike),
				"review_or_missing_changes": int(review_changes),
				"pce_affected_but_rescued_by_edge": int(pce_rescued_by_edge),
				"edge_affected": int(edge_affected),
			},
			changed_rows,
		)

	scenarios = [
		("pce_threshold", ss_blue, ss_red, suggestions["pce"], rve_red),
		("edge_rve_threshold", ss_blue, ss_red, pce_red, suggestions["edge_rve"]),
		("ss1_blue_threshold", suggestions["ss1_blue"], ss_red, pce_red, rve_red),
		("ss1_red_threshold", ss_blue, suggestions["ss1_red"], pce_red, rve_red),
	]
	summaries: List[Dict[str, Any]] = []
	all_changed: List[Dict[str, Any]] = []
	for scenario_name, ss_blue_thr, ss_red_thr, pce_thr, rve_thr in scenarios:
		summary, changed_rows = _scenario_decisions(
			scenario_name,
			ss_blue_thr=float(ss_blue_thr),
			ss_red_thr=float(ss_red_thr),
			pce_thr=float(pce_thr),
			rve_thr=float(rve_thr),
		)
		summary["current_threshold"] = {
			"ss_blue_max": ss_blue,
			"ss_red_min": ss_red,
			"pce_red_min": pce_red,
			"rve_red_max": rve_red,
		}
		summary["suggested_threshold"] = {
			"ss_blue_max": float(ss_blue_thr),
			"ss_red_min": float(ss_red_thr),
			"pce_red_min": float(pce_thr),
			"rve_red_max": float(rve_thr),
		}
		summaries.append(summary)
		all_changed.extend(changed_rows)

	_write_simple_csv(
		base_dir / "threshold_calibration_counts.csv",
		[
			"scenario",
			"current_spike_count",
			"suggested_spike_count",
			"spike_to_non_spike",
			"non_spike_to_spike",
			"review_or_missing_changes",
			"pce_affected_but_rescued_by_edge",
			"edge_affected",
		],
		summaries,
	)
	_write_simple_csv(
		base_dir / "threshold_calibration_changed_candidates.csv",
		[
			"scenario",
			"source_y",
			"source_x",
			"peak_index",
			"spike_score_v1",
			"pce_negpref_t098_evidence_signed",
			"recdw_sum_0_90_raman_veto_evidence_signed",
			"old_ss4",
			"old_decision",
			"new_ss4",
			"new_decision",
			"old_reason",
			"new_reason",
		],
		all_changed,
	)
	(base_dir / "threshold_calibration_summary.json").write_text(
		json.dumps(
			{
				"suggestions": suggestions,
				"summaries": summaries,
				"n_rows": len(rows),
			},
			indent=2,
			ensure_ascii=False,
		),
		encoding="utf-8",
	)
	print(f"Threshold calibration written to {base_dir}")


def run(cfg: Dict[str, Any]) -> None:

	start = time.time()

	# 0) config file
	in_path = Path(cfg['input_path'])

	config = time.time()
	print(f"Config file loaded in {config - start:.2f} seconds")

	# 1) load and optionally resample data
	ds = load_dataset(in_path)
	x_axis = ds.x_axis
	raw = ds.spectra

	if bool(cfg.get("resample_enabled", False)):
		x_axis, raw = resample_axis_and_spectra(
			x_axis=x_axis,
			spectra=raw,
			factor=int(cfg.get("resample_factor", 2)),
		)

	load_and_resample = time.time()
	print(f"Data loaded and resampled in {load_and_resample - config:.2f} seconds")

	# 2) morphology
	overlays = compute_morph_overlays(raw, se_size=int(cfg['se_size']))

	# 3a) score map + candidates
	score_map = score_map_from_top_hat(overlays['top_hat'], mode=str(cfg['score_mode']))
	thr = threshold_score_map(
		score_map=score_map,
		method=str(cfg['threshold_method']),
		quantile=float(cfg['threshold_quantile']),
		k_mad=float(cfg['threshold_k_mad']),
		min_abs=cfg['threshold_min_abs'],
	)
	candidate_mask = score_map >= thr

	target_coords: Optional[List[Tuple[int, int]]] = None
	if cfg.get("coords_csv"):
		target_coords = load_target_coords_csv(Path(cfg['coords_csv']), shape_hw=score_map.shape)
		coord_mask = np.zeros_like(candidate_mask, dtype=bool)
		for yy, xx in target_coords:
			coord_mask[yy, xx] = True
		# For coordinate-based optimization we run extraction only on requested spectra
		candidate_mask = coord_mask

	score_map_candidates = time.time()
	print(f"Score map computed in {score_map_candidates - load_and_resample:.2f} seconds")

	# 3b) spike segments (narrow bands) for candidate pixels
	spikes, spikes_by_pixel = extract_spikes_for_candidates(
		x_axis=x_axis,
		top_hat=overlays['top_hat'],
		candidate_mask=candidate_mask,
		raw_spectra=raw,
		max_width_pts=int(cfg['max_width_pts']),
		k_mad_pixel=float(cfg['k_mad_pixel']),
		min_peak=float(cfg['min_peak']),
		baseline_se_size=cfg['baseline_se_size'],
		edge_k_mad=float(cfg['edge_k_mad']),
		pad_pts=int(cfg['pad_pts']),
	)

	spike_segments = time.time()
	print(f"Spike segments extracted in {spike_segments - score_map_candidates:.2f} seconds")

	# 3) comparison with neighbors
	if bool(cfg['neighbor_filter_enabled']):
		accepted = neighbour_filter_ratio(
			top_hat=overlays['top_hat'],
			spikes=spikes,
			radius=int(cfg['neighbor_radius']),
			ratio_min=float(cfg['neighbor_ratio_min']),
		)
		accepted_set = set(accepted)

		# rebuild spikes_by_pixel + rebuild cand from kept spikes
		spikes_by_pixel = {
			pix: [s for s in segs if s in accepted_set]
			for pix, segs in spikes_by_pixel.items()
		}
		spikes_by_pixel = {pix: segs for pix, segs in spikes_by_pixel.items() if segs}
		spikes = accepted

	comparison = time.time()
	print(f"Comparison with neighbors completed in {comparison - score_map_candidates:.2f} seconds")

	# 4) other stuff
	corrected = raw
	merge_dups = bool(cfg.get("merge_duplicate_segments", cfg.get("debug_merge_duplicate_segments", False)))
	labels_by_candidate = None
	if cfg.get("labels_csv"):
		labels_by_candidate = load_labels_csv(Path(cfg["labels_csv"]))
	report: Optional[Dict[str, Any]] = None
	ss4_selected_metadata: Dict[Tuple[int, int, int, int, int], Dict[str, Any]] = {}
	ss4_candidate_metrics_by_pixel: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
	candidate_prefilter_rows_by_pixel: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
	candidate_prefilter_summary_by_pixel: Dict[Tuple[int, int], Dict[str, object]] = {}
	primary_profile_overlap_summary: Dict[str, Any] = {}
	small_morphology_cache: Dict[Tuple[int, int], SmallMorphologyBundle] = {}
	debug_feature_report_enabled = bool(cfg.get("debug_feature_report_enabled", cfg.get("debug_report_enabled", False)))
	debug_feature_report_path = cfg.get("debug_feature_report_path", cfg.get("debug_report_path"))
	despike_debug_lite_enabled = bool(cfg.get("despike_debug_lite_enabled", True))
	despike_debug_lite_path = cfg.get("despike_debug_lite_path")
	despike_debug_full_enabled = bool(cfg.get("despike_debug_full_enabled", False))
	despike_debug_full_path = cfg.get("despike_debug_full_path")
	debug_store_arrays = bool(cfg.get("debug_store_arrays", False))
	save_viewer_cache_enabled = bool(cfg.get("save_viewer_cache", True))
	viewer_cache_path = cfg.get("viewer_cache_path")

	def _robust_center_scale(vals: np.ndarray) -> Tuple[float, float]:
		x = np.asarray(vals, dtype=float)
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

	def _sigmoid_support(z: float, scale: float, clip: float) -> float:
		if not np.isfinite(z):
			return float("nan")
		z_clip = float(np.clip(float(z), -abs(float(clip)), abs(float(clip))))
		return float(1.0 / (1.0 + np.exp(-z_clip / max(float(scale), 1e-12))))

	def _save_ss4_primary_histograms(records: List[Dict[str, Any]]) -> None:
		if not bool(cfg.get("ss4_histogram_enabled", False)):
			return
		out_dir_raw = cfg.get("ss4_histogram_dir")
		if not out_dir_raw:
			base_dir = Path(str(cfg.get("output_dir", "") or "."))
			out_dir = base_dir / "ss4_histograms"
		else:
			out_dir = Path(str(out_dir_raw))
		out_dir.mkdir(parents=True, exist_ok=True)
		prefix = str(cfg.get("ss4_histogram_prefix", "ss4_primary")).strip() or "ss4_primary"
		series = {
			"ss1": [float(r["features"].get("spike_score_v1", np.nan)) for r in records],
			"pce": [float(r["features"].get("pce_negpref_t098_evidence_signed", np.nan)) for r in records],
			"edge": [float(r["features"].get("recdw_sum_0_90_raman_veto_evidence_signed", np.nan)) for r in records],
		}
		try:
			import matplotlib.pyplot as plt
		except Exception as exc:
			print(f"[ss4-hist] skipped: matplotlib unavailable ({exc})")
			return
		for suffix, vals in series.items():
			arr = np.asarray(vals, dtype=float)
			arr = arr[np.isfinite(arr)]
			if arr.size == 0:
				continue
			fig_h, ax_h = plt.subplots(figsize=(7.0, 4.2))
			ax_h.hist(arr, bins=min(80, max(10, int(np.sqrt(arr.size) * 2))), color="#4c78a8", alpha=0.85)
			ax_h.set_title(f"SS4 primary {suffix}")
			ax_h.set_xlabel(suffix)
			ax_h.set_ylabel("count")
			ax_h.grid(alpha=0.18)
			fig_h.tight_layout()
			path = out_dir / f"{prefix}_{suffix}.png"
			fig_h.savefig(path, dpi=150)
			plt.close(fig_h)
		print(f"[ss4-hist] saved primary SS4 histograms to {out_dir}")

	def _primary_candidate_id(y: int, x: int, peak: int, start: int, end: int) -> str:
		return f"primary:{int(y)}:{int(x)}:{int(peak)}:{int(start)}:{int(end)}"

	def _primary_ss4_record(y: int, x: int, peak: int, start: int, end: int, features: Dict[str, Any]) -> Dict[str, Any]:
		candidate_id = _primary_candidate_id(y, x, peak, start, end)
		return {
			"candidate_id": candidate_id,
			"parent_id": candidate_id,
			"y": int(y),
			"x": int(x),
			"peak_index": int(peak),
			"start": int(start),
			"end": int(end),
			"primary_spike_score_v1": features.get("spike_score_v1"),
			"primary_pce_negpref_t098_evidence_signed": features.get("pce_negpref_t098_evidence_signed"),
			"primary_recdw_sum_0_90_raman_veto_evidence_signed": features.get("recdw_sum_0_90_raman_veto_evidence_signed"),
			"primary_ss4": features.get("ss4"),
			"primary_ss4_decision": features.get("ss4_decision"),
			"primary_ss4_reason": features.get("ss4_reason"),
			"primary_ss4_rve_feature": features.get("ss4_rve_feature"),
			"primary_spike_score_v5": features.get("spike_score_v5", features.get("ss5")),
			"primary_ss5": features.get("ss5"),
			"primary_ss5_decision": features.get("ss5_decision"),
			"primary_ss5_reason": features.get("ss5_reason"),
			"primary_ss5_ss1_vote": features.get("ss5_ss1_vote"),
			"primary_ss5_pce_vote": features.get("ss5_pce_vote"),
			"primary_ss5_edge_vote": features.get("ss5_edge_vote"),
			"primary_ss5_ss1_threshold": features.get("ss5_ss1_threshold"),
			"primary_ss5_pce_spike_min": features.get("ss5_pce_spike_min"),
			"primary_ss5_edge_spike_max": features.get("ss5_edge_spike_max"),
			"primary_active_decision_profile": features.get("primary_active_decision_profile", "ss4"),
			"primary_active_score": features.get("primary_active_score"),
			"primary_active_decision": features.get("primary_active_decision"),
			"primary_active_reason": features.get("primary_active_reason"),
			# Legacy aliases are kept read-only for older debug/report consumers.
			"spike_score_v1": features.get("spike_score_v1"),
			"pce_negpref_t098_evidence_signed": features.get("pce_negpref_t098_evidence_signed"),
			"recdw_sum_0_90_raman_veto_evidence_signed": features.get("recdw_sum_0_90_raman_veto_evidence_signed"),
			"ss4": features.get("ss4"),
			"ss4_decision": features.get("ss4_decision"),
			"ss4_reason": features.get("ss4_reason"),
			"ss4_rve_feature": features.get("ss4_rve_feature"),
			"spike_score_v5": features.get("spike_score_v5", features.get("ss5")),
			"ss5": features.get("ss5"),
			"ss5_decision": features.get("ss5_decision"),
			"ss5_reason": features.get("ss5_reason"),
			"ss5_ss1_vote": features.get("ss5_ss1_vote"),
			"ss5_pce_vote": features.get("ss5_pce_vote"),
			"ss5_edge_vote": features.get("ss5_edge_vote"),
			"ss5_ss1_threshold": features.get("ss5_ss1_threshold"),
			"ss5_pce_spike_min": features.get("ss5_pce_spike_min"),
			"ss5_edge_spike_max": features.get("ss5_edge_spike_max"),
		}

	def _merge_candidate_rows(base: Dict[str, object], extra: Dict[str, object]) -> Dict[str, object]:
		out = dict(base)
		out.update(extra)
		return out

	def _minimal_ss4_base_features(raw_sig: np.ndarray, grad_sig: Optional[np.ndarray], seg: SpikeSegment) -> Dict[str, float]:
		src = str(cfg.get("debug_feature_signal_source", "gradient")).strip().lower()
		if src == "raw" or grad_sig is None or grad_sig.size != raw_sig.size:
			sig = raw_sig
		else:
			sig = grad_sig
		n = int(sig.size)
		a = int(np.clip(seg.start, 0, n - 1))
		b = int(np.clip(seg.end, 0, n - 1))
		p = int(np.clip(seg.peak_index, 0, n - 1))
		if b < a:
			a, b = b, a
		out: Dict[str, float] = {
			"spike_score_v1": float("nan"),
			"pce_negpref_t098_evidence_signed": float("nan"),
		}
		if not (a < p < b):
			return out
		segment = np.asarray(sig[a:b + 1], dtype=float)
		if segment.size < 3:
			return out
		d = np.diff(segment)
		rise = d[: max(1, p - a)]
		fall = d[max(1, p - a):]
		rise_slope = float(np.nanmax(rise)) if rise.size else 0.0
		fall_slope = float(np.nanmin(fall)) if fall.size else 0.0
		context = max(10, 3 * max(1, b - a + 1))
		l0 = max(0, a - context)
		r1 = min(n, b + context + 1)
		bg = np.concatenate([sig[l0:a], sig[b + 1:r1]])
		if bg.size < 5:
			bg = np.concatenate([sig[:a], sig[b + 1:]])
		if bg.size < 5:
			bg = sig
		bg_med = float(np.nanmedian(bg))
		bg_mad = max(float(np.nanmedian(np.abs(bg - bg_med))), 1e-12)
		out["rise_slope_z"] = float(rise_slope / bg_mad)
		out["fall_slope_z"] = float(abs(fall_slope) / bg_mad)
		sr = float(np.tanh(out["rise_slope_z"] / 6.0))
		sf = float(np.tanh(out["fall_slope_z"] / 6.0))
		out["spike_score_v1"] = float(0.50 * sr + 0.50 * sf)
		peak_rel = int(np.clip(p - a, 0, segment.size - 1))
		out.update(compute_peak_curvature_features(segment, bg_mad, peak_rel=peak_rel))
		out.update(compute_spike_score_v2_features(out))
		return out

	def _ss4_selected_spikes_fast() -> List[SpikeSegment]:
		nonlocal primary_profile_overlap_summary
		t0 = time.time()
		rows: List[SpikeSegment] = [seg for segs in ss4_input_candidates_by_pixel.values() for seg in segs]
		print(f"[ss4-fast] computing minimal ss4 features for {len(rows)} candidates")
		records: List[Dict[str, Any]] = []
		recdw_population = str(cfg.get("ss4_fast_recdw_population", "needed")).strip().lower()
		if recdw_population not in {"needed", "edge_needed", "all"}:
			recdw_population = "needed"
		recdw_for_all = recdw_population == "all"
		progress_bar = None
		if bool(cfg.get("debug_progress", True)):
			try:
				from tqdm import tqdm  # type: ignore
				progress_bar = tqdm(rows, desc="SS4 minimal features", unit="spike", dynamic_ncols=True, mininterval=0.25, miniters=1)
			except Exception:
				progress_bar = None
		iterable = progress_bar if progress_bar is not None else rows
		for prepared_seg in iterable:
			yv = int(prepared_seg.y)
			xv = int(prepared_seg.x)
			raw_sig = raw[yv, xv, :].astype(float)
			grad_sig = overlays["gradient"][yv, xv, :].astype(float) if "gradient" in overlays else None
			features = _minimal_ss4_base_features(raw_sig, grad_sig, prepared_seg)
			ss_v = float(features.get("spike_score_v1", np.nan))
			pce_v = float(features.get("pce_negpref_t098_evidence_signed", np.nan))
			needs_edge = bool(np.isfinite(ss_v) and ss_v >= float(cfg.get("ss4_ss_blue_max", 0.95)) and (not np.isfinite(pce_v) or pce_v < float(cfg.get("ss4_pce_red_min", 0.4))))
			if not needs_edge and not recdw_for_all:
				features["recdw_sum_0_90"] = float("nan")
				records.append({"original": original, "prepared": prepared_seg, "features": features, "needs_edge": False})
				continue
			edge_ctx_pad = int(cfg.get("edge_dense_context_pad_pts", 20))
			edge_ctx_pad = max(edge_ctx_pad, int(cfg.get("edge_dense_context_min_pad_pts", 10)))
			edge_ctx_pad = min(edge_ctx_pad, int(cfg.get("edge_dense_context_max_pad_pts", 80)))
			edge_left = max(0, int(prepared_seg.start) - edge_ctx_pad)
			edge_right = min(int(raw_sig.size) - 1, int(prepared_seg.end) + edge_ctx_pad)
			feature_context = max(10, 3 * max(1, int(prepared_seg.end) - int(prepared_seg.start) + 1))
			l0 = max(0, int(prepared_seg.start) - feature_context)
			r1 = min(int(raw_sig.size), int(prepared_seg.end) + feature_context + 1)
			raw_bg = np.concatenate([raw_sig[l0:int(prepared_seg.start)], raw_sig[int(prepared_seg.end) + 1:r1]])
			if raw_bg.size < 5:
				raw_bg = np.concatenate([raw_sig[:int(prepared_seg.start)], raw_sig[int(prepared_seg.end) + 1:]])
			if raw_bg.size < 5:
				raw_bg = raw_sig
			raw_bg_med = float(np.nanmedian(raw_bg)) if raw_bg.size else 0.0
			raw_bg_mad = max(float(np.nanmedian(np.abs(raw_bg - raw_bg_med))) if raw_bg.size else 0.0, 1e-12)
			edge_metrics = compute_edge_width_metrics(
				raw_sig,
				detection_left=int(edge_left),
				detection_right=int(edge_right),
				prefix="raw_edge_ctx",
				apex_idx=int(prepared_seg.peak_index),
				bg_mad=raw_bg_mad,
				include_low_root_metrics=True,
				low_root_noise_k_mad=float(cfg.get("edge_dense_min_snr", 1.0)),
				use_enhanced_spike_mapping=bool(cfg.get("edge_use_enhanced_spike_mapping", False)) and bool(cfg.get("edge_enhanced_in_debug_report", True)),
				mapping_levels_desc=tuple(int(v) for v in cfg.get("edge_mapping_levels_desc", [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5])),
				mapping_refine_step_percent=int(cfg.get("edge_mapping_refine_step_percent", 1)),
				mapping_min_level_percent=int(cfg.get("edge_mapping_min_level_percent", 1)),
				mapping_require_closed_interval=bool(cfg.get("edge_mapping_require_closed_interval", True)),
				mapping_use_apex_component=bool(cfg.get("edge_mapping_use_apex_component", True)),
				mapping_enable_merge_guard=bool(cfg.get("edge_mapping_enable_merge_guard", True)),
				mapping_max_width_jump_factor=float(cfg.get("edge_mapping_max_width_jump_factor", 2.5)),
				mapping_max_width_jump_points=float(cfg.get("edge_mapping_max_width_jump_points", 8)),
				mapping_fallback_to_old=bool(cfg.get("edge_mapping_fallback_to_old", False)),
				mapping_noise_guard_enabled=bool(cfg.get("edge_mapping_noise_guard_enabled", False)),
				robust_reference_enabled=bool(cfg.get("edge_robust_reference_enabled", True)),
				robust_reference_noise=float(
					next(
						(
							float(row.get("candidate_noise_estimate_used"))
							for row in candidate_prefilter_rows_by_pixel.get((yv, xv), [])
							if str(row.get("candidate_id")) == candidate_segment_id(prepared_seg)
							and row.get("candidate_noise_estimate_used") is not None
						),
						raw_bg_mad,
					)
				),
			)
			features["recdw_sum_0_90"] = float(edge_metrics.get("raw_edge_ctx_dense_width_sum_0_90", np.nan))
			edge_debug = edge_metrics.get("raw_edge_ctx_debug")
			if isinstance(edge_debug, dict):
				for dbg_key in (
					"edge_robust_reference_enabled",
					"edge_reference_original",
					"edge_reference_robust",
					"edge_reference_delta",
					"edge_reference_noise_used",
					"edge_reference_adjusted",
					"edge_reference_reason",
				):
					features[dbg_key] = edge_debug.get(dbg_key)
			records.append({"prepared": prepared_seg, "features": features, "needs_edge": needs_edge})
		if progress_bar is not None:
			progress_bar.close()
		edge_needed_n = int(sum(1 for r in records if bool(r.get("needs_edge", False))))
		edge_computed_n = int(sum(1 for r in records if np.isfinite(float(r["features"].get("recdw_sum_0_90", np.nan)))))
		print(f"[ss4-fast] edge/RVE computed for {edge_computed_n}/{len(records)} candidates ({edge_needed_n} needed by ss4)")
		if recdw_for_all:
			vals = np.asarray([float(r["features"].get("recdw_sum_0_90", np.nan)) for r in records], dtype=float)
		else:
			vals = np.asarray([float(r["features"].get("recdw_sum_0_90", np.nan)) for r in records if bool(r.get("needs_edge", False))], dtype=float)
		center, scale = _robust_center_scale(vals)
		selected: List[SpikeSegment] = []
		missing = 0
		for rec in records:
			features = rec["features"]
			xv = float(features.get("recdw_sum_0_90", np.nan))
			if np.isfinite(xv) and np.isfinite(center) and np.isfinite(scale) and scale > 1e-12:
				z = float((xv - center) / scale)
				support = _sigmoid_support(z, float(cfg.get("recdw_support_z_scale", 1.0)), float(cfg.get("recdw_z_clip", 6.0)))
				features["recdw_sum_0_90_z"] = z
				features["recdw_sum_0_90_support01"] = support
				features["recdw_sum_0_90_raman_veto_evidence_signed"] = float(2.0 * support - 1.0)
			else:
				features["recdw_sum_0_90_z"] = float("nan")
				features["recdw_sum_0_90_support01"] = float("nan")
				features["recdw_sum_0_90_raman_veto_evidence_signed"] = float("nan")
			features.update(
				annotate_feature_dict_with_spike_score_v4(
					features,
					edge_feature=str(cfg.get("ss4_rve_feature", cfg.get("spike_score_v4_edge_feature", "recdw_sum_0_90_raman_veto_evidence_signed"))),
					ss_blue_max=float(cfg.get("ss4_ss_blue_max", cfg.get("spike_score_v4_ss_blue_max", 0.95))),
					ss_red_min=float(cfg.get("ss4_ss_red_min", cfg.get("spike_score_v4_ss_red_min", 0.9999))),
					pce_red_min=float(cfg.get("ss4_pce_red_min", cfg.get("spike_score_v4_pce_red_min", 0.4))),
					edge_red_min=float(cfg.get("ss4_rve_red_max", cfg.get("spike_score_v4_edge_red_min", -0.1))),
					pce_dead_zone_enabled=bool(cfg.get("ss4_pce_dead_zone_enabled", True)),
					pce_dead_zone_low=float(cfg.get("ss4_pce_dead_zone_low", -0.8)),
					pce_dead_zone_high=float(cfg.get("ss4_pce_dead_zone_high", -0.2)),
					missing_policy=str(cfg.get("ss4_missing_policy", cfg.get("spike_score_v4_missing_policy", "review"))),
				)
			)
			features.update(
				annotate_feature_dict_with_spike_score_v5(
					features,
					edge_feature=str(cfg.get("ss4_rve_feature", cfg.get("spike_score_v4_edge_feature", "recdw_sum_0_90_raman_veto_evidence_signed"))),
					ss1_threshold=float(cfg.get("ss5_ss1_threshold", 0.95)),
					pce_spike_min=float(cfg.get("ss5_pce_spike_min", 0.8)),
					edge_spike_max=float(cfg.get("ss5_edge_spike_max", -0.4)),
				)
			)
			active_profile = "ss5" if str(decision_profile).strip().lower() == "ss5" else "ss4"
			if active_profile == "ss5":
				features["primary_active_decision_profile"] = "ss5"
				features["primary_active_score"] = features.get("ss5")
				features["primary_active_decision"] = features.get("ss5_decision")
				features["primary_active_reason"] = features.get("ss5_reason")
			else:
				features["primary_active_decision_profile"] = "ss4"
				features["primary_active_score"] = features.get("ss4")
				features["primary_active_decision"] = features.get("ss4_decision")
				features["primary_active_reason"] = features.get("ss4_reason")
			v = float(features.get("primary_active_score", np.nan))
			prepared_seg = rec["prepared"]
			primary_record = _primary_ss4_record(
				int(prepared_seg.y),
				int(prepared_seg.x),
				int(prepared_seg.peak_index),
				int(prepared_seg.start),
				int(prepared_seg.end),
				features,
			)
			existing_rows = ss4_candidate_metrics_by_pixel.setdefault((int(prepared_seg.y), int(prepared_seg.x)), [])
			merged = False
			for idx, row in enumerate(existing_rows):
				if str(row.get("candidate_id")) == primary_record["candidate_id"]:
					existing_rows[idx] = _merge_candidate_rows(row, primary_record)
					merged = True
					break
			if not merged:
				existing_rows.append(primary_record)
			if not np.isfinite(v):
				missing += 1
				continue
			if v >= 0.5:
				key = (int(prepared_seg.y), int(prepared_seg.x), int(prepared_seg.peak_index), int(prepared_seg.start), int(prepared_seg.end))
				ss4_selected_metadata[key] = dict(primary_record)
				selected.append(prepared_seg)
		apply_global_metric_ranks(row for rows in ss4_candidate_metrics_by_pixel.values() for row in rows)
		primary_profile_overlap_summary = _summarize_ss4_ss5_overlap(
			row for rows in ss4_candidate_metrics_by_pixel.values() for row in rows
		)
		print(
			"[primary-compare] "
			f"ss4={primary_profile_overlap_summary.get('ss4_spikes', 0)} "
			f"ss5={primary_profile_overlap_summary.get('ss5_spikes', 0)} "
			f"both={primary_profile_overlap_summary.get('accepted_by_both', 0)} "
			f"ss4_only={primary_profile_overlap_summary.get('accepted_by_ss4_only', 0)} "
			f"ss5_only={primary_profile_overlap_summary.get('accepted_by_ss5_only', 0)} "
			f"neither={primary_profile_overlap_summary.get('rejected_by_both', 0)}"
		)
		accepted_rows = [
			row
			for rows in ss4_candidate_metrics_by_pixel.values()
			for row in rows
			if str(row.get("primary_active_decision", row.get("primary_ss4_decision", row.get("ss4_decision", "")))).strip().lower() == "spike"
		]
		accepted_with_noise_ratio = 0
		for row in accepted_rows:
			try:
				ratio_v = float(row.get("candidate_noise_height_ratio", np.nan))
			except Exception:
				ratio_v = float("nan")
			if np.isfinite(ratio_v):
				accepted_with_noise_ratio += 1
		print(
			f"[candidate-noise] accepted spikes with valid height ratio: "
			f"{accepted_with_noise_ratio}/{len(accepted_rows)}"
		)
		_save_ss4_primary_histograms(records)
		print(
			f"[{str(decision_profile).strip().lower()}-fast] selected {len(selected)} accepted spike candidates "
			f"({missing} missing) in {time.time() - t0:.2f} seconds"
		)
		return selected

	def _build_current_report(include_per_spectrum: Optional[bool] = None) -> Dict[str, Any]:
		return build_debug_report(
			score_map=score_map,
			candidate_mask=candidate_mask,
			spikes_by_pixel=prepared_candidates_by_pixel,
			threshold=float(thr),
			target_coords=target_coords,
			include_per_spectrum=bool(cfg['debug_include_per_spectrum']) if include_per_spectrum is None else bool(include_per_spectrum),
			max_top_pixels=int(cfg['debug_top_pixels']),
			raw_spectra=raw,
			overlays=overlays,
			x_axis=x_axis,
			feature_signal_source=str(cfg.get("debug_feature_signal_source", "gradient")),
			merge_duplicate_segments=merge_dups,
			feature_expand_to_gradient_foot=bool(cfg.get("feature_expand_to_gradient_foot", True)),
			feature_foot_k_mad=float(cfg.get("feature_foot_k_mad", 2.0)),
			feature_foot_min_run=int(cfg.get("feature_foot_min_run", 2)),
			feature_window_method=cfg.get("feature_window_method", "mad_run"),
			feature_erosion_se_size=int(cfg.get("feature_erosion_se_size", 5)),
			ss5_ss1_threshold=float(cfg.get("ss5_ss1_threshold", 0.95)),
			ss5_pce_spike_min=float(cfg.get("ss5_pce_spike_min", 0.8)),
			ss5_edge_spike_max=float(cfg.get("ss5_edge_spike_max", -0.4)),
			candidate_prefilter_rows_by_pixel=candidate_prefilter_rows_by_pixel,
			candidate_prefilter_summary_by_pixel=candidate_prefilter_summary_by_pixel,
			boundary_minimum_source=cfg.get("boundary_minimum_source", "gradient"),
			gws_split_overlapping_contexts=bool(cfg.get("gws_split_overlapping_contexts", False)),
			gws_split_source=str(cfg.get("gws_split_source", "gradient")),
			gws_split_smooth_pts=int(cfg.get("gws_split_smooth_pts", 3)),
			gws_split_valley_alpha=float(cfg.get("gws_split_valley_alpha", 0.75)),
			gws_split_min_distance_from_apex=int(cfg.get("gws_split_min_distance_from_apex", 1)),
			gws_split_min_context_width=int(cfg.get("gws_split_min_context_width", 3)),
			gws_split_debug=bool(cfg.get("gws_split_debug", True)),
			gws_source_modes=tuple(str(v) for v in cfg.get("gws_source_modes", ["morph_gradient", "morph_gradient_med3", "morph_gradient_med5", "morph_gradient_mean3", "morph_gradient_mean5"])),
			gws_include_scale_zero=bool(cfg.get("gws_include_scale_zero", False)),
			gws_measure_region=str(cfg.get("gws_measure_region", "mask")),
			gws_threshold_region=str(cfg.get("gws_threshold_region", "spike_edges")),
			edge_dense_enabled=bool(cfg.get("edge_dense_enabled", True)),
			edge_dense_levels=tuple(int(v) for v in cfg.get("edge_dense_levels", [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])),
			edge_dense_min_snr=float(cfg.get("edge_dense_min_snr", 1.0)),
			edge_dense_context_pad_pts=int(cfg.get("edge_dense_context_pad_pts", 20)),
			edge_dense_context_min_pad_pts=int(cfg.get("edge_dense_context_min_pad_pts", 10)),
			edge_dense_context_max_pad_pts=int(cfg.get("edge_dense_context_max_pad_pts", 80)),
			edge_use_enhanced_spike_mapping=bool(cfg.get("edge_use_enhanced_spike_mapping", False)),
			edge_mapping_levels_desc=tuple(int(v) for v in cfg.get("edge_mapping_levels_desc", [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5])),
			edge_mapping_refine_step_percent=int(cfg.get("edge_mapping_refine_step_percent", 1)),
			edge_mapping_min_level_percent=int(cfg.get("edge_mapping_min_level_percent", 1)),
			edge_mapping_require_closed_interval=bool(cfg.get("edge_mapping_require_closed_interval", True)),
			edge_mapping_use_apex_component=bool(cfg.get("edge_mapping_use_apex_component", True)),
			edge_mapping_enable_merge_guard=bool(cfg.get("edge_mapping_enable_merge_guard", True)),
			edge_mapping_max_width_jump_factor=float(cfg.get("edge_mapping_max_width_jump_factor", 2.5)),
			edge_mapping_max_width_jump_points=int(cfg.get("edge_mapping_max_width_jump_points", 8)),
			edge_mapping_fallback_to_old=bool(cfg.get("edge_mapping_fallback_to_old", False)),
			edge_mapping_noise_guard_enabled=bool(cfg.get("edge_mapping_noise_guard_enabled", False)),
			edge_robust_reference_enabled=bool(cfg.get("edge_robust_reference_enabled", True)),
			edge_enhanced_in_debug_report=bool(cfg.get("edge_enhanced_in_debug_report", True)),
			recdw_evidence_enabled=bool(cfg.get("recdw_evidence_enabled", True)),
			recdw_evidence_metrics=tuple(str(v) for v in cfg.get("recdw_evidence_metrics", ["recdw_sum_0_90"])),
			recdw_z_clip=float(cfg.get("recdw_z_clip", 6.0)),
			recdw_support_z_scale=float(cfg.get("recdw_support_z_scale", 1.0)),
			rucdw_enabled=bool(cfg.get("rucdw_enabled", True)),
			rucdw_context_pad_pts=int(cfg.get("rucdw_context_pad_pts", 20)),
			rucdw_context_max_pad_pts=int(cfg.get("rucdw_context_max_pad_pts", 80)),
			rucdw_levels=tuple(int(v) for v in cfg.get("rucdw_levels", [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90])),
			rucdw_min_snr=float(cfg.get("rucdw_min_snr", 1.0)),
			rucdw_noise_fallback_rel_amp=float(cfg.get("rucdw_noise_fallback_rel_amp", 0.05)),
			rucdw_anchor_mode=str(cfg.get("rucdw_anchor_mode", "max_in_candidate")),
			rucdw_baseline_mode=str(cfg.get("rucdw_baseline_mode", "context_low_percentile")),
			rucdw_baseline_percentile=float(cfg.get("rucdw_baseline_percentile", 5.0)),
			rucdw_z_clip=float(cfg.get("rucdw_z_clip", 6.0)),
			rucdw_support_z_scale=float(cfg.get("rucdw_support_z_scale", 1.0)),
			spike_score_v4_enabled=bool(cfg.get("ss4_enabled", cfg.get("spike_score_v4_enabled", True))),
			spike_score_v4_ss_blue_max=float(cfg.get("ss4_ss_blue_max", cfg.get("spike_score_v4_ss_blue_max", 0.95))),
			spike_score_v4_ss_red_min=float(cfg.get("ss4_ss_red_min", cfg.get("spike_score_v4_ss_red_min", 0.9999))),
			spike_score_v4_pce_red_min=float(cfg.get("ss4_pce_red_min", cfg.get("spike_score_v4_pce_red_min", 0.4))),
			spike_score_v4_edge_feature=str(cfg.get("ss4_rve_feature", cfg.get("spike_score_v4_edge_feature", "recdw_sum_0_90_raman_veto_evidence_signed"))),
			spike_score_v4_edge_red_min=float(cfg.get("ss4_rve_red_max", cfg.get("spike_score_v4_edge_red_min", -0.1))),
			spike_score_v4_pce_dead_zone_enabled=bool(cfg.get("ss4_pce_dead_zone_enabled", True)),
			spike_score_v4_pce_dead_zone_low=float(cfg.get("ss4_pce_dead_zone_low", -0.8)),
			spike_score_v4_pce_dead_zone_high=float(cfg.get("ss4_pce_dead_zone_high", -0.2)),
			spike_score_v4_missing_policy=str(cfg.get("ss4_missing_policy", cfg.get("spike_score_v4_missing_policy", "review"))),
			merge_max_width_pts=int(cfg.get("max_width_pts", 20)),
			labels_by_candidate=labels_by_candidate,
			show_progress=bool(cfg.get("debug_progress", True)),
			progress_print_every=int(cfg.get("debug_progress_print_every", 1000)),
		)

	def _prepare_decision_segments(y: int, x: int, segs: List[SpikeSegment]) -> List[SpikeSegment]:
		if not segs:
			return []
		n = int(raw.shape[2])
		grad_sig = overlays["gradient"][int(y), int(x), :].astype(float) if "gradient" in overlays else None
		bsrc = str(cfg.get("boundary_minimum_source", "gradient")).strip().lower()
		if bsrc == "gradient" and grad_sig is not None:
			boundary_sig = grad_sig
		else:
			boundary_sig = raw[int(y), int(x), :].astype(float)
		return prepare_primary_ss4_segments(
			y=int(y),
			x=int(x),
			segs=segs,
			feature_signal=(grad_sig if str(cfg.get("debug_feature_signal_source", "gradient")).strip().lower() != "raw" else raw[int(y), int(x), :].astype(float)),
			boundary_signal=boundary_sig,
			merge_signal=grad_sig,
			feature_expand_to_gradient_foot=bool(cfg.get("feature_expand_to_gradient_foot", True)),
			feature_foot_k_mad=float(cfg.get("feature_foot_k_mad", 2.0)),
			feature_foot_min_run=int(cfg.get("feature_foot_min_run", 2)),
			feature_window_method=cfg.get("feature_window_method", "mad_run"),
			feature_erosion_se_size=int(cfg.get("feature_erosion_se_size", 5)),
			merge_duplicate_segments=merge_dups,
			merge_max_width_pts=int(cfg.get("max_width_pts", 20)),
		)

	prepared_candidates_by_pixel: Dict[Tuple[int, int], List[SpikeSegment]] = {}
	for (y, x), segs in spikes_by_pixel.items():
		prepared = _prepare_decision_segments(int(y), int(x), list(segs))
		if prepared:
			prepared_candidates_by_pixel[(int(y), int(x))] = prepared

	ss4_input_candidates_by_pixel: Dict[Tuple[int, int], List[SpikeSegment]] = {}
	n_prefilter_before = 0
	n_prefilter_after = 0
	n_prefilter_rejected = 0
	n_prefilter_ratio_kept_mismatch = 0
	n_prefilter_ratio_rejected_mismatch = 0
	n_prefilter_not_evaluated = 0
	for (y, x), segs in prepared_candidates_by_pixel.items():
		raw_sig = raw[int(y), int(x), :].astype(float)
		small_morph = get_or_compute_small_morphology(
			cache=small_morphology_cache,
			raw_spectra=raw,
			y=int(y),
			x=int(x),
			window_size=3,
		)
		rows, summary = evaluate_candidate_noise_prefilter(
			y=int(y),
			x=int(x),
			segs=segs,
			raw_signal=raw_sig,
			small_morphology=small_morph,
			enabled=bool(cfg.get("candidate_noise_prefilter_enabled", True)),
			mode=str(cfg.get("candidate_noise_prefilter_mode", "morph_range_chord")),
			height_factor=float(cfg.get("candidate_noise_height_factor", cfg.get("candidate_noise_prefilter_sensitivity", 3.0))),
		)
		if rows:
			candidate_prefilter_rows_by_pixel[(int(y), int(x))] = [dict(row) for row in rows]
			ss4_candidate_metrics_by_pixel[(int(y), int(x))] = [dict(row) for row in rows]
		for row in rows:
			status = str(row.get("candidate_noise_prefilter_status", "not_evaluated") or "not_evaluated")
			try:
				ratio_v = float(row.get("candidate_noise_height_ratio", np.nan))
			except Exception:
				ratio_v = float("nan")
			try:
				factor_v = float(row.get("candidate_noise_height_factor", np.nan))
			except Exception:
				factor_v = float("nan")
			if status == "not_evaluated":
				n_prefilter_not_evaluated += 1
			elif np.isfinite(ratio_v) and np.isfinite(factor_v):
				if ratio_v < factor_v and status == "kept":
					n_prefilter_ratio_kept_mismatch += 1
				elif ratio_v >= factor_v and status == "rejected_noise":
					n_prefilter_ratio_rejected_mismatch += 1
		candidate_prefilter_summary_by_pixel[(int(y), int(x))] = dict(summary)
		kept: List[SpikeSegment] = []
		row_by_id = {str(row.get("candidate_id")): row for row in rows}
		for seg in segs:
			row = row_by_id.get(candidate_segment_id(seg))
			if row is None:
				kept.append(seg)
				continue
			if str(row.get("candidate_noise_prefilter_status", "kept")) != "rejected_noise":
				kept.append(seg)
		if kept:
			ss4_input_candidates_by_pixel[(int(y), int(x))] = kept
		n_prefilter_before += int(summary.get("n_candidates_before_noise_prefilter", 0))
		n_prefilter_after += int(summary.get("n_candidates_after_noise_prefilter", 0))
		n_prefilter_rejected += int(summary.get("n_candidates_rejected_by_noise_prefilter", 0))
	if n_prefilter_before:
		sufficient_specs = int(sum(1 for s in candidate_prefilter_summary_by_pixel.values() if str(s.get("noise_reference_status")) == "ok"))
		insufficient_specs = int(sum(1 for s in candidate_prefilter_summary_by_pixel.values() if str(s.get("noise_reference_status")) != "ok"))
		noise_vals = np.asarray([float(s.get("noise_height_morph_range", np.nan)) for s in candidate_prefilter_summary_by_pixel.values()], dtype=float)
		noise_vals = noise_vals[np.isfinite(noise_vals)]
		print(
			f"[candidate-noise] kept {n_prefilter_after}/{n_prefilter_before} prepared candidates "
			f"({n_prefilter_rejected} rejected as noise) | "
			f"spectra noise-ref ok={sufficient_specs} insufficient={insufficient_specs} | "
			f"median morph-range noise={float(np.median(noise_vals)) if noise_vals.size else float('nan'):.4g}"
		)
		print(
			f"[candidate-noise] ratio/status mismatches kept_below_factor={n_prefilter_ratio_kept_mismatch} "
			f"rejected_at_or_above_factor={n_prefilter_ratio_rejected_mismatch} "
			f"not_evaluated={n_prefilter_not_evaluated}"
		)

	decision_profile = str(cfg.get("decision_profile", "muon_rule_v3")).strip().lower()
	if decision_profile == "legacy_v3":
		decision_profile = "muon_rule_v3"
	if decision_profile not in {"muon_rule_v3", "ss4", "ss5"}:
		raise ValueError(f"Unsupported decision_profile: {decision_profile!r}")
	decision_action = str(cfg.get("despike_decision_action", "auto_only" if decision_profile == "muon_rule_v3" else "spike_only")).strip().lower()
	if decision_profile in {"ss4", "ss5"} and decision_action == "auto_only":
		decision_action = "spike_only"
	if decision_action not in {"auto_only", "auto_and_maybe", "none", "spike_only", "spike_and_review"}:
		raise ValueError(f"Unsupported despike_decision_action: {decision_action!r}")

	spike_decisions: Dict[SpikeSegment, Dict[str, Any]] = {}
	if decision_profile == "muon_rule_v3":
		for (y, x), segs in spikes_by_pixel.items():
			prepared = _prepare_decision_segments(int(y), int(x), list(segs))
			raw_sig = raw[int(y), int(x), :].astype(float)
			grad_sig = overlays["gradient"][int(y), int(x), :].astype(float) if "gradient" in overlays else None
			for original, prepared_seg in zip(segs, prepared):
				spike_decisions[original] = classify_segment_with_muon_rule_v3(
					raw_signal=raw_sig,
					gradient_signal=grad_sig,
					start=int(prepared_seg.start),
					end=int(prepared_seg.end),
					peak_index=int(prepared_seg.peak_index),
				)

	def _ss4_selected_spikes() -> List[SpikeSegment]:
		nonlocal report
		nonlocal primary_profile_overlap_summary
		if bool(cfg.get("ss4_fast_selection", True)):
			return _ss4_selected_spikes_fast()
		if report is None:
			t0_report = time.time()
			print("[debug-report] building per-spectrum feature report...")
			report = _build_current_report(include_per_spectrum=True)
			print(f"[debug-report] completed in {time.time() - t0_report:.2f} seconds")
		selected: List[SpikeSegment] = []
		missing = 0
		for spec in report.get("per_spectrum", []):
			if not isinstance(spec, dict):
				continue
			for sp in spec.get("spikes", []):
				if not isinstance(sp, dict):
					continue
				yv = int(spec.get("y", sp.get("y", -1)))
				xv = int(spec.get("x", sp.get("x", -1)))
				peak = int(sp.get("peak_index", -1))
				start = int(sp.get("start", sp.get("feature_window_start", -1)))
				end = int(sp.get("end", sp.get("feature_window_end", -1)))
				record = {
					"candidate_id": _primary_candidate_id(yv, xv, peak, start, end),
					"parent_id": _primary_candidate_id(yv, xv, peak, start, end),
					"y": yv,
					"x": xv,
					"peak_index": peak,
					"start": start,
					"end": end,
					"primary_ss4": sp.get("ss4"),
					"primary_ss4_reason": sp.get("ss4_reason"),
					"primary_ss4_decision": sp.get("ss4_decision"),
					"primary_ss4_rve_feature": sp.get("ss4_rve_feature"),
					"primary_spike_score_v5": sp.get("spike_score_v5", sp.get("ss5")),
					"primary_ss5": sp.get("ss5"),
					"primary_ss5_decision": sp.get("ss5_decision"),
					"primary_ss5_reason": sp.get("ss5_reason"),
					"primary_spike_score_v1": sp.get("spike_score_v1"),
					"primary_pce_negpref_t098_evidence_signed": sp.get("pce_negpref_t098_evidence_signed"),
					"primary_recdw_sum_0_90_raman_veto_evidence_signed": sp.get("recdw_sum_0_90_raman_veto_evidence_signed"),
					"ss4": sp.get("ss4"),
					"ss4_reason": sp.get("ss4_reason"),
					"ss4_decision": sp.get("ss4_decision"),
					"ss4_rve_feature": sp.get("ss4_rve_feature"),
					"spike_score_v5": sp.get("spike_score_v5", sp.get("ss5")),
					"ss5": sp.get("ss5"),
					"ss5_decision": sp.get("ss5_decision"),
					"ss5_reason": sp.get("ss5_reason"),
					"spike_score_v1": sp.get("spike_score_v1"),
					"pce_negpref_t098_evidence_signed": sp.get("pce_negpref_t098_evidence_signed"),
					"recdw_sum_0_90_raman_veto_evidence_signed": sp.get("recdw_sum_0_90_raman_veto_evidence_signed"),
					"primary_active_decision_profile": "ss5" if decision_profile == "ss5" else "ss4",
					"primary_active_score": sp.get("ss5") if decision_profile == "ss5" else sp.get("ss4"),
					"primary_active_decision": sp.get("ss5_decision") if decision_profile == "ss5" else sp.get("ss4_decision"),
					"primary_active_reason": sp.get("ss5_reason") if decision_profile == "ss5" else sp.get("ss4_reason"),
				}
				existing_rows = ss4_candidate_metrics_by_pixel.setdefault((yv, xv), [])
				merged = False
				for idx, row in enumerate(existing_rows):
					if str(row.get("candidate_id")) == record["candidate_id"]:
						existing_rows[idx] = _merge_candidate_rows(row, record)
						merged = True
						break
				if not merged:
					existing_rows.append(record)
				decision_value = sp.get("ss5") if decision_profile == "ss5" else sp.get(str(cfg.get("despike_decision_feature", "ss4")), sp.get("ss4"))
				try:
					v = float(decision_value)
				except Exception:
					missing += 1
					continue
				if not np.isfinite(v):
					missing += 1
					continue
				if bool(v >= 0.5):
					ss4_selected_metadata[(yv, xv, peak, start, end)] = {
						"candidate_id": _primary_candidate_id(yv, xv, peak, start, end),
						"parent_id": _primary_candidate_id(yv, xv, peak, start, end),
						"primary_ss4": float(v),
						"primary_ss4_reason": sp.get("ss4_reason"),
						"primary_ss4_decision": sp.get("ss4_decision"),
						"primary_ss4_rve_feature": sp.get("ss4_rve_feature"),
						"primary_spike_score_v5": sp.get("spike_score_v5", sp.get("ss5")),
						"primary_ss5": sp.get("ss5"),
						"primary_ss5_reason": sp.get("ss5_reason"),
						"primary_ss5_decision": sp.get("ss5_decision"),
						"primary_spike_score_v1": sp.get("spike_score_v1"),
						"primary_pce_negpref_t098_evidence_signed": sp.get("pce_negpref_t098_evidence_signed"),
						"primary_recdw_sum_0_90_raman_veto_evidence_signed": sp.get("recdw_sum_0_90_raman_veto_evidence_signed"),
						"ss4": float(v),
						"ss4_reason": sp.get("ss4_reason"),
						"ss4_decision": sp.get("ss4_decision"),
						"ss4_rve_feature": sp.get("ss4_rve_feature"),
						"spike_score_v5": sp.get("spike_score_v5", sp.get("ss5")),
						"ss5": sp.get("ss5"),
						"ss5_reason": sp.get("ss5_reason"),
						"ss5_decision": sp.get("ss5_decision"),
						"primary_active_decision_profile": "ss5" if decision_profile == "ss5" else "ss4",
						"primary_active_score": sp.get("ss5") if decision_profile == "ss5" else float(v),
						"primary_active_decision": sp.get("ss5_decision") if decision_profile == "ss5" else sp.get("ss4_decision"),
						"primary_active_reason": sp.get("ss5_reason") if decision_profile == "ss5" else sp.get("ss4_reason"),
						"spike_score_v1": sp.get("spike_score_v1"),
						"pce_negpref_t098_evidence_signed": sp.get("pce_negpref_t098_evidence_signed"),
						"recdw_sum_0_90_raman_veto_evidence_signed": sp.get("recdw_sum_0_90_raman_veto_evidence_signed"),
					}
					selected.append(
						SpikeSegment(
							y=yv,
							x=xv,
							peak_index=peak,
							start=start,
							end=end,
							peak_height=float(sp.get("peak_height", 0.0)),
							area=float(sp.get("area", 0.0)),
						)
					)
		apply_global_metric_ranks(row for rows in ss4_candidate_metrics_by_pixel.values() for row in rows)
		primary_profile_overlap_summary = _summarize_ss4_ss5_overlap(
			row for rows in ss4_candidate_metrics_by_pixel.values() for row in rows
		)
		if missing:
			print(f"[despike:ss4] skipped {missing} candidates with missing/non-finite ss4 decision value")
		print(
			"[primary-compare] "
			f"ss4={primary_profile_overlap_summary.get('ss4_spikes', 0)} "
			f"ss5={primary_profile_overlap_summary.get('ss5_spikes', 0)} "
			f"both={primary_profile_overlap_summary.get('accepted_by_both', 0)} "
			f"ss4_only={primary_profile_overlap_summary.get('accepted_by_ss4_only', 0)} "
			f"ss5_only={primary_profile_overlap_summary.get('accepted_by_ss5_only', 0)} "
			f"neither={primary_profile_overlap_summary.get('rejected_by_both', 0)}"
		)
		print(f"[despike:{decision_profile}] selected {len(selected)} accepted spike candidates")
		return selected

	contact_parent_spikes: List[SpikeSegment] = []
	despike_contact_analysis: Optional[Dict[str, Any]] = None
	if bool(cfg.get("despike_contact_analysis_enabled", True)):
		if decision_profile in {"ss4", "ss5"}:
			contact_parent_spikes = _ss4_selected_spikes() if decision_action in {"spike_only", "spike_and_review"} else []
		else:
			contact_parent_spikes = [
				s for s in spikes
				if str(spike_decisions.get(s, {}).get("muon_rule_v3_decision", "no_muon")) == "auto_muon"
			]
		despike_contact_analysis = analyze_erosion_dilation_contact_cells(
			x_axis=x_axis,
			raw_spectra=raw,
			erosion=overlays["erosion"],
			dilation=overlays["dilation"],
			gradient=overlays.get("gradient"),
			small_morphology_by_pixel=small_morphology_cache,
			parents=contact_parent_spikes,
			parent_metadata=ss4_selected_metadata,
			context_pad_pts=int(cfg.get("despike_contact_context_pad_pts", 4)),
			strict_equal=bool(cfg.get("despike_contact_strict_equal", True)),
			despike_sensitivity=float(cfg.get("despike_sensitivity", 1.0)),
			secondary_noise_thr=float(cfg.get("secondary_noise_thr", 0.05)),
			secondary_uncertain_thr=float(cfg.get("secondary_uncertain_thr", 0.5)),
			ss4_ss_blue_max=float(cfg.get("ss4_ss_blue_max", cfg.get("spike_score_v4_ss_blue_max", 0.95))),
			ss4_ss_red_min=float(cfg.get("ss4_ss_red_min", cfg.get("spike_score_v4_ss_red_min", 0.9999))),
			ss4_pce_red_min=float(cfg.get("ss4_pce_red_min", cfg.get("spike_score_v4_pce_red_min", 0.4))),
			ss4_rve_red_max=float(cfg.get("ss4_rve_red_max", cfg.get("spike_score_v4_edge_red_min", -0.1))),
			ss4_missing_policy=str(cfg.get("ss4_missing_policy", cfg.get("spike_score_v4_missing_policy", "review"))),
			secondary_edge_rescue_ss_min=float(cfg.get("secondary_edge_rescue_ss_min", 0.85)),
			candidate_noise_height_factor=float(cfg.get("candidate_noise_height_factor", cfg.get("candidate_noise_prefilter_sensitivity", 3.0))),
		)
		corrected = apply_contact_cell_despike_chords(raw, despike_contact_analysis)
		if report is not None:
			report["despike_contact_analysis_summary"] = {
				"method": despike_contact_analysis.get("method"),
				"n_parent_segments": despike_contact_analysis.get("n_parent_segments"),
				"debug_json_path": str(despike_debug_lite_path or ""),
			}


	reps = time.time()
	print(f"Despike contact analysis completed in {reps - comparison:.2f} seconds")

	# 5) Despike
	if bool(cfg['despike_enabled']) and despike_contact_analysis is None:
		if decision_action == "none":
			despike_spikes: List[SpikeSegment] = []
		elif decision_profile in {"ss4", "ss5"}:
			despike_spikes = list(contact_parent_spikes) if contact_parent_spikes else (_ss4_selected_spikes() if decision_action in {"spike_only", "spike_and_review"} else [])
		elif decision_action == "auto_and_maybe":
			despike_spikes = [
				s for s in spikes
				if str(spike_decisions.get(s, {}).get("muon_rule_v3_decision", "no_muon")) in {"auto_muon", "maybe_muon"}
			]
		else:
			despike_spikes = [
				s for s in spikes
				if str(spike_decisions.get(s, {}).get("muon_rule_v3_decision", "no_muon")) == "auto_muon"
			]
		corrected, despike_diagnostics = apply_despike(
			x_axis=x_axis,
			spectra=raw,
			accepted_spikes=despike_spikes,
			method=str(cfg.get("despike_method", "segment_linear_expanded")),
			fallback_method=str(cfg.get("despike_fallback_method", "simple_linear_existing")),
			allow_fallback=bool(cfg.get("despike_allow_fallback", True)),
			return_diagnostics=True,
			expand_left_pts=int(cfg.get("despike_expand_left_pts", 1)),
			expand_right_pts=int(cfg.get("despike_expand_right_pts", 1)),
			preserve_anchor_points=bool(cfg.get("despike_preserve_anchor_points", False)),
			patch_method=str(cfg.get("despike_patch_method", "plain_line_capped")),
			cap_strategy=str(cfg.get("despike_cap_strategy", "vertical_shift")),
			auto_expand_edges=bool(cfg.get("despike_auto_expand_edges", True)),
			auto_expand_max_pts=int(cfg.get("despike_auto_expand_max_pts", 3)),
			auto_expand_rel=float(cfg.get("despike_auto_expand_rel", 0.10)),
			auto_expand_k_mad=float(cfg.get("despike_auto_expand_k_mad", 2.0)),
			support_mode=str(cfg.get("despike_support_mode", "apex_component")),
			segment_as_max_bounds=bool(cfg.get("despike_segment_as_max_bounds", True)),
			support_min_width=int(cfg.get("despike_support_min_width", 1)),
			support_max_width=int(cfg.get("despike_support_max_width", 12)),
			group_supports_before_patch=bool(cfg.get("despike_group_supports_before_patch", True)),
			support_group_gap_pts=int(cfg.get("despike_support_group_gap_pts", 2)),
			support_group_max_width=int(cfg.get("despike_support_group_max_width", 28)),
			support_edge_expand_pts=int(cfg.get("despike_support_edge_expand_pts", 1)),
			anchor_avoid_all_supports=bool(cfg.get("despike_anchor_avoid_all_supports", True)),
			anchor_max_shift_pts=int(cfg.get("despike_anchor_max_shift_pts", 4)),
			use_external_anchor_zones=bool(cfg.get("despike_use_external_anchor_zones", True)),
			enforce_not_above_raw=bool(cfg.get("despike_enforce_not_above_raw", True)),
			max_overshoot_eps=float(cfg.get("despike_max_overshoot_eps", 0.0)),
			context_pad_pts=int(cfg.get("despike_context_pad_pts", 8)),
			baseline_method=str(cfg.get("despike_baseline_method", "opening")),
			baseline_width=int(cfg.get("despike_baseline_width", 9)),
			high_rel=float(cfg.get("despike_high_rel", 0.35)),
			low_rel=float(cfg.get("despike_support_low_rel", cfg.get("despike_low_rel", 0.03))),
			high_k_mad=float(cfg.get("despike_high_k_mad", 5.0)),
			low_k_mad=float(cfg.get("despike_support_low_k_mad", cfg.get("despike_low_k_mad", 1.5))),
			merge_gap_pts=int(cfg.get("despike_merge_gap_pts", 2)),
			min_support_width=int(cfg.get("despike_min_support_width", 1)),
			max_support_width=int(cfg.get("despike_max_support_width", 20)),
			anchor_pad_pts=int(cfg.get("despike_anchor_pad_pts", 1)),
			anchor_width_pts=int(cfg.get("despike_anchor_width_pts", 3)),
			guard_tolerance_k_mad=float(cfg.get("despike_guard_tolerance_k_mad", 1.0)),
			feather_pts=int(cfg.get("despike_feather_pts", 0)),
			group_merge_gap_pts=int(cfg.get("despike_group_merge_gap_pts", 2)),
			group_merge_peak_distance_pts=int(cfg.get("despike_group_merge_peak_distance_pts", 6)),
		)
		if report is not None:
			report["despike_diagnostics"] = despike_diagnostics
			report["despike_summary"] = {
				"decision_profile": decision_profile,
				"decision_action": decision_action,
				"selected_count": int(len(despike_spikes)),
				"diagnostics_count": int(len(despike_diagnostics)),
				"applied_count": int(sum(1 for d in despike_diagnostics if not str(d.get("skipped_reason", "")).strip())),
				"fallback_count": int(sum(1 for d in despike_diagnostics if bool(d.get("fallback_used", False)))),
				"primary_profile_comparison": dict(primary_profile_overlap_summary),
			}

	despike = time.time()
	print(f"Despike contact analysis completed in {despike - reps:.2f} seconds")

	# 6) Viewer (hover)
	view_x = x_axis
	view_spectra = raw
	view_score = score_map
	view_mask = candidate_mask
	view_overlays = overlays
	view_spikes_by_pixel = prepared_candidates_by_pixel
	source_coords_map = None
	has_corrected = bool(cfg['despike_enabled']) or despike_contact_analysis is not None
	view_corrected = corrected if has_corrected else None

	if target_coords and bool(cfg['use_compact_coords_view']):
		(
			view_spectra,
			view_score,
			view_mask,
			view_overlays,
			view_spikes_by_pixel,
			source_coords_map,
			view_corrected,
		) = build_compact_subset(
			x_axis=view_x,
			raw_spectra=view_spectra,
			score_map=view_score,
			candidate_mask=view_mask,
			overlays=view_overlays,
			spikes_by_pixel=view_spikes_by_pixel,
			coords=target_coords,
			corrected_spectra=view_corrected,
		)

	source_to_compact = _invert_coord_map(source_coords_map)
	viewer_contact_analysis = (
		build_despike_contact_debug_payload(
			despike_contact_analysis,
			mode="full",
			store_arrays=False,
			source_to_compact=source_to_compact,
		)
		if isinstance(despike_contact_analysis, dict)
		else None
	)
	viewer_diagnostics: List[Dict[str, Any]] = []
	if isinstance(report, dict):
		viewer_diagnostics = list(report.get("despike_diagnostics", []) or [])

	# 7) Save data and reports
	if cfg.get("save_npz_path"):
		save_result_npz(
			out_path=Path(cfg['save_npz_path']),
			ds=ds,
			score_map=score_map,
			threshold=float(thr),
			candidate_mask=candidate_mask,
			spikes=spikes,
			corrected_spectra=corrected if (bool(cfg['save_corrected_in_npz']) and has_corrected) else None,
			overlays=overlays if bool(cfg['save_overlays_in_npz']) else None,
		)

	if cfg.get("save_spikes_csv_path"):
		save_spikes_csv(path=Path(cfg["save_spikes_csv_path"]), spikes=spikes)

	if despike_debug_lite_enabled and despike_contact_analysis is not None and despike_debug_lite_path:
		lite_path = _ensure_parent_dir(despike_debug_lite_path)
		if lite_path is not None:
			lite_payload = build_despike_contact_debug_payload(
				despike_contact_analysis,
				mode="lite",
				store_arrays=False,
				source_to_compact=source_to_compact,
			)
			save_despike_contact_debug_json(lite_path, lite_payload)
			print(f"Lightweight despike debug written to {lite_path}")
	else:
		print("Lightweight despike debug disabled.")

	if despike_debug_full_enabled and despike_contact_analysis is not None and despike_debug_full_path:
		full_path = _ensure_parent_dir(despike_debug_full_path)
		if full_path is not None:
			full_payload = build_despike_contact_debug_payload(
				despike_contact_analysis,
				mode="full",
				store_arrays=debug_store_arrays,
				source_to_compact=source_to_compact,
			)
			save_despike_contact_debug_json(full_path, full_payload)
			print(f"Full despike debug written to {full_path}")
	else:
		print("Full debug disabled.")

	if debug_feature_report_enabled and debug_feature_report_path:
		report_path = _ensure_parent_dir(debug_feature_report_path)
		if report is None:
			report = _build_current_report()
		if report_path is not None:
			save_debug_report_json(report_path, report)
			print(f"Debug feature report written to {report_path}")
	else:
		print("Debug feature report disabled.")

	if save_viewer_cache_enabled and viewer_cache_path:
		cache_path = _ensure_parent_dir(viewer_cache_path)
		if cache_path is not None:
			save_viewer_cache(
				cache_path,
				x_axis=view_x,
				spectra=view_spectra,
				score_map=view_score,
				candidate_mask=view_mask,
				spikes_by_pixel=view_spikes_by_pixel,
				overlays=view_overlays,
				corrected_spectra=view_corrected,
				source_coords_map=source_coords_map,
				despike_contact_analysis=viewer_contact_analysis,
				ss4_candidate_metrics_by_pixel=ss4_candidate_metrics_by_pixel,
				despike_diagnostics=viewer_diagnostics,
				viewer_status_text="PREVIEW FROM CACHE - pipeline was not run now",
			)
			print(f"Viewer cache written to {cache_path}")
	else:
		print("Viewer cache disabled.")

	calib_dir = cfg.get("threshold_calibration_dir")
	if not calib_dir:
		calib_dir = str(Path("outputs") / "threshold_calibration")
	_run_threshold_calibration(
		rows=_flatten_ss4_candidate_rows(ss4_candidate_metrics_by_pixel),
		cfg=cfg,
		base_dir=Path(str(calib_dir)),
	)

	show_hover_map(
		x_axis=view_x,
		spectra=view_spectra,
		score_map=view_score,
		candidate_mask=view_mask,
		spikes_by_pixel=view_spikes_by_pixel,
		overlays=view_overlays,
		source_coords_map=source_coords_map,
		corrected_spectra=view_corrected,
		despike_diagnostics=viewer_diagnostics,
		despike_contact_analysis=viewer_contact_analysis,
		ss4_candidate_metrics_by_pixel=ss4_candidate_metrics_by_pixel,
		initial_checked={
			"raw": True,
			"top_hat": False,
			"gradient": False,
			"dilation_minus_opening": False,
			"corrected": True,
		},
		merge_duplicate_segments=merge_dups,
		feature_expand_to_gradient_foot=bool(cfg.get("feature_expand_to_gradient_foot", True)),
		feature_foot_k_mad=float(cfg.get("feature_foot_k_mad", 2.0)),
		feature_foot_min_run=int(cfg.get("feature_foot_min_run", 2)),
		feature_window_method=cfg.get("feature_window_method", "mad_run"),
		feature_erosion_se_size=int(cfg.get("feature_erosion_se_size", 5)),
		boundary_minimum_source=cfg.get("boundary_minimum_source", "gradient"),
		gws_split_overlapping_contexts=bool(cfg.get("gws_split_overlapping_contexts", False)),
		gws_split_source=str(cfg.get("gws_split_source", "gradient")),
		gws_split_smooth_pts=int(cfg.get("gws_split_smooth_pts", 3)),
		gws_split_valley_alpha=float(cfg.get("gws_split_valley_alpha", 0.75)),
		gws_split_min_distance_from_apex=int(cfg.get("gws_split_min_distance_from_apex", 1)),
		gws_split_min_context_width=int(cfg.get("gws_split_min_context_width", 3)),
		gws_split_debug=bool(cfg.get("gws_split_debug", True)),
		gws_measure_region=str(cfg.get("gws_measure_region", "mask")),
		gws_threshold_region=str(cfg.get("gws_threshold_region", "spike_edges")),
		edge_dense_levels=tuple(int(v) for v in cfg.get("edge_dense_levels", [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])),
		edge_dense_min_snr=float(cfg.get("edge_dense_min_snr", 1.0)),
		edge_dense_context_pad_pts=int(cfg.get("edge_dense_context_pad_pts", 20)),
		edge_dense_context_min_pad_pts=int(cfg.get("edge_dense_context_min_pad_pts", 10)),
		edge_dense_context_max_pad_pts=int(cfg.get("edge_dense_context_max_pad_pts", 80)),
		edge_use_enhanced_spike_mapping=bool(cfg.get("edge_use_enhanced_spike_mapping", False)),
		edge_mapping_levels_desc=tuple(int(v) for v in cfg.get("edge_mapping_levels_desc", [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5])),
		edge_mapping_refine_step_percent=int(cfg.get("edge_mapping_refine_step_percent", 1)),
		edge_mapping_min_level_percent=int(cfg.get("edge_mapping_min_level_percent", 1)),
		edge_mapping_require_closed_interval=bool(cfg.get("edge_mapping_require_closed_interval", True)),
		edge_mapping_use_apex_component=bool(cfg.get("edge_mapping_use_apex_component", True)),
		edge_mapping_enable_merge_guard=bool(cfg.get("edge_mapping_enable_merge_guard", True)),
		edge_mapping_max_width_jump_factor=float(cfg.get("edge_mapping_max_width_jump_factor", 2.5)),
		edge_mapping_max_width_jump_points=int(cfg.get("edge_mapping_max_width_jump_points", 8)),
		edge_mapping_fallback_to_old=bool(cfg.get("edge_mapping_fallback_to_old", False)),
		edge_mapping_noise_guard_enabled=bool(cfg.get("edge_mapping_noise_guard_enabled", False)),
		rucdw_enabled=bool(cfg.get("rucdw_enabled", True)),
		rucdw_context_pad_pts=int(cfg.get("rucdw_context_pad_pts", 20)),
		rucdw_context_max_pad_pts=int(cfg.get("rucdw_context_max_pad_pts", 80)),
		rucdw_levels=tuple(int(v) for v in cfg.get("rucdw_levels", [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90])),
		rucdw_min_snr=float(cfg.get("rucdw_min_snr", 1.0)),
		rucdw_noise_fallback_rel_amp=float(cfg.get("rucdw_noise_fallback_rel_amp", 0.05)),
		rucdw_anchor_mode=str(cfg.get("rucdw_anchor_mode", "max_in_candidate")),
		rucdw_baseline_mode=str(cfg.get("rucdw_baseline_mode", "context_low_percentile")),
		rucdw_baseline_percentile=float(cfg.get("rucdw_baseline_percentile", 5.0)),
		ball_context_pad_pts=int(cfg.get("ball_context_pad_pts", 20)),
		ball_context_min_pad_pts=int(cfg.get("ball_context_min_pad_pts", 10)),
		ball_context_max_pad_pts=int(cfg.get("ball_context_max_pad_pts", 80)),
		ball_stop_k_mad=float(cfg.get("ball_stop_k_mad", 1.0)),
		ball_stop_rel_amp=float(cfg.get("ball_stop_rel_amp", 0.05)),
		ball_prevent_crossing_neighbor_peak=bool(cfg.get("ball_prevent_crossing_neighbor_peak", True)),
		exp_context_pad_pts=int(cfg.get("exp_context_pad_pts", 20)),
		exp_foot_low_rel=float(cfg.get("exp_foot_low_rel", 0.05)),
		exp_foot_high_rel=float(cfg.get("exp_foot_high_rel", 0.45)),
		exp_foot_noise_k_mad=float(cfg.get("exp_foot_noise_k_mad", 1.0)),
		exp_min_points=int(cfg.get("exp_min_points", 3)),
		exp_prevent_apex_region=bool(cfg.get("exp_prevent_apex_region", True)),
		despike_contact_candidates_enabled=bool(cfg.get("despike_contact_candidates_enabled", True)),
		despike_contact_context_pad_pts=int(cfg.get("despike_contact_context_pad_pts", 4)),
		merge_max_width_pts=int(cfg.get("max_width_pts", 20)),
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Muon finder")
	parser.add_argument("--config", type=Path, default=None, help="Path to JSON config file.")
	parser.add_argument("--input", type=Path, default=None, help="Input .wdf or .npz path (overrides config).")
	parser.add_argument("--coords-csv", type=Path, default=None, help="CSV with columns y,x for targeted spectra.")
	args = parser.parse_args()

	cfg = load_config(args.config)
	if args.input is not None:
		cfg['input_path'] = str(args.input)
	if args.coords_csv is not None:
		cfg['coords_csv'] = str(args.coords_csv)
	if not cfg.get("input_path"):
		raise ValueError("Set input path via config 'input_path' or CLI --input.")

	run(cfg)


if __name__ == "__main__":
	main()
