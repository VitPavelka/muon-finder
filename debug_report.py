# debug_report.py
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Literal, Sequence

import numpy as np

from muon_pipeline import SpikeSegment
from primary_candidate_preparation import prepare_primary_ss4_segments
from feature_window import expand_interval_to_signal_foot
from feature_discrimination import (
	compute_peak_curvature_features,
	compute_peak_curvature_d3_features,
	compute_support_after_core_removal_features,
	compute_curvature_support_features,
	compute_local_shape_features,
	compute_gws_context_infos,
	compute_gws_diagnostics,
	compute_gws_feature_bundle_from_diagnostics,
	compute_multiscale_tophat_features,
	compute_gws_granulometric_shape_features,
	filter_multiscale_tophat_export_features,
	compute_residual_component_features,
	compute_optional_foot_shape_features,
	compute_residual_anatomy_features,
	compute_spike_score_v2_features,
	compute_context_window_experiment_features,
	compute_candidate_tophat_shape_metrics,
	compute_edge_width_metrics,
	compute_edge_dense_width_metrics,
	compute_raw_upper_component_dense_width_metrics,
	compute_ball_descent_metrics,
	compute_exponential_decay_metrics,
	compute_shape_top_hat_signal,
	CONTEXT_PADS,
	EDGE_WIDTH_ACTIVE_METRICS,
	GWS_EXPERIMENT_SOURCE_MODES,
	GWS_COMPACT_ACTIVE_METRICS,
	GWS_MEASURE_REGION,
	GWS_THRESHOLD_REGION,
	TH_SHAPE_ACTIVE_METRICS,
	GWS_SOURCE_PREFIX_BY_MODE,
	GWS_GRANULO_SCALES,
	GWS_INCLUDE_SCALE_ZERO,
	GWS_SPLIT_DEBUG,
	GWS_SPLIT_MIN_CONTEXT_WIDTH,
	GWS_SPLIT_MIN_DISTANCE_FROM_APEX,
	GWS_SPLIT_OVERLAPPING_CONTEXTS,
	GWS_SPLIT_SMOOTH_PTS,
	GWS_SPLIT_SOURCE,
	GWS_SPLIT_VALLEY_ALPHA,
)
from muon_decision import (
	annotate_feature_dict_with_muon_rule_v3,
	annotate_feature_dict_with_spike_score_v4,
)


def build_debug_report(
		score_map: np.ndarray,
		candidate_mask: np.ndarray,
		spikes_by_pixel: Dict[Tuple[int, int], List[SpikeSegment]],
		threshold: float,
		target_coords: Optional[List[Tuple[int, int]]] = None,
		include_per_spectrum: bool = False,
		max_top_pixels: int = 25,
		raw_spectra: Optional[np.ndarray] = None,  # (H,W,N)
		overlays: Optional[Dict[str, np.ndarray]] = None,  # expects at least 'gradient' if available
		x_axis: Optional[np.ndarray] = None,
		feature_signal_source: str = "gradient",  # gradient | raw
		merge_duplicate_segments: bool = False,
		feature_expand_to_gradient_foot: bool = True,
		feature_foot_k_mad: float = 2.0,
		feature_foot_min_run: int = 2,
		feature_window_method: Literal["mad_run", "erosion_touch"] = "mad_run",
		feature_erosion_se_size: int = 5,
		boundary_minimum_source: Literal["raw", "gradient"] = "gradient",
		gws_split_overlapping_contexts: bool = GWS_SPLIT_OVERLAPPING_CONTEXTS,
		gws_split_source: str = GWS_SPLIT_SOURCE,
		gws_split_smooth_pts: int = GWS_SPLIT_SMOOTH_PTS,
		gws_split_valley_alpha: float = GWS_SPLIT_VALLEY_ALPHA,
		gws_split_min_distance_from_apex: int = GWS_SPLIT_MIN_DISTANCE_FROM_APEX,
		gws_split_min_context_width: int = GWS_SPLIT_MIN_CONTEXT_WIDTH,
		gws_split_debug: bool = GWS_SPLIT_DEBUG,
		gws_source_modes: Sequence[str] = GWS_EXPERIMENT_SOURCE_MODES,
		gws_include_scale_zero: bool = GWS_INCLUDE_SCALE_ZERO,
		gws_measure_region: str = GWS_MEASURE_REGION,
		gws_threshold_region: str = GWS_THRESHOLD_REGION,
		edge_dense_enabled: bool = True,
		edge_dense_levels: Sequence[int] = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95),
		edge_dense_min_snr: float = 1.0,
		edge_dense_context_pad_pts: int = 20,
		edge_dense_context_min_pad_pts: int = 10,
		edge_dense_context_max_pad_pts: int = 80,
		edge_use_enhanced_spike_mapping: bool = False,
		edge_mapping_levels_desc: Sequence[int] = (95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5),
		edge_mapping_refine_step_percent: int = 1,
		edge_mapping_min_level_percent: int = 1,
		edge_mapping_require_closed_interval: bool = True,
		edge_mapping_use_apex_component: bool = True,
		edge_mapping_enable_merge_guard: bool = True,
		edge_mapping_max_width_jump_factor: float = 2.5,
		edge_mapping_max_width_jump_points: int = 8,
		edge_mapping_fallback_to_old: bool = False,
		edge_mapping_noise_guard_enabled: bool = False,
		recdw_evidence_enabled: bool = True,
		recdw_evidence_metrics: Sequence[str] = ("recdw_sum_0_90",),
		recdw_z_clip: float = 6.0,
		recdw_support_z_scale: float = 1.0,
		rucdw_enabled: bool = True,
		rucdw_context_pad_pts: int = 20,
		rucdw_context_max_pad_pts: int = 80,
		rucdw_levels: Sequence[int] = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90),
		rucdw_min_snr: float = 1.0,
		rucdw_noise_fallback_rel_amp: float = 0.05,
		rucdw_anchor_mode: str = "max_in_candidate",
		rucdw_baseline_mode: str = "context_low_percentile",
		rucdw_baseline_percentile: float = 5.0,
		rucdw_z_clip: float = 6.0,
		rucdw_support_z_scale: float = 1.0,
		spike_score_v4_enabled: bool = True,
		spike_score_v4_ss_blue_max: float = 0.95,
		spike_score_v4_ss_red_min: float = 0.9999,
		spike_score_v4_pce_red_min: float = 0.4,
		spike_score_v4_edge_feature: str = "recdw_sum_0_90_raman_veto_evidence_signed",
		spike_score_v4_edge_red_min: float = -0.1,
		spike_score_v4_missing_policy: str = "review",
		ball_context_pad_pts: int = 20,
		ball_context_min_pad_pts: int = 10,
		ball_context_max_pad_pts: int = 80,
		ball_stop_k_mad: float = 1.0,
		ball_stop_rel_amp: float = 0.05,
		ball_prevent_crossing_neighbor_peak: bool = True,
		exp_context_pad_pts: int = 20,
		exp_foot_low_rel: float = 0.05,
		exp_foot_high_rel: float = 0.45,
		exp_foot_noise_k_mad: float = 1.0,
		exp_min_points: int = 3,
		exp_prevent_apex_region: bool = True,
		merge_max_width_pts: Optional[int] = None,
		labels_by_candidate: Optional[Dict[Tuple[int, int, int], bool]] = None,
		show_progress: bool = False,
		edge_enhanced_in_debug_report: bool = True,
		progress_print_every: int = 1000,
) -> Dict[str, Any]:
	EXPORT_KEYWORD_EXCLUDES = ("curv2", "opres")
	EXPORT_ALWAYS_KEEP = {
		"width_pts",
		"peak_index",
		"start",
		"end",
		"peak_height",
		"area",
		"raw_peak_value",
		"feature_source",
		"feature_window_start",
		"feature_window_end",
		"peak_position_cm1",
		"is_muon_label",
		"muon_rule_v3_decision",
		"muon_rule_v3_reason",
		"muon_rule_v3_value",
		"muon_rule_v3_score",
		"ss_category_v3",
		"pce_category_v3",
		"ss4_decision",
		"ss4_reason",
		"ss4_ss_zone",
		"ss4_pce_zone",
		"ss4_rve_zone",
		"ss4_rve_feature",
		"spike_score_v4_decision",
		"spike_score_v4_reason",
		"spike_score_v4_ss_zone",
		"spike_score_v4_pce_zone",
		"spike_score_v4_edge_zone",
		"spike_score_v4_edge_feature",
	}

	def _add_edge_width_aliases(dst: Dict[str, Any], metrics: Dict[str, object], *, source_prefix: str, alias_prefix: str) -> None:
		dense_prefix = f"{source_prefix}_dense_width_"
		width_prefix = f"{source_prefix}_width_at_"
		for key, value in metrics.items():
			name = str(key)
			if name.endswith("_debug"):
				continue
			if name.startswith(dense_prefix):
				dst[f"{alias_prefix}_{name[len(dense_prefix):]}"] = value
				continue
			if name.startswith(width_prefix):
				level = name[len(width_prefix):]
				if level in {"5", "10", "15", "20", "25"}:
					dst[f"{alias_prefix}_at_{level}"] = value

	def _is_pipeline_export_metric(key: str) -> bool:
		name = str(key).strip()
		nm = name.lower()
		if name in EXPORT_ALWAYS_KEEP:
			return True
		if any(token in nm for token in EXPORT_KEYWORD_EXCLUDES):
			return False
		if nm.endswith("_debug"):
			return False
		if name.startswith("spike_score_v1"):
			return True
		if nm.startswith("rise_slope") or nm.startswith("fall_slope"):
			return True
		if nm.startswith("ss_"):
			return True
		if nm == "ss4" or nm.startswith("ss4_"):
			return True
		if nm.startswith("spike_score_v4"):
			return True
		if name == "pce_negpref_t098_evidence_signed":
			return True
		if name in {
			"peak_curvature_extreme",
			"peak_curvature_extreme_z",
			"peak_curvature_extreme_negpref_t098",
			"peak_curvature_extreme_negpref_t098_z",
			"peak_curvature_extreme_negpref_t098_switched",
		}:
			return True
		if nm.startswith("pce_negpref_t098_"):
			return True
		if name in {
			"recdw_sum_0_90",
			"recdw_sum_0_90_z",
			"recdw_sum_0_90_support01",
			"recdw_sum_0_90_raman_veto_evidence_signed",
		}:
			return True
		if name in {
			"rucdw_sum_0_90",
			"rucdw_sum_0_90_z",
			"rucdw_sum_0_90_support01",
			"rucdw_sum_0_90_raman_veto_evidence_signed",
			"rucdw_valid_n_0_90",
		}:
			return True
		return False

	def _filter_export_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
		return {
			key: value
			for key, value in row.items()
			if _is_pipeline_export_metric(str(key))
		}

	def _robust_center_scale(values: np.ndarray) -> Tuple[float, float, str]:
		x = np.asarray(values, dtype=float)
		x = x[np.isfinite(x)]
		if x.size == 0:
			return float("nan"), float("nan"), "none"
		center = float(np.median(x))
		mad = float(np.median(np.abs(x - center)))
		scale = float(1.4826 * mad)
		if np.isfinite(scale) and scale > 1e-12:
			return center, scale, "mad"
		q25, q75 = np.percentile(x, [25.0, 75.0])
		scale = float((float(q75) - float(q25)) / 1.349)
		if np.isfinite(scale) and scale > 1e-12:
			return center, scale, "iqr"
		scale = float(np.std(x))
		if np.isfinite(scale) and scale > 1e-12:
			return center, scale, "std"
		return center, float("nan"), "invalid"

	def _sigmoid_support(z: float, scale: float, clip: float) -> float:
		if not np.isfinite(z):
			return float("nan")
		z_clip = float(np.clip(float(z), -abs(float(clip)), abs(float(clip))))
		s = max(float(scale), 1e-12)
		return float(1.0 / (1.0 + np.exp(-z_clip / s)))

	def _add_recdw_evidence(per_spec_rows: List[Dict[str, Any]]) -> None:
		if not bool(recdw_evidence_enabled):
			return
		enabled = {str(v) for v in recdw_evidence_metrics}
		if bool(rucdw_enabled):
			enabled.add("rucdw_sum_0_90")
		targets = {
			"recdw_sum_0_90": {
				"z": "recdw_sum_0_90_z",
				"support": "recdw_sum_0_90_support01",
				"positive": "recdw_sum_0_90_raman_veto_evidence_signed",
				"negative": "",
			},
			"recdw_initial_0_90": {
				"z": "recdw_initial_0_90_z",
				"support": "recdw_initial_0_90_support01",
				"positive": "recdw_initial_0_90_structure_evidence_signed",
				"negative": "recdw_initial_0_90_noise_evidence_signed",
			},
			"rucdw_sum_0_90": {
				"z": "rucdw_sum_0_90_z",
				"support": "rucdw_sum_0_90_support01",
				"positive": "rucdw_sum_0_90_raman_veto_evidence_signed",
				"negative": "",
			},
		}
		spikes: List[Dict[str, Any]] = []
		for spec in per_spec_rows:
			for spike in spec.get("spikes", []):
				if isinstance(spike, dict):
					spikes.append(spike)
		diag: Dict[str, Any] = {}
		for raw_name, names in targets.items():
			if raw_name not in enabled:
				continue
			vals = np.asarray([float(sp.get(raw_name, np.nan)) for sp in spikes], dtype=float)
			center, scale, method = _robust_center_scale(vals)
			valid_n = int(np.count_nonzero(np.isfinite(vals)))
			diag[raw_name] = {
				"center": float(center) if np.isfinite(center) else float("nan"),
				"scale": float(scale) if np.isfinite(scale) else float("nan"),
				"scale_method": method,
				"valid_n": int(valid_n),
				"z_clip": float(rucdw_z_clip if raw_name.startswith("rucdw_") else recdw_z_clip),
				"support_z_scale": float(rucdw_support_z_scale if raw_name.startswith("rucdw_") else recdw_support_z_scale),
			}
			for sp in spikes:
				xv = float(sp.get(raw_name, np.nan))
				if np.isfinite(xv) and np.isfinite(center) and np.isfinite(scale) and scale > 1e-12:
					z = float((xv - center) / scale)
					if raw_name.startswith("rucdw_"):
						support = _sigmoid_support(z, float(rucdw_support_z_scale), float(rucdw_z_clip))
					else:
						support = _sigmoid_support(z, float(recdw_support_z_scale), float(recdw_z_clip))
					evidence = float(2.0 * support - 1.0) if np.isfinite(support) else float("nan")
				else:
					z = float("nan")
					support = float("nan")
					evidence = float("nan")
				sp[names["z"]] = z
				sp[names["support"]] = support
				sp[names["positive"]] = evidence
				if names.get("negative"):
					sp[names["negative"]] = -evidence if np.isfinite(evidence) else float("nan")
		if diag:
			report["recdw_evidence_diagnostics"] = diag

	def _add_spike_score_v4(per_spec_rows: List[Dict[str, Any]]) -> None:
		if not bool(spike_score_v4_enabled):
			return
		for spec in per_spec_rows:
			for spike in spec.get("spikes", []):
				if not isinstance(spike, dict):
					continue
				spike.update(
					annotate_feature_dict_with_spike_score_v4(
						spike,
						edge_feature=str(spike_score_v4_edge_feature),
						ss_blue_max=float(spike_score_v4_ss_blue_max),
						ss_red_min=float(spike_score_v4_ss_red_min),
						pce_red_min=float(spike_score_v4_pce_red_min),
						edge_red_min=float(spike_score_v4_edge_red_min),
						missing_policy=str(spike_score_v4_missing_policy),
					)
				)

	def _feature_source_signal(y: int, x: int) -> Optional[np.ndarray]:
		src = str(feature_signal_source).lower().strip()
		if src == "raw":
			if raw_spectra is None:
				return None
			return raw_spectra[int(y), int(x), :].astype(float)
		if overlays is not None and "gradient" in overlays:
			return overlays['gradient'][int(y), int(x), :].astype(float)
		if raw_spectra is not None:
			return raw_spectra[int(y), int(x), :].astype(float)
		return None

	def _boundary_source_signal(y: int, x: int) -> Optional[np.ndarray]:
		bsrc = str(boundary_minimum_source).strip().lower()
		if bsrc == "gradient" and overlays is not None and "gradient" in overlays:
			return overlays['gradient'][int(y), int(x), :].astype(float)
		if raw_spectra is not None:
			return raw_spectra[int(y), int(x), :].astype(float)
		if overlays is not None and "gradient" in overlays:
			return overlays['gradient'][int(y), int(x), :].astype(float)
		return None

	def _prepare_pixel_segments(y: int, x: int, segs: List[SpikeSegment]) -> List[SpikeSegment]:
		merge_sig = overlays['gradient'][int(y), int(x), :].astype(float) if overlays is not None and "gradient" in overlays else None
		return prepare_primary_ss4_segments(
			y=int(y),
			x=int(x),
			segs=segs,
			feature_signal=_feature_source_signal(int(y), int(x)),
			boundary_signal=_boundary_source_signal(int(y), int(x)),
			merge_signal=merge_sig,
			feature_expand_to_gradient_foot=bool(feature_expand_to_gradient_foot),
			feature_foot_k_mad=float(feature_foot_k_mad),
			feature_foot_min_run=int(feature_foot_min_run),
			feature_window_method=feature_window_method,
			feature_erosion_se_size=int(feature_erosion_se_size),
			merge_duplicate_segments=bool(merge_duplicate_segments),
			merge_max_width_pts=merge_max_width_pts,
		)

	merged_spikes_by_pixel: Dict[Tuple[int, int], List[SpikeSegment]] = {
		pix: _prepare_pixel_segments(int(pix[0]), int(pix[1]), list(segs))
		for pix, segs in spikes_by_pixel.items()
	}
	raw_n_spikes_total = int(sum(len(segs) for segs in spikes_by_pixel.values()))

	finite = score_map[np.isfinite(score_map)]
	if finite.size == 0:
		score_stats = {"min": None, "max": None, "median": None, "p95": None}
	else:
		score_stats = {
			"min": float(np.min(finite)),
			"max": float(np.max(finite)),
			"median": float(np.median(finite)),
			"p95": float(np.percentile(finite, 95.0)),
		}

	all_spikes = [s for segs in merged_spikes_by_pixel.values() for s in segs]
	pixel_rows = []
	for (y, x), segs in merged_spikes_by_pixel.items():
		if not segs:
			continue
		pixel_rows.append(
			{
				"y": int(y),
				"x": int(x),
				"n_spikes": int(len(segs)),
				"max_peak_height": float(max(s.peak_height for s in segs)),
				"sum_area": float(sum(s.area for s in segs)),
			}
		)
	pixel_rows.sort(key=lambda r: (r['n_spikes'], r['max_peak_height']), reverse=True)
	top_pixels = pixel_rows[: int(max_top_pixels)]

	report: Dict[str, Any] = {
		"shape_hw": [int(score_map.shape[0]), int(score_map.shape[1])],
		"threshold": float(threshold),
		"score_stats": score_stats,
		"n_candidates": int(np.count_nonzero(candidate_mask)),
		"n_pixels_with_spikes": int(sum(1 for segs in merged_spikes_by_pixel.values() if segs)),
		"n_spikes_total": int(len(all_spikes)),
		"n_spikes_total_raw": raw_n_spikes_total,
		"top_pixels": top_pixels,
	}

	def _muon_score(feat: Dict[str, Any]) -> Dict[str, Any]:
		# bounded transforms to avoid domination by any single large feature
		rise = float(feat.get("rise_slope", 0.0))
		fall = float(feat.get("fall_slope", 0.0))
		gabs = float(feat.get("gradient_max", 0.0))
		gz = float(feat.get("gradient_max_z", 0.0))
		pw = float(feat.get("plateau_width_90", 0.0))
		asym = abs(float(feat.get("edge_asymmetry", 1.0)))

		s_grad = float(np.tanh(gz / 6.0))
		s_grad_abs = float(np.tanh(gabs / 2000.0))
		s_rise_abs = float(np.tanh(max(rise, 0.0) / 1200.0))
		s_fall_abs = float(np.tanh(abs(min(fall, 0.0)) / 1200.0))
		s_plateau = float(np.exp(-((pw - 3.0) / 4.0) ** 2))
		s_asym = float(np.exp(-asym / 0.5))

		weights = {
			"s_rise_abs": 0.4,
			"s_fall_abs": 0.4,
			"s_grad_abs": 0.15,
			"s_grad": 0.03,
			"s_plateau": 0.01,
			"s_asym": 0.01
		}
		score = (
			weights['s_rise_abs'] * s_rise_abs
			+ weights['s_fall_abs'] * s_fall_abs
			+ weights['s_grad_abs'] * s_grad_abs
			+ weights['s_grad'] * s_grad
			+ weights['s_plateau'] * s_plateau
			+ weights['s_asym'] * s_asym
		)
		return {
			"muon_score": float(score),
			"muon_score_components": {
				"s_rise_abs": s_rise_abs,
				"s_fall_abs": s_fall_abs,
				"s_grad_abs": s_grad_abs,
				"s_grad": s_grad,
				"s_plateau": s_plateau,
				"s_asym": s_asym,
				"weights": weights,
			},
		}

	def _spike_features(
			y: int, x: int, s: SpikeSegment,
			window_override: Optional[Tuple[int, int]] = None,
			gws_context_info: Optional[Dict[str, object]] = None,
			gws_context_infos_by_pad: Optional[Dict[int, Dict[str, object]]] = None,
	) -> Dict[str, Any]:
		out: Dict[str, Any] = {
			"width_pts": int(max(0, s.end - s.start - 1)),
			"peak_index": int(s.peak_index),
			"start": int(s.start),
			"end": int(s.end),
			"peak_height": float(s.peak_height),
			"area": float(s.area),
		}
		if raw_spectra is not None:
			out['raw_peak_value'] = float(raw_spectra[int(y), int(x), int(s.peak_index)])
		top_hat_full = None
		if overlays is not None and "top_hat" in overlays:
			top_hat_full = overlays["top_hat"][int(y), int(x), :].astype(float)

		src = str(feature_signal_source).lower().strip()
		if src == "raw":
			if raw_spectra is None:
				return out
			sig_spec = raw_spectra[int(y), int(x), :].astype(float)
		else:
			# Primary features are computed from morphological gradient.
			if overlays is None or "gradient" not in overlays:
				return out
			sig_spec = overlays['gradient'][int(y), int(x), :].astype(float)
			src = "gradient"

		n = sig_spec.size
		a = int(np.clip(s.start, 0, n - 1))
		b = int(np.clip(s.end, 0, n - 1))
		p = int(np.clip(s.peak_index, 0, n - 1))
		if not (a < p < b):
			return out

		if window_override is None:
			a, b = expand_interval_to_signal_foot(
				sig=sig_spec,
				left=a,
				right=b,
				peak=p,
				enabled=bool(feature_expand_to_gradient_foot),
				k_mad=float(feature_foot_k_mad),
				min_run=int(feature_foot_min_run),
				method=feature_window_method,
				erosion_se_size=int(feature_erosion_se_size),
			)
		else:
			a = int(window_override[0])
			b = int(window_override[1])

		segment = sig_spec[a:b + 1]
		if segment.size < 3:
			return out
		peak_rel = int(np.clip(p - a, 0, segment.size - 1))

		d = np.diff(segment)
		rise = d[: max(1, p - a)]
		fall = d[max(1, p - a) :]
		rise_slope = float(np.max(rise)) if rise.size else 0.0
		fall_slope = float(np.min(fall)) if fall.size else 0.0
		rise_total_change = float(np.sum(np.maximum(rise, 0.0))) if rise.size else 0.0
		fall_total_change = float(np.sum(np.maximum(-fall, 0.0))) if fall.size else 0.0

		context = max(10, 3 * max(1, b - a + 1))
		l0 = max(0, a - context)
		r1 = min(n, b + context + 1)
		bg = np.concatenate([sig_spec[l0:a], sig_spec[b + 1:r1]])
		if bg.size < 5:
			bg = np.concatenate([sig_spec[:a], sig_spec[b + 1:]])
		if bg.size < 5:
			bg = sig_spec
		bg_med = float(np.median(bg))
		bg_mad = float(np.median(np.abs(bg - bg_med)))
		bg_mad = max(bg_mad, 1e-12)
		out['rise_slope'] = rise_slope
		out['fall_slope'] = fall_slope
		out['rise_total_change'] = rise_total_change
		out['fall_total_change'] = fall_total_change
		out['rise_slope_z'] = float(rise_slope / bg_mad)
		out['fall_slope_z'] = float(abs(fall_slope / bg_mad))
		out['rise_total_change_z'] = float(rise_total_change / bg_mad)
		out['fall_total_change_z'] = float(fall_total_change / bg_mad)
		area_z = float(out.get("area", 0.0) / (bg_mad * max(1.0, float(out.get("width_pts", 1)))))
		out['area_z'] = area_z
		out['noise_mad_signal'] = float(bg_mad)
		# Backward-compatible key kept for existing tooling
		out['noise_mad_gradient'] = float(bg_mad)

		peak_val = float(sig_spec[p])
		lvl = 0.9 * peak_val
		plateau = int(np.count_nonzero(segment >= lvl))
		out['plateau_width_90'] = plateau
		out['edge_asymmetry'] = float(
			(abs(rise_slope) - abs(fall_slope)) / (abs(rise_slope) + abs(fall_slope) + 1e-12)
		)
		out['edge_asymmetry_total_change'] = float(
			(rise_total_change - fall_total_change) / (rise_total_change + fall_total_change + 1e-12)
		)

		out['gradient_max'] = float(np.max(segment))
		out['gradient_max_z'] = float(np.max(segment) / bg_mad)
		# Additional morphology-like descriptors around the selected peak:
		# 1) strongest curvature (2nd difference) across whole feature interval
		out.update(compute_peak_curvature_features(segment, bg_mad, peak_rel=peak_rel))

		# 2) prominence on selected signal (vs local minima around the peak)
		left_min = float(np.min(sig_spec[a:p + 1])) if p >= a else float(sig_spec[p])
		right_min = float(np.min(sig_spec[p:b + 1])) if b >= p else float(sig_spec[p])
		prom = float(sig_spec[p] - max(left_min, right_min))
		out['prominence_local'] = prom
		out['prominence_local_z'] = float(prom / bg_mad)

		# 2b) raw-spectrum higher derivatives over whole feature interval
		if raw_spectra is not None:
			raw_seg = raw_spectra[int(y), int(x), a:b + 1].astype(float)
			raw_full = raw_spectra[int(y), int(x), :].astype(float)
			raw_bg = np.concatenate([raw_full[l0:a], raw_full[b + 1:r1]])
			if raw_bg.size < 5:
				raw_bg = np.concatenate([raw_full[:a], raw_full[b + 1:]])
			if raw_bg.size < 5:
				raw_bg = raw_full
			raw_bg_med = float(np.median(raw_bg))
			raw_bg_mad = max(float(np.median(np.abs(raw_bg - raw_bg_med))), 1e-12)
			raw_curv = compute_peak_curvature_features(raw_seg, bg_mad, peak_rel=peak_rel)
			out['peak_raw_curvature_extreme'] = float(raw_curv['peak_curvature_extreme'])
			out['peak_raw_curvature_extreme_z'] = float(raw_curv['peak_curvature_extreme_z'])
			if raw_seg.size >= 3:
				d2r = np.diff(raw_seg, n=2)
				out['raw_d2_max_abs'] = float(np.max(np.abs(d2r)))
				out['raw_d2_max_abs_z'] = float(out['raw_d2_max_abs'] / bg_mad)
			else:
				out['raw_d2_max_abs'] = 0.0
				out['raw_d2_max_abs_z'] = 0.0
			if raw_seg.size >= 4:
				d3r = np.diff(raw_seg, n=3)
				out['raw_d3_max_abs'] = float(np.max(np.abs(d3r)))
				out['raw_d3_max_abs_z'] = float(out['raw_d3_max_abs'] / bg_mad)
			else:
				out['raw_d3_max_abs'] = 0.0
				out['raw_d3_max_abs_z'] = 0.0
		else:
			raw_full = sig_spec.astype(float)
			raw_bg_mad = float(bg_mad)
			out['peak_raw_curvature_extreme'] = 0.0
			out['peak_raw_curvature_extreme_z'] = 0.0
		full_gradient = overlays['gradient'][int(y), int(x), :].astype(float) if (overlays is not None and "gradient" in overlays) else np.array([], dtype=float)

		half = 0.5 * peak_val
		above_half = np.where(segment >= half)[0]
		if above_half.size:
			fwhm_pts = int(above_half[-1] - above_half[0] + 1)
		else:
			fwhm_pts = 0
		out['fwhm_pts'] = fwhm_pts
		if x_axis is not None and 0 <= p < x_axis.size and above_half.size:
			l_ix = int(a + above_half[0])
			r_ix = int(a + above_half[-1])
			out['fwhm_cm1'] = float(abs(float(x_axis[r_ix]) - float(x_axis[l_ix])))
		else:
			out['fwhm_cm1'] = float("nan")

		top_mask = segment >= lvl
		top_vals = segment[top_mask]
		out['top_points_n'] = int(top_vals.size)
		if top_vals.size >= 2:
			top_std = float(np.std(top_vals))
			top_med = float(np.median(top_vals))
			top_mad = float(np.median(np.abs(top_vals - top_med)))
		else:
			top_std = 0.0
			top_mad = 0.0
		out['top_fluct_std'] = top_std
		out['top_fluct_mad'] = top_mad
		out['top_fluck_rel_std'] = float(top_std / (abs(peak_val) + 1e-12))

		# 3) width shape across multiple levels + concentration
		def _width_at_level(frac: float) -> int:
			thr = float(frac * peak_val)
			idx = np.where(segment >= thr)[0]
			return int(idx[-1] + 1) if idx.size else 0

		w30 = int(_width_at_level(0.3))
		w50 = int(_width_at_level(0.5))
		w70 = int(_width_at_level(0.7))
		w90 = int(_width_at_level(0.9))
		out['width_30_pts'] = w30
		out['width_50_pts'] = w50
		out['width_70_pts'] = w70
		out['width_90_pts'] = w90
		out['width_ratio_90_50'] = float(w90 / max(1, w50))
		out['width_ratio_70_50'] = float(w70 / max(1, w50))

		# energy concentration near peak versus full feature interval
		r = max(1, int(round((b - a + 1) * 0.12)))
		lpk = max(a, p - r)
		rpk = min(b, p + r)
		e_full = float(np.sum(np.maximum(segment, 0.0)))
		e_core = float(np.sum(np.maximum(sig_spec[lpk:rpk + 1], 0.0)))
		out['energy_core_ratio'] = float(e_core / max(e_full, 1e-12))

		edge_ctx_pad = int(edge_dense_context_pad_pts)
		if edge_ctx_pad > 0:
			edge_ctx_pad = max(edge_ctx_pad, int(edge_dense_context_min_pad_pts))
			edge_ctx_pad = min(edge_ctx_pad, int(edge_dense_context_max_pad_pts))
			edge_ctx_left = max(0, int(a) - edge_ctx_pad)
			edge_ctx_right = min(int(raw_full.size) - 1, int(b) + edge_ctx_pad)
			raw_edge_ctx_metrics = compute_edge_width_metrics(
				raw_full,
				detection_left=int(edge_ctx_left),
				detection_right=int(edge_ctx_right),
				prefix="raw_edge_ctx",
				apex_idx=int(p),
				bg_mad=raw_bg_mad,
				include_low_root_metrics=True,
				low_root_noise_k_mad=float(edge_dense_min_snr),
				use_enhanced_spike_mapping=bool(edge_use_enhanced_spike_mapping) and bool(edge_enhanced_in_debug_report),
				mapping_levels_desc=tuple(int(v) for v in edge_mapping_levels_desc),
				mapping_refine_step_percent=int(edge_mapping_refine_step_percent),
				mapping_min_level_percent=int(edge_mapping_min_level_percent),
				mapping_require_closed_interval=bool(edge_mapping_require_closed_interval),
				mapping_use_apex_component=bool(edge_mapping_use_apex_component),
				mapping_enable_merge_guard=bool(edge_mapping_enable_merge_guard),
				mapping_max_width_jump_factor=float(edge_mapping_max_width_jump_factor),
				mapping_max_width_jump_points=float(edge_mapping_max_width_jump_points),
				mapping_fallback_to_old=bool(edge_mapping_fallback_to_old),
				mapping_noise_guard_enabled=bool(edge_mapping_noise_guard_enabled),
			)
			_add_edge_width_aliases(out, raw_edge_ctx_metrics, source_prefix="raw_edge_ctx", alias_prefix="recdw")
		if bool(rucdw_enabled) and raw_spectra is not None:
			rucdw_metrics = compute_raw_upper_component_dense_width_metrics(
				raw_full,
				detection_left=int(a),
				detection_right=int(b),
				prefix="rucdw",
				bg_mad=raw_bg_mad,
				context_pad_pts=int(rucdw_context_pad_pts),
				context_max_pad_pts=int(rucdw_context_max_pad_pts),
				levels=tuple(int(v) for v in rucdw_levels),
				min_snr=float(rucdw_min_snr),
				noise_fallback_rel_amp=float(rucdw_noise_fallback_rel_amp),
				anchor_mode=str(rucdw_anchor_mode),
				baseline_mode=str(rucdw_baseline_mode),
				baseline_percentile=float(rucdw_baseline_percentile),
			)
			for key, value in rucdw_metrics.items():
				if str(key).endswith("_debug"):
					continue
				out[str(key)] = value
		out['feature_source'] = src
		out['feature_window_start'] = int(a)
		out['feature_window_end'] = int(b)

		# SPIKE SCORES
		def _get_value(key: str, div: float):
			return float(np.tanh(float(out.get(key, 0.0)) / div))

		# Versioned spike score (v1) from empirically strongest features
		sr = _get_value('rise_slope_z', 6.0)
		sf = _get_value('fall_slope_z', 6.0)
		sg = _get_value('gradient_max_z', 6.0)
		sa = _get_value('area_z', 4.0)

		spike_score_v1 = float(
			0.50 * sr
			+ 0.50 * sf
		)
		out['spike_score_v1'] = spike_score_v1
		out['spike_score_v1_components'] = {
			'rise_slope_z_tanh': sr,
			'fall_slope_z_tanh': sf,
			'gradient_max_z_tanh': sg,
			'area_z_tanh': sa,
			'weights': {
				'rise_slope_z': 0.36,
				'fall_slope_z': 0.36,
				'gradient_max_z': 0.18,
				'area_z': 0.10,
			},
		}
		out.update(compute_spike_score_v2_features(out))
		out["gws_support01"] = float("nan")
		out["gws_evidence_signed"] = float("nan")

		if x_axis is not None and 0 <= p < x_axis.size:
			out['peak_position_cm1'] = float(x_axis[p])
		out.update(_muon_score(out))
		out.update(annotate_feature_dict_with_muon_rule_v3(out))
		if labels_by_candidate is not None:
			label_key = (int(y), int(x), int(out.get("peak_index", -1)))
			if label_key in labels_by_candidate:
				out["is_muon_label"] = bool(labels_by_candidate[label_key])

		return _filter_export_metrics(out)

	def _candidate_eval_row(
			spec_index: int,
			candidate_index: int,
			y: int,
			x: int,
			spike: Dict[str, Any],
			is_muon: bool,
	) -> Dict[str, Any]:
		return {
			"spectrum_index": int(spec_index),
			"candidate_index": int(candidate_index),
			"candidate_position": spike.get("peak_position_cm1"),
			"spike_score_v1": spike.get("spike_score_v1"),
			"pce_negpref_t098_evidence_signed": spike.get("pce_negpref_t098_evidence_signed"),
			"gws_evidence_signed": spike.get("gws_evidence_signed"),
			"ss_category_v3": spike.get("ss_category_v3"),
			"pce_category_v3": spike.get("pce_category_v3"),
			"gws_category_v3": spike.get("gws_category_v3"),
			"muon_rule_v3_decision": spike.get("muon_rule_v3_decision"),
			"muon_rule_v3_reason": spike.get("muon_rule_v3_reason"),
			"spike_score_v4_three_friends": spike.get("spike_score_v4_three_friends"),
			"ss4": spike.get("ss4"),
			"ss4_decision": spike.get("ss4_decision"),
			"ss4_reason": spike.get("ss4_reason"),
			"ss4_ss_zone": spike.get("ss4_ss_zone"),
			"ss4_pce_zone": spike.get("ss4_pce_zone"),
			"ss4_rve_zone": spike.get("ss4_rve_zone"),
			"spike_score_v4_decision": spike.get("spike_score_v4_decision"),
			"spike_score_v4_reason": spike.get("spike_score_v4_reason"),
			"spike_score_v4_ss_zone": spike.get("spike_score_v4_ss_zone"),
			"spike_score_v4_pce_zone": spike.get("spike_score_v4_pce_zone"),
			"spike_score_v4_edge_zone": spike.get("spike_score_v4_edge_zone"),
			"y": int(y),
			"x": int(x),
			"peak_index": int(spike.get("peak_index", -1)),
			"is_muon_label": bool(is_muon),
		}

	def _build_muon_rule_v3_summary(per_spec: List[Dict[str, Any]]) -> Dict[str, Any]:
		reason_counts: Dict[str, int] = {}
		decision_counts = {"auto_muon": 0, "maybe_muon": 0, "no_muon": 0}
		fp_rows: List[Dict[str, Any]] = []
		fn_rows: List[Dict[str, Any]] = []
		tp = fp = fn = tn = 0
		maybe_muon_true = 0
		maybe_muon_false = 0
		n_labeled = 0

		for spec_index, spec in enumerate(per_spec):
			y = int(spec.get("y", -1))
			x = int(spec.get("x", -1))
			spikes = spec.get("spikes", [])
			if not isinstance(spikes, list):
				continue
			for candidate_index, spike in enumerate(spikes):
				decision = str(spike.get("muon_rule_v3_decision", "no_muon"))
				reason = str(spike.get("muon_rule_v3_reason", ""))
				if decision in decision_counts:
					decision_counts[decision] += 1
				reason_counts[reason] = int(reason_counts.get(reason, 0) + 1)
				if "is_muon_label" not in spike:
					continue
				n_labeled += 1
				is_muon = bool(spike["is_muon_label"])
				if decision == "maybe_muon":
					if is_muon:
						maybe_muon_true += 1
					else:
						maybe_muon_false += 1
				is_auto = decision == "auto_muon"
				row = _candidate_eval_row(spec_index, candidate_index, y, x, spike, is_muon)
				if is_auto and is_muon:
					tp += 1
				elif is_auto and not is_muon:
					fp += 1
					fp_rows.append(row)
				elif (not is_auto) and is_muon:
					fn += 1
					fn_rows.append(row)
				else:
					tn += 1

		return {
			"counts": decision_counts,
			"reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
			"auto_muon_confusion": {
				"tp": int(tp),
				"fp": int(fp),
				"fn": int(fn),
				"tn": int(tn),
				"precision": float(tp / max(tp + fp, 1)),
				"recall": float(tp / max(tp + fn, 1)),
				"specificity": float(tn / max(tn + fp, 1)),
			},
			"maybe_muon_label_counts": {
				"muon": int(maybe_muon_true),
				"non_muon": int(maybe_muon_false),
			},
			"false_positives": fp_rows,
			"false_negatives": fn_rows,
			"n_labeled_candidates": int(n_labeled),
		}

	if include_per_spectrum:
		if target_coords is not None:
			iter_coords = [(int(y), int(x)) for (y, x) in target_coords]
		else:
			iter_coords = sorted(
				{(int(y), int(x)) for (y, x) in spikes_by_pixel.keys()} | {(int(y), int(x)) for (y, x) in merged_spikes_by_pixel.keys()},
				key=lambda t: (t[0], t[1]),
			)

		per_spec = []
		total_spikes_for_progress = int(sum(len(merged_spikes_by_pixel.get((int(y), int(x)), [])) for y, x in iter_coords))
		progress_bar = None
		progress_t0 = time.time()
		progress_done = 0
		print_every = max(0, int(progress_print_every))
		if bool(show_progress):
			try:
				from tqdm import tqdm  # type: ignore
				progress_bar = tqdm(
					total=total_spikes_for_progress,
					desc="Debug feature report",
					unit="spike",
					file=sys.stdout,
					dynamic_ncols=True,
					mininterval=0.25,
					miniters=1,
					smoothing=0.1,
					leave=True,
				)
			except Exception:
				progress_bar = None
		for y, x in iter_coords:
			prepared_segs = list(merged_spikes_by_pixel.get((int(y), int(x)), []))
			spikes = []
			for idx, s in enumerate(prepared_segs):
				progress_done += 1
				if progress_bar is not None:
					progress_bar.set_postfix_str(f"y={int(y)} x={int(x)} peak={int(s.peak_index)}")
				row = _spike_features(
					int(y),
					int(x),
					s,
					window_override=(int(s.start), int(s.end)),
				)
				if isinstance(row, dict) and row:
					spikes.append(row)
				if progress_bar is not None:
					progress_bar.update(1)
				elif bool(show_progress) and print_every > 0 and (progress_done == 1 or progress_done % print_every == 0):
					elapsed = max(time.time() - progress_t0, 1e-9)
					rate = float(progress_done / elapsed)
					remaining = max(total_spikes_for_progress - progress_done, 0)
					eta = float(remaining / rate) if rate > 0 else float("nan")
					print(
						f"[debug-report] {progress_done}/{total_spikes_for_progress} spikes "
						f"({100.0 * progress_done / max(total_spikes_for_progress, 1):.2f}%) "
						f"rate={rate:.1f}/s eta={eta/60.0:.1f} min "
						f"current y={int(y)} x={int(x)} peak={int(s.peak_index)}",
						flush=True,
					)

			per_spec.append(
				{
					"y": int(y),
					"x": int(x),
					"is_candidate": bool(candidate_mask[int(y), int(x)]),
					"score": float(score_map[int(y), int(x)]),
					"n_spikes": int(len(spikes)),
					"spikes": spikes,
				}
			)
		if progress_bar is not None:
			progress_bar.close()
		_add_recdw_evidence(per_spec)
		_add_spike_score_v4(per_spec)
		report['per_spectrum'] = per_spec
		if labels_by_candidate is not None:
			report["muon_rule_v3_summary"] = _build_muon_rule_v3_summary(per_spec)

	return report


def save_debug_report_json(path: Path, report: Dict[str, Any]) -> None:
	path = Path(path)
	path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
