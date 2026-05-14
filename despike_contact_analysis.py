from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from feature_discrimination import (
	compute_edge_width_metrics,
	compute_peak_curvature_features,
	compute_spike_score_v2_features,
	estimate_background_mad,
)
from morph1d import dilation_1d, erosion_1d
from muon_pipeline import SpikeSegment
from muon_decision import compute_ss4, compute_ss5


def _clean_value(value: Any) -> Any:
	if isinstance(value, (np.integer,)):
		return int(value)
	if isinstance(value, (np.floating,)):
		value = float(value)
	if isinstance(value, float):
		return value if math.isfinite(value) else None
	if isinstance(value, dict):
		return {str(k): _clean_value(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_clean_value(v) for v in value]
	return value


def _mad(values: np.ndarray) -> float:
	x = np.asarray(values, dtype=float)
	x = x[np.isfinite(x)]
	if x.size == 0:
		return 1.0
	med = float(np.median(x))
	mad = float(np.median(np.abs(x - med)))
	if not np.isfinite(mad) or mad <= 1e-12:
		std = float(np.nanstd(x))
		return std if np.isfinite(std) and std > 1e-12 else 1.0
	return 1.4826 * mad


def _cell_kind(
		*,
		contains_parent_apex: bool,
		overlaps_parent_segment: bool,
		n_dilation_contacts: int,
		salience: float,
		high_salience: float,
) -> str:
	if contains_parent_apex:
		return "parent_apex_cell"
	if n_dilation_contacts >= 2 and salience >= high_salience:
		return "candidate_multispike_cell"
	if n_dilation_contacts >= 1 and salience >= high_salience:
		return "candidate_spike_cell"
	if salience < 1.0:
		return "noise_like_cell"
	if overlaps_parent_segment:
		return "mixed_or_uncertain_cell"
	return "context_cell"


def _finite_minmax_norm(values: List[float]) -> List[float]:
	arr = np.asarray(values, dtype=float)
	finite = arr[np.isfinite(arr)]
	if finite.size == 0:
		return [float("nan") for _ in values]
	lo = float(np.min(finite))
	hi = float(np.max(finite))
	if hi <= lo + 1e-12:
		fill = 1.0 if hi > 0.0 else 0.0
		return [float(fill) if np.isfinite(v) else float("nan") for v in arr]
	return [float((v - lo) / (hi - lo)) if np.isfinite(v) else float("nan") for v in arr]


def _secondary_edge_rve_proxy(raw: np.ndarray, left: int, right: int, local_noise: float) -> Tuple[float, Dict[str, Any]]:
	"""Small diagnostic Raman-veto proxy for one contact cell.

	The production RVE metric is dataset-normalized and cannot be reproduced for
	an isolated contact cell. This proxy keeps the same orientation/range for
	secondary diagnostics only: narrow cells trend negative/spike-compatible,
	broad upper-level cells trend positive/Raman-like.
	"""
	n = int(raw.size)
	a = int(np.clip(left, 0, n - 1))
	b = int(np.clip(right, 0, n - 1))
	if b < a:
		a, b = b, a
	seg = np.asarray(raw[a:b + 1], dtype=float)
	debug: Dict[str, Any] = {
		"secondary_edge_kind": "local_cell_upper_width_proxy",
		"left": int(a),
		"right": int(b),
		"valid_n": 0,
	}
	if seg.size < 2 or not np.any(np.isfinite(seg)):
		debug["reason"] = "invalid_segment"
		return float("nan"), debug
	baseline = float(np.nanmin(seg))
	ymax = float(np.nanmax(seg))
	amp = float(ymax - baseline)
	noise = float(local_noise) if np.isfinite(local_noise) and local_noise > 1e-12 else 0.0
	debug.update({"baseline": baseline, "ymax": ymax, "amp": amp, "noise": noise})
	if not np.isfinite(amp) or amp <= max(noise, 1e-12):
		debug["reason"] = "below_noise_or_flat"
		return float("nan"), debug
	widths: List[float] = []
	levels: List[int] = []
	for pct in range(5, 95, 5):
		level = baseline + (float(pct) / 100.0) * amp
		if level <= baseline + noise:
			continue
		idx = np.flatnonzero(seg >= level)
		if idx.size == 0:
			continue
		widths.append(float(int(idx[-1]) - int(idx[0]) + 1))
		levels.append(int(pct))
	max_sum = float(max(seg.size, 1) * max(len(widths), 1))
	width_sum = float(np.nansum(widths)) if widths else float("nan")
	structure01 = float(width_sum / max_sum) if np.isfinite(width_sum) and max_sum > 0 else float("nan")
	rve = float(np.clip(2.0 * structure01 - 1.0, -1.0, 1.0)) if np.isfinite(structure01) else float("nan")
	debug.update({
		"reason": "ok" if np.isfinite(rve) else "no_valid_levels",
		"levels": levels,
		"widths": widths,
		"width_sum": width_sum,
		"structure01": structure01,
		"rve_proxy": rve,
		"valid_n": int(len(widths)),
	})
	return rve, debug


def _secondary_ss4_for_cell(
		*,
		raw: np.ndarray,
		gradient: Optional[np.ndarray],
		left: int,
		right: int,
		preferred_apex: Optional[int],
		dilation_contacts: List[int],
		local_noise: float,
		ss_blue_max: float,
		ss_red_min: float,
		pce_red_min: float,
		rve_red_max: float,
		missing_policy: str,
		edge_rescue_ss_min: float,
) -> Dict[str, Any]:
	n = int(raw.size)
	a = int(np.clip(left, 0, n - 1))
	b = int(np.clip(right, 0, n - 1))
	if b < a:
		a, b = b, a
	raw_seg = np.asarray(raw[a:b + 1], dtype=float)
	if raw_seg.size < 3:
		return {
			"secondary_ss4_ran": True,
			"secondary_ss4_invalid_reason": "cell_too_short",
			"secondary_spike_score_v1": float("nan"),
			"secondary_pce_negpref_t098_evidence_signed": float("nan"),
			"secondary_recdw_sum_0_90_raman_veto_evidence_signed": float("nan"),
			"secondary_rve_value_used_for_ss4": float("nan"),
			"secondary_rve_metric_source": "not_computed_cell_too_short",
			"secondary_ss1": float("nan"),
			"secondary_pce": float("nan"),
			"secondary_edge": float("nan"),
			"secondary_ss4": float("nan"),
			"secondary_ss4_decision": "review",
			"secondary_ss4_reason": "review_missing",
		}
	if preferred_apex is not None and a <= int(preferred_apex) <= b:
		apex = int(preferred_apex)
		anchor_source = "parent_apex"
	else:
		anchor_candidates = [int(v) for v in dilation_contacts if a <= int(v) <= b]
		if anchor_candidates:
			apex = max(anchor_candidates, key=lambda idx: float(raw[int(idx)]))
			anchor_source = "dilation_contact_max_raw"
		else:
			apex = int(a + int(np.nanargmax(raw_seg)))
			anchor_source = "cell_max_raw"
	apex_rel = int(np.clip(apex - a, 0, raw_seg.size - 1))
	sig = np.asarray(gradient, dtype=float) if gradient is not None and np.asarray(gradient).size == n else raw
	sig_seg = np.asarray(sig[a:b + 1], dtype=float)
	if sig_seg.size < 3 or not (0 < apex_rel < sig_seg.size - 1):
		ss1 = float("nan")
	else:
		rise = np.diff(sig_seg)[: max(1, apex_rel)]
		fall = np.diff(sig_seg)[max(1, apex_rel):]
		rise_slope = float(np.nanmax(rise)) if rise.size else 0.0
		fall_slope = float(np.nanmin(fall)) if fall.size else 0.0
		bg = estimate_background_mad(sig, a, b)
		bg = bg if np.isfinite(bg) and bg > 1e-12 else max(float(local_noise), 1e-12)
		ss1 = float(0.5 * np.tanh((rise_slope / bg) / 6.0) + 0.5 * np.tanh((abs(fall_slope) / bg) / 6.0))
	pce = float("nan")
	try:
		bg = max(float(local_noise), 1e-12)
		pce_features = compute_peak_curvature_features(sig_seg, bg, peak_rel=apex_rel)
		tmp = dict(pce_features)
		tmp["spike_score_v1"] = ss1
		tmp.update(compute_spike_score_v2_features(tmp))
		pce = float(tmp.get("pce_negpref_t098_evidence_signed", np.nan))
	except Exception:
		pce = float("nan")
	rve, edge_debug = _secondary_edge_rve_proxy(raw, a, b, local_noise)
	ss4 = compute_ss4(
		ss1,
		pce,
		rve,
		ss_blue_max=ss_blue_max,
		ss_red_min=ss_red_min,
		pce_red_min=pce_red_min,
		rve_red_max=rve_red_max,
		missing_policy=missing_policy,
	)
	if (
			str(ss4.get("ss4_decision", "")) != "spike"
			and np.isfinite(ss1)
			and np.isfinite(rve)
			and float(ss1) >= float(edge_rescue_ss_min)
			and float(rve) <= float(rve_red_max)
	):
		ss4 = dict(ss4)
		ss4.update({
			"ss4": 1.0,
			"ss4_decision": "spike",
			"ss4_reason": "secondary_ss_low_but_edge_red",
			"ss4_rve_zone": "red",
			"ss4_is_rve_red": 1.0,
			"ss4_is_raman_veto": 0.0,
		})
	return {
		"secondary_ss4_ran": True,
		"secondary_anchor_index": int(apex),
		"secondary_anchor_source": anchor_source,
		"secondary_spike_score_v1": float(ss1),
		"secondary_pce_negpref_t098_evidence_signed": float(pce),
		"secondary_recdw_sum_0_90_raman_veto_evidence_signed": float("nan"),
		"secondary_rve_value_used_for_ss4": float(rve),
		"secondary_rve_metric_source": "local_cell_upper_width_proxy_not_primary_recdw",
		"secondary_rve_metric_note": "Primary recdw_sum_0_90_raman_veto_evidence_signed is dataset-normalized and is not recomputed exactly for isolated contact cells.",
		"secondary_edge_rve_proxy": float(rve),
		"secondary_ss1": float(ss1),
		"secondary_pce": float(pce),
		"secondary_edge": float(rve),
		"secondary_edge_debug": edge_debug,
		"secondary_ss4": ss4.get("ss4"),
		"secondary_ss4_decision": ss4.get("ss4_decision"),
		"secondary_ss4_reason": ss4.get("ss4_reason"),
		"secondary_ss4_ss_zone": ss4.get("ss4_ss_zone"),
		"secondary_ss4_pce_zone": ss4.get("ss4_pce_zone"),
		"secondary_ss4_rve_zone": ss4.get("ss4_rve_zone"),
	}


def _compute_local_morph_contacts(signal: np.ndarray, *, strict_equal: bool = True, window_size: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	raw = np.asarray(signal, dtype=float)
	raw2 = raw.reshape(1, 1, -1)
	ero = erosion_1d(raw2, int(window_size)).reshape(-1).astype(float)
	dil = dilation_1d(raw2, int(window_size)).reshape(-1).astype(float)
	if bool(strict_equal):
		eq = raw == ero
		dq = raw == dil
	else:
		eq = np.isclose(raw, ero)
		dq = np.isclose(raw, dil)
	if not np.any(eq):
		eq = np.isclose(raw, ero)
	if not np.any(dq):
		dq = np.isclose(raw, dil)
	return ero, dil, np.flatnonzero(eq).astype(int), np.flatnonzero(dq).astype(int)


def _build_working_cells(
		raw: np.ndarray,
		*,
		context_left: int,
		context_right: int,
		parent_start: int,
		parent_end: int,
		parent_apex: int,
		strict_equal: bool,
		local_noise: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], Optional[int]]:
	ctx = np.asarray(raw[int(context_left):int(context_right) + 1], dtype=float)
	ero_ctx, dil_ctx, ero_local, dil_local = _compute_local_morph_contacts(ctx, strict_equal=bool(strict_equal))
	erosion_contacts = (ero_local + int(context_left)).astype(int)
	dilation_contacts = (dil_local + int(context_left)).astype(int)
	dilation_set = set(int(v) for v in dilation_contacts.tolist())
	cells: List[Dict[str, Any]] = []
	parent_apex_cell_index: Optional[int] = None
	if erosion_contacts.size >= 2:
		for idx, (left, right) in enumerate(zip(erosion_contacts[:-1], erosion_contacts[1:])):
			left = int(left)
			right = int(right)
			if right <= left:
				continue
			ys = raw[left:right + 1].astype(float)
			chord = np.linspace(float(raw[left]), float(raw[right]), right - left + 1)
			excess = ys - chord
			above = np.maximum(excess, 0.0)
			below = np.maximum(-excess, 0.0)
			chord_area_above = float(np.nansum(above))
			chord_height_above = float(np.nanmax(above)) if above.size else 0.0
			chord_area_below = float(np.nansum(below))
			chord_area_total = float(chord_area_above + chord_area_below)
			height_abs = float(np.nanmax(np.abs(excess))) if excess.size else 0.0
			width = int(right - left + 1)
			height_z = chord_height_above / max(float(local_noise), 1e-12)
			area_z = chord_area_above / max(float(local_noise) * max(width, 1), 1e-12)
			salience = float(max(height_z, math.sqrt(max(area_z, 0.0))))
			height_abs_z = float(height_abs / max(float(local_noise), 1e-12))
			area_total_z = float(chord_area_total / max(float(local_noise) * max(width, 1), 1e-12))
			salience_total = float(max(height_abs_z, math.sqrt(max(area_total_z, 0.0))))
			salience_density = float(salience_total / math.sqrt(max(width, 1)))
			dil_inside = [int(v) for v in dilation_contacts.tolist() if left <= int(v) <= right]
			contains_apex = bool(left <= int(parent_apex) <= right)
			if contains_apex:
				parent_apex_cell_index = int(idx)
			cells.append({
				"cell_index": int(idx),
				"cell_left": int(left),
				"cell_right": int(right),
				"cell_width": int(width),
				"contains_parent_apex": contains_apex,
				"overlaps_parent_segment": bool(max(left, int(parent_start)) <= min(right, int(parent_end))),
				"dilation_contact_indices_inside_cell": dil_inside,
				"n_dilation_contacts_inside_cell": int(len(dil_inside)),
				"contains_dilation_contact": bool(len(dil_inside) > 0),
				"parent_apex_is_dilation_contact": bool(int(parent_apex) in dilation_set),
				"chord_area_above": chord_area_above,
				"chord_height_above": chord_height_above,
				"chord_area_below": chord_area_below,
				"chord_area_total": chord_area_total,
				"height_abs": height_abs,
				"height_z": float(height_z),
				"area_z": float(area_z),
				"salience": salience,
				"height_abs_z": height_abs_z,
				"area_total_z": area_total_z,
				"salience_total": salience_total,
				"salience_density": salience_density,
				"chord_y": [float(v) for v in chord],
			})
	return ero_ctx, dil_ctx, erosion_contacts, dilation_contacts, cells, parent_apex_cell_index


def _recdw_evidence_from_sum(value: float, *, center: float, scale: float, z_clip: float, support_z_scale: float) -> Tuple[float, float, float]:
	if not (np.isfinite(value) and np.isfinite(center) and np.isfinite(scale) and float(scale) > 1e-12):
		return float("nan"), float("nan"), float("nan")
	z = float((float(value) - float(center)) / float(scale))
	z_clipped = float(np.clip(z, -float(z_clip), float(z_clip)))
	support = float(1.0 / (1.0 + math.exp(-z_clipped / max(float(support_z_scale), 1e-12))))
	return z, support, float(2.0 * support - 1.0)


def _evaluate_active_residual_candidate(
		*,
		raw: np.ndarray,
		gradient: Optional[np.ndarray],
		context_left: int,
		context_right: int,
		candidate_left: int,
		candidate_right: int,
		apex: int,
		local_noise: float,
		decision_profile: str,
		ss4_ss_blue_max: float,
		ss4_ss_red_min: float,
		ss4_pce_red_min: float,
		ss4_rve_red_max: float,
		ss4_missing_policy: str,
		ss5_ss1_threshold: float,
		ss5_pce_spike_min: float,
		ss5_edge_spike_max: float,
		recdw_center: float,
		recdw_scale: float,
		recdw_z_clip: float,
		recdw_support_z_scale: float,
		edge_use_enhanced_spike_mapping: bool,
		edge_mapping_levels_desc: Tuple[int, ...],
		edge_mapping_refine_step_percent: int,
		edge_mapping_min_level_percent: int,
		edge_mapping_require_closed_interval: bool,
		edge_mapping_use_apex_component: bool,
		edge_mapping_enable_merge_guard: bool,
		edge_mapping_max_width_jump_factor: float,
		edge_mapping_max_width_jump_points: float,
		edge_mapping_fallback_to_old: bool,
		edge_mapping_noise_guard_enabled: bool,
		edge_robust_reference_enabled: bool,
		edge_noise_guard_enabled: bool,
		edge_noise_guard_factor: float,
		edge_noise_guard_value: float,
) -> Dict[str, Any]:
	n = int(raw.size)
	a = int(np.clip(int(candidate_left), 0, n - 1))
	b = int(np.clip(int(candidate_right), 0, n - 1))
	if b < a:
		a, b = b, a
	apex = int(np.clip(int(apex), a, b))
	raw_seg = np.asarray(raw[a:b + 1], dtype=float)
	if raw_seg.size < 3:
		return {"decision_score": float("nan"), "decision": "review", "reason": "candidate_too_short"}
	if gradient is not None and np.asarray(gradient).size == n:
		sig = np.asarray(gradient, dtype=float)
	else:
		sig = np.asarray(raw, dtype=float)
	sig_seg = np.asarray(sig[a:b + 1], dtype=float)
	apex_rel = int(np.clip(apex - a, 0, raw_seg.size - 1))
	if sig_seg.size < 3 or not (0 < apex_rel < sig_seg.size - 1):
		ss1 = float("nan")
	else:
		rise = np.diff(sig_seg)[: max(1, apex_rel)]
		fall = np.diff(sig_seg)[max(1, apex_rel):]
		rise_slope = float(np.nanmax(rise)) if rise.size else 0.0
		fall_slope = float(np.nanmin(fall)) if fall.size else 0.0
		bg = estimate_background_mad(sig, a, b)
		bg = bg if np.isfinite(bg) and bg > 1e-12 else max(float(local_noise), 1e-12)
		ss1 = float(0.5 * np.tanh((rise_slope / bg) / 6.0) + 0.5 * np.tanh((abs(fall_slope) / bg) / 6.0))
	try:
		bg = max(float(local_noise), 1e-12)
		pce_features = compute_peak_curvature_features(sig_seg, bg, peak_rel=apex_rel)
		tmp = dict(pce_features)
		tmp["spike_score_v1"] = ss1
		tmp.update(compute_spike_score_v2_features(tmp))
		pce = float(tmp.get("pce_negpref_t098_evidence_signed", np.nan))
	except Exception:
		pce = float("nan")
	edge_metrics = compute_edge_width_metrics(
		raw,
		detection_left=int(context_left),
		detection_right=int(context_right),
		prefix="raw_edge_ctx",
		apex_idx=int(apex),
		bg_mad=float(local_noise),
		include_low_root_metrics=True,
		low_root_noise_k_mad=1.0,
		use_enhanced_spike_mapping=bool(edge_use_enhanced_spike_mapping),
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
		robust_reference_enabled=bool(edge_robust_reference_enabled),
		robust_reference_noise=float(local_noise),
		edge_noise_guard_enabled=bool(edge_noise_guard_enabled),
		edge_noise_guard_factor=float(edge_noise_guard_factor),
		edge_noise_guard_value=float(edge_noise_guard_value),
	)
	recdw_sum_before_guard = float(edge_metrics.get("raw_edge_ctx_dense_width_sum_0_90", np.nan))
	edge_debug = edge_metrics.get("raw_edge_ctx_debug")
	dense_debug = edge_debug.get("dense_width_0_90", {}) if isinstance(edge_debug, dict) else {}
	edge_noise_ratio = float(dense_debug.get("root_snr", np.nan)) if isinstance(dense_debug, dict) else float("nan")
	edge_guard_passed = bool((not bool(edge_noise_guard_enabled)) or (np.isfinite(edge_noise_ratio) and edge_noise_ratio >= float(edge_noise_guard_factor)))
	if not np.isfinite(recdw_sum_before_guard):
		edge_guard_reason = "edge_value_missing"
	elif not bool(edge_noise_guard_enabled):
		edge_guard_reason = "guard_disabled"
	elif not np.isfinite(edge_noise_ratio):
		edge_guard_reason = "edge_noise_ratio_missing"
	elif edge_guard_passed:
		edge_guard_reason = "guard_passed"
	else:
		edge_guard_reason = "below_edge_noise_guard"
	recdw_sum = float(recdw_sum_before_guard) if edge_guard_passed else float("nan")
	edge_z, edge_support, edge_signed = _recdw_evidence_from_sum(
		recdw_sum,
		center=float(recdw_center),
		scale=float(recdw_scale),
		z_clip=float(recdw_z_clip),
		support_z_scale=float(recdw_support_z_scale),
	)
	if str(decision_profile).strip().lower() == "ss5":
		dec = compute_ss5(
			ss1,
			pce,
			edge_signed,
			ss1_threshold=float(ss5_ss1_threshold),
			pce_spike_min=float(ss5_pce_spike_min),
			edge_spike_max=float(ss5_edge_spike_max),
		)
		return {
			"ss1": float(ss1),
			"pce": float(pce),
			"edge_raw_sum": float(recdw_sum),
			"edge_noise_guard_factor": float(edge_noise_guard_factor),
			"edge_noise_ratio": float(edge_noise_ratio) if np.isfinite(edge_noise_ratio) else float("nan"),
			"edge_guard_passed": bool(edge_guard_passed),
			"edge_guard_reason": str(edge_guard_reason),
			"edge_value_before_guard": float(recdw_sum_before_guard) if np.isfinite(recdw_sum_before_guard) else float("nan"),
			"edge_value_after_guard": float(recdw_sum) if np.isfinite(recdw_sum) else float("nan"),
			"edge": float(edge_signed),
			"edge_z": float(edge_z),
			"edge_support01": float(edge_support),
			"decision_profile": "ss5",
			"decision_score": dec.get("ss5"),
			"decision": dec.get("ss5_decision"),
			"reason": dec.get("ss5_reason"),
			"decision_debug": dec,
		}
	dec = compute_ss4(
		ss1,
		pce,
		edge_signed,
		ss_blue_max=float(ss4_ss_blue_max),
		ss_red_min=float(ss4_ss_red_min),
		pce_red_min=float(ss4_pce_red_min),
		rve_red_max=float(ss4_rve_red_max),
		missing_policy=str(ss4_missing_policy),
	)
	return {
		"ss1": float(ss1),
		"pce": float(pce),
		"edge_raw_sum": float(recdw_sum),
		"edge_noise_guard_factor": float(edge_noise_guard_factor),
		"edge_noise_ratio": float(edge_noise_ratio) if np.isfinite(edge_noise_ratio) else float("nan"),
		"edge_guard_passed": bool(edge_guard_passed),
		"edge_guard_reason": str(edge_guard_reason),
		"edge_value_before_guard": float(recdw_sum_before_guard) if np.isfinite(recdw_sum_before_guard) else float("nan"),
		"edge_value_after_guard": float(recdw_sum) if np.isfinite(recdw_sum) else float("nan"),
		"edge": float(edge_signed),
		"edge_z": float(edge_z),
		"edge_support01": float(edge_support),
		"decision_profile": "ss4",
		"decision_score": dec.get("ss4"),
		"decision": dec.get("ss4_decision"),
		"reason": dec.get("ss4_reason"),
		"decision_debug": dec,
	}


def _line_between(raw: np.ndarray, left: int, right: int) -> np.ndarray:
	if right <= left:
		return np.asarray([float(raw[left])], dtype=float)
	return np.linspace(float(raw[left]), float(raw[right]), int(right - left + 1))


def _chord_y_at_index(raw: np.ndarray, left: int, right: int, apex: int) -> float:
	if right <= left:
		return float("nan")
	t = float((int(apex) - int(left)) / max(int(right) - int(left), 1))
	return float((1.0 - t) * float(raw[int(left)]) + t * float(raw[int(right)]))


def _cell_height_above_chord(raw: np.ndarray, left: int, right: int, apex: int) -> Tuple[float, float]:
	chord_y = _chord_y_at_index(raw, left, right, apex)
	if not np.isfinite(chord_y):
		return float("nan"), float("nan")
	return float(raw[int(apex)] - chord_y), float(chord_y)


def _has_only_trivial_binary_norm(values: List[float]) -> bool:
	finite = sorted(set(float(v) for v in values if np.isfinite(v)))
	if not finite:
		return False
	return all(any(abs(v - ref) <= 1e-9 for ref in (0.0, 1.0)) for v in finite)


def _chord_stats(raw: np.ndarray, left: int, right: int, chord: np.ndarray, eps: float) -> Dict[str, Any]:
	seg = np.asarray(raw[left:right + 1], dtype=float)
	overshoot = np.asarray(chord, dtype=float) - seg
	pos = overshoot > float(eps)
	return {
		"max_overshoot": float(np.nanmax(overshoot)) if overshoot.size else 0.0,
		"crossing_count": int(np.count_nonzero(pos)),
	}


def _required_points_supported(raw: np.ndarray, left: int, right: int, chord: np.ndarray, required_indices: Iterable[int], eps: float) -> bool:
	for idx in required_indices:
		try:
			i = int(idx)
		except Exception:
			continue
		if not (int(left) <= i <= int(right)):
			return False
		rel = int(i - int(left))
		if rel < 0 or rel >= int(chord.size):
			return False
		if float(chord[rel]) > float(raw[i]) + float(eps):
			return False
	return True


def _best_supporting_tangent(raw: np.ndarray, left: int, right: int, required_indices: Iterable[int], eps: float) -> Dict[str, Any]:
	best: Optional[Dict[str, Any]] = None
	required = [int(v) for v in required_indices]

	def _consider(fixed_side: str, a: int, b: int, touch_index: int) -> None:
		nonlocal best
		if b <= a:
			return
		chord = _line_between(raw, a, b)
		if not _required_points_supported(raw, a, b, chord, required, eps):
			return
		stats = _chord_stats(raw, a, b, chord, eps)
		if int(stats["crossing_count"]) > 0 or float(stats["max_overshoot"]) > float(eps):
			return
		score = (int(b - a + 1), float(np.nansum(chord)))
		cand = {
			"score": score,
			"chord_method": "supporting_tangent",
			"fixed_side": fixed_side,
			"final_left_edge": int(a),
			"final_right_edge": int(b),
			"touch_index": int(touch_index),
			"chord_x_index": [int(v) for v in range(a, b + 1)],
			"chord_y": [float(v) for v in chord],
			"max_overshoot_after": float(stats["max_overshoot"]),
			"crossing_count_after": int(stats["crossing_count"]),
		}
		if best is None or score > best["score"]:
			best = cand

	for k in range(right, left, -1):
		_consider("left", left, int(k), int(k))
	for k in range(left, right):
		_consider("right", int(k), right, int(k))
	if best is not None:
		best.pop("score", None)
		return best
	chord = _line_between(raw, left, right)
	raw_span = np.asarray(raw[left:right + 1], dtype=float)
	overshoot = chord - raw_span
	max_overshoot = float(np.nanmax(overshoot)) if overshoot.size else 0.0
	if max_overshoot > float(eps):
		chord = chord - max_overshoot - float(eps)
	stats = _chord_stats(raw, left, right, chord, eps)
	return {
		"chord_method": "supporting_tangent_fallback_shifted_chord",
		"fixed_side": "both_shifted",
		"final_left_edge": int(left),
		"final_right_edge": int(right),
		"touch_index": None,
		"vertical_shift": float(max(max_overshoot, 0.0) + float(eps)),
		"chord_x_index": [int(v) for v in range(left, right + 1)],
		"chord_y": [float(v) for v in chord],
		"max_overshoot_after": float(stats["max_overshoot"]),
		"crossing_count_after": int(stats["crossing_count"]),
	}


def _build_final_chord(raw: np.ndarray, left: int, right: int, required_indices: Optional[Iterable[int]] = None, eps: float = 1e-12) -> Dict[str, Any]:
	left = int(left)
	right = int(right)
	if right < left:
		left, right = right, left
	required = [int(v) for v in (required_indices or []) if left <= int(v) <= right]
	ordinary = _line_between(raw, left, right)
	before = _chord_stats(raw, left, right, ordinary, eps)
	base = {
		"original_left_edge": int(left),
		"original_right_edge": int(right),
		"max_overshoot_before": float(before["max_overshoot"]),
		"crossing_count_before": int(before["crossing_count"]),
	}
	if (
			int(before["crossing_count"]) == 0
			and float(before["max_overshoot"]) <= float(eps)
			and _required_points_supported(raw, left, right, ordinary, required, eps)
	):
		after = _chord_stats(raw, left, right, ordinary, eps)
		return {
			**base,
			"chord_method": "ordinary",
			"fixed_side": "both",
			"final_left_edge": int(left),
			"final_right_edge": int(right),
			"touch_index": None,
			"chord_x_index": [int(v) for v in range(left, right + 1)],
			"chord_y": [float(v) for v in ordinary],
			"max_overshoot_after": float(after["max_overshoot"]),
			"crossing_count_after": int(after["crossing_count"]),
			"required_indices": required,
		}
	out = {**base, **_best_supporting_tangent(raw, left, right, required, eps)}
	out["required_indices"] = required
	return out


def apply_contact_cell_despike_chords(raw_spectra: np.ndarray, analysis: Mapping[str, Any]) -> np.ndarray:
	"""Apply final contact-cell despike chords to a corrected copy."""
	out = np.asarray(raw_spectra, dtype=float).copy()
	for parent in analysis.get("parents", []) or []:
		if not isinstance(parent, Mapping):
			continue
		try:
			y = int(parent.get("y"))
			x = int(parent.get("x"))
		except Exception:
			continue
		if not (0 <= y < out.shape[0] and 0 <= x < out.shape[1]):
			continue
		cells_by_index = {
			int(c.get("cell_index")): c
			for c in (parent.get("cells", []) or [])
			if isinstance(c, Mapping) and c.get("cell_index") is not None
		}
		for chord in (parent.get("summary", {}) or {}).get("final_despike_chords", []) or []:
			try:
				cell_indices = [int(v) for v in chord.get("cell_indices", [])]
				if cell_indices and any(not bool(cells_by_index.get(int(ci), {}).get("secondary_final_is_spike", False)) for ci in cell_indices):
					continue
				left = int(chord.get("final_left_edge"))
				right = int(chord.get("final_right_edge"))
				vals = np.asarray(chord.get("chord_y", []), dtype=float)
			except Exception:
				continue
			if right < left:
				left, right = right, left
			left = int(np.clip(left, 0, out.shape[2] - 1))
			right = int(np.clip(right, 0, out.shape[2] - 1))
			if vals.size != right - left + 1 or vals.size == 0:
				continue
			out[y, x, left:right + 1] = vals
	return out


def analyze_erosion_dilation_contact_cells(
		*,
		x_axis: np.ndarray,
		raw_spectra: np.ndarray,
		erosion: np.ndarray,
		dilation: np.ndarray,
		gradient: Optional[np.ndarray] = None,
		small_morphology_by_pixel: Optional[Mapping[Tuple[int, int], Any]] = None,
		parents: Iterable[SpikeSegment],
		parent_metadata: Optional[Mapping[Tuple[int, int, int, int, int], Mapping[str, Any]]] = None,
		context_pad_pts: int = 4,
		strict_equal: bool = True,
		despike_sensitivity: float = 1.0,
		secondary_noise_thr: float = 0.05,
		secondary_uncertain_thr: float = 0.5,
		ss4_ss_blue_max: float = 0.95,
		ss4_ss_red_min: float = 0.9999,
		ss4_pce_red_min: float = 0.4,
		ss4_rve_red_max: float = -0.1,
		ss4_missing_policy: str = "review",
		secondary_edge_rescue_ss_min: float = 0.85,
		candidate_noise_height_factor: float = 3.0,
		decision_profile: str = "ss4",
		recdw_center: float = np.nan,
		recdw_scale: float = np.nan,
		recdw_z_clip: float = 6.0,
		recdw_support_z_scale: float = 1.0,
		ss5_ss1_threshold: float = 0.95,
		ss5_pce_spike_min: float = 0.8,
		ss5_edge_spike_max: float = -0.4,
		despike_iterative_refinement_enabled: bool = True,
		despike_iterative_max_removals_per_parent: int = 4,
		despike_cluster_cleanup_enabled: bool = True,
		despike_cluster_cleanup_max_passes: int = 1,
		despike_residual_check_enabled: bool = True,
		edge_use_enhanced_spike_mapping: bool = False,
		edge_mapping_levels_desc: Sequence[int] = (95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5),
		edge_mapping_refine_step_percent: int = 1,
		edge_mapping_min_level_percent: int = 1,
		edge_mapping_require_closed_interval: bool = True,
		edge_mapping_use_apex_component: bool = True,
		edge_mapping_enable_merge_guard: bool = True,
		edge_mapping_max_width_jump_factor: float = 2.5,
		edge_mapping_max_width_jump_points: float = 8.0,
		edge_mapping_fallback_to_old: bool = False,
		edge_mapping_noise_guard_enabled: bool = False,
		edge_robust_reference_enabled: bool = True,
		edge_noise_guard_enabled: bool = True,
		edge_noise_guard_factor: float = 3.0,
) -> Dict[str, Any]:
	"""Build diagnostic erosion-contact cells for already accepted parent spikes."""
	parent_metadata = parent_metadata or {}
	h, w, n = raw_spectra.shape
	pad = max(0, int(context_pad_pts))
	sensitivity = max(float(despike_sensitivity), 1e-6)
	parent_rows: List[Dict[str, Any]] = []
	for parent in parents:
		y = int(parent.y)
		x = int(parent.x)
		if not (0 <= y < h and 0 <= x < w):
			continue
		raw = raw_spectra[y, x, :].astype(float)
		bundle = None if small_morphology_by_pixel is None else small_morphology_by_pixel.get((int(y), int(x)))
		if bundle is not None and hasattr(bundle, "erosion") and hasattr(bundle, "dilation"):
			ero = np.asarray(bundle.erosion, dtype=float)
			dil = np.asarray(bundle.dilation, dtype=float)
		else:
			ero = erosion[y, x, :].astype(float)
			dil = dilation[y, x, :].astype(float)
		grad = gradient[y, x, :].astype(float) if gradient is not None and np.asarray(gradient).shape == raw_spectra.shape else None
		start = int(np.clip(parent.start, 0, n - 1))
		end = int(np.clip(parent.end, 0, n - 1))
		apex = int(np.clip(parent.peak_index, 0, n - 1))
		if start > end:
			start, end = end, start
		parent_id = f"primary:{int(y)}:{int(x)}:{int(apex)}:{int(start)}:{int(end)}"
		key = (y, x, apex, start, end)
		meta = dict(parent_metadata.get(key, {}))
		cl0 = max(0, start - pad)
		cr0 = min(n - 1, end + pad)
		all_erosion_contacts = None
		all_dilation_contacts = None
		if bundle is not None and hasattr(bundle, "erosion_contacts") and hasattr(bundle, "dilation_contacts"):
			all_erosion_contacts = np.asarray(bundle.erosion_contacts, dtype=int)
			all_dilation_contacts = np.asarray(bundle.dilation_contacts, dtype=int)
		cl = int(cl0)
		cr = int(cr0)
		left_candidates = np.asarray([], dtype=int)
		right_candidates = np.asarray([], dtype=int)
		if all_erosion_contacts is not None and all_erosion_contacts.size:
			left_candidates = all_erosion_contacts[all_erosion_contacts < int(apex)]
			right_candidates = all_erosion_contacts[all_erosion_contacts > int(apex)]
			if left_candidates.size and left_candidates[-1] < cl:
				cl = int(max(0, int(left_candidates[-1])))
			elif left_candidates.size == 0:
				cl = 0
			if right_candidates.size and right_candidates[0] > cr:
				cr = int(min(n - 1, int(right_candidates[0])))
			elif right_candidates.size == 0:
				cr = n - 1
		raw_ctx = raw[cl:cr + 1]
		local_noise = _mad(np.diff(raw_ctx)) / math.sqrt(2.0) if raw_ctx.size >= 3 else _mad(raw_ctx)
		local_noise = local_noise if np.isfinite(local_noise) and local_noise > 1e-12 else 1.0
		if all_erosion_contacts is not None and all_dilation_contacts is not None:
			erosion_contacts = all_erosion_contacts[(all_erosion_contacts >= cl) & (all_erosion_contacts <= cr)].astype(int)
			dilation_contacts = all_dilation_contacts[(all_dilation_contacts >= cl) & (all_dilation_contacts <= cr)].astype(int)
		elif bool(strict_equal):
			erosion_contacts = (np.flatnonzero(raw[cl:cr + 1] == ero[cl:cr + 1]) + cl).astype(int)
			dilation_contacts = (np.flatnonzero(raw[cl:cr + 1] == dil[cl:cr + 1]) + cl).astype(int)
		else:
			erosion_contacts = (np.flatnonzero(np.isclose(raw[cl:cr + 1], ero[cl:cr + 1])) + cl).astype(int)
			dilation_contacts = (np.flatnonzero(np.isclose(raw[cl:cr + 1], dil[cl:cr + 1])) + cl).astype(int)
		parent_left_foot_found = bool(np.any(erosion_contacts < int(apex)))
		parent_right_foot_found = bool(np.any(erosion_contacts > int(apex)))
		parent_used_spectrum_edge_as_foot = bool((not parent_left_foot_found and cl == 0) or (not parent_right_foot_found and cr == n - 1))
		if not parent_left_foot_found and cl == 0:
			erosion_contacts = np.unique(np.concatenate([np.asarray([0], dtype=int), erosion_contacts.astype(int)])).astype(int)
		if not parent_right_foot_found and cr == n - 1:
			erosion_contacts = np.unique(np.concatenate([erosion_contacts.astype(int), np.asarray([n - 1], dtype=int)])).astype(int)
		parent_left_foot_found = bool(np.any(erosion_contacts < int(apex))) or bool(cl == 0)
		parent_right_foot_found = bool(np.any(erosion_contacts > int(apex))) or bool(cr == n - 1)
		dilation_set = set(int(v) for v in dilation_contacts.tolist())
		cells: List[Dict[str, Any]] = []
		parent_apex_cell_index: Optional[int] = None
		if erosion_contacts.size >= 2:
			for idx, (left, right) in enumerate(zip(erosion_contacts[:-1], erosion_contacts[1:])):
				left = int(left)
				right = int(right)
				if right <= left:
					continue
				xs = np.asarray(x_axis[left:right + 1], dtype=float)
				ys = raw[left:right + 1].astype(float)
				chord = np.linspace(float(raw[left]), float(raw[right]), right - left + 1)
				excess = ys - chord
				above = np.maximum(excess, 0.0)
				below = np.maximum(-excess, 0.0)
				chord_area_above = float(np.nansum(above))
				chord_height_above = float(np.nanmax(above)) if above.size else 0.0
				chord_area_below = float(np.nansum(below))
				chord_area_total = float(chord_area_above + chord_area_below)
				height_abs = float(np.nanmax(np.abs(excess))) if excess.size else 0.0
				width = int(right - left + 1)
				height_z = chord_height_above / local_noise
				area_z = chord_area_above / (local_noise * max(width, 1))
				salience = float(max(height_z, math.sqrt(max(area_z, 0.0))))
				height_abs_z = float(height_abs / local_noise)
				area_total_z = float(chord_area_total / (local_noise * max(width, 1)))
				salience_total = float(max(height_abs_z, math.sqrt(max(area_total_z, 0.0))))
				salience_density = float(salience_total / math.sqrt(max(width, 1)))
				dil_inside = [int(v) for v in dilation_contacts.tolist() if left <= int(v) <= right]
				contains_apex = bool(left <= apex <= right)
				overlaps_parent = bool(max(left, start) <= min(right, end))
				if contains_apex:
					parent_apex_cell_index = int(idx)
				kind = _cell_kind(
					contains_parent_apex=contains_apex,
					overlaps_parent_segment=overlaps_parent,
					n_dilation_contacts=len(dil_inside),
					salience=salience,
					high_salience=3.0 / sensitivity,
				)
				cells.append({
					"cell_id": f"{parent_id}:cell:{int(idx)}",
					"parent_id": parent_id,
					"y": int(y),
					"x": int(x),
					"cell_index": int(idx),
					"cell_left": int(left),
					"cell_right": int(right),
					"cell_width": int(width),
					"contains_parent_apex": contains_apex,
					"overlaps_parent_segment": overlaps_parent,
					"is_left_of_parent": bool(right < start),
					"is_right_of_parent": bool(left > end),
					"dilation_contact_indices_inside_cell": dil_inside,
					"n_dilation_contacts_inside_cell": int(len(dil_inside)),
					"contains_dilation_contact": bool(len(dil_inside) > 0),
					"parent_apex_is_dilation_contact": bool(apex in dilation_set),
					"chord_area_above": chord_area_above,
					"chord_height_above": chord_height_above,
					"chord_area_below": chord_area_below,
					"chord_area_total": chord_area_total,
					"height_abs": height_abs,
					"chord_crosses_raw": bool(np.any(below > 1e-12)),
					"chord_compactness": float(chord_area_above / max(width * chord_height_above, 1e-12)) if chord_height_above > 0 else 0.0,
					"height_z": float(height_z),
					"area_z": float(area_z),
					"salience": salience,
					"height_abs_z": height_abs_z,
					"area_total_z": area_total_z,
					"salience_total": salience_total,
					"salience_density": salience_density,
					"cell_label": kind,
					"chord_x": [float(v) for v in xs],
					"chord_y": [float(v) for v in chord],
				})
		spectrum_noise_height = meta.get("noise_height_morph_range", meta.get("candidate_noise_estimate_used"))
		try:
			spectrum_noise_height = float(spectrum_noise_height)
		except Exception:
			spectrum_noise_height = float("nan")
		if not np.isfinite(spectrum_noise_height) or float(spectrum_noise_height) <= 0.0:
			spectrum_noise_height = float(local_noise)
		if cells:
			s_norm = _finite_minmax_norm([float(c.get("salience", np.nan)) for c in cells])
			t_norm = _finite_minmax_norm([float(c.get("salience_total", np.nan)) for c in cells])
			d_norm = _finite_minmax_norm([float(c.get("salience_density", np.nan)) for c in cells])
			for c, sv, tv, dv in zip(cells, s_norm, t_norm, d_norm):
				c["salience_norm"] = sv
				c["salience_total_norm"] = tv
				c["salience_density_norm"] = dv
				dil_inside = [int(v) for v in c.get("dilation_contact_indices_inside_cell", []) or []]
				if bool(c.get("contains_parent_apex", False)):
					anchor_idx = int(apex)
					anchor_source = "parent_apex"
				elif dil_inside:
					anchor_idx = int(max(dil_inside, key=lambda idx: float(raw[int(idx)])))
					anchor_source = "dilation_contact_max_raw"
				else:
					anchor_idx = int((int(c["cell_left"]) + int(c["cell_right"])) // 2)
					anchor_source = "cell_center"
				c["secondary_anchor_index"] = int(anchor_idx)
				c["secondary_anchor_source"] = str(anchor_source)
				cell_height, cell_chord_y = _cell_height_above_chord(raw, int(c["cell_left"]), int(c["cell_right"]), anchor_idx)
				cell_noise_ratio = (
					float(cell_height / spectrum_noise_height)
					if np.isfinite(cell_height) and np.isfinite(spectrum_noise_height) and spectrum_noise_height > 0.0
					else float("nan")
				)
				c["cell_height_above_chord"] = float(cell_height) if np.isfinite(cell_height) else float("nan")
				c["cell_chord_y_at_anchor"] = float(cell_chord_y) if np.isfinite(cell_chord_y) else float("nan")
				c["cell_height_ratio_to_noise"] = float(cell_noise_ratio) if np.isfinite(cell_noise_ratio) else float("nan")
				c["cell_noise_height_threshold"] = float(candidate_noise_height_factor)
				c["secondary_preclassification"] = "diagnostic_only"
				c["secondary_final_class"] = "unknown"
				c["secondary_final_is_spike"] = False
				c["secondary_final_source"] = "iterative_despike_pending"
				c["secondary_ss4_ran"] = False

		cells_overlapping = [int(c["cell_index"]) for c in cells if bool(c["overlaps_parent_segment"])]
		cells_inside = [int(c["cell_index"]) for c in cells if int(c["cell_left"]) >= start and int(c["cell_right"]) <= end]
		cells_left = [int(c["cell_index"]) for c in cells if bool(c["is_left_of_parent"])]
		cells_right = [int(c["cell_index"]) for c in cells if bool(c["is_right_of_parent"])]
		candidate_spike = [int(c["cell_index"]) for c in cells if c["cell_label"] == "candidate_spike_cell"]
		candidate_multi = [int(c["cell_index"]) for c in cells if c["cell_label"] == "candidate_multispike_cell"]
		uncertain = [int(c["cell_index"]) for c in cells if c["cell_label"] == "mixed_or_uncertain_cell"]
		stage_rows: List[Dict[str, Any]] = [{
			"stage_index": 0,
			"stage_name": "raw",
			"removed_by": "none",
			"working_stage_name": "raw",
			"removed_regions": [],
		}]
		final_chords: List[Dict[str, Any]] = []
		removed_regions: List[Tuple[int, int]] = []
		dirty_left = int(start)
		dirty_right = int(end)
		cells_by_index = {int(c["cell_index"]): c for c in cells}
		parent_cell = cells_by_index.get(int(parent_apex_cell_index)) if parent_apex_cell_index is not None else None
		if parent_cell is not None:
			parent_left = int(parent_cell["cell_left"])
			parent_right = int(parent_cell["cell_right"])
		else:
			left_ero = [int(v) for v in erosion_contacts.tolist() if int(v) < int(apex)]
			right_ero = [int(v) for v in erosion_contacts.tolist() if int(v) > int(apex)]
			parent_left = int(left_ero[-1]) if left_ero else int(start)
			parent_right = int(right_ero[0]) if right_ero else int(end)
		parent_chord = _build_final_chord(raw, parent_left, parent_right, required_indices=[int(apex)])
		parent_chord.update({
			"chord_id": f"{parent_id}:chord:{len(final_chords)}",
			"parent_id": parent_id,
			"cell_indices": ([] if parent_apex_cell_index is None else [int(parent_apex_cell_index)]),
			"y": int(y),
			"x": int(x),
			"chord_x": [float(x_axis[int(v)]) for v in parent_chord.get("chord_x_index", [])],
			"removed_by": "primary_parent",
			"removed_reason": "primary_parent",
			"removed_apex": int(apex),
		})
		final_chords.append(parent_chord)
		removed_regions.append((int(parent_chord["final_left_edge"]), int(parent_chord["final_right_edge"])))
		dirty_left = min(dirty_left, int(parent_chord["final_left_edge"]))
		dirty_right = max(dirty_right, int(parent_chord["final_right_edge"]))
		stage_rows.append({
			"stage_index": 1,
			"stage_name": "after_parent_removal",
			"working_stage_name": "primary_parent",
			"removed_by": "primary_parent",
			"removed_interval_left": int(parent_chord["final_left_edge"]),
			"removed_interval_right": int(parent_chord["final_right_edge"]),
			"removed_apex": int(apex),
			"removed_regions": [[int(lv), int(rv)] for lv, rv in removed_regions],
		})
		cluster_cleanup_applied = False
		cluster_cleanup_passes = 0
		iteration_rows: List[Dict[str, Any]] = []
		active_profile = str(decision_profile).strip().lower()
		iter_limit = max(1, int(despike_iterative_max_removals_per_parent))
		if bool(despike_iterative_refinement_enabled):
			for iter_idx in range(1, iter_limit):
				working = raw.copy()
				for chord in final_chords:
					try:
						fl = int(chord.get("final_left_edge"))
						fr = int(chord.get("final_right_edge"))
						chy = np.asarray(chord.get("chord_y", []), dtype=float)
					except Exception:
						continue
					if chy.size == max(fr - fl + 1, 0):
						working[fl:fr + 1] = chy
				_, _, _, _, iter_cells, _ = _build_working_cells(
					working,
					context_left=int(cl),
					context_right=int(cr),
					parent_start=int(start),
					parent_end=int(end),
					parent_apex=int(apex),
					strict_equal=bool(strict_equal),
					local_noise=float(local_noise),
				)
				candidates_iter: List[Dict[str, Any]] = []
				uncertain_dirty: List[Dict[str, Any]] = []
				for cell in iter_cells:
					left_i = int(cell["cell_left"])
					right_i = int(cell["cell_right"])
					if any(max(left_i, lv) < min(right_i, rv) for lv, rv in removed_regions):
						continue
					dil_inside = [int(v) for v in cell.get("dilation_contact_indices_inside_cell", []) or []]
					if dil_inside:
						apex_i = int(max(dil_inside, key=lambda idx: float(working[int(idx)])))
					else:
						apex_i = int(left_i + int(np.nanargmax(working[left_i:right_i + 1])))
					height_above = float(cell.get("height_abs", np.nan))
					height_ratio = float(height_above / max(float(spectrum_noise_height), 1e-12)) if np.isfinite(height_above) else float("nan")
					base_row = {
						"iteration_index": int(iter_idx),
						"candidate_left": int(left_i),
						"candidate_right": int(right_i),
						"candidate_apex": int(apex_i),
						"height_above_chord": float(height_above) if np.isfinite(height_above) else np.nan,
						"height_ratio_to_noise": float(height_ratio) if np.isfinite(height_ratio) else np.nan,
					}
					if not np.isfinite(height_ratio) or height_ratio < float(candidate_noise_height_factor):
						base_row["residual_status"] = "below_noise"
						iteration_rows.append(base_row)
						continue
					decision_row = _evaluate_active_residual_candidate(
						raw=working,
						gradient=grad,
						context_left=int(cl),
						context_right=int(cr),
						candidate_left=int(left_i),
						candidate_right=int(right_i),
						apex=int(apex_i),
						local_noise=float(local_noise),
						decision_profile=active_profile,
						ss4_ss_blue_max=float(ss4_ss_blue_max),
						ss4_ss_red_min=float(ss4_ss_red_min),
						ss4_pce_red_min=float(ss4_pce_red_min),
						ss4_rve_red_max=float(ss4_rve_red_max),
						ss4_missing_policy=str(ss4_missing_policy),
						ss5_ss1_threshold=float(ss5_ss1_threshold),
						ss5_pce_spike_min=float(ss5_pce_spike_min),
						ss5_edge_spike_max=float(ss5_edge_spike_max),
						recdw_center=float(recdw_center),
						recdw_scale=float(recdw_scale),
						recdw_z_clip=float(recdw_z_clip),
						recdw_support_z_scale=float(recdw_support_z_scale),
						edge_use_enhanced_spike_mapping=bool(edge_use_enhanced_spike_mapping),
						edge_mapping_levels_desc=tuple(int(v) for v in edge_mapping_levels_desc),
						edge_mapping_refine_step_percent=int(edge_mapping_refine_step_percent),
						edge_mapping_min_level_percent=int(edge_mapping_min_level_percent),
						edge_mapping_require_closed_interval=bool(edge_mapping_require_closed_interval),
						edge_mapping_use_apex_component=bool(edge_mapping_use_apex_component),
						edge_mapping_enable_merge_guard=bool(edge_mapping_enable_merge_guard),
						edge_mapping_max_width_jump_factor=float(edge_mapping_max_width_jump_factor),
						edge_mapping_max_width_jump_points=float(edge_mapping_max_width_jump_points),
						edge_mapping_fallback_to_old=bool(edge_mapping_fallback_to_old),
						edge_mapping_noise_guard_enabled=bool(edge_mapping_noise_guard_enabled),
						edge_robust_reference_enabled=bool(edge_robust_reference_enabled),
						edge_noise_guard_enabled=bool(edge_noise_guard_enabled),
						edge_noise_guard_factor=float(edge_noise_guard_factor),
						edge_noise_guard_value=float(spectrum_noise_height),
					)
					decision_payload = {**base_row, **decision_row}
					decision_payload["residual_status"] = "above_noise"
					decision_payload["inside_dirty_region"] = bool(max(left_i, dirty_left) <= min(right_i, dirty_right))
					iteration_rows.append(decision_payload)
					if str(decision_row.get("decision", "")).strip().lower() == "spike":
						candidates_iter.append(decision_payload)
					elif bool(decision_payload["inside_dirty_region"]):
						uncertain_dirty.append(decision_payload)
				if candidates_iter:
					best = max(candidates_iter, key=lambda row: float(row.get("height_ratio_to_noise", -np.inf)))
					chord = _build_final_chord(raw, int(best["candidate_left"]), int(best["candidate_right"]), required_indices=[int(best["candidate_apex"])])
					chord.update({
						"chord_id": f"{parent_id}:chord:{len(final_chords)}",
						"parent_id": parent_id,
						"cell_indices": [],
						"y": int(y),
						"x": int(x),
						"chord_x": [float(x_axis[int(v)]) for v in chord.get("chord_x_index", [])],
						"removed_by": f"residual_{active_profile}",
						"removed_reason": str(best.get("reason", "")),
						"removed_apex": int(best["candidate_apex"]),
					})
					final_chords.append(chord)
					removed_regions.append((int(chord["final_left_edge"]), int(chord["final_right_edge"])))
					dirty_left = min(dirty_left, int(chord["final_left_edge"]))
					dirty_right = max(dirty_right, int(chord["final_right_edge"]))
					stage_rows.append({
						"stage_index": int(len(stage_rows)),
						"stage_name": f"after_residual_removal_{iter_idx}",
						"working_stage_name": f"residual_{active_profile}",
						"removed_by": f"residual_{active_profile}",
						"removed_interval_left": int(chord["final_left_edge"]),
						"removed_interval_right": int(chord["final_right_edge"]),
						"removed_apex": int(best["candidate_apex"]),
						"height_above_chord": best.get("height_above_chord"),
						"height_ratio_to_noise": best.get("height_ratio_to_noise"),
						"ss1": best.get("ss1"),
						"pce": best.get("pce"),
						"edge": best.get("edge"),
						"decision_score": best.get("decision_score"),
						"decision": best.get("decision"),
						"reason": best.get("reason"),
						"removed_regions": [[int(lv), int(rv)] for lv, rv in removed_regions],
					})
					continue
				if bool(despike_cluster_cleanup_enabled) and int(cluster_cleanup_passes) < max(1, int(despike_cluster_cleanup_max_passes)) and uncertain_dirty:
					cluster_left = min([int(v[0]) for v in removed_regions] + [int(r["candidate_left"]) for r in uncertain_dirty])
					cluster_right = max([int(v[1]) for v in removed_regions] + [int(r["candidate_right"]) for r in uncertain_dirty])
					chord = _build_final_chord(raw, int(cluster_left), int(cluster_right), required_indices=[int(apex)])
					chord.update({
						"chord_id": f"{parent_id}:chord:{len(final_chords)}",
						"parent_id": parent_id,
						"cell_indices": [],
						"y": int(y),
						"x": int(x),
						"chord_x": [float(x_axis[int(v)]) for v in chord.get("chord_x_index", [])],
						"removed_by": "cluster_cleanup",
						"removed_reason": "uncertain_residual_inside_dirty_region",
						"removed_apex": int(apex),
					})
					final_chords.append(chord)
					removed_regions.append((int(chord["final_left_edge"]), int(chord["final_right_edge"])))
					dirty_left = min(dirty_left, int(chord["final_left_edge"]))
					dirty_right = max(dirty_right, int(chord["final_right_edge"]))
					cluster_cleanup_applied = True
					cluster_cleanup_passes += 1
					stage_rows.append({
						"stage_index": int(len(stage_rows)),
						"stage_name": "after_cluster_cleanup",
						"working_stage_name": "cluster_cleanup",
						"removed_by": "cluster_cleanup",
						"removed_interval_left": int(chord["final_left_edge"]),
						"removed_interval_right": int(chord["final_right_edge"]),
						"removed_apex": int(apex),
						"removed_regions": [[int(lv), int(rv)] for lv, rv in removed_regions],
					})
				break
		secondary_spike_cells = []
		secondary_uncertain_cells = []
		chord_cell_map: Dict[int, Dict[str, Any]] = {}
		for chord_idx, chord in enumerate(final_chords):
			chord["applied_to_corrected"] = True
			try:
				fl = int(chord.get("final_left_edge"))
				fr = int(chord.get("final_right_edge"))
			except Exception:
				continue
			for cell in cells:
				try:
					left_i = int(cell["cell_left"])
					right_i = int(cell["cell_right"])
				except Exception:
					continue
				if max(left_i, fl) < min(right_i, fr):
					cell_idx = int(cell["cell_index"])
					if cell_idx not in chord_cell_map:
						chord_cell_map[cell_idx] = {
							"chord_index": int(chord_idx),
							"chord_id": str(chord.get("chord_id", f"{parent_id}:chord:{chord_idx}")),
							"final_interval_id": str(chord.get("chord_id", f"{parent_id}:chord:{chord_idx}")),
							"removed_by": str(chord.get("removed_by", "")),
							"chord_left": fl,
							"chord_right": fr,
						}
		for cell in cells:
			cell_idx = int(cell["cell_index"])
			chord_info = chord_cell_map.get(cell_idx)
			active_for_current_chord = bool(chord_info is not None)
			cell["cell_classification"] = str(cell.get("secondary_preclassification", "diagnostic_only"))
			cell["active_for_current_chord"] = bool(active_for_current_chord)
			cell["final_interval_id"] = chord_info.get("final_interval_id") if chord_info is not None else None
			cell["assigned_interval_id"] = cell.get("final_interval_id")
			cell["chord_id"] = chord_info.get("chord_id") if chord_info is not None else None
			cell["chord_left"] = chord_info.get("chord_left") if chord_info is not None else None
			cell["chord_right"] = chord_info.get("chord_right") if chord_info is not None else None
			cell["applied_to_corrected"] = bool(active_for_current_chord)
			if active_for_current_chord:
				cell["secondary_final_is_spike"] = True
				cell["secondary_final_class"] = "spike"
				cell["secondary_final_source"] = str(chord_info.get("removed_by", "final_chord"))
				cell["skipped_reason"] = None
				secondary_spike_cells.append(cell_idx)
			else:
				cell["secondary_final_is_spike"] = False
				cell["secondary_final_class"] = "non_spike"
				cell["secondary_final_source"] = "iterative_preserved"
				cell["skipped_reason"] = "not_final_spike"
		secondary_groups = [
			{
				"cell_indices": [],
				"left": int(ch.get("final_left_edge")),
				"right": int(ch.get("final_right_edge")),
				"merge_rule": str(ch.get("removed_by", "iterative")),
			}
			for ch in final_chords
		]
		saliences = [float(c["salience"]) for c in cells]
		parent_salience = None
		if parent_apex_cell_index is not None:
			for c in cells:
				if int(c["cell_index"]) == int(parent_apex_cell_index):
					parent_salience = float(c["salience"])
					break
		island = []
		if parent_apex_cell_index is not None:
			by_idx = {int(c["cell_index"]): c for c in cells}
			island = [int(parent_apex_cell_index)]
			base_salience = max(float(parent_salience or 0.0), 1e-12)
			for direction in (-1, 1):
				cur = int(parent_apex_cell_index)
				while True:
					cur += direction
					c = by_idx.get(cur)
					if c is None:
						break
					compatible = float(c["salience"]) >= max(1.0, 0.35 * base_salience / sensitivity)
					has_dilation = bool(c["contains_dilation_contact"])
					if compatible or has_dilation:
						island.append(int(cur))
					else:
						break
			island = sorted(set(island))
		if island:
			island_cells = [c for c in cells if int(c["cell_index"]) in set(island)]
			island_left = min(int(c["cell_left"]) for c in island_cells)
			island_right = max(int(c["cell_right"]) for c in island_cells)
			island_reason = "grown_from_parent_apex_cell"
		else:
			island_left = None
			island_right = None
			island_reason = "no_parent_apex_cell"
		final_working = raw.copy()
		for chord in final_chords:
			try:
				fl = int(chord.get("final_left_edge"))
				fr = int(chord.get("final_right_edge"))
				chy = np.asarray(chord.get("chord_y", []), dtype=float)
			except Exception:
				continue
			if chy.size == max(fr - fl + 1, 0):
				final_working[fl:fr + 1] = chy
		residual_check_status = "not_run"
		residual_check_n_above_noise = 0
		residual_check_n_spike_like = 0
		if bool(despike_residual_check_enabled):
			_, _, _, _, final_cells_iter, _ = _build_working_cells(
				final_working,
				context_left=int(cl),
				context_right=int(cr),
				parent_start=int(start),
				parent_end=int(end),
				parent_apex=int(apex),
				strict_equal=bool(strict_equal),
				local_noise=float(local_noise),
			)
			for cell in final_cells_iter:
				left_i = int(cell["cell_left"])
				right_i = int(cell["cell_right"])
				if any(max(left_i, lv) < min(right_i, rv) for lv, rv in removed_regions):
					continue
				height_ratio = float(cell.get("height_abs", np.nan) / max(float(spectrum_noise_height), 1e-12)) if np.isfinite(cell.get("height_abs", np.nan)) else float("nan")
				if np.isfinite(height_ratio) and height_ratio >= float(candidate_noise_height_factor):
					residual_check_n_above_noise += 1
					if bool(cell.get("contains_dilation_contact", False)):
						residual_check_n_spike_like += 1
			if residual_check_n_above_noise == 0:
				residual_check_status = "clean"
			elif residual_check_n_spike_like > 0:
				residual_check_status = "residual_spike_like_found"
			else:
				residual_check_status = "ambiguous_residual_manual_review"
		primary_ss4 = meta.get("primary_ss4", meta.get("ss4", 1.0))
		primary_ss4_reason = meta.get("primary_ss4_reason", meta.get("ss4_reason"))
		primary_ss4_decision = meta.get("primary_ss4_decision", meta.get("ss4_decision"))
		primary_ss1 = meta.get("primary_spike_score_v1", meta.get("spike_score_v1"))
		primary_pce = meta.get("primary_pce_negpref_t098_evidence_signed", meta.get("pce_negpref_t098_evidence_signed"))
		primary_edge = meta.get("primary_recdw_sum_0_90_raman_veto_evidence_signed", meta.get("recdw_sum_0_90_raman_veto_evidence_signed"))
		primary_edge_feature = meta.get("primary_ss4_rve_feature", meta.get("ss4_rve_feature"))
		primary_ss5 = meta.get("primary_ss5", meta.get("ss5"))
		primary_ss5_reason = meta.get("primary_ss5_reason", meta.get("ss5_reason"))
		primary_ss5_decision = meta.get("primary_ss5_decision", meta.get("ss5_decision"))
		primary_active_profile = meta.get("primary_active_decision_profile", "ss4")
		primary_active_score = meta.get("primary_active_score", primary_ss5 if str(primary_active_profile) == "ss5" else primary_ss4)
		primary_active_decision = meta.get("primary_active_decision", primary_ss5_decision if str(primary_active_profile) == "ss5" else primary_ss4_decision)
		primary_active_reason = meta.get("primary_active_reason", primary_ss5_reason if str(primary_active_profile) == "ss5" else primary_ss4_reason)
		summary = {
			"n_erosion_contacts": int(erosion_contacts.size),
			"n_dilation_contacts": int(dilation_contacts.size),
			"n_cells": int(len(cells)),
			"parent_apex_cell_index": parent_apex_cell_index,
			"cells_overlapping_parent_segment": cells_overlapping,
			"cells_inside_parent_segment": cells_inside,
			"cells_left_of_parent": cells_left,
			"cells_right_of_parent": cells_right,
			"n_dilation_contacts_inside_parent_segment": int(sum(1 for v in dilation_contacts.tolist() if start <= int(v) <= end)),
			"dilation_contacts_inside_parent_segment": [int(v) for v in dilation_contacts.tolist() if start <= int(v) <= end],
			"max_cell_salience": max(saliences) if saliences else None,
			"salience_of_parent_apex_cell": parent_salience,
			"candidate_spike_cell_indices": candidate_spike,
			"candidate_multispike_cell_indices": candidate_multi,
			"uncertain_cell_indices": uncertain,
			"secondary_noise_thr": float(secondary_noise_thr),
			"secondary_uncertain_thr": float(secondary_uncertain_thr),
			"secondary_spike_cell_indices": secondary_spike_cells,
			"secondary_uncertain_cell_indices": secondary_uncertain_cells,
			"secondary_final_spike_intervals": secondary_groups,
			"final_despike_chords": final_chords,
			"despike_stages": stage_rows,
			"iteration_rows": iteration_rows,
			"iterative_removals_count": int(max(0, len(final_chords) - 1)),
			"cluster_cleanup_applied": bool(cluster_cleanup_applied),
			"cluster_cleanup_passes": int(cluster_cleanup_passes),
			"dirty_region_left": int(dirty_left),
			"dirty_region_right": int(dirty_right),
			"residual_check_status": str(residual_check_status),
			"residual_check_n_above_noise": int(residual_check_n_above_noise),
			"residual_check_n_spike_like": int(residual_check_n_spike_like),
			"degenerate_salience_context": bool(len(cells) <= 2 or _has_only_trivial_binary_norm([float(c.get("salience_total_norm", np.nan)) for c in cells])),
			"preliminary_island_cell_indices": island,
			"preliminary_island_left": island_left,
			"preliminary_island_right": island_right,
			"preliminary_island_reason": island_reason,
			"incomplete_contact_segmentation": bool(erosion_contacts.size < 2),
		}
		parent_rows.append({
			"parent_id": parent_id,
			"candidate_id": meta.get("candidate_id", parent_id),
			"y": y,
			"x": x,
			"parent_start": int(start),
			"parent_end": int(end),
			"parent_apex": int(apex),
			"parent_peak_height": float(parent.peak_height),
			"primary_spike_score_v1": _clean_value(primary_ss1),
			"primary_pce_negpref_t098_evidence_signed": _clean_value(primary_pce),
			"primary_recdw_sum_0_90_raman_veto_evidence_signed": _clean_value(primary_edge),
			"primary_ss4": _clean_value(primary_ss4),
			"primary_ss4_decision": primary_ss4_decision,
			"primary_ss4_reason": primary_ss4_reason,
			"primary_ss4_rve_feature": primary_edge_feature,
			"primary_ss5": _clean_value(primary_ss5),
			"primary_ss5_decision": primary_ss5_decision,
			"primary_ss5_reason": primary_ss5_reason,
			"primary_active_decision_profile": primary_active_profile,
			"primary_active_score": _clean_value(primary_active_score),
			"primary_active_decision": primary_active_decision,
			"primary_active_reason": primary_active_reason,
			"parent_ss4_value": _clean_value(primary_ss4),
			"parent_ss4_reason": primary_ss4_reason,
			"parent_ss1": _clean_value(primary_ss1),
			"parent_pce": _clean_value(primary_pce),
			"parent_edge": _clean_value(primary_edge),
			"parent_edge_feature": primary_edge_feature,
			"decision_profile_used": active_profile,
			"context_left_initial": int(cl0),
			"context_right_initial": int(cr0),
			"context_left_final": int(cl),
			"context_right_final": int(cr),
			"context_expanded_left_pts": int(max(0, cl0 - cl)),
			"context_expanded_right_pts": int(max(0, cr - cr0)),
			"parent_left_foot_found": bool(parent_left_foot_found),
			"parent_right_foot_found": bool(parent_right_foot_found),
			"parent_used_spectrum_edge_as_foot": bool(parent_used_spectrum_edge_as_foot),
			"context_left": int(cl),
			"context_right": int(cr),
			"local_noise": float(local_noise),
			"noise_height_morph_range": float(spectrum_noise_height),
			"candidate_noise_height_factor": float(candidate_noise_height_factor),
			"erosion_contacts": [int(v) for v in erosion_contacts.tolist()],
			"erosion_contact_x": [float(x_axis[int(v)]) for v in erosion_contacts.tolist()],
			"erosion_contact_y": [float(raw[int(v)]) for v in erosion_contacts.tolist()],
			"dilation_contacts": [int(v) for v in dilation_contacts.tolist()],
			"dilation_contact_x": [float(x_axis[int(v)]) for v in dilation_contacts.tolist()],
			"dilation_contact_y": [float(raw[int(v)]) for v in dilation_contacts.tolist()],
			"cells": cells,
			"summary": summary,
		})
	result = {
		"method": "erosion_dilation_contact_cell_analysis",
		"n_parent_segments": int(len(parent_rows)),
		"config": {
			"despike_contact_context_pad_pts": int(context_pad_pts),
			"despike_contact_strict_equal": bool(strict_equal),
			"despike_sensitivity": float(despike_sensitivity),
			"decision_profile": str(decision_profile),
			"secondary_noise_thr": float(secondary_noise_thr),
			"secondary_uncertain_thr": float(secondary_uncertain_thr),
			"ss4_ss_blue_max": float(ss4_ss_blue_max),
			"ss4_ss_red_min": float(ss4_ss_red_min),
			"ss4_pce_red_min": float(ss4_pce_red_min),
			"ss4_rve_red_max": float(ss4_rve_red_max),
			"ss5_ss1_threshold": float(ss5_ss1_threshold),
			"ss5_pce_spike_min": float(ss5_pce_spike_min),
			"ss5_edge_spike_max": float(ss5_edge_spike_max),
			"despike_iterative_refinement_enabled": bool(despike_iterative_refinement_enabled),
			"despike_iterative_max_removals_per_parent": int(despike_iterative_max_removals_per_parent),
			"despike_cluster_cleanup_enabled": bool(despike_cluster_cleanup_enabled),
			"despike_cluster_cleanup_max_passes": int(despike_cluster_cleanup_max_passes),
			"despike_residual_check_enabled": bool(despike_residual_check_enabled),
			"secondary_edge_rescue_ss_min": float(secondary_edge_rescue_ss_min),
			"candidate_noise_height_factor": float(candidate_noise_height_factor),
			"edge_noise_guard_enabled": bool(edge_noise_guard_enabled),
			"edge_noise_guard_factor": float(edge_noise_guard_factor),
		},
		"parents": parent_rows,
	}
	return _clean_value(result)


def save_despike_contact_debug_json(path: Path, analysis: Mapping[str, Any]) -> None:
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	Path(path).write_text(json.dumps(_clean_value(dict(analysis)), indent=2), encoding="utf-8")


def build_despike_contact_debug_payload(
		analysis: Mapping[str, Any],
		*,
		mode: str = "lite",
		store_arrays: bool = False,
		source_to_compact: Optional[Mapping[Tuple[int, int], Tuple[int, int]]] = None,
) -> Dict[str, Any]:
	mode_norm = str(mode).strip().lower()
	base = _clean_value(dict(analysis))
	if mode_norm not in {"lite", "full"}:
		mode_norm = "lite"
	if mode_norm == "full" and store_arrays:
		return dict(base)

	def _strip_cell(cell: Mapping[str, Any], parent_idx: int) -> Dict[str, Any]:
		row = {
			"parent_id": f"parent_{parent_idx}",
			"cell_id": f"parent_{parent_idx}_cell_{int(cell.get('cell_index', -1))}",
			"cell_index": cell.get("cell_index"),
			"cell_left": cell.get("cell_left"),
			"cell_right": cell.get("cell_right"),
			"cell_width": cell.get("cell_width"),
			"contains_parent_apex": cell.get("contains_parent_apex"),
			"overlaps_parent_segment": cell.get("overlaps_parent_segment"),
			"contains_dilation_contact": cell.get("contains_dilation_contact"),
			"n_dilation_contacts_inside_cell": cell.get("n_dilation_contacts_inside_cell"),
			"salience_norm": cell.get("salience_norm"),
			"salience_total_norm": cell.get("salience_total_norm"),
			"salience_density_norm": cell.get("salience_density_norm"),
			"secondary_preclassification": cell.get("secondary_preclassification"),
			"cell_classification": cell.get("cell_classification"),
			"secondary_final_class": cell.get("secondary_final_class"),
			"secondary_final_is_spike": cell.get("secondary_final_is_spike"),
			"secondary_final_source": cell.get("secondary_final_source"),
			"active_for_current_chord": cell.get("active_for_current_chord"),
			"skipped_reason": cell.get("skipped_reason"),
			"final_interval_id": cell.get("final_interval_id"),
			"assigned_interval_id": cell.get("assigned_interval_id"),
			"chord_id": cell.get("chord_id"),
			"chord_left": cell.get("chord_left"),
			"chord_right": cell.get("chord_right"),
			"applied_to_corrected": cell.get("applied_to_corrected"),
			"secondary_ss4_ran": cell.get("secondary_ss4_ran"),
			"secondary_ss4": cell.get("secondary_ss4"),
			"secondary_ss4_decision": cell.get("secondary_ss4_decision"),
			"secondary_ss4_reason": cell.get("secondary_ss4_reason"),
			"secondary_ss4_reason_for_run": cell.get("secondary_ss4_reason_for_run"),
			"secondary_ss1": cell.get("secondary_ss1"),
			"secondary_pce": cell.get("secondary_pce"),
			"secondary_edge": cell.get("secondary_edge"),
			"secondary_anchor_index": cell.get("secondary_anchor_index"),
			"secondary_anchor_source": cell.get("secondary_anchor_source"),
			"cell_low_t_degenerate_context": cell.get("cell_low_t_degenerate_context"),
			"cell_height_above_chord": cell.get("cell_height_above_chord"),
			"cell_height_ratio_to_noise": cell.get("cell_height_ratio_to_noise"),
			"cell_noise_height_threshold": cell.get("cell_noise_height_threshold"),
		}
		if mode_norm == "full":
			for key, value in cell.items():
				if key in row:
					continue
				if not store_arrays and isinstance(value, list) and len(value) > 32:
					continue
				if not store_arrays and key in {"chord_x", "chord_y"}:
					continue
				row[key] = value
		return row

	def _strip_chord(chord: Mapping[str, Any], parent_idx: int, chord_idx: int) -> Dict[str, Any]:
		chord_id = chord.get("chord_id", f"parent_{parent_idx}_chord_{chord_idx}")
		row = {
			"parent_id": f"parent_{parent_idx}",
			"chord_id": chord_id,
			"cell_indices": chord.get("cell_indices"),
			"chord_method": chord.get("chord_method"),
			"fixed_side": chord.get("fixed_side"),
			"original_left_edge": chord.get("original_left_edge"),
			"original_right_edge": chord.get("original_right_edge"),
			"final_left_edge": chord.get("final_left_edge"),
			"final_right_edge": chord.get("final_right_edge"),
			"touch_index": chord.get("touch_index"),
			"max_overshoot_before": chord.get("max_overshoot_before"),
			"max_overshoot_after": chord.get("max_overshoot_after"),
			"crossing_count_before": chord.get("crossing_count_before"),
			"crossing_count_after": chord.get("crossing_count_after"),
			"applied_to_corrected": chord.get("applied_to_corrected"),
		}
		if mode_norm == "full":
			for key, value in chord.items():
				if key in row:
					continue
				if not store_arrays and key in {"chord_x", "chord_y", "chord_x_index"}:
					continue
				row[key] = value
		return row

	parents_out: List[Dict[str, Any]] = []
	for parent_idx, parent in enumerate(base.get("parents", []) or []):
		if not isinstance(parent, dict):
			continue
		py = int(parent.get("y", -1)) if parent.get("y") is not None else -1
		px = int(parent.get("x", -1)) if parent.get("x") is not None else -1
		compact = source_to_compact.get((py, px)) if source_to_compact is not None else None
		summary = dict(parent.get("summary", {}) or {})
		chords = [_strip_chord(ch, parent_idx, chord_idx) for chord_idx, ch in enumerate(summary.get("final_despike_chords", []) or []) if isinstance(ch, dict)]
		summary["final_despike_chords"] = chords
		parent_row: Dict[str, Any] = {
			"parent_id": f"parent_{parent_idx}",
			"source_y": py,
			"source_x": px,
			"compact_y": int(compact[0]) if compact is not None else None,
			"compact_x": int(compact[1]) if compact is not None else None,
			"parent_start": parent.get("parent_start"),
			"parent_end": parent.get("parent_end"),
			"parent_apex": parent.get("parent_apex"),
			"parent_peak_height": parent.get("parent_peak_height"),
			"parent_ss4_value": parent.get("parent_ss4_value"),
			"parent_ss4_reason": parent.get("parent_ss4_reason"),
			"parent_ss1": parent.get("parent_ss1"),
			"parent_pce": parent.get("parent_pce"),
			"parent_edge": parent.get("parent_edge"),
			"parent_edge_feature": parent.get("parent_edge_feature"),
			"context_left_initial": parent.get("context_left_initial"),
			"context_right_initial": parent.get("context_right_initial"),
			"context_left_final": parent.get("context_left_final"),
			"context_right_final": parent.get("context_right_final"),
			"context_expanded_left_pts": parent.get("context_expanded_left_pts"),
			"context_expanded_right_pts": parent.get("context_expanded_right_pts"),
			"parent_left_foot_found": parent.get("parent_left_foot_found"),
			"parent_right_foot_found": parent.get("parent_right_foot_found"),
			"parent_used_spectrum_edge_as_foot": parent.get("parent_used_spectrum_edge_as_foot"),
			"context_left": parent.get("context_left"),
			"context_right": parent.get("context_right"),
			"n_erosion_contacts": summary.get("n_erosion_contacts"),
			"n_dilation_contacts": summary.get("n_dilation_contacts"),
			"n_cells": summary.get("n_cells"),
			"summary": summary,
			"cells": [_strip_cell(c, parent_idx) for c in (parent.get("cells", []) or []) if isinstance(c, dict)],
		}
		if mode_norm == "full":
			for key, value in parent.items():
				if key in parent_row:
					continue
				if not store_arrays and key in {"erosion_contact_x", "erosion_contact_y", "dilation_contact_x", "dilation_contact_y"}:
					continue
				parent_row[key] = value
		parents_out.append(parent_row)
	return {
		"method": base.get("method"),
		"n_parent_segments": base.get("n_parent_segments"),
		"config": base.get("config", {}),
		"parents": parents_out,
	}
