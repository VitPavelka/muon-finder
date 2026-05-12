from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


TOPHAT_RAW_EXPORT_FEATURES = {
	"mt_area_small_z",
	"mt_area_large_fraction",
	"mt_large_dominance",
	"mt_scale_com",
	"mt_width_d1_mean",
	"mt_area_d1_min",
}

TOPHAT_GRADIENT_EXPORT_FEATURES = {
	"gmt_area_d2_max_abs",
	"gmt_width_soft_d1_mean",
	"gmt_width_soft_slope",
	"gmt_scale_curvature",
	"gws_width_auc_norm",
	"gws_width_scale_at_50",
	"gws_width_scale_at_80",
	"gws_width_stepiness",
	"gws_width_max_d1_norm",
	"gws_area_auc_norm",
	"gws_area_scale_at_50",
	"gws_area_scale_at_80",
	"gws_area_stepiness",
	"gws_area_max_d1_norm",
	"gws_area_above_thr_auc_norm",
	"gws_area_above_thr_scale_at_50",
	"gws_area_above_thr_stepiness",
	"gws_width_auc_norm_evidence_signed",
	"gws_width_scale_at_50_evidence_signed",
	"gws_width_scale_at_80_evidence_signed",
	"gws_width_stepiness_evidence_signed",
	"gws_width_max_d1_norm_evidence_signed",
	"gws_area_auc_norm_evidence_signed",
	"gws_area_scale_at_50_evidence_signed",
	"gws_area_scale_at_80_evidence_signed",
	"gws_area_stepiness_evidence_signed",
	"gws_area_max_d1_norm_evidence_signed",
	"gws_area_above_thr_auc_norm_evidence_signed",
	"gws_area_above_thr_scale_at_50_evidence_signed",
	"gws_area_above_thr_stepiness_evidence_signed",
}

TOPHAT_LOG1P_EXPORT_MAP = {
	"mt_log1p_area_small_z": "mt_area_small_z",
	"mt_log1p_area_mean": "mt_area_mean",
	"mt_log1p_area_auc_scale": "mt_area_auc_scale",
	"mt_log1p_area_d1_max_abs": "mt_area_d1_max_abs",
	"mt_log1p_area_d2_max_abs": "mt_area_d2_max_abs",
	"gmt_log1p_area_small_z": "gmt_area_small_z",
	"gmt_log1p_area_mean": "gmt_area_mean",
	"gmt_log1p_area_auc_scale": "gmt_area_auc_scale",
	"gmt_log1p_area_d1_max_abs": "gmt_area_d1_max_abs",
	"gmt_log1p_area_d2_max_abs": "gmt_area_d2_max_abs",
}

TOPHAT_FEATURE_WHITELIST = set().union(
	TOPHAT_RAW_EXPORT_FEATURES,
	TOPHAT_GRADIENT_EXPORT_FEATURES,
	set(TOPHAT_LOG1P_EXPORT_MAP.keys()),
)

# Empirical calibration values from the current reference dataset.
# These may need adjustment for a different experiment, laser, grating,
# spectral resolution, or preprocessing chain.
SS_CANDIDATE_LOW = 0.95
SS_FULL = 0.9913
SS_STRONG_LOW = 0.9913
SS_STRONG_HIGH = 1.0

PCE_SUPPORT_FULL = -1992.0
PCE_SUPPORT_ZERO = -273.0
PCE_VETO_ZERO = -273.0
PCE_VETO_FULL = 503.0

GWS_LEFT_OUTER = -0.55
GWS_LEFT_INNER = -0.375
GWS_RIGHT_INNER = 0.20
GWS_RIGHT_OUTER = 0.35
GWS_GRANULO_SCALES = (3, 5, 7, 9, 11, 13, 15)
GWS_SOURCE = "morph_gradient"
GWS_EXPERIMENT_SOURCE_MODES = (
	"morph_gradient",
	"morph_gradient_med3",
	"morph_gradient_med5",
	"morph_gradient_mean3",
	"morph_gradient_mean5",
)
GWS_SOURCE_PREFIX_BY_MODE = {
	"morph_gradient": "gws_mg",
	"morph_gradient_med3": "gws_medgrad3",
	"morph_gradient_med5": "gws_medgrad5",
	"morph_gradient_mean3": "gws_meangrad3",
	"morph_gradient_mean5": "gws_meangrad5",
}
GWS_MEASURE_REGION = "mask"
GWS_THRESHOLD_REGION = "spike_edges"
GWS_COMPACT_ACTIVE_METRICS = (
	"gmt_width_soft_slope_ctx10",
	"gmt_width_soft_d1_mean_ctx10",
	"gws_support01_ctx10",
	"gws_evidence_signed_ctx10",
	"gws_support_initial_fraction",
	"gws_support_total_increase",
	"gws_support_total_increase_fraction",
	"gws_support_final",
	"gws_support_final_fraction",
	"gws_support_final_minus_initial",
	"gws_support_final_minus_initial_fraction",
	"gws_support_max",
	"gws_support_max_fraction",
	"gws_support_total_abs_change",
	"gws_support_total_abs_change_fraction",
	"gws_support_max_abs_change",
	"gws_support_num_increases",
	"gws_support_num_changes",
	"gws_support_change_density",
	"gws_support_longest_constant_run_norm",
)
EDGE_WIDTH_ACTIVE_METRICS = (
	"raw_edge_width_at_80",
	"raw_edge_width_at_50",
	"raw_edge_width_at_20",
	"raw_edge_width_50_over_80",
	"raw_edge_width_20_over_80",
	"raw_edge_width_20_minus_80",
	"raw_edge_width_50_minus_80",
	"raw_edge_base_expansion_rate",
	"mg_edge_width_at_80",
	"mg_edge_width_at_50",
	"mg_edge_width_at_20",
	"mg_edge_width_50_over_80",
	"mg_edge_width_20_over_80",
	"mg_edge_width_20_minus_80",
	"mg_edge_width_50_minus_80",
	"mg_edge_base_expansion_rate",
)
TH_SHAPE_ACTIVE_METRICS = (
	"th_width_at_80",
	"th_width_at_50",
	"th_width_at_20",
	"th_width_50_over_80",
	"th_width_20_over_80",
	"th_width_20_minus_80",
	"th_width_50_minus_80",
	"th_base_expansion_rate",
	"th_area_total",
	"th_area_above_80",
	"th_area_above_50",
	"th_area_above_20",
	"th_area_above_80_fraction",
	"th_area_above_50_fraction",
	"th_area_above_20_fraction",
	"th_area_core_fraction_1pt",
	"th_area_core_fraction_2pt",
	"th_area_core_fraction_3pt",
	"th_points_to_50_area",
	"th_points_to_80_area",
	"th_points_to_50_area_norm",
	"th_points_to_80_area_norm",
)
GWS_INCLUDE_SCALE_ZERO = False
GWS_SPLIT_OVERLAPPING_CONTEXTS = False
GWS_SPLIT_SOURCE = "gradient"
GWS_SPLIT_SMOOTH_PTS = 3
GWS_SPLIT_VALLEY_ALPHA = 0.75
GWS_SPLIT_MIN_DISTANCE_FROM_APEX = 1
GWS_SPLIT_MIN_CONTEXT_WIDTH = 3
GWS_SPLIT_DEBUG = True

MDWS510_SUPPORT_FULL = 0.0
MDWS510_SUPPORT_ZERO = 3.0
MDWS510_VETO_ZERO = 3.0
MDWS510_VETO_FULL = 12.0

CONTEXT_PADS = (5, 10)
CURVATURE_NEGPREF_LOCAL_RADIUS = 3
CURVATURE_NEGPREF_TOLERANCES = (0.90, 0.95, 0.98)


def _tolerance_tag(value: float) -> str:
	"""Encode a tolerance like 0.95 as a stable metric suffix like t095."""
	v = int(round(100.0 * float(value)))
	return f"t{v:03d}"


def _components_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
	"""Return contiguous [start, end] components for a 1D boolean mask."""
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


def _component_spacing(comps: List[Tuple[int, int]]) -> float:
	"""Mean spacing between neighboring component centers in index units."""
	if len(comps) < 2:
		return 0.0
	centers = np.array([(a + b) * 0.5 for (a, b) in comps], dtype=float)
	d = np.diff(np.sort(centers))
	return float(np.mean(d)) if d.size else 0.0


def ramp_up(x: float, low: float, high: float) -> float:
	"""Return 0 below low, 1 above high, and linear interpolation between."""
	try:
		xv = float(x)
		lo = float(low)
		hi = float(high)
	except Exception:
		return 0.0
	if not np.isfinite(xv) or not np.isfinite(lo) or not np.isfinite(hi):
		return 0.0
	if hi <= lo:
		return 1.0 if xv >= hi else 0.0
	if xv <= lo:
		return 0.0
	if xv >= hi:
		return 1.0
	return float((xv - lo) / (hi - lo))


def ramp_down(x: float, low: float, high: float) -> float:
	"""Return 1 below low, 0 above high, and linear interpolation between."""
	return float(1.0 - ramp_up(x, low, high))


def interval_membership(
		x: float,
		left_outer: float,
		left_inner: float,
		right_inner: float,
		right_outer: float,
) -> float:
	"""Return fuzzy interval membership with outer zero zones and inner full support."""
	try:
		xv = float(x)
		lo = float(left_outer)
		li = float(left_inner)
		ri = float(right_inner)
		ro = float(right_outer)
	except Exception:
		return 0.0
	vals = np.array([xv, lo, li, ri, ro], dtype=float)
	if not np.all(np.isfinite(vals)):
		return 0.0
	if lo > li:
		lo, li = li, lo
	if ri > ro:
		ri, ro = ro, ri
	if li > ri:
		li, ri = ri, li
	if xv <= lo or xv >= ro:
		return 0.0
	if li <= xv <= ri:
		return 1.0
	if xv < li:
		return ramp_up(xv, lo, li)
	return ramp_down(xv, ri, ro)


def to_signed_evidence(support01: float, veto01: float = 0.0) -> float:
	"""Convert bounded support and veto components to signed evidence in [-1, 1]."""
	try:
		support = float(support01)
		veto = float(veto01)
	except Exception:
		return 0.0
	if not np.isfinite(support):
		support = 0.0
	if not np.isfinite(veto):
		veto = 0.0
	return float(np.clip(support - veto, -1.0, 1.0))


def estimate_background_mad(
		signal: np.ndarray,
		detection_left: int,
		detection_right: int,
		*,
		context_pad_factor: float = 3.0,
		context_min_pad: int = 10,
) -> float:
	"""Estimate background MAD using the same candidate-centered rule as the viewer."""
	x = np.asarray(signal, dtype=float)
	n = int(x.size)
	if n <= 0:
		return 1e-12
	a = int(np.clip(detection_left, 0, n - 1))
	b = int(np.clip(detection_right, 0, n - 1))
	width = max(1, b - a + 1)
	context = max(int(context_min_pad), int(round(float(context_pad_factor) * float(width))))
	l0 = max(0, a - context)
	r1 = min(n, b + context + 1)
	bg = np.concatenate([x[l0:a], x[b + 1:r1]])
	if bg.size < 5:
		bg = np.concatenate([x[:a], x[b + 1:]])
	if bg.size < 5:
		bg = x
	bg_med = float(np.median(bg))
	bg_mad = float(np.median(np.abs(bg - bg_med)))
	return max(bg_mad, 1e-12)


def compute_spike_score_v2_features(
		features: Dict[str, float],
		*,
		ss_candidate_low: float = SS_CANDIDATE_LOW,
		ss_full: float = SS_FULL,
		ss_strong_low: float = SS_STRONG_LOW,
		ss_strong_high: float = SS_STRONG_HIGH,
		pce_support_full: float = PCE_SUPPORT_FULL,
		pce_support_zero: float = PCE_SUPPORT_ZERO,
		pce_veto_zero: float = PCE_VETO_ZERO,
		pce_veto_full: float = PCE_VETO_FULL,
		gws_left_outer: float = GWS_LEFT_OUTER,
		gws_left_inner: float = GWS_LEFT_INNER,
		gws_right_inner: float = GWS_RIGHT_INNER,
		gws_right_outer: float = GWS_RIGHT_OUTER,
) -> Dict[str, float]:
	"""Build experimental bounded support/evidence features for spike decision logic."""
	ss_v1 = float(features.get("spike_score_v1", 0.0))
	pce = float(features.get("peak_curvature_extreme", 0.0))
	pce_apex = float(features.get("peak_curvature_apex_at_peak", 0.0))
	gws = float(features.get("gmt_width_soft_d1_mean", 0.0))

	ss_gate01 = ramp_up(ss_v1, ss_candidate_low, ss_full)
	ss_strong01 = ramp_up(ss_v1, ss_strong_low, ss_strong_high)

	pce_support01 = ramp_down(pce, pce_support_full, pce_support_zero)
	pce_veto01 = ramp_up(pce, pce_veto_zero, pce_veto_full)
	pce_evidence_signed = to_signed_evidence(pce_support01, pce_veto01)
	pce_apex_support01 = ramp_down(pce_apex, pce_support_full, pce_support_zero)
	pce_apex_veto01 = ramp_up(pce_apex, pce_veto_zero, pce_veto_full)
	pce_apex_evidence_signed = to_signed_evidence(pce_apex_support01, pce_apex_veto01)
	pce_z = float(features.get("peak_curvature_extreme_z", 0.0))
	if np.isfinite(pce_z) and abs(pce_z) > 1e-12:
		bg_scale = max(abs(pce / pce_z), 1e-12)
	else:
		pce_apex_z = float(features.get("peak_curvature_apex_at_peak_z", 0.0))
		if np.isfinite(pce_apex_z) and abs(pce_apex_z) > 1e-12:
			bg_scale = max(abs(pce_apex / pce_apex_z), 1e-12)
		else:
			bg_scale = 1.0

	gws_support01 = interval_membership(
		gws,
		gws_left_outer,
		gws_left_inner,
		gws_right_inner,
		gws_right_outer,
	)
	gws_evidence_signed = float(np.clip(2.0 * gws_support01 - 1.0, -1.0, 1.0))

	spike_score_v2_or = max(
		ss_strong01,
		ss_gate01 * max(pce_support01, gws_support01),
	)
	spike_score_v2_weighted = max(
		ss_strong01,
		ss_gate01 * (0.65 * pce_support01 + 0.35 * gws_support01),
	)
	spike_score_v2_conservative = max(
		ss_strong01,
		ss_gate01 * (0.75 * pce_support01 + 0.25 * gws_support01),
	)
	spike_score_v2_signed_evidence = float(
		ss_gate01 * (0.75 * pce_evidence_signed + 0.25 * gws_evidence_signed)
	)
	out = {
		"ss_gate01": float(ss_gate01),
		"ss_strong01": float(ss_strong01),
		"pce_support01": float(pce_support01),
		"pce_veto01": float(pce_veto01),
		"pce_evidence_signed": float(pce_evidence_signed),
		"pce_apex_support01": float(pce_apex_support01),
		"pce_apex_veto01": float(pce_apex_veto01),
		"pce_apex_evidence_signed": float(pce_apex_evidence_signed),
		"gws_support01": float(gws_support01),
		"gws_evidence_signed": float(gws_evidence_signed),
		"spike_score_v2_or": float(spike_score_v2_or),
		"spike_score_v2_weighted": float(spike_score_v2_weighted),
		"spike_score_v2_conservative": float(spike_score_v2_conservative),
		"spike_score_v2_signed_evidence": float(spike_score_v2_signed_evidence),
	}
	for tol in CURVATURE_NEGPREF_TOLERANCES:
		tag = _tolerance_tag(tol)
		raw_key = f"peak_curvature_extreme_negpref_{tag}"
		raw_z_key = f"{raw_key}_z"
		raw_val = float(features.get(raw_key, 0.0))
		raw_z_val = float(features.get(raw_z_key, 0.0))
		support01 = ramp_down(raw_val, pce_support_full, pce_support_zero)
		veto01 = ramp_up(raw_val, pce_veto_zero, pce_veto_full)
		out[f"pce_negpref_{tag}_support01"] = float(support01)
		out[f"pce_negpref_{tag}_veto01"] = float(veto01)
		out[f"pce_negpref_{tag}_evidence_signed"] = float(to_signed_evidence(support01, veto01))
		pce_support_full_z = float(pce_support_full / bg_scale)
		pce_support_zero_z = float(pce_support_zero / bg_scale)
		pce_veto_zero_z = float(pce_veto_zero / bg_scale)
		pce_veto_full_z = float(pce_veto_full / bg_scale)
		support01_z = ramp_down(raw_z_val, pce_support_full_z, pce_support_zero_z)
		veto01_z = ramp_up(raw_z_val, pce_veto_zero_z, pce_veto_full_z)
		out[f"pce_negpref_{tag}_z_support01"] = float(support01_z)
		out[f"pce_negpref_{tag}_z_veto01"] = float(veto01_z)
		out[f"pce_negpref_{tag}_z_evidence_signed"] = float(to_signed_evidence(support01_z, veto01_z))
	return out


def compute_context_window_experiment_features(
		raw_signal: np.ndarray,
		gradient_signal: np.ndarray,
		*,
		apex_idx: int,
		detection_left: int,
		detection_right: int,
		bg_mad: float,
		context_pads: Sequence[int] = CONTEXT_PADS,
		curvature_search_radius: int = 3,
		gws_context_infos_by_pad: Optional[Dict[int, Dict[str, object]]] = None,
		gws_source_modes: Sequence[str] = GWS_EXPERIMENT_SOURCE_MODES,
		gws_include_scale_zero: bool = GWS_INCLUDE_SCALE_ZERO,
		gws_measure_region: str = GWS_MEASURE_REGION,
		gws_threshold_region: str = GWS_THRESHOLD_REGION,
) -> Dict[str, float]:
	"""Compute a small whitelist of context-window diagnostics around a fixed apex."""
	raw = np.asarray(raw_signal, dtype=float)
	grad = np.asarray(gradient_signal, dtype=float)
	pads = []
	for p in context_pads:
		try:
			pad = int(p)
		except Exception:
			continue
		if pad < 0 or pad in pads:
			continue
		pads.append(pad)

	zero_key_prefixes = (
		"gmt_width_soft_d1_mean",
		"gmt_width_soft_slope",
		"gws_support01",
		"gws_evidence_signed",
		"medres_local_base_support_w5",
		"medres_width_w5",
		"medres_width_soft_w5",
		"medres_support_after_center_removal_w5",
		"opres_support_after_center_removal_w125",
		"opres_width_w125",
		"opres_area_after_center_removal_w125",
		"peak_curvature_extreme_centered",
		"pce_support01_centered",
	)
	zero_keys = tuple(
		f"{prefix}_ctx{pad}"
		for pad in pads
		for prefix in zero_key_prefixes
	)
	if 10 in pads:
		zero_keys = zero_keys + ("mdws510_evidence_signed",)
	out = {key: 0.0 for key in zero_keys}
	if raw.ndim != 1 or raw.size < 5:
		return out
	if grad.ndim != 1 or grad.size != raw.size:
		grad = np.zeros_like(raw)
	opening_full = compute_opening_residual_features(raw, opening_windows=(125,))
	opening_residual_full = np.asarray(opening_full.get("opening_residual_w125", np.zeros_like(raw)), dtype=float)

	n = int(raw.size)
	apex = int(np.clip(apex_idx, 0, n - 1))
	left = int(np.clip(detection_left, 0, n - 1))
	right = int(np.clip(detection_right, 0, n - 1))
	if right < left:
		left, right = right, left

	for pad in pads:
		ctx_l = max(0, left - pad)
		ctx_r = min(n - 1, right + pad)
		raw_ctx = raw[ctx_l:ctx_r + 1]
		apex_rel = int(np.clip(apex - ctx_l, 0, raw_ctx.size - 1))

		gws_diag = compute_gws_diagnostics(
			grad,
			raw_signal=raw,
			apex_idx=apex,
			detection_left=left,
			detection_right=right,
			bg_mad=float(bg_mad),
			scales=GWS_GRANULO_SCALES,
			context_mode="fixed_pad",
			context_fixed_pad=int(pad),
			context_info=(None if gws_context_infos_by_pad is None else gws_context_infos_by_pad.get(int(pad))),
			include_scale_zero=False,
			measure_region=gws_measure_region,
			threshold_region=gws_threshold_region,
		)
		gmt_width_soft_d1_mean = float(gws_diag.get("width_soft_d1_mean", 0.0))
		gmt_width_soft_slope = float(gws_diag.get("width_soft_slope", 0.0))
		gws_support01 = float(gws_diag.get("gws_support01", 0.0))
		gws_evidence_signed = float(gws_diag.get("gws_evidence_signed", -1.0))
		out[f"gmt_width_soft_d1_mean_ctx{pad}"] = gmt_width_soft_d1_mean
		out[f"gmt_width_soft_slope_ctx{pad}"] = gmt_width_soft_slope
		out[f"gws_support01_ctx{pad}"] = gws_support01
		out[f"gws_evidence_signed_ctx{pad}"] = gws_evidence_signed
		if pad == 10:
			for source_mode in gws_source_modes:
				mode_name = _normalize_gws_source_mode(source_mode)
				prefix = GWS_SOURCE_PREFIX_BY_MODE.get(mode_name, "")
				if not prefix:
					continue
				mode_diag = compute_gws_diagnostics(
					grad,
					raw_signal=raw,
					apex_idx=apex,
					detection_left=left,
					detection_right=right,
					bg_mad=float(bg_mad),
					scales=GWS_GRANULO_SCALES,
					context_mode="fixed_pad",
					context_fixed_pad=int(pad),
					context_info=(None if gws_context_infos_by_pad is None else gws_context_infos_by_pad.get(int(pad))),
					source_mode=mode_name,
					include_scale_zero=False,
					measure_region=gws_measure_region,
					threshold_region=gws_threshold_region,
				)
				out.update(compute_compact_gws_metrics(mode_diag, ctx_suffix="ctx10", prefix=prefix))
				if bool(gws_include_scale_zero):
					mode_diag_z0 = compute_gws_diagnostics(
						grad,
						raw_signal=raw,
						apex_idx=apex,
						detection_left=left,
						detection_right=right,
						bg_mad=float(bg_mad),
						scales=GWS_GRANULO_SCALES,
						context_mode="fixed_pad",
						context_fixed_pad=int(pad),
						context_info=(None if gws_context_infos_by_pad is None else gws_context_infos_by_pad.get(int(pad))),
						source_mode=mode_name,
						include_scale_zero=True,
						measure_region=gws_measure_region,
						threshold_region=gws_threshold_region,
					)
					out.update(compute_compact_gws_metrics(mode_diag_z0, ctx_suffix="ctx10", prefix=f"{prefix}_z0"))

		med_ctx = compute_residual_anatomy_features(raw_ctx, apex_rel, bg_mad, mode="median", windows=(5,))
		for name in (
			"medres_local_base_support_w5",
			"medres_width_w5",
			"medres_width_soft_w5",
			"medres_support_after_center_removal_w5",
		):
			out[f"{name}_ctx{pad}"] = float(med_ctx.get(name, 0.0))
		if pad == 10:
			mdws510 = float(med_ctx.get("medres_width_soft_w5", 0.0))
			mdws510_support01 = float(ramp_down(mdws510, MDWS510_SUPPORT_FULL, MDWS510_SUPPORT_ZERO))
			mdws510_veto01 = float(ramp_up(mdws510, MDWS510_VETO_ZERO, MDWS510_VETO_FULL))
			out["mdws510_evidence_signed"] = float(to_signed_evidence(mdws510_support01, mdws510_veto01))

		op_ctx = compute_single_residual_context_metrics(
			opening_residual_full[ctx_l:ctx_r + 1],
			apex_rel,
			bg_mad,
			center_radius=2,
			shoulder_inner=4,
			shoulder_outer=14,
			local_radius=28,
		)
		for name in (
			"support_after_center_removal",
			"width",
			"area_after_center_removal",
		):
			out[f"opres_{name}_w125_ctx{pad}"] = float(op_ctx.get(name, 0.0))

		shape_ctx = compute_local_shape_features(
			raw_ctx,
			apex_rel,
			core_radius=1,
			center_radius=2,
			shoulder_inner=3,
			shoulder_outer=8,
			curvature_search_radius=int(curvature_search_radius),
		)
		pce_centered = float(shape_ctx.get("curvature_extreme_centered", 0.0))
		out[f"peak_curvature_extreme_centered_ctx{pad}"] = pce_centered
		out[f"pce_support01_centered_ctx{pad}"] = float(
			ramp_down(
				pce_centered,
				PCE_SUPPORT_FULL,
				PCE_SUPPORT_ZERO,
			)
		)

	return out


def compute_support_after_core_removal_features(
		raw_segment: np.ndarray,
		peak_rel: int,
		bg_mad: float,
) -> Dict[str, float]:
	"""Measure how much coherent support remains after removing a narrow apex core."""
	seg = np.asarray(raw_segment, dtype=float)
	if seg.size < 5:
		return {
			'support_after_core_removal_ratio': 0.0,
			'residual_area_after_core_removal': 0.0,
			'residual_component_count_after_core': 0.0,
		}

	peak_rel = int(np.clip(peak_rel, 0, seg.size - 1))
	base = float(np.median(seg))
	pos = np.maximum(seg - base, 0.0)
	peak_amp = float(np.max(pos))
	thr = max(0.20 * peak_amp, 1.5 * float(bg_mad))
	support_mask = pos >= thr
	total_support = int(np.count_nonzero(support_mask))

	core_hw = max(1, int(round(seg.size * 0.10)))
	l = max(0, peak_rel - core_hw)
	r = min(seg.size - 1, peak_rel + core_hw)
	rem_mask = support_mask.copy()
	rem_mask[l : r + 1] = False

	remaining_support = int(np.count_nonzero(rem_mask))
	rem_area = float(np.sum(pos[rem_mask]))
	comps = _components_from_mask(rem_mask)

	return {
		'support_after_core_removal_ratio': float(remaining_support / max(1, total_support)),
		'residual_area_after_core_removal': rem_area,
		'residual_component_count_after_core': float(len(comps)),
	}


def compute_curvature_support_features(
		raw_wide: np.ndarray,
		peak_rel: int,
		bg_mad: float,
) -> Dict[str, float]:
	"""Describe how second-derivative support is distributed in a wider neighborhood."""
	x = np.asarray(raw_wide, dtype=float)
	if x.size < 7:
		return {
			'curv2_support_width': 0.0,
			'curv2_abs_integral': 0.0,
			'curv2_apex_to_surround_ratio': 0.0,
			'curv2_component_count': 0.0,
		}

	d2 = np.diff(x, n=2)
	ad2 = np.abs(d2)
	med = float(np.median(ad2))
	mad = float(np.median(np.abs(d2 - med)))
	thr = med + max(1.5 * mad, 1.0 * float(bg_mad))
	mask = ad2 >= thr
	comps = _components_from_mask(mask)

	# map peak index from x to d2 index domain (offset by 1)
	pk = int(np.clip(peak_rel - 1, 0, ad2.size - 1))
	core_hw = max(1, int(round(ad2.size * 0.08)))
	la = max(0, pk - core_hw)
	ra = min(ad2.size - 1, pk + core_hw)
	core = ad2[la : ra + 1]
	sur = np.concatenate([ad2[:la], ad2[ra + 1 :]])

	apex = float(np.max(core)) if core.size else 0.0
	surround = float(np.mean(sur)) if sur.size else 0.0
	return {
		'curv2_support_width': float(np.count_nonzero(mask)),
		'curv2_abs_integral': float(np.sum(ad2)),
		'curv2_apex_to_surround_ratio': float(apex / max(surround, 1e-12)),
		'curv2_component_count': float(len(comps)),
	}


def _valid_odd_windows(length: int, windows: Sequence[int]) -> List[int]:
	"""Normalize and keep valid odd window sizes for a 1D signal length."""
	n = int(length)
	out: List[int] = []
	for w in windows:
		try:
			wi = int(w)
		except Exception:
			continue
		if wi <= 1 or wi % 2 == 0 or wi > n:
			continue
		if wi not in out:
			out.append(wi)
	return out


def compute_median_residual_features(
		y: np.ndarray,
		*,
		median_windows: Sequence[int] = (5, 7, 9),
) -> Dict[str, np.ndarray]:
	"""Compute median-filtered signals and raw-minus-median residuals."""
	x = np.asarray(y, dtype=float)
	if x.ndim != 1 or x.size < 3:
		return {}

	out: Dict[str, np.ndarray] = {}
	for w in _valid_odd_windows(x.size, median_windows):
		med = ndimage.median_filter(x, size=int(w), mode="nearest")
		out[f"median_w{w}"] = med.astype(float, copy=False)
		out[f"median_residual_w{w}"] = (x - med).astype(float, copy=False)
	return out


def compute_opening_residual_features(
		y: np.ndarray,
		*,
		opening_windows: Sequence[int] = (5, 7, 9, 15, 21),
) -> Dict[str, np.ndarray]:
	"""Compute grey-opening signals and residual representations."""
	x = np.asarray(y, dtype=float)
	if x.ndim != 1 or x.size < 3:
		return {}

	out: Dict[str, np.ndarray] = {}
	for w in _valid_odd_windows(x.size, opening_windows):
		opn = ndimage.grey_opening(x, size=int(w))
		res = x - opn
		out[f"opening_w{w}"] = opn.astype(float, copy=False)
		out[f"opening_residual_w{w}"] = res.astype(float, copy=False)
		out[f"opening_tophat_w{w}"] = np.maximum(res, 0.0).astype(float, copy=False)
	return out


def compute_large_window_median_residuals(
		y: np.ndarray,
		*,
		median_windows: Sequence[int] = (81, 101, 125),
) -> Dict[str, np.ndarray]:
	"""Compute large-window median-filter diagnostic representations."""
	return compute_median_residual_features(y, median_windows=median_windows)


def compute_large_window_opening_residuals(
		y: np.ndarray,
		*,
		opening_windows: Sequence[int] = (81, 101, 125),
) -> Dict[str, np.ndarray]:
	"""Compute large-window opening-based diagnostic representations."""
	return compute_opening_residual_features(y, opening_windows=opening_windows)


def compute_single_residual_context_metrics(
		residual_signal: np.ndarray,
		apex_idx: int,
		bg_mad: float,
		*,
		center_radius: int = 2,
		shoulder_inner: int = 4,
		shoulder_outer: int = 14,
		local_radius: int = 28,
) -> Dict[str, float]:
	"""Compute a small residual-anatomy set from one precomputed residual signal."""
	x = np.asarray(residual_signal, dtype=float)
	eps = 1e-12
	out = {
		"apex": 0.0,
		"area": 0.0,
		"width": 0.0,
		"width_soft": 0.0,
		"support_after_center_removal": 0.0,
		"area_after_center_removal": 0.0,
		"center_to_total_area_ratio": 0.0,
		"shoulder_to_center_ratio": 0.0,
		"local_base_support": 0.0,
		"area_left_shoulder": 0.0,
		"area_right_shoulder": 0.0,
		"left_right_balance": 0.0,
	}
	if x.ndim != 1 or x.size < 5:
		return out

	apex = int(np.clip(apex_idx, 0, x.size - 1))
	bg = max(float(bg_mad), eps)
	center_radius = max(0, int(center_radius))
	shoulder_inner = max(center_radius + 1, int(shoulder_inner))
	shoulder_outer = max(shoulder_inner, int(shoulder_outer))
	local_radius = max(shoulder_outer + 2, int(local_radius))

	pos = np.maximum(x, 0.0)
	l0 = max(0, apex - local_radius)
	r0 = min(x.size - 1, apex + local_radius)
	pos_local = pos[l0:r0 + 1]
	if pos_local.size == 0:
		return out

	center_l = max(l0, apex - center_radius)
	center_r = min(r0, apex + center_radius)
	left_sh_l = max(l0, apex - shoulder_outer)
	left_sh_r = max(l0, apex - shoulder_inner + 1)
	right_sh_l = min(r0 + 1, apex + shoulder_inner)
	right_sh_r = min(r0 + 1, apex + shoulder_outer + 1)

	center_area = float(np.sum(pos[center_l:center_r + 1]))
	left_area = float(np.sum(pos[left_sh_l:left_sh_r])) if left_sh_l < left_sh_r else 0.0
	right_area = float(np.sum(pos[right_sh_l:right_sh_r])) if right_sh_l < right_sh_r else 0.0
	shoulder_area = left_area + right_area
	total_area = float(np.sum(pos_local))

	apex_pos = float(max(pos[apex], 0.0))
	thr = max(0.15 * apex_pos, bg)
	thr_soft = max(0.08 * apex_pos, 0.5 * bg)
	width = float(np.count_nonzero(pos_local >= thr)) if apex_pos > 0.0 else 0.0
	width_soft = float(np.count_nonzero(pos_local >= thr_soft)) if apex_pos > 0.0 else 0.0

	keep_mask = np.ones(pos_local.size, dtype=bool)
	keep_mask[(center_l - l0):(center_r - l0 + 1)] = False
	support_after_center = float(np.count_nonzero(pos_local[keep_mask] >= thr)) if apex_pos > 0.0 else 0.0
	area_after_center = float(np.sum(pos_local[keep_mask]))
	local_base_support = float(np.count_nonzero(pos_local[keep_mask] >= thr_soft)) if apex_pos > 0.0 else 0.0
	lr_max = max(left_area, right_area)

	out.update(
		{
			"apex": float(x[apex]),
			"area": total_area,
			"width": width,
			"width_soft": width_soft,
			"support_after_center_removal": support_after_center,
			"area_after_center_removal": area_after_center,
			"center_to_total_area_ratio": float(center_area / max(total_area, eps)),
			"shoulder_to_center_ratio": float(shoulder_area / max(center_area, eps)),
			"local_base_support": local_base_support,
			"area_left_shoulder": left_area,
			"area_right_shoulder": right_area,
			"left_right_balance": float(min(left_area, right_area) / max(lr_max, eps)) if lr_max > 0.0 else 0.0,
		}
	)
	return out


def compute_cross_window_change_features(
		values_by_window: Dict[int, float],
		*,
		prefix: str,
		windows: Sequence[int] = (81, 101, 125),
) -> Dict[str, float]:
	"""Summarize how one scalar residual metric changes across large windows."""
	eps = 1e-12
	req_windows = [int(w) for w in windows]
	out: Dict[str, float] = {}
	for a, b in zip(req_windows[:-1], req_windows[1:]):
		out[f"{prefix}_delta_{a}_{b}"] = 0.0
	out[f"{prefix}_slope_large_windows"] = 0.0
	out[f"{prefix}_ratio_{req_windows[0]}_{req_windows[-1]}"] = 0.0
	out[f"{prefix}_curvature_large_windows"] = 0.0

	valid_pairs = [(int(w), float(values_by_window[w])) for w in req_windows if int(w) in values_by_window]
	if len(valid_pairs) < 2:
		return out

	scales = np.asarray([w for w, _ in valid_pairs], dtype=float)
	vals = np.asarray([v for _, v in valid_pairs], dtype=float)
	for a, b in zip(req_windows[:-1], req_windows[1:]):
		if a in values_by_window and b in values_by_window:
			out[f"{prefix}_delta_{a}_{b}"] = float(values_by_window[b] - values_by_window[a])
	if scales.size >= 2:
		out[f"{prefix}_slope_large_windows"] = float((vals[-1] - vals[0]) / max(scales[-1] - scales[0], eps))
		out[f"{prefix}_ratio_{req_windows[0]}_{req_windows[-1]}"] = float(vals[-1] / max(vals[0], eps))
	if len(req_windows) >= 3 and all(w in values_by_window for w in req_windows[:3]):
		out[f"{prefix}_curvature_large_windows"] = float(
			values_by_window[req_windows[0]] - 2.0 * values_by_window[req_windows[1]] + values_by_window[req_windows[2]]
		)
	return out


def compute_residual_anatomy_features(
		raw_signal: np.ndarray,
		apex_idx: int,
		bg_mad: float,
		*,
		mode: str,
		windows: Sequence[int] = (81, 101, 125),
		center_radius: int = 2,
		shoulder_inner: int = 4,
		shoulder_outer: int = 14,
		local_radius: int = 28,
) -> Dict[str, float]:
	"""Describe residual apex/body anatomy for large median/opening windows."""
	x = np.asarray(raw_signal, dtype=float)
	eps = 1e-12
	prefix = "medres" if str(mode).strip().lower() == "median" else "opres"
	req_windows = [int(w) for w in windows]

	per_window_keys = (
		"apex",
		"area",
		"width",
		"width_soft",
		"support_after_center_removal",
		"area_after_center_removal",
		"center_to_total_area_ratio",
		"shoulder_to_center_ratio",
		"local_base_support",
		"area_left_shoulder",
		"area_right_shoulder",
		"left_right_balance",
	)
	out: Dict[str, float] = {
		f"{prefix}_{name}_w{w}": 0.0
		for w in req_windows
		for name in per_window_keys
	}
	for metric_name in ("area", "width", "width_soft", "support_after_center_removal", "area_after_center_removal", "local_base_support"):
		out.update(compute_cross_window_change_features({}, prefix=f"{prefix}_{metric_name}", windows=req_windows))
	if x.ndim != 1 or x.size < 5:
		return out

	apex = int(np.clip(apex_idx, 0, x.size - 1))
	bg = max(float(bg_mad), eps)
	center_radius = max(0, int(center_radius))
	shoulder_inner = max(center_radius + 1, int(shoulder_inner))
	shoulder_outer = max(shoulder_inner, int(shoulder_outer))
	local_radius = max(shoulder_outer + 2, int(local_radius))

	if prefix == "medres":
		residual_bank = compute_large_window_median_residuals(x, median_windows=req_windows)
		residual_key = lambda w: f"median_residual_w{w}"
	else:
		residual_bank = compute_large_window_opening_residuals(x, opening_windows=req_windows)
		residual_key = lambda w: f"opening_residual_w{w}"

	metric_maps: Dict[str, Dict[int, float]] = {
		"area": {},
		"width": {},
		"width_soft": {},
		"support_after_center_removal": {},
		"area_after_center_removal": {},
		"local_base_support": {},
	}

	for w in req_windows:
		residual = residual_bank.get(residual_key(w))
		if residual is None or residual.size != x.size:
			continue
		res = np.asarray(residual, dtype=float)
		pos = np.maximum(res, 0.0)
		l0 = max(0, apex - local_radius)
		r0 = min(x.size - 1, apex + local_radius)
		pos_local = pos[l0:r0 + 1]
		if pos_local.size == 0:
			continue

		center_l = max(l0, apex - center_radius)
		center_r = min(r0, apex + center_radius)
		left_sh_l = max(l0, apex - shoulder_outer)
		left_sh_r = max(l0, apex - shoulder_inner + 1)
		right_sh_l = min(r0 + 1, apex + shoulder_inner)
		right_sh_r = min(r0 + 1, apex + shoulder_outer + 1)

		center_area = float(np.sum(pos[center_l:center_r + 1]))
		left_area = float(np.sum(pos[left_sh_l:left_sh_r])) if left_sh_l < left_sh_r else 0.0
		right_area = float(np.sum(pos[right_sh_l:right_sh_r])) if right_sh_l < right_sh_r else 0.0
		shoulder_area = left_area + right_area
		total_area = float(np.sum(pos_local))

		apex_pos = float(max(pos[apex], 0.0))
		thr = max(0.15 * apex_pos, bg)
		thr_soft = max(0.08 * apex_pos, 0.5 * bg)
		width = float(np.count_nonzero(pos_local >= thr)) if apex_pos > 0.0 else 0.0
		width_soft = float(np.count_nonzero(pos_local >= thr_soft)) if apex_pos > 0.0 else 0.0

		keep_mask = np.ones(pos_local.size, dtype=bool)
		keep_mask[(center_l - l0):(center_r - l0 + 1)] = False
		support_after_center = float(np.count_nonzero(pos_local[keep_mask] >= thr)) if apex_pos > 0.0 else 0.0
		area_after_center = float(np.sum(pos_local[keep_mask]))
		local_base_support = float(np.count_nonzero(pos_local[keep_mask] >= thr_soft)) if apex_pos > 0.0 else 0.0
		lr_max = max(left_area, right_area)

		out[f"{prefix}_apex_w{w}"] = float(res[apex])
		out[f"{prefix}_area_w{w}"] = total_area
		out[f"{prefix}_width_w{w}"] = width
		out[f"{prefix}_width_soft_w{w}"] = width_soft
		out[f"{prefix}_support_after_center_removal_w{w}"] = support_after_center
		out[f"{prefix}_area_after_center_removal_w{w}"] = area_after_center
		out[f"{prefix}_center_to_total_area_ratio_w{w}"] = float(center_area / max(total_area, eps))
		out[f"{prefix}_shoulder_to_center_ratio_w{w}"] = float(shoulder_area / max(center_area, eps))
		out[f"{prefix}_local_base_support_w{w}"] = local_base_support
		out[f"{prefix}_area_left_shoulder_w{w}"] = left_area
		out[f"{prefix}_area_right_shoulder_w{w}"] = right_area
		out[f"{prefix}_left_right_balance_w{w}"] = float(min(left_area, right_area) / max(lr_max, eps)) if lr_max > 0.0 else 0.0

		metric_maps["area"][w] = total_area
		metric_maps["width"][w] = width
		metric_maps["width_soft"][w] = width_soft
		metric_maps["support_after_center_removal"][w] = support_after_center
		metric_maps["area_after_center_removal"][w] = area_after_center
		metric_maps["local_base_support"][w] = local_base_support

	for metric_name, values in metric_maps.items():
		out.update(
			compute_cross_window_change_features(
				values,
				prefix=f"{prefix}_{metric_name}",
				windows=req_windows,
			)
		)
	return out


def compute_local_shape_features(
		segment: np.ndarray,
		apex_idx: int,
		*,
		core_radius: int = 1,
		center_radius: int = 2,
		shoulder_inner: int = 3,
		shoulder_outer: int = 8,
		curvature_search_radius: int = 3,
) -> Dict[str, float]:
	"""Compute low-cost local shape features around one candidate apex."""
	x = np.asarray(segment, dtype=float)
	eps = 1e-12
	zero_out = {
		'apex_excess_over_shoulders': 0.0,
		'center_removed_support_ratio': 0.0,
		'local_line_residual_at_apex': 0.0,
		'curvature_at_apex': 0.0,
		'curvature_extreme_centered': 0.0,
		'curvature_extreme_offset': 0.0,
		'curvature_extreme_abs_offset': 0.0,
		'central_negative_curvature_ratio': 0.0,
		'positive_side_lobe_ratio': 0.0,
		'side_lobe_balance': 0.0,
		'support_after_center_removal': 0.0,
	}
	if x.size < 5:
		return zero_out

	apex = int(np.clip(apex_idx, 0, x.size - 1))
	core_radius = max(0, int(core_radius))
	center_radius = max(0, int(center_radius))
	shoulder_inner = max(1, int(shoulder_inner))
	shoulder_outer = max(shoulder_inner, int(shoulder_outer))
	curvature_search_radius = max(1, int(curvature_search_radius))

	local_med = float(np.median(x))
	local_mad = float(np.median(np.abs(x - local_med)))

	left_sh_start = max(0, apex - shoulder_outer)
	left_sh_end = max(0, apex - shoulder_inner + 1)
	right_sh_start = min(x.size, apex + shoulder_inner)
	right_sh_end = min(x.size, apex + shoulder_outer + 1)
	left_shoulder = x[left_sh_start:left_sh_end] if left_sh_start < left_sh_end else np.array([], dtype=float)
	right_shoulder = x[right_sh_start:right_sh_end] if right_sh_start < right_sh_end else np.array([], dtype=float)
	shoulder_vals = np.concatenate([left_shoulder, right_shoulder]) if (left_shoulder.size or right_shoulder.size) else np.array([], dtype=float)
	shoulder_median = float(np.median(shoulder_vals)) if shoulder_vals.size >= 2 else local_med
	apex_excess = float((x[apex] - shoulder_median) / max(local_mad, eps)) if shoulder_vals.size >= 2 else 0.0

	win_l = max(0, apex - shoulder_outer)
	win_r = min(x.size - 1, apex + shoulder_outer)
	window = x[win_l:win_r + 1]
	full_pos = np.maximum(window - shoulder_median, 0.0)
	full_support = float(np.sum(full_pos))
	rem_l = max(win_l, apex - center_radius)
	rem_r = min(win_r, apex + center_radius)
	keep_mask = np.ones(window.size, dtype=bool)
	keep_mask[(rem_l - win_l):(rem_r - win_l + 1)] = False
	support_after_removal = float(np.sum(full_pos[keep_mask])) if full_pos.size else 0.0
	center_removed_support_ratio = float(support_after_removal / max(full_support, eps)) if full_support > 0.0 else 0.0

	left_core_start = max(0, apex - shoulder_outer)
	left_core_end = max(0, apex - core_radius)
	right_core_start = min(x.size, apex + core_radius + 1)
	right_core_end = min(x.size, apex + shoulder_outer + 1)
	left_anchor = x[left_core_start:left_core_end] if left_core_start < left_core_end else np.array([], dtype=float)
	right_anchor = x[right_core_start:right_core_end] if right_core_start < right_core_end else np.array([], dtype=float)
	if left_anchor.size >= 1 and right_anchor.size >= 1:
		left_idx = int(round(0.5 * (left_core_start + left_core_end - 1)))
		right_idx = int(round(0.5 * (right_core_start + right_core_end - 1)))
		left_val = float(np.median(left_anchor))
		right_val = float(np.median(right_anchor))
		if right_idx != left_idx:
			t = float((apex - left_idx) / (right_idx - left_idx))
			line_value = float((1.0 - t) * left_val + t * right_val)
			local_line_residual = float((x[apex] - line_value) / max(local_mad, eps))
		else:
			local_line_residual = 0.0
	else:
		local_line_residual = 0.0

	if not (0 < apex < x.size - 1):
		return {
			**zero_out,
			'apex_excess_over_shoulders': apex_excess,
			'center_removed_support_ratio': center_removed_support_ratio,
			'local_line_residual_at_apex': local_line_residual,
			'support_after_center_removal': support_after_removal,
		}

	d2 = x[:-2] - 2.0 * x[1:-1] + x[2:]
	apex_d2_idx = apex - 1
	d2_at_apex = float(d2[apex_d2_idx]) if 0 <= apex_d2_idx < d2.size else 0.0
	search_l = max(0, apex_d2_idx - curvature_search_radius)
	search_r = min(d2.size - 1, apex_d2_idx + curvature_search_radius)
	d2_local = d2[search_l:search_r + 1]
	if d2_local.size:
		i_ext = int(np.argmax(np.abs(d2_local)))
		d2_ext = float(d2_local[i_ext])
		ext_d2_idx = search_l + i_ext
		ext_signal_idx = ext_d2_idx + 1
		curvature_extreme_offset = float(ext_signal_idx - apex)
	else:
		d2_ext = 0.0
		curvature_extreme_offset = 0.0

	total_abs = float(np.sum(np.abs(d2_local))) if d2_local.size else 0.0
	central_neg = float(max(0.0, -d2_at_apex))
	central_negative_curvature_ratio = float(central_neg / max(total_abs, eps)) if total_abs > 0.0 else 0.0

	left_lobe = 0.0
	if search_l <= apex_d2_idx - 1:
		left_lobe = float(max(0.0, np.max(d2[search_l:apex_d2_idx])))
	right_lobe = 0.0
	if apex_d2_idx + 1 <= search_r:
		right_lobe = float(max(0.0, np.max(d2[apex_d2_idx + 1:search_r + 1])))
	positive_side_lobe_ratio = float((left_lobe + right_lobe) / max(central_neg, eps))
	lobe_max = max(left_lobe, right_lobe)
	side_lobe_balance = float(min(left_lobe, right_lobe) / max(lobe_max, eps)) if lobe_max > 0.0 else 0.0

	return {
		'apex_excess_over_shoulders': apex_excess,
		'center_removed_support_ratio': center_removed_support_ratio,
		'local_line_residual_at_apex': local_line_residual,
		'curvature_at_apex': d2_at_apex,
		'curvature_extreme_centered': d2_ext,
		'curvature_extreme_offset': curvature_extreme_offset,
		'curvature_extreme_abs_offset': float(abs(curvature_extreme_offset)),
		'central_negative_curvature_ratio': central_negative_curvature_ratio,
		'positive_side_lobe_ratio': positive_side_lobe_ratio,
		'side_lobe_balance': side_lobe_balance,
		'support_after_center_removal': support_after_removal,
	}


def compute_multiscale_tophat_features(raw_wide: np.ndarray, bg_mad: float) -> Dict[str, float]:
	"""Compute low-cost multiscale white top-hat and granulometric descriptors."""
	x = np.asarray(raw_wide, dtype=float)
	eps = 1e-12
	zero_out = {
		'multiscale_tophat_area_small': 0.0,
		'multiscale_tophat_area_medium': 0.0,
		'multiscale_tophat_area_large': 0.0,
		'multiscale_tophat_decay_ratio': 0.0,
		'multiscale_tophat_persistence': 0.0,
		'multiscale_tophat_persistence_relmax': 0.0,
		'multiscale_tophat_area_small_z': 0.0,
		'multiscale_tophat_log_decay_small_large': 0.0,
		'multiscale_tophat_ratio_small_medium': 0.0,
		'multiscale_tophat_ratio_medium_large': 0.0,
		'multiscale_tophat_linear_slope': 0.0,
		'multiscale_tophat_linear_slope_abs': 0.0,
		'multiscale_tophat_scale_curvature': 0.0,
		'multiscale_tophat_scale_curvature_norm': 0.0,
		'multiscale_tophat_scale_com': 0.0,
		'multiscale_tophat_area_mean': 0.0,
		'multiscale_tophat_area_std': 0.0,
		'multiscale_tophat_area_cv': 0.0,
		'multiscale_tophat_area_auc_scale': 0.0,
		'multiscale_tophat_area_com_scale': 0.0,
		'multiscale_tophat_area_small_fraction': 0.0,
		'multiscale_tophat_area_large_fraction': 0.0,
		'multiscale_tophat_area_d1_mean': 0.0,
		'multiscale_tophat_area_d1_min': 0.0,
		'multiscale_tophat_area_d1_max_abs': 0.0,
		'multiscale_tophat_area_d2_mean': 0.0,
		'multiscale_tophat_area_d2_max_abs': 0.0,
		'multiscale_tophat_small_dominance': 0.0,
		'multiscale_tophat_medium_dominance': 0.0,
		'multiscale_tophat_large_dominance': 0.0,
		'multiscale_tophat_scale_entropy': 0.0,
		'multiscale_tophat_scale_entropy_norm': 0.0,
		'multiscale_tophat_width_small': 0.0,
		'multiscale_tophat_width_medium': 0.0,
		'multiscale_tophat_width_large': 0.0,
		'multiscale_tophat_width_ratio_small_large': 0.0,
		'multiscale_tophat_width_slope': 0.0,
		'multiscale_tophat_width_mean': 0.0,
		'multiscale_tophat_width_std': 0.0,
		'multiscale_tophat_width_d1_mean': 0.0,
		'multiscale_tophat_width_d1_max_abs': 0.0,
		'multiscale_tophat_width_soft_small': 0.0,
		'multiscale_tophat_width_soft_medium': 0.0,
		'multiscale_tophat_width_soft_large': 0.0,
		'multiscale_tophat_width_soft_mean': 0.0,
		'multiscale_tophat_width_soft_std': 0.0,
		'multiscale_tophat_width_soft_d1_mean': 0.0,
		'multiscale_tophat_width_soft_d1_max_abs': 0.0,
		'multiscale_tophat_width_soft_ratio_small_large': 0.0,
		'multiscale_tophat_width_soft_slope': 0.0,
	}
	def _with_mt_aliases(features: Dict[str, float]) -> Dict[str, float]:
		out = dict(features)
		for key, value in list(features.items()):
			if key.startswith("multiscale_tophat_"):
				out["mt_" + key[len("multiscale_tophat_"):]] = float(value)
		return out
	if x.size < 3:
		return _with_mt_aliases(zero_out)

	curves = _compute_multiscale_tophat_residual_curves(x, float(bg_mad), scales=GWS_GRANULO_SCALES)
	if curves is None:
		return _with_mt_aliases(zero_out)

	scales = curves["scales"]
	areas_arr = curves["areas"]
	widths_arr = curves["widths"]
	widths_soft_arr = curves["widths_soft"]
	mid_idx = int(len(scales) // 2)

	a_small = float(areas_arr[0])
	a_med = float(areas_arr[mid_idx])
	a_large = float(areas_arr[-1])
	w_small = float(widths_arr[0])
	w_med = float(widths_arr[mid_idx])
	w_large = float(widths_arr[-1])
	w_soft_small = float(widths_soft_arr[0])
	w_soft_med = float(widths_soft_arr[mid_idx])
	w_soft_large = float(widths_soft_arr[-1])
	size_small = float(scales[0])
	size_med = float(scales[mid_idx])
	size_large = float(scales[-1])

	max_area = max(float(np.max(areas_arr)), eps)
	persist = float(np.count_nonzero(areas_arr >= 0.15 * max_area) / len(areas_arr))
	persist_relmax = float(np.count_nonzero(areas_arr >= 0.5 * max_area) / len(areas_arr))
	total_area = float(np.sum(areas_arr))
	probs = areas_arr / max(total_area, eps)
	scale_entropy = float(-np.sum(probs * np.log(probs + eps))) if total_area > 0.0 else 0.0
	scale_entropy_norm = float(scale_entropy / np.log(float(scales.size))) if scales.size > 1 and total_area > 0.0 else 0.0

	if scales.size >= 2:
		dscale = np.diff(scales)
		area_d1 = np.diff(areas_arr) / dscale
		width_d1 = np.diff(widths_arr) / dscale
		width_soft_d1 = np.diff(widths_soft_arr) / dscale
	else:
		area_d1 = np.array([], dtype=float)
		width_d1 = np.array([], dtype=float)
		width_soft_d1 = np.array([], dtype=float)

	if scales.size >= 3 and area_d1.size >= 2:
		mid_scales = 0.5 * (scales[:-1] + scales[1:])
		area_d2 = np.diff(area_d1) / np.diff(mid_scales)
	else:
		area_d2 = np.array([], dtype=float)

	area_mean = float(np.mean(areas_arr))
	area_std = float(np.std(areas_arr))
	width_mean = float(np.mean(widths_arr))
	width_std = float(np.std(widths_arr))
	width_soft_mean = float(np.mean(widths_soft_arr))
	width_soft_std = float(np.std(widths_soft_arr))
	linear_slope = float((a_large - a_small) / max(size_large - size_small, eps))
	scale_curvature = float(a_small - 2.0 * a_med + a_large)
	result = {
		'multiscale_tophat_area_small': a_small,
		'multiscale_tophat_area_medium': a_med,
		'multiscale_tophat_area_large': a_large,
		'multiscale_tophat_decay_ratio': float(a_small / max(a_large, eps)),
		'multiscale_tophat_persistence': persist,
		'multiscale_tophat_persistence_relmax': persist_relmax,
		'multiscale_tophat_area_small_z': float(a_small / max(float(bg_mad), eps)),
		'multiscale_tophat_area_mean': area_mean,
		'multiscale_tophat_area_std': area_std,
		'multiscale_tophat_area_cv': float(area_std / max(area_mean, eps)),
		'multiscale_tophat_area_auc_scale': float(np.trapezoid(areas_arr, scales)) if scales.size >= 2 else 0.0,
		'multiscale_tophat_area_com_scale': float(np.sum(scales * areas_arr) / max(total_area, eps)),
		'multiscale_tophat_area_small_fraction': float(a_small / max(total_area, eps)),
		'multiscale_tophat_area_large_fraction': float(a_large / max(total_area, eps)),
		'multiscale_tophat_area_d1_mean': float(np.mean(area_d1)) if area_d1.size else 0.0,
		'multiscale_tophat_area_d1_min': float(np.min(area_d1)) if area_d1.size else 0.0,
		'multiscale_tophat_area_d1_max_abs': float(np.max(np.abs(area_d1))) if area_d1.size else 0.0,
		'multiscale_tophat_area_d2_mean': float(np.mean(area_d2)) if area_d2.size else 0.0,
		'multiscale_tophat_area_d2_max_abs': float(np.max(np.abs(area_d2))) if area_d2.size else 0.0,
		'multiscale_tophat_log_decay_small_large': float(np.log(a_small + eps) - np.log(a_large + eps)),
		'multiscale_tophat_ratio_small_medium': float(a_small / max(a_med, eps)),
		'multiscale_tophat_ratio_medium_large': float(a_med / max(a_large, eps)),
		'multiscale_tophat_linear_slope': linear_slope,
		'multiscale_tophat_linear_slope_abs': float(abs(linear_slope)),
		'multiscale_tophat_scale_curvature': scale_curvature,
		'multiscale_tophat_scale_curvature_norm': float(scale_curvature / max(total_area, eps)),
		'multiscale_tophat_scale_com': float(
			(size_small * a_small + size_med * a_med + size_large * a_large) / max(total_area, eps)
		),
		'multiscale_tophat_small_dominance': float(a_small / max(total_area, eps)),
		'multiscale_tophat_medium_dominance': float(a_med / max(total_area, eps)),
		'multiscale_tophat_large_dominance': float(a_large / max(total_area, eps)),
		'multiscale_tophat_scale_entropy': scale_entropy,
		'multiscale_tophat_scale_entropy_norm': scale_entropy_norm,
		'multiscale_tophat_width_small': w_small,
		'multiscale_tophat_width_medium': w_med,
		'multiscale_tophat_width_large': w_large,
		'multiscale_tophat_width_ratio_small_large': float(w_small / max(w_large, eps)),
		'multiscale_tophat_width_slope': float((w_large - w_small) / max(size_large - size_small, eps)),
		'multiscale_tophat_width_mean': width_mean,
		'multiscale_tophat_width_std': width_std,
		'multiscale_tophat_width_d1_mean': float(np.mean(width_d1)) if width_d1.size else 0.0,
		'multiscale_tophat_width_d1_max_abs': float(np.max(np.abs(width_d1))) if width_d1.size else 0.0,
		'multiscale_tophat_width_soft_small': w_soft_small,
		'multiscale_tophat_width_soft_medium': w_soft_med,
		'multiscale_tophat_width_soft_large': w_soft_large,
		'multiscale_tophat_width_soft_mean': width_soft_mean,
		'multiscale_tophat_width_soft_std': width_soft_std,
		'multiscale_tophat_width_soft_d1_mean': float(np.mean(width_soft_d1)) if width_soft_d1.size else 0.0,
		'multiscale_tophat_width_soft_d1_max_abs': float(np.max(np.abs(width_soft_d1))) if width_soft_d1.size else 0.0,
		'multiscale_tophat_width_soft_ratio_small_large': float(w_soft_small / max(w_soft_large, eps)),
		'multiscale_tophat_width_soft_slope': float((w_soft_large - w_soft_small) / max(size_large - size_small, eps)),
	}
	return _with_mt_aliases(result)


def _normalize_curve(y: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, bool]:
	"""
	Normalize a curve to 0-1.
	Return normalized curve and a flag indicating whether the curve was flat.
	"""
	arr = np.asarray(y, dtype=float)
	if arr.size == 0:
		return np.zeros(0, dtype=float), True
	y_min = float(np.min(arr))
	y_max = float(np.max(arr))
	span = float(y_max - y_min)
	if not np.isfinite(span) or span <= float(eps):
		return np.zeros_like(arr, dtype=float), True
	return (arr - y_min) / span, False


def _scale_axis_norm(scales: np.ndarray, eps: float = 1e-12) -> np.ndarray:
	"""Normalize scale axis to 0-1."""
	arr = np.asarray(scales, dtype=float)
	if arr.size == 0:
		return np.zeros(0, dtype=float)
	x_min = float(np.min(arr))
	x_max = float(np.max(arr))
	return (arr - x_min) / max(float(x_max - x_min), float(eps))


def _curve_auc_norm(y_norm: np.ndarray, x_norm: np.ndarray) -> float:
	"""Return normalized area under the curve."""
	y = np.asarray(y_norm, dtype=float)
	x = np.asarray(x_norm, dtype=float)
	if y.size < 2 or x.size != y.size:
		return 0.0
	return float(np.trapezoid(y, x))


def _scale_at_level(y_norm: np.ndarray, x_norm: np.ndarray, level: float) -> float:
	"""
	Return the first normalized scale where y_norm reaches the selected level.
	Use linear interpolation where possible.
	"""
	y = np.asarray(y_norm, dtype=float)
	x = np.asarray(x_norm, dtype=float)
	if y.size == 0 or x.size != y.size:
		return 1.0
	lvl = float(level)
	if y[0] >= lvl:
		return float(x[0])
	for i in range(1, y.size):
		y0 = float(y[i - 1])
		y1 = float(y[i])
		if y1 < lvl:
			continue
		x0 = float(x[i - 1])
		x1 = float(x[i])
		dy = y1 - y0
		if abs(dy) <= 1e-12:
			return float(x1)
		t = float((lvl - y0) / dy)
		return float(np.clip(x0 + t * (x1 - x0), 0.0, 1.0))
	return 1.0


def _curve_stepiness(y_norm: np.ndarray) -> float:
	"""Return max positive increment divided by sum of positive increments."""
	y = np.asarray(y_norm, dtype=float)
	if y.size < 2:
		return 0.0
	dy = np.diff(y)
	pos_dy = np.maximum(dy, 0.0)
	den = float(np.sum(pos_dy))
	if den <= 1e-12:
		return 0.0
	return float(np.max(pos_dy) / den)


def _curve_max_d1_norm(y_norm: np.ndarray) -> float:
	"""Return maximum positive normalized increment."""
	y = np.asarray(y_norm, dtype=float)
	if y.size < 2:
		return 0.0
	pos_dy = np.maximum(np.diff(y), 0.0)
	if pos_dy.size == 0:
		return 0.0
	return float(np.max(pos_dy))


def _effective_opening_size(signal_size: int, requested_scale: int) -> int:
	"""Match viewer behavior: cap opening size to the largest valid odd size."""
	n = int(signal_size)
	if n < 3:
		return 0
	max_odd = max(3, n - (1 - (n % 2)))
	s_eff = int(min(int(requested_scale), int(max_odd)))
	if s_eff % 2 == 0:
		s_eff -= 1
	return s_eff if s_eff >= 3 else 0


def _signed_evidence_high_is_muon(value: float) -> float:
	"""Map 0-1 metric to -1/+1 evidence where high means muon-like."""
	return float(np.clip(2.0 * float(value) - 1.0, -1.0, 1.0))


def _signed_evidence_low_is_muon(value: float) -> float:
	"""Map 0-1 metric to -1/+1 evidence where low means muon-like."""
	return float(np.clip(1.0 - 2.0 * float(value), -1.0, 1.0))


def _valid_granulo_scales(signal: np.ndarray, scales: Sequence[int]) -> np.ndarray:
	"""Return valid odd opening sizes while preserving the original scale order."""
	x = np.asarray(signal, dtype=float)
	n = int(x.size)
	if n < 3:
		return np.zeros(0, dtype=float)
	valid: List[float] = []
	for s in scales:
		si = int(s)
		if si > 1 and si % 2 == 1 and si <= n:
			valid.append(float(si))
	return np.asarray(valid, dtype=float)


def _compute_multiscale_tophat_residual_curves(
	signal: np.ndarray,
	bg_mad: float,
	scales: Sequence[int] = GWS_GRANULO_SCALES,
	*,
	support_fraction: float = 0.15,
	bg_fraction: float = 0.5,
	hard_bg_fraction: float = 1.0,
	include_scale_zero: bool = False,
	threshold_left_idx: Optional[int] = None,
	threshold_right_idx: Optional[int] = None,
) -> Dict[str, np.ndarray] | None:
	"""Compute multiscale opening residual curve summaries on the provided signal."""
	x = np.asarray(signal, dtype=float)
	if x.size < 3:
		return None
	n = int(x.size)
	thr_l = 0 if threshold_left_idx is None else int(np.clip(int(threshold_left_idx), 0, n - 1))
	thr_r = n - 1 if threshold_right_idx is None else int(np.clip(int(threshold_right_idx), 0, n - 1))
	if thr_r < thr_l:
		thr_l, thr_r = thr_r, thr_l
	rows: List[Dict[str, object]] = []
	scale_values: List[float] = []
	areas: List[float] = []
	areas_above_thr: List[float] = []
	widths: List[float] = []
	widths_soft: List[float] = []
	if bool(include_scale_zero):
		residual0 = np.maximum(x, 0.0).astype(float, copy=False)
		r_max0 = float(np.max(residual0)) if residual0.size else 0.0
		r_max_thr0 = float(np.max(residual0[thr_l:thr_r + 1])) if residual0.size else 0.0
		r_thr0 = max(float(support_fraction) * r_max_thr0, float(hard_bg_fraction) * float(bg_mad))
		r_thr_soft0 = max(float(support_fraction) * r_max_thr0, float(bg_fraction) * float(bg_mad))
		support_mask0 = residual0 >= r_thr_soft0
		width_soft0 = float(np.count_nonzero(support_mask0)) if r_max0 > 0.0 else 0.0
		width_hard0 = float(np.count_nonzero(residual0 >= r_thr0)) if r_max0 > 0.0 else 0.0
		rows.append(
			{
				"scale": 0,
				"scale_eff": 0,
				"opening": np.zeros_like(x, dtype=float),
				"residual": np.maximum(x, 0.0).astype(float, copy=False),
				"support_mask": support_mask0,
				"width_soft": float(width_soft0),
				"width": float(width_hard0),
				"r_max": float(r_max0),
				"r_max_threshold_region": float(r_max_thr0),
				"r_thr": float(r_thr0),
				"r_thr_soft": float(r_thr_soft0),
			}
		)
		scale_values.append(0.0)
		areas.append(float(np.sum(np.maximum(x, 0.0))))
		areas_above_thr.append(float(np.sum(np.maximum(x - r_thr_soft0, 0.0))))
		widths.append(float(width_hard0))
		widths_soft.append(float(width_soft0))
	for scale in scales:
		try:
			scale_label = int(scale)
		except Exception:
			continue
		s_eff = _effective_opening_size(int(x.size), scale_label)
		if s_eff <= 0:
			continue
		opening_s = ndimage.grey_opening(x, size=int(s_eff)).astype(float)
		residual_s = np.maximum(x - opening_s, 0.0).astype(float)
		r_max = float(np.max(residual_s)) if residual_s.size else 0.0
		r_max_thr = float(np.max(residual_s[thr_l:thr_r + 1])) if residual_s.size else 0.0
		r_thr = max(float(support_fraction) * r_max_thr, float(hard_bg_fraction) * float(bg_mad))
		r_thr_soft = max(float(support_fraction) * r_max_thr, float(bg_fraction) * float(bg_mad))
		support_mask = residual_s >= r_thr_soft
		width_soft = float(np.count_nonzero(support_mask)) if r_max > 0.0 else 0.0
		width_hard = float(np.count_nonzero(residual_s >= r_thr)) if r_max > 0.0 else 0.0
		rows.append(
			{
				"scale": int(scale_label),
				"scale_eff": int(s_eff),
				"opening": opening_s,
				"residual": residual_s,
				"support_mask": support_mask,
				"width_soft": float(width_soft),
				"width": float(width_hard),
				"r_max": float(r_max),
				"r_max_threshold_region": float(r_max_thr),
				"r_thr": float(r_thr),
				"r_thr_soft": float(r_thr_soft),
			}
		)
		scale_values.append(float(scale_label))
		areas.append(float(np.sum(residual_s)))
		areas_above_thr.append(float(np.sum(np.maximum(residual_s - r_thr_soft, 0.0))))
		widths.append(float(width_hard))
		widths_soft.append(float(width_soft))
	if not rows:
		return None
	scales_arr = np.asarray(scale_values, dtype=float)
	return {
		"rows": np.asarray(rows, dtype=object),
		"scales": scales_arr.astype(float),
		"areas": np.asarray(areas, dtype=float),
		"areas_above_thr": np.asarray(areas_above_thr, dtype=float),
		"widths": np.asarray(widths, dtype=float),
		"widths_soft": np.asarray(widths_soft, dtype=float),
	}


def _curve_shape_defaults(
	*,
	include_scale80: bool,
	include_max_d1: bool,
) -> Dict[str, float]:
	"""Return safe defaults for normalized curve-shape descriptors."""
	out = {
		"auc_norm": 0.0,
		"scale_at_50": 1.0,
		"stepiness": 0.0,
	}
	if include_scale80:
		out["scale_at_80"] = 1.0
	if include_max_d1:
		out["max_d1_norm"] = 0.0
	return out


def _curve_shape_metrics(
	y: np.ndarray,
	x_norm: np.ndarray,
	*,
	include_scale80: bool,
	include_max_d1: bool,
) -> Dict[str, float]:
	"""Compute normalized curve-shape descriptors for a granulometric activation curve."""
	y_norm, is_flat = _normalize_curve(y)
	if is_flat or x_norm.size != y_norm.size or y_norm.size == 0:
		return _curve_shape_defaults(include_scale80=include_scale80, include_max_d1=include_max_d1)

	out = {
		"auc_norm": _curve_auc_norm(y_norm, x_norm),
		"scale_at_50": _scale_at_level(y_norm, x_norm, 0.5),
		"stepiness": _curve_stepiness(y_norm),
	}
	if include_scale80:
		out["scale_at_80"] = _scale_at_level(y_norm, x_norm, 0.8)
	if include_max_d1:
		out["max_d1_norm"] = _curve_max_d1_norm(y_norm)
	return out


def _compute_gws_context_bounds(
		n: int,
		detection_left: int,
		detection_right: int,
		*,
		context_mode: str,
		context_pad_factor: float,
		context_fixed_pad: Optional[int],
		context_min_pad: Optional[int],
		context_max_pad: Optional[int],
) -> Tuple[int, int, int]:
	"""Return context bounds and the applied pad for one candidate."""
	if n <= 0:
		return 0, 0, 0
	a = int(np.clip(detection_left, 0, n - 1))
	b = int(np.clip(detection_right, 0, n - 1))
	if b < a:
		a, b = b, a
	width = max(1, b - a + 1)
	mode = str(context_mode).strip().lower()
	if mode == "fixed_pad":
		pad = int(context_fixed_pad if context_fixed_pad is not None else round(float(context_pad_factor)))
	else:
		pad = int(round(float(width) * float(context_pad_factor)))
	if context_min_pad is not None:
		pad = max(int(context_min_pad), pad)
	if context_max_pad is not None:
		pad = min(int(context_max_pad), pad)
	pad = max(0, int(pad))
	return max(0, a - pad), min(n - 1, b + pad), int(pad)


def _smooth_gws_split_source(signal: np.ndarray, smooth_pts: int) -> np.ndarray:
	"""Apply a small odd moving-average smoother for valley detection."""
	x = np.asarray(signal, dtype=float)
	pts = int(smooth_pts)
	if x.ndim != 1 or x.size == 0 or pts <= 1:
		return x.astype(float, copy=True)
	if pts % 2 == 0:
		pts += 1
	kernel = np.ones(int(pts), dtype=float) / float(pts)
	return np.convolve(x, kernel, mode="same")


def _normalize_gws_source_mode(mode: str) -> str:
	"""Normalize one configured GWS source mode to a supported stable name."""
	name = str(mode).strip().lower()
	legacy_aliases = {
		"median_smoothed_gradient_w3": "morph_gradient_med3",
		"median_smoothed_gradient_w5": "morph_gradient_med5",
		"medgrad3": "morph_gradient_med3",
		"medgrad5": "morph_gradient_med5",
		"meangrad3": "morph_gradient_mean3",
		"meangrad5": "morph_gradient_mean5",
		"raw": "morph_gradient",
		"raw_tophat_residual": "morph_gradient",
		"median_residual_w9": "morph_gradient",
		"median_residual_w15": "morph_gradient",
	}
	name = legacy_aliases.get(name, name)
	if name in GWS_SOURCE_PREFIX_BY_MODE:
		return name
	return str(GWS_SOURCE).strip().lower()


def _normalize_gws_measure_region(mode: str) -> str:
	"""Normalize the measurement region used for component-like GWS scalar summaries."""
	name = str(mode).strip().lower()
	if name in {"mask", "component", "apex_component"}:
		return "mask"
	if name in {"spike_edges", "edges", "candidate"}:
		return "spike_edges"
	return str(GWS_MEASURE_REGION).strip().lower()


def _normalize_gws_threshold_region(mode: str) -> str:
	"""Normalize the region used to derive the relative GWS support threshold."""
	name = str(mode).strip().lower()
	if name in {"context"}:
		return "context"
	if name in {"spike_edges", "edges", "candidate"}:
		return "spike_edges"
	if name in {"measure_region", "measure"}:
		return "measure_region"
	return str(GWS_THRESHOLD_REGION).strip().lower()


def _compute_local_morph_gradient(signal: np.ndarray, size: int = 5) -> np.ndarray:
	"""Compute a simple 1D morphological gradient for experimental GWS sources."""
	x = np.asarray(signal, dtype=float)
	if x.ndim != 1 or x.size == 0:
		return np.asarray([], dtype=float)
	si = max(3, int(size))
	if si % 2 == 0:
		si += 1
	dil = ndimage.grey_dilation(x, size=si).astype(float)
	ero = ndimage.grey_erosion(x, size=si).astype(float)
	return (dil - ero).astype(float, copy=False)


def _centered_moving_average(signal: np.ndarray, window: int) -> np.ndarray:
	"""Compute an edge-safe centered moving average with reflect padding."""
	x = np.asarray(signal, dtype=float)
	if x.ndim != 1 or x.size == 0:
		return np.asarray([], dtype=float)
	w = max(1, int(window))
	if w % 2 == 0:
		w += 1
	if w <= 1:
		return x.astype(float, copy=True)
	pad = int(w // 2)
	x_pad = np.pad(x, (pad, pad), mode="reflect")
	kernel = np.ones(int(w), dtype=float) / float(w)
	return np.convolve(x_pad, kernel, mode="valid").astype(float, copy=False)


def _compute_gws_source_context(
		gradient_context: np.ndarray,
		raw_context: Optional[np.ndarray],
		*,
		source_mode: str,
) -> np.ndarray:
	"""Build one shared source curve used by GWS support-trace experiments."""
	mode = _normalize_gws_source_mode(source_mode)
	grad_ctx = np.asarray(gradient_context, dtype=float)
	if mode == "morph_gradient":
		return grad_ctx.astype(float, copy=True)
	if mode == "morph_gradient_med3":
		return ndimage.median_filter(grad_ctx, size=3, mode="reflect").astype(float, copy=False)
	if mode == "morph_gradient_med5":
		return ndimage.median_filter(grad_ctx, size=5, mode="reflect").astype(float, copy=False)
	if mode == "morph_gradient_mean3":
		return _centered_moving_average(grad_ctx, window=3)
	if mode == "morph_gradient_mean5":
		return _centered_moving_average(grad_ctx, window=5)
	return grad_ctx.astype(float, copy=True)


def compute_gws_support_trace(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
	"""Return the discrete GWS support-count trace used by the viewer and metrics."""
	scale_values: List[int] = []
	support_counts: List[int] = []
	support_masks: List[np.ndarray] = []
	support_indices: List[np.ndarray] = []
	for row in rows:
		if not isinstance(row, dict):
			continue
		try:
			scale_i = int(row.get("scale", 0))
		except Exception:
			continue
		mask = np.asarray(row.get("support_mask", []), dtype=bool)
		scale_values.append(int(scale_i))
		support_masks.append(mask)
		support_indices.append(np.where(mask)[0].astype(int))
		support_counts.append(int(np.count_nonzero(mask)))
	scales_arr = np.asarray(scale_values, dtype=int)
	support_counts_arr = np.asarray(support_counts, dtype=int)
	diffs_arr = np.diff(support_counts_arr).astype(int) if support_counts_arr.size >= 2 else np.asarray([], dtype=int)
	return {
		"scales": scales_arr,
		"support_counts": support_counts_arr,
		"support_masks": support_masks,
		"support_indices": support_indices,
		"diffs": diffs_arr,
	}


def _true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
	"""Return contiguous True runs as inclusive (left, right) index pairs."""
	m = np.asarray(mask, dtype=bool)
	if m.ndim != 1 or m.size == 0 or not np.any(m):
		return []
	idx = np.flatnonzero(m).astype(int)
	splits = np.where(np.diff(idx) > 1)[0] + 1
	parts = np.split(idx, splits)
	return [(int(part[0]), int(part[-1])) for part in parts if part.size]


def _select_component_from_mask(mask: np.ndarray, apex_idx: int) -> Dict[str, object]:
	"""Select the connected support component owned by the candidate apex."""
	m = np.asarray(mask, dtype=bool)
	n = int(m.size)
	if n <= 0:
		return {
			"selected_mask": np.zeros(0, dtype=bool),
			"selected_indices": np.asarray([], dtype=int),
			"apex_inside_support": 0.0,
			"nearest_component_used": 0.0,
			"distance_to_selected_component": np.nan,
			"n_components": 0.0,
			"selected_left": np.nan,
			"selected_right": np.nan,
		}
	p = int(np.clip(apex_idx, 0, max(n - 1, 0)))
	runs = _true_runs(m)
	if not runs:
		return {
			"selected_mask": np.zeros_like(m, dtype=bool),
			"selected_indices": np.asarray([], dtype=int),
			"apex_inside_support": 0.0,
			"nearest_component_used": 0.0,
			"distance_to_selected_component": np.nan,
			"n_components": 0.0,
			"selected_left": np.nan,
			"selected_right": np.nan,
		}
	selected = None
	apex_inside = 0.0
	for left, right in runs:
		if left <= p <= right:
			selected = (int(left), int(right))
			apex_inside = 1.0
			break
	if selected is None:
		best = None
		best_key = None
		for left, right in runs:
			if p < left:
				dist = int(left - p)
			elif p > right:
				dist = int(p - right)
			else:
				dist = 0
			center = 0.5 * float(left + right)
			key = (dist, abs(center - float(p)), left)
			if best_key is None or key < best_key:
				best_key = key
				best = (int(left), int(right), int(dist))
		if best is None:
			return {
				"selected_mask": np.zeros_like(m, dtype=bool),
				"selected_indices": np.asarray([], dtype=int),
				"apex_inside_support": 0.0,
				"nearest_component_used": 0.0,
				"distance_to_selected_component": np.nan,
				"n_components": float(len(runs)),
				"selected_left": np.nan,
				"selected_right": np.nan,
			}
		selected = (best[0], best[1])
		distance = float(best[2])
		nearest_used = 1.0
	else:
		distance = 0.0
		nearest_used = 0.0
	left, right = int(selected[0]), int(selected[1])
	selected_mask = np.zeros_like(m, dtype=bool)
	selected_mask[left:right + 1] = True
	selected_indices = np.arange(left, right + 1, dtype=int)
	return {
		"selected_mask": selected_mask,
		"selected_indices": selected_indices,
		"apex_inside_support": float(apex_inside),
		"nearest_component_used": float(nearest_used),
		"distance_to_selected_component": float(distance),
		"n_components": float(len(runs)),
		"selected_left": float(left),
		"selected_right": float(right),
	}


def compute_gws_component_trace(
		rows: Sequence[Dict[str, object]],
		*,
		apex_rel_idx: int,
) -> Dict[str, object]:
	"""Restrict GWS support/area/width traces to the connected component owned by the apex."""
	scale_values: List[int] = []
	support_counts: List[int] = []
	areas: List[float] = []
	widths: List[float] = []
	apex_inside_flags: List[float] = []
	nearest_flags: List[float] = []
	distances: List[float] = []
	n_components: List[float] = []
	selected_lefts: List[float] = []
	selected_rights: List[float] = []
	selected_masks: List[np.ndarray] = []
	for row in rows:
		if not isinstance(row, dict):
			continue
		try:
			scale_i = int(row.get("scale", 0))
		except Exception:
			continue
		mask = np.asarray(row.get("support_mask", []), dtype=bool)
		residual = np.asarray(row.get("residual", []), dtype=float)
		sel = _select_component_from_mask(mask, int(apex_rel_idx))
		sel_mask = np.asarray(sel["selected_mask"], dtype=bool)
		scale_values.append(int(scale_i))
		selected_masks.append(sel_mask)
		support_counts.append(int(np.count_nonzero(sel_mask)))
		widths.append(float(np.count_nonzero(sel_mask)))
		areas.append(float(np.sum(residual[sel_mask])) if residual.size == sel_mask.size else np.nan)
		apex_inside_flags.append(float(sel["apex_inside_support"]))
		nearest_flags.append(float(sel["nearest_component_used"]))
		distances.append(float(sel["distance_to_selected_component"]))
		n_components.append(float(sel["n_components"]))
		selected_lefts.append(float(sel["selected_left"]))
		selected_rights.append(float(sel["selected_right"]))
	support_counts_arr = np.asarray(support_counts, dtype=int)
	return {
		"scales": np.asarray(scale_values, dtype=int),
		"support_counts": support_counts_arr,
		"areas": np.asarray(areas, dtype=float),
		"widths": np.asarray(widths, dtype=float),
		"diffs": np.diff(support_counts_arr).astype(int) if support_counts_arr.size >= 2 else np.asarray([], dtype=int),
		"selected_masks": selected_masks,
		"apex_inside_flags": np.asarray(apex_inside_flags, dtype=float),
		"nearest_component_flags": np.asarray(nearest_flags, dtype=float),
		"distance_to_selected_component": np.asarray(distances, dtype=float),
		"n_components": np.asarray(n_components, dtype=float),
		"selected_lefts": np.asarray(selected_lefts, dtype=float),
		"selected_rights": np.asarray(selected_rights, dtype=float),
	}


def compute_gws_measure_trace(
		rows: Sequence[Dict[str, object]],
		*,
		apex_rel_idx: int,
		measure_left_idx: int,
		measure_right_idx: int,
		measure_region: str = GWS_MEASURE_REGION,
) -> Dict[str, object]:
	"""Measure GWS support/area/width either on the apex-owned component or on spike edges."""
	mode = _normalize_gws_measure_region(measure_region)
	if mode == "mask":
		return compute_gws_component_trace(rows, apex_rel_idx=apex_rel_idx)

	scale_values: List[int] = []
	support_counts: List[int] = []
	areas: List[float] = []
	widths: List[float] = []
	apex_inside_flags: List[float] = []
	nearest_flags: List[float] = []
	distances: List[float] = []
	n_components: List[float] = []
	selected_lefts: List[float] = []
	selected_rights: List[float] = []
	selected_masks: List[np.ndarray] = []
	region_lefts: List[float] = []
	region_rights: List[float] = []
	for row in rows:
		if not isinstance(row, dict):
			continue
		try:
			scale_i = int(row.get("scale", 0))
		except Exception:
			continue
		mask = np.asarray(row.get("support_mask", []), dtype=bool)
		residual = np.asarray(row.get("residual", []), dtype=float)
		n = int(mask.size)
		if n <= 0:
			continue
		left = int(np.clip(measure_left_idx, 0, n - 1))
		right = int(np.clip(measure_right_idx, 0, n - 1))
		if right < left:
			left, right = right, left
		region_mask = np.zeros_like(mask, dtype=bool)
		region_mask[left:right + 1] = True
		selected_mask = mask & region_mask
		scale_values.append(int(scale_i))
		selected_masks.append(selected_mask)
		support_counts.append(int(np.count_nonzero(selected_mask)))
		widths.append(float(np.count_nonzero(selected_mask)))
		areas.append(float(np.sum(residual[selected_mask])) if residual.size == selected_mask.size else np.nan)
		runs = _true_runs(selected_mask)
		n_components.append(float(len(runs)))
		region_lefts.append(float(left))
		region_rights.append(float(right))
		if runs:
			selected_lefts.append(float(runs[0][0]))
			selected_rights.append(float(runs[-1][1]))
		else:
			selected_lefts.append(np.nan)
			selected_rights.append(np.nan)
		apex_local = int(np.clip(apex_rel_idx, 0, n - 1))
		apex_inside = bool(selected_mask[apex_local]) if 0 <= apex_local < n else False
		apex_inside_flags.append(1.0 if apex_inside else 0.0)
		nearest_flags.append(0.0)
		if runs:
			best_dist = min(
				0 if left_i <= apex_local <= right_i else min(abs(apex_local - left_i), abs(apex_local - right_i))
				for left_i, right_i in runs
			)
			distances.append(float(best_dist))
		else:
			distances.append(np.nan)
	return {
		"scales": np.asarray(scale_values, dtype=int),
		"support_counts": np.asarray(support_counts, dtype=int),
		"areas": np.asarray(areas, dtype=float),
		"widths": np.asarray(widths, dtype=float),
		"diffs": np.diff(np.asarray(support_counts, dtype=int)).astype(int) if len(support_counts) >= 2 else np.asarray([], dtype=int),
		"selected_masks": selected_masks,
		"apex_inside_flags": np.asarray(apex_inside_flags, dtype=float),
		"nearest_component_flags": np.asarray(nearest_flags, dtype=float),
		"distance_to_selected_component": np.asarray(distances, dtype=float),
		"n_components": np.asarray(n_components, dtype=float),
		"selected_lefts": np.asarray(selected_lefts, dtype=float),
		"selected_rights": np.asarray(selected_rights, dtype=float),
		"region_lefts": np.asarray(region_lefts, dtype=float),
		"region_rights": np.asarray(region_rights, dtype=float),
	}


def _gws_support_count_metrics(support_counts: np.ndarray) -> Dict[str, float]:
	"""Measure discrete changes in detected GWS support counts across granulometric scales."""
	out = {
		"gws_support_num_changes": np.nan,
		"gws_support_change_density": np.nan,
		"gws_support_total_abs_change": np.nan,
		"gws_support_max_abs_change": np.nan,
		"gws_support_num_increases": np.nan,
		"gws_support_num_decreases": np.nan,
		"gws_support_total_increase": np.nan,
		"gws_support_total_decrease": np.nan,
		"gws_support_longest_constant_run": np.nan,
		"gws_support_longest_constant_run_norm": np.nan,
		"gws_support_initial": np.nan,
		"gws_support_final": np.nan,
		"gws_support_final_minus_initial": np.nan,
		"gws_support_max": np.nan,
		"gws_support_min": np.nan,
	}
	counts = np.asarray(support_counts, dtype=float)
	if counts.ndim != 1:
		return out
	mask = np.isfinite(counts)
	if not np.any(mask):
		return out
	counts = counts[mask]
	n = int(counts.size)
	if n <= 0:
		return out
	if n == 1:
		return {
			"gws_support_num_changes": 0.0,
			"gws_support_change_density": 0.0,
			"gws_support_total_abs_change": 0.0,
			"gws_support_max_abs_change": 0.0,
			"gws_support_num_increases": 0.0,
			"gws_support_num_decreases": 0.0,
			"gws_support_total_increase": 0.0,
			"gws_support_total_decrease": 0.0,
			"gws_support_longest_constant_run": 1.0,
			"gws_support_longest_constant_run_norm": 1.0,
			"gws_support_initial": float(counts[0]),
			"gws_support_final": float(counts[0]),
			"gws_support_final_minus_initial": 0.0,
			"gws_support_max": float(counts[0]),
			"gws_support_min": float(counts[0]),
		}
	diffs = np.diff(counts)
	abs_diffs = np.abs(diffs)
	longest_const = 1
	cur = 1
	for dv in diffs:
		if float(dv) == 0.0:
			cur += 1
		else:
			longest_const = max(longest_const, cur)
			cur = 1
	longest_const = max(longest_const, cur)
	pos = np.maximum(diffs, 0.0)
	neg = np.minimum(diffs, 0.0)
	return {
		"gws_support_num_changes": float(np.count_nonzero(diffs != 0.0)),
		"gws_support_change_density": float(np.count_nonzero(diffs != 0.0) / max(n - 1, 1)),
		"gws_support_total_abs_change": float(np.sum(abs_diffs)),
		"gws_support_max_abs_change": float(np.max(abs_diffs)) if abs_diffs.size else 0.0,
		"gws_support_num_increases": float(np.count_nonzero(diffs > 0.0)),
		"gws_support_num_decreases": float(np.count_nonzero(diffs < 0.0)),
		"gws_support_total_increase": float(np.sum(pos)),
		"gws_support_total_decrease": float(np.sum(np.abs(neg))),
		"gws_support_longest_constant_run": float(longest_const),
		"gws_support_longest_constant_run_norm": float(longest_const / max(n, 1)),
		"gws_support_initial": float(counts[0]),
		"gws_support_final": float(counts[-1]),
		"gws_support_final_minus_initial": float(counts[-1] - counts[0]),
		"gws_support_max": float(np.max(counts)),
		"gws_support_min": float(np.min(counts)),
	}


def _gws_support_fraction_metrics(
		support_metrics: Dict[str, float],
		context_width: float,
) -> Dict[str, float]:
	"""Normalize support-count metrics by the available local context width."""
	out = {
		"gws_support_initial_fraction": np.nan,
		"gws_support_final_fraction": np.nan,
		"gws_support_max_fraction": np.nan,
		"gws_support_final_minus_initial_fraction": np.nan,
		"gws_support_total_increase_fraction": np.nan,
		"gws_support_total_abs_change_fraction": np.nan,
	}
	try:
		den = float(context_width)
	except Exception:
		return out
	if not np.isfinite(den) or den <= 0.0:
		return out
	for src_key, dst_key in (
		("gws_support_initial", "gws_support_initial_fraction"),
		("gws_support_final", "gws_support_final_fraction"),
		("gws_support_max", "gws_support_max_fraction"),
		("gws_support_final_minus_initial", "gws_support_final_minus_initial_fraction"),
		("gws_support_total_increase", "gws_support_total_increase_fraction"),
		("gws_support_total_abs_change", "gws_support_total_abs_change_fraction"),
	):
		value = float(support_metrics.get(src_key, np.nan))
		if np.isfinite(value):
			out[dst_key] = float(value / den)
	return out


def _gws_simple_trace_metrics(trace: np.ndarray, prefix: str) -> Dict[str, float]:
	"""Summarize a numeric trace with a compact initial/final/change family."""
	out = {
		f"{prefix}_initial": np.nan,
		f"{prefix}_final": np.nan,
		f"{prefix}_total_increase": np.nan,
		f"{prefix}_total_abs_change": np.nan,
	}
	x = np.asarray(trace, dtype=float)
	if x.ndim != 1:
		return out
	mask = np.isfinite(x)
	if not np.any(mask):
		return out
	x = x[mask]
	if x.size == 0:
		return out
	if x.size == 1:
		return {
			f"{prefix}_initial": float(x[0]),
			f"{prefix}_final": float(x[0]),
			f"{prefix}_total_increase": 0.0,
			f"{prefix}_total_abs_change": 0.0,
		}
	d = np.diff(x)
	return {
		f"{prefix}_initial": float(x[0]),
		f"{prefix}_final": float(x[-1]),
		f"{prefix}_total_increase": float(np.sum(np.maximum(d, 0.0))),
		f"{prefix}_total_abs_change": float(np.sum(np.abs(d))),
	}


def width_between_outer_level_crossings(seg: np.ndarray, level: float) -> Tuple[float, float, float, bool, str]:
	"""Measure width between outermost crossings of a level inside one candidate interval."""
	y = np.asarray(seg, dtype=float)
	if y.ndim != 1 or y.size == 0:
		return np.nan, np.nan, np.nan, False, "empty_segment"
	mask = np.isfinite(y)
	if not np.all(mask):
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
	"""Interpolate horizontal-level crossings for one selected above-level run."""
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


def compute_raw_upper_component_dense_width_metrics(
		signal: np.ndarray,
		*,
		detection_left: int,
		detection_right: int,
		prefix: str = "rucdw",
		bg_mad: float | None = None,
		context_pad_pts: int = 20,
		context_max_pad_pts: int = 80,
		levels: Sequence[int] = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90),
		min_snr: float = 1.0,
		noise_fallback_rel_amp: float = 0.05,
		anchor_mode: str = "max_in_candidate",
		baseline_mode: str = "context_low_percentile",
		baseline_percentile: float = 5.0,
) -> Dict[str, object]:
	"""Raw upper-level-set apex-component dense width, a compact Raman-veto experiment."""
	x = np.asarray(signal, dtype=float)
	out: Dict[str, object] = {
		f"{prefix}_sum_0_90": np.nan,
		f"{prefix}_valid_n_0_90": 0.0,
	}
	debug: Dict[str, object] = {
		"reason": "ok",
		"levels": [int(v) for v in levels],
		"valid_levels": [],
		"skipped": {},
		"components": {},
	}
	if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
		debug["reason"] = "invalid_signal"
		out[f"{prefix}_debug"] = debug
		return out
	n = int(x.size)
	left = int(np.clip(detection_left, 0, n - 1))
	right = int(np.clip(detection_right, 0, n - 1))
	if right < left:
		left, right = right, left
	pad = max(0, min(int(context_pad_pts), int(context_max_pad_pts)))
	ctx_l = max(0, int(left) - pad)
	ctx_r = min(n - 1, int(right) + pad)
	cand = x[left:right + 1]
	if cand.size == 0:
		debug["reason"] = "empty_candidate"
		out[f"{prefix}_debug"] = debug
		return out
	if str(anchor_mode).strip().lower() == "max_in_candidate":
		anchor = int(left + int(np.argmax(cand)))
	else:
		anchor = int(left + int(np.argmax(cand)))
	ctx = x[ctx_l:ctx_r + 1]
	mode = str(baseline_mode).strip().lower()
	if mode == "context_low_percentile":
		baseline = float(np.percentile(ctx, float(baseline_percentile)))
	else:
		baseline = float(np.min(ctx))
	y_ctx = ctx - baseline
	anchor_rel = int(anchor - ctx_l)
	amp = float(x[anchor] - baseline) if 0 <= anchor < n else np.nan
	dseg = np.diff(ctx)
	if dseg.size:
		dmed = float(np.median(dseg))
		diff_noise_mad = float(np.median(np.abs(dseg - dmed)) / max(np.sqrt(2.0), 1e-12))
	else:
		diff_noise_mad = np.nan
	if bg_mad is None:
		noise_mad = diff_noise_mad
	else:
		candidates_noise = [float(bg_mad)]
		if np.isfinite(diff_noise_mad) and diff_noise_mad > 0.0:
			candidates_noise.append(float(diff_noise_mad))
		noise_mad = float(min(candidates_noise))
	noise_level = (
		float(max(0.0, float(noise_fallback_rel_amp)) * max(float(amp), 0.0))
		if not np.isfinite(noise_mad)
		else float(max(0.0, float(min_snr)) * 1.4826 * max(float(noise_mad), 0.0))
	)
	debug.update(
		{
			"context_left": int(ctx_l),
			"context_right": int(ctx_r),
			"detection_left": int(left),
			"detection_right": int(right),
			"anchor_index": int(anchor),
			"anchor_mode": str(anchor_mode),
			"baseline": float(baseline),
			"baseline_mode": str(baseline_mode),
			"baseline_percentile": float(baseline_percentile),
			"amp": float(amp) if np.isfinite(amp) else np.nan,
			"input_bg_mad": np.nan if bg_mad is None else float(bg_mad),
			"diff_noise_mad": float(diff_noise_mad) if np.isfinite(diff_noise_mad) else np.nan,
			"noise_mad": float(noise_mad) if np.isfinite(noise_mad) else np.nan,
			"noise_level": float(noise_level) if np.isfinite(noise_level) else np.nan,
		}
	)
	if not np.isfinite(amp) or amp <= 1e-12:
		debug["reason"] = "flat_or_negative_anchor"
		out[f"{prefix}_debug"] = debug
		return out
	widths: List[float] = []
	valid_levels: List[int] = []
	for raw_pct in levels:
		pct = int(raw_pct)
		level_value = float((float(pct) / 100.0) * float(amp))
		if (not np.isfinite(level_value)) or level_value <= max(float(noise_level), 1e-12):
			debug["skipped"][str(pct)] = "below_noise_level"
			continue
		intervals = _components_from_mask(y_ctx >= level_value)
		interval = None
		for a0, b0 in intervals:
			if int(a0) <= anchor_rel <= int(b0):
				interval = (int(a0), int(b0))
				break
		if interval is None:
			debug["skipped"][str(pct)] = "anchor_component_not_reached"
			continue
		comp_l = int(ctx_l + interval[0])
		comp_r = int(ctx_l + interval[1])
		width = float(max(0, comp_r - comp_l))
		widths.append(width)
		valid_levels.append(int(pct))
		debug["components"][str(pct)] = {
			"level_value": float(level_value + baseline),
			"relative_level_value": float(level_value),
			"component_left": int(comp_l),
			"component_right": int(comp_r),
			"width": float(width),
			"touches_context_boundary": bool(comp_l <= ctx_l or comp_r >= ctx_r),
			"touches_spike_edges_boundary": bool(comp_l <= left or comp_r >= right),
		}
	out[f"{prefix}_valid_n_0_90"] = float(len(widths))
	if widths:
		out[f"{prefix}_sum_0_90"] = float(np.sum(np.asarray(widths, dtype=float)))
	debug["valid_levels"] = [int(v) for v in valid_levels]
	out[f"{prefix}_debug"] = debug
	return out


def _edge_component_width_at_level(
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
	"""Track the apex-owned peak root so low-level widths do not absorb neighbors."""
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
		"""Shrink the searchable interval when a lower level merges into a neighbor."""
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

	def _clip_at_neighbor_crossings(
			level_percent: int,
			level_value: float,
			selected_interval: Tuple[int, int],
			intervals: Sequence[Tuple[int, int]],
	) -> None:
		"""Do not let lower levels look past the first neighboring curve crossing."""
		nonlocal active_left, active_right
		sel_l, sel_r = int(selected_interval[0]), int(selected_interval[1])
		left_neighbors = [(int(a), int(b)) for a, b in intervals if int(b) < sel_l]
		right_neighbors = [(int(a), int(b)) for a, b in intervals if int(a) > sel_r]
		diag["neighbor_clip_per_level"] = diag.get("neighbor_clip_per_level", {})
		level_clip: Dict[str, object] = {}
		if left_neighbors:
			neighbor = max(left_neighbors, key=lambda t: int(t[1]))
			_, _, right_cross, _, _ = _crossings_for_interval(
				y,
				float(level_value),
				neighbor,
				bounds=(active_left, active_right),
			)
			if np.isfinite(right_cross):
				new_left = int(np.clip(int(math.ceil(float(right_cross))), active_left, active_right))
				if new_left > active_left and new_left <= ap:
					active_left = new_left
					level_clip["left"] = float(right_cross)
					level_clip["left_index"] = int(active_left)
		if right_neighbors:
			neighbor = min(right_neighbors, key=lambda t: int(t[0]))
			_, left_cross, _, _, _ = _crossings_for_interval(
				y,
				float(level_value),
				neighbor,
				bounds=(active_left, active_right),
			)
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
		interval = _select_tracked_interval(
			intervals,
			apex_local=ap,
			prev_interval=ref,
			allow_fallback=False,
		)
		diag["tested_levels"].append(int(level_percent))
		diag["crossing_count_per_level"][str(int(level_percent))] = int(2 * len(intervals))
		if interval is None:
			diag["invalid_levels"].append(int(level_percent))
			diag["reason_invalid_per_level"][str(int(level_percent))] = "no_component"
			return False, "no_component", None, level_value, np.nan
		l, r = int(interval[0]), int(interval[1])
		width, left_cross, right_cross, _used_interp, width_reason = _crossings_for_interval(
			y,
			level_value,
			interval,
			bounds=(active_left, active_right),
		)
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
				# After an active boundary has already been clipped, width can still grow
				# within the candidate-owned side. Keep tracking instead of raising the
				# local zero back into the peak.
				diag["reason_invalid_per_level"][str(int(level_percent))] = "merge_guard_unclipped_but_accepted"
		diag["valid_levels"].append(int(level_percent))
		return True, "ok", interval, level_value, width

	for idx, pct in enumerate(raw_levels):
		ok, reason, interval, level_value, width = _test_level(int(pct))
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
		mapping_levels_desc: Sequence[int] = (95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5),
		mapping_refine_step_percent: int = 1,
		mapping_min_level_percent: int = 1,
		mapping_require_closed_interval: bool = True,
		mapping_use_apex_component: bool = True,
		mapping_enable_merge_guard: bool = True,
		mapping_max_width_jump_factor: float = 2.5,
		mapping_max_width_jump_points: float = 8.0,
		mapping_fallback_to_old: bool = False,
		mapping_noise_guard_enabled: bool = False,
) -> Dict[str, object]:
	"""Measure width-at-relative-level inside the original spike-edges interval."""
	x = np.asarray(signal, dtype=float)
	out: Dict[str, object] = {
		f"{prefix}_width_at_10": np.nan,
		f"{prefix}_width_at_25": np.nan,
		f"{prefix}_width_at_80": np.nan,
		f"{prefix}_width_at_50": np.nan,
		f"{prefix}_width_at_20": np.nan,
		f"{prefix}_width_at_75": np.nan,
		f"{prefix}_width_at_90": np.nan,
		f"{prefix}_width_50_over_80": np.nan,
		f"{prefix}_width_20_over_80": np.nan,
		f"{prefix}_width_20_minus_80": np.nan,
		f"{prefix}_width_50_minus_80": np.nan,
		f"{prefix}_base_expansion_rate": np.nan,
	}
	if bool(include_low_root_metrics):
		out.update(
			{
				f"{prefix}_width_at_0": np.nan,
				f"{prefix}_width_at_5": np.nan,
				f"{prefix}_width_at_15": np.nan,
				f"{prefix}_root_width_mean_5_25": np.nan,
				f"{prefix}_root_width_median_5_25": np.nan,
				f"{prefix}_root_width_weighted_5_25": np.nan,
				f"{prefix}_root_width_valid_n_5_25": 0.0,
				f"{prefix}_root_width_mean_10_25": np.nan,
				f"{prefix}_root_width_median_10_25": np.nan,
				f"{prefix}_root_width_weighted_10_25": np.nan,
				f"{prefix}_root_width_valid_n_10_25": 0.0,
				f"{prefix}_dense_width_sum_0_90": np.nan,
				f"{prefix}_dense_width_mean_0_90": np.nan,
				f"{prefix}_dense_width_median_0_90": np.nan,
				f"{prefix}_dense_width_weighted_low_0_90": np.nan,
				f"{prefix}_dense_width_valid_n_0_90": 0.0,
				f"{prefix}_dense_width_missing_n_0_90": 19.0,
				f"{prefix}_dense_width_complete_0_90": 0.0,
				f"{prefix}_dense_width_total_abs_change_0_90": np.nan,
				f"{prefix}_dense_width_max_abs_change_0_90": np.nan,
				f"{prefix}_dense_width_num_changes_0_90": 0.0,
				f"{prefix}_dense_width_change_density_0_90": 0.0,
				f"{prefix}_dense_width_initial_0_90": np.nan,
				f"{prefix}_dense_width_final_0_90": np.nan,
				f"{prefix}_dense_width_final_minus_initial_0_90": np.nan,
				f"{prefix}_dense_width_ratio_sum_0_90": np.nan,
				f"{prefix}_dense_width_ratio_mean_0_90": np.nan,
				f"{prefix}_dense_width_ratio_valid_n_0_90": 0.0,
			}
		)
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
	amp = float(y_max - y_min)
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
	debug = {
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
	}
	if not np.isfinite(amp) or amp <= 1e-12:
		debug["reason"] = "flat_segment"
		out[f"{prefix}_debug"] = debug
		return out
	level_fracs = {"10": 0.10, "20": 0.20, "25": 0.25, "50": 0.50, "75": 0.75, "80": 0.80, "90": 0.90}
	if bool(include_low_root_metrics):
		level_fracs = {"0": 0.0, "5": 0.05, "10": 0.10, "15": 0.15, **level_fracs}
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
	widths: Dict[str, float] = {}
	levels: Dict[str, float] = {}
	noise_level = (
		float(y_min + 0.05 * amp)
		if not np.isfinite(edge_noise_mad)
		else float(y_min + max(0.0, float(low_root_noise_k_mad)) * max(float(edge_noise_mad), 0.0))
	)
	debug["low_root_noise_level"] = float(noise_level)
	debug["edge_mapping"] = mapping_diag
	root_amp_noise_reject = False
	if bool(include_low_root_metrics):
		if bool(use_enhanced_spike_mapping) and mapping_reference_interval is not None:
			local_zero_for_noise = float(mapping_diag.get("local_zero_level_value", np.nan))
			local_top_for_noise = float(mapping_diag.get("local_top_value", np.nan))
			root_amp_for_noise = (
				float(local_top_for_noise - local_zero_for_noise)
				if np.isfinite(local_zero_for_noise) and np.isfinite(local_top_for_noise)
				else np.nan
			)
		else:
			root_amp_for_noise = float(amp)
		root_noise_amp = (
			float(0.05 * max(root_amp_for_noise, 0.0))
			if not np.isfinite(edge_noise_mad)
			else float(max(0.0, float(low_root_noise_k_mad)) * 1.4826 * max(float(edge_noise_mad), 0.0))
		)
		root95_amp_for_noise = float(0.95 * root_amp_for_noise) if np.isfinite(root_amp_for_noise) else np.nan
		debug["root_noise_amp"] = float(root_noise_amp) if np.isfinite(root_noise_amp) else np.nan
		debug["root_amp_for_noise"] = float(root_amp_for_noise) if np.isfinite(root_amp_for_noise) else np.nan
		debug["root95_amp_for_noise"] = float(root95_amp_for_noise) if np.isfinite(root95_amp_for_noise) else np.nan
		debug["root_snr"] = float(root95_amp_for_noise / max(root_noise_amp, 1e-12)) if np.isfinite(root95_amp_for_noise) and np.isfinite(root_noise_amp) else np.nan
		root_amp_noise_reject = (not np.isfinite(root95_amp_for_noise)) or root95_amp_for_noise <= max(float(root_noise_amp), 1e-12)
		debug["root_amp_below_noise"] = bool(root_amp_noise_reject)
	for tag, frac in level_fracs.items():
		if bool(use_enhanced_spike_mapping) and mapping_reference_interval is not None:
			local_zero = float(mapping_diag.get("local_zero_level_value", np.nan))
			local_top = float(mapping_diag.get("local_top_value", np.nan))
			level = float(local_zero + frac * (local_top - local_zero))
		else:
			level = float(y_min + frac * amp)
		levels[tag] = float(level)
		low_root_level = tag in {"0", "5", "10", "15", "20", "25"}
		if bool(include_low_root_metrics) and low_root_level and bool(root_amp_noise_reject):
			width = np.nan
			left_cross = np.nan
			right_cross = np.nan
			used_interp = False
			reason = "root_amp_below_noise"
		elif (not bool(use_enhanced_spike_mapping)) and bool(include_low_root_metrics) and low_root_level and level <= noise_level:
			width = np.nan
			left_cross = np.nan
			right_cross = np.nan
			used_interp = False
			reason = "below_noise_level"
		elif bool(use_enhanced_spike_mapping) and mapping_reference_interval is not None:
			width, left_cross, right_cross, used_interp, reason, selected_interval, crossing_count = _edge_component_width_at_level(
				seg,
				level,
				apex_local=int(apex_local),
				reference_interval=mapping_reference_interval,
				bounds=mapping_active_bounds,
			)
			debug[f"selected_interval_{tag}"] = (
				None if selected_interval is None else [int(selected_interval[0]), int(selected_interval[1])]
			)
			debug[f"crossing_count_{tag}"] = int(crossing_count)
		else:
			width, left_cross, right_cross, used_interp, reason = width_between_outer_level_crossings(seg, level)
		debug[f"level_{tag}"] = float(level)
		debug[f"left_cross_{tag}"] = (np.nan if not np.isfinite(left_cross) else float(left + left_cross))
		debug[f"right_cross_{tag}"] = (np.nan if not np.isfinite(right_cross) else float(left + right_cross))
		debug[f"width_reason_{tag}"] = str(reason)
		debug[f"level_below_noise_{tag}"] = bool(low_root_level and level <= noise_level)
		debug["interpolation_used"] = bool(debug["interpolation_used"] or used_interp)
		widths[tag] = float(width)
		out[f"{prefix}_width_at_{tag}"] = float(width)
	w80 = float(widths.get("80", np.nan))
	w50 = float(widths.get("50", np.nan))
	w20 = float(widths.get("20", np.nan))
	out[f"{prefix}_width_50_over_80"] = float(w50 / w80) if np.isfinite(w50) and np.isfinite(w80) and abs(w80) > 1e-12 else np.nan
	out[f"{prefix}_width_20_over_80"] = float(w20 / w80) if np.isfinite(w20) and np.isfinite(w80) and abs(w80) > 1e-12 else np.nan
	out[f"{prefix}_width_20_minus_80"] = float(w20 - w80) if np.isfinite(w20) and np.isfinite(w80) else np.nan
	out[f"{prefix}_width_50_minus_80"] = float(w50 - w80) if np.isfinite(w50) and np.isfinite(w80) else np.nan
	out[f"{prefix}_base_expansion_rate"] = float((w20 - w80) / max(abs(w80), 1e-12)) if np.isfinite(w20) and np.isfinite(w80) else np.nan
	if bool(include_low_root_metrics):
		def _root_summary(tags: Sequence[str], suffix: str) -> None:
			weights = {"5": 0.5, "10": 1.0, "15": 1.2, "20": 1.0, "25": 0.7}
			vals = []
			ws = []
			for tag in tags:
				v = float(widths.get(tag, np.nan))
				if np.isfinite(v):
					vals.append(v)
					ws.append(float(weights.get(tag, 1.0)))
			out[f"{prefix}_root_width_valid_n_{suffix}"] = float(len(vals))
			if not vals:
				out[f"{prefix}_root_width_mean_{suffix}"] = np.nan
				out[f"{prefix}_root_width_median_{suffix}"] = np.nan
				out[f"{prefix}_root_width_weighted_{suffix}"] = np.nan
				return
			arr = np.asarray(vals, dtype=float)
			w_arr = np.asarray(ws, dtype=float)
			out[f"{prefix}_root_width_mean_{suffix}"] = float(np.mean(arr))
			out[f"{prefix}_root_width_median_{suffix}"] = float(np.median(arr))
			out[f"{prefix}_root_width_weighted_{suffix}"] = float(np.sum(arr * w_arr) / max(float(np.sum(w_arr)), 1e-12))

		_root_summary(("5", "10", "15", "20", "25"), "5_25")
		_root_summary(("10", "15", "20", "25"), "10_25")
		def _dense_root_summary() -> None:
			dense_level_values = tuple(range(0, 95, 5))
			dense_tags = tuple(str(v) for v in dense_level_values)
			dense_widths: List[float] = []
			dense_levels: List[int] = []
			dense_debug: Dict[str, object] = {
				"levels": [int(v) for v in dense_level_values],
				"widths": {},
				"ratios": {},
				"skipped": {},
			}
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
				return
			for tag in dense_tags:
				pct = int(tag)
				if tag in widths:
					width = float(widths.get(tag, np.nan))
					left_cross = float(debug.get(f"left_cross_{tag}", np.nan))
					right_cross = float(debug.get(f"right_cross_{tag}", np.nan))
					level = float(levels.get(tag, np.nan))
					reason = str(debug.get(f"width_reason_{tag}", "ok"))
				else:
					frac = float(pct) / 100.0
					level = float(local_zero + frac * (local_top - local_zero))
					if bool(use_enhanced_spike_mapping) and mapping_reference_interval is not None:
						width, left_rel, right_rel, used_interp, reason, selected_interval, crossing_count = _edge_component_width_at_level(
							seg,
							level,
							apex_local=int(apex_local),
							reference_interval=mapping_reference_interval,
							bounds=mapping_active_bounds,
						)
						left_cross = np.nan if not np.isfinite(left_rel) else float(left + left_rel)
						right_cross = np.nan if not np.isfinite(right_rel) else float(left + right_rel)
						debug["interpolation_used"] = bool(debug["interpolation_used"] or used_interp)
						debug[f"dense_selected_interval_{tag}"] = (
							None if selected_interval is None else [int(selected_interval[0]), int(selected_interval[1])]
						)
						debug[f"dense_crossing_count_{tag}"] = int(crossing_count)
					else:
						width, left_rel, right_rel, used_interp, reason = width_between_outer_level_crossings(seg, level)
						left_cross = np.nan if not np.isfinite(left_rel) else float(left + left_rel)
						right_cross = np.nan if not np.isfinite(right_rel) else float(left + right_rel)
						debug["interpolation_used"] = bool(debug["interpolation_used"] or used_interp)
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
				dense_levels.append(int(pct))
			out[f"{prefix}_dense_width_valid_n_0_90"] = float(len(dense_widths))
			out[f"{prefix}_dense_width_missing_n_0_90"] = float(max(0, len(dense_tags) - len(dense_widths)))
			if not dense_widths:
				dense_debug["reason"] = "no_valid_widths"
				debug["dense_width_0_90"] = dense_debug
				return
			if len(dense_widths) != len(dense_tags):
				dense_debug["reason"] = "incomplete_dense_levels"
				debug["dense_width_0_90"] = dense_debug
				return
			out[f"{prefix}_dense_width_complete_0_90"] = 1.0
			out[f"{prefix}_dense_width_missing_n_0_90"] = 0.0
			w = np.asarray(dense_widths, dtype=float)
			lv = np.asarray(dense_levels, dtype=float)
			# Low relative levels describe the root/foot, so they get the largest weight.
			weights = 1.0 + (90.0 - np.clip(lv, 0.0, 90.0)) / 90.0
			diffs = np.diff(w)
			out[f"{prefix}_dense_width_sum_0_90"] = float(np.sum(w))
			out[f"{prefix}_dense_width_mean_0_90"] = float(np.mean(w))
			out[f"{prefix}_dense_width_median_0_90"] = float(np.median(w))
			out[f"{prefix}_dense_width_weighted_low_0_90"] = float(np.sum(w * weights) / max(float(np.sum(weights)), 1e-12))
			out[f"{prefix}_dense_width_total_abs_change_0_90"] = float(np.sum(np.abs(diffs))) if diffs.size else 0.0
			out[f"{prefix}_dense_width_max_abs_change_0_90"] = float(np.max(np.abs(diffs))) if diffs.size else 0.0
			out[f"{prefix}_dense_width_num_changes_0_90"] = float(np.sum(np.abs(diffs) > 1e-12)) if diffs.size else 0.0
			out[f"{prefix}_dense_width_change_density_0_90"] = float(np.sum(np.abs(diffs) > 1e-12) / max(1, diffs.size)) if diffs.size else 0.0
			out[f"{prefix}_dense_width_initial_0_90"] = float(w[0])
			out[f"{prefix}_dense_width_final_0_90"] = float(w[-1])
			out[f"{prefix}_dense_width_final_minus_initial_0_90"] = float(w[-1] - w[0])
			ratio_levels = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90)
			ratio_widths: Dict[int, float] = {}
			for pct in ratio_levels:
				tag = str(int(pct))
				if tag in widths:
					ratio_widths[int(pct)] = float(widths.get(tag, np.nan))
				elif tag in dense_debug["widths"]:
					row = dense_debug["widths"].get(tag, {})
					ratio_widths[int(pct)] = float(row.get("width", np.nan)) if isinstance(row, dict) else np.nan
				else:
					ratio_widths[int(pct)] = np.nan
			ratios: List[float] = []
			for prev_pct, curr_pct in zip(ratio_levels[:-1], ratio_levels[1:]):
				prev_w = float(ratio_widths.get(int(prev_pct), np.nan))
				curr_w = float(ratio_widths.get(int(curr_pct), np.nan))
				key = f"{int(curr_pct)}_over_{int(prev_pct)}"
				if np.isfinite(prev_w) and np.isfinite(curr_w) and abs(prev_w) > 1e-12:
					ratio = float(curr_w / prev_w)
					ratios.append(ratio)
					dense_debug["ratios"][key] = ratio
				else:
					dense_debug["ratios"][key] = np.nan
			out[f"{prefix}_dense_width_ratio_valid_n_0_90"] = float(len(ratios))
			if ratios:
				r_arr = np.asarray(ratios, dtype=float)
				out[f"{prefix}_dense_width_ratio_sum_0_90"] = float(np.sum(r_arr))
				out[f"{prefix}_dense_width_ratio_mean_0_90"] = float(np.mean(r_arr))
			dense_debug["valid_levels"] = [int(v) for v in dense_levels]
			dense_debug["weights"] = {str(int(v)): float(weights[i]) for i, v in enumerate(dense_levels)}
			debug["dense_width_0_90"] = dense_debug

		_dense_root_summary()
	pair_values: List[float] = []
	pair_values_snr: List[float] = []
	pair_broadening: List[float] = []
	bg = None if bg_mad is None else max(float(bg_mad), 1e-12)
	for lo, hi in (("10", "90"), ("25", "75"), ("50", "90"), ("20", "80")):
		w_lo = float(widths.get(lo, np.nan))
		w_hi = float(widths.get(hi, np.nan))
		i_lo = float(levels.get(lo, np.nan))
		i_hi = float(levels.get(hi, np.nan))
		d_i = float(i_hi - i_lo) if np.isfinite(i_lo) and np.isfinite(i_hi) else np.nan
		d_w = float(w_lo - w_hi) if np.isfinite(w_lo) and np.isfinite(w_hi) else np.nan
		nonpos = bool(np.isfinite(d_w) and d_w <= 0.0)
		debug[f"dI_{lo}_{hi}"] = float(d_i) if np.isfinite(d_i) else np.nan
		debug[f"dW_{lo}_{hi}"] = float(d_w) if np.isfinite(d_w) else np.nan
		debug[f"nonpositive_narrowing_{lo}_{hi}"] = bool(nonpos)
		if np.isfinite(d_i) and np.isfinite(d_w):
			ipwl = float(d_i / max(d_w, float(min_width_delta)))
			bpi = float(d_w / max(d_i, 1e-12))
			out[f"{prefix}_intensity_per_width_loss_{lo}_{hi}"] = ipwl
			out[f"{prefix}_broadening_per_intensity_{lo}_{hi}"] = bpi
			pair_values.append(ipwl)
			pair_broadening.append(bpi)
			if bg is not None:
				ipwl_snr = float(d_i / max(bg * max(d_w, float(min_width_delta)), 1e-12))
				out[f"{prefix}_intensity_per_width_loss_{lo}_{hi}_snr"] = ipwl_snr
				pair_values_snr.append(ipwl_snr)
			else:
				out[f"{prefix}_intensity_per_width_loss_{lo}_{hi}_snr"] = np.nan
		else:
			out[f"{prefix}_intensity_per_width_loss_{lo}_{hi}"] = np.nan
			out[f"{prefix}_broadening_per_intensity_{lo}_{hi}"] = np.nan
			out[f"{prefix}_intensity_per_width_loss_{lo}_{hi}_snr"] = np.nan
	if pair_values:
		arr = np.asarray(pair_values, dtype=float)
		out[f"{prefix}_profile_intensity_per_width_loss_mean"] = float(np.mean(arr))
		out[f"{prefix}_profile_intensity_per_width_loss_median"] = float(np.median(arr))
		out[f"{prefix}_profile_intensity_per_width_loss_max"] = float(np.max(arr))
	else:
		out[f"{prefix}_profile_intensity_per_width_loss_mean"] = np.nan
		out[f"{prefix}_profile_intensity_per_width_loss_median"] = np.nan
		out[f"{prefix}_profile_intensity_per_width_loss_max"] = np.nan
	if pair_broadening:
		arr_b = np.asarray(pair_broadening, dtype=float)
		out[f"{prefix}_profile_broadening_per_intensity_mean"] = float(np.mean(arr_b))
		out[f"{prefix}_profile_broadening_per_intensity_median"] = float(np.median(arr_b))
	else:
		out[f"{prefix}_profile_broadening_per_intensity_mean"] = np.nan
		out[f"{prefix}_profile_broadening_per_intensity_median"] = np.nan
	if pair_values_snr:
		out[f"{prefix}_profile_intensity_per_width_loss_median_snr"] = float(np.median(np.asarray(pair_values_snr, dtype=float)))
	else:
		out[f"{prefix}_profile_intensity_per_width_loss_median_snr"] = np.nan
	out[f"{prefix}_debug"] = debug
	return out


def compute_edge_dense_width_metrics(
		signal: np.ndarray,
		*,
		detection_left: int,
		detection_right: int,
		prefix: str,
		bg_mad: float | None = None,
		levels: Sequence[int] = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95),
		context_pad_pts: int = 0,
		context_min_pad_pts: int = 0,
		context_max_pad_pts: int | None = None,
		edge_dense_min_snr: float = 1.0,
) -> Dict[str, object]:
	"""Summarize dense outer-envelope widths across relative height levels.

	Dense raw edge-width metrics are experimental Raman-veto diagnostics: noise,
	muon, and Raman-like candidates can be ordered non-monotonically, so binary
	muon-vs-nonmuon AUC alone can be misleading. Use ternary/pairwise analysis for
	these metrics when label_class is available.
	"""
	x = np.asarray(signal, dtype=float)
	pfx = str(prefix).strip().rstrip("_")
	metric_names = (
		"width_sum",
		"width_mean",
		"width_median",
		"width_max",
		"width_min",
		"width_valid_n",
		"width_auc_norm",
		"width_weighted_mean",
		"width_low_sum",
		"width_mid_sum",
		"width_high_sum",
		"width_low_high_ratio",
		"width_mid_high_ratio",
	)
	out: Dict[str, object] = {f"{pfx}_{name}": np.nan for name in metric_names}
	out[f"{pfx}_width_valid_n"] = 0.0
	debug: Dict[str, object] = {"reason": "ok"}
	if x.ndim != 1 or x.size == 0:
		debug["reason"] = "empty_signal"
		out[f"{pfx}_debug"] = debug
		return out
	n = int(x.size)
	left0 = int(np.clip(detection_left, 0, n - 1))
	right0 = int(np.clip(detection_right, 0, n - 1))
	if right0 < left0:
		left0, right0 = right0, left0
	pad = max(int(context_pad_pts), int(context_min_pad_pts))
	if context_max_pad_pts is not None:
		pad = min(pad, int(context_max_pad_pts))
	left = max(0, left0 - max(0, pad))
	right = min(n - 1, right0 + max(0, pad))
	seg = np.asarray(x[left:right + 1], dtype=float)
	debug.update({
		"measurement_left": int(left),
		"measurement_right": int(right),
		"detection_left": int(left0),
		"detection_right": int(right0),
		"context_pad_pts": int(pad),
		"measurement_mode": "expanded" if pad > 0 else "strict",
	})
	if seg.size == 0 or not np.all(np.isfinite(seg)):
		debug["reason"] = "invalid_segment"
		out[f"{pfx}_debug"] = debug
		return out
	baseline = float(np.min(seg))
	y_max = float(np.max(seg))
	amp = float(y_max - baseline)
	if not np.isfinite(amp) or amp <= 1e-12:
		debug.update({"reason": "flat_segment", "baseline": baseline, "y_max": y_max, "amp": amp})
		out[f"{pfx}_debug"] = debug
		return out
	noise_delta = float(edge_dense_min_snr) * max(float(bg_mad), 1e-12) if bg_mad is not None else 0.05 * amp
	noise_level = float(baseline + noise_delta)
	level_list = [int(v) for v in levels if 0 < int(v) < 100]
	widths: List[float] = []
	valid_levels: List[int] = []
	skipped: Dict[str, str] = {}
	level_debug: Dict[str, Dict[str, object]] = {}
	for level_pct in level_list:
		level_value = float(baseline + (float(level_pct) / 100.0) * amp)
		if level_value <= noise_level:
			skipped[str(level_pct)] = "below_noise"
			level_debug[str(level_pct)] = {"level_value": level_value, "reason": "below_noise"}
			continue
		width, left_cross, right_cross, used_interp, reason = width_between_outer_level_crossings(seg, level_value)
		if not np.isfinite(width):
			skipped[str(level_pct)] = str(reason)
			level_debug[str(level_pct)] = {"level_value": level_value, "reason": str(reason)}
			continue
		widths.append(float(width))
		valid_levels.append(int(level_pct))
		level_debug[str(level_pct)] = {
			"level_value": level_value,
			"width": float(width),
			"left_cross": float(left + left_cross),
			"right_cross": float(left + right_cross),
			"interpolation_used": bool(used_interp),
			"reason": "ok",
		}
	debug.update({
		"baseline": baseline,
		"y_max": y_max,
		"amp": amp,
		"noise_level": noise_level,
		"edge_dense_min_snr": float(edge_dense_min_snr),
		"levels": level_list,
		"valid_levels": valid_levels,
		"skipped_levels": skipped,
		"level_debug": level_debug,
	})
	if not widths:
		debug["reason"] = "no_valid_levels"
		out[f"{pfx}_debug"] = debug
		return out
	w = np.asarray(widths, dtype=float)
	lv = np.asarray(valid_levels, dtype=float) / 100.0
	weights = np.asarray(valid_levels, dtype=float)
	low_mask = np.asarray([(5 <= v <= 35) for v in valid_levels], dtype=bool)
	mid_mask = np.asarray([(40 <= v <= 65) for v in valid_levels], dtype=bool)
	high_mask = np.asarray([(70 <= v <= 95) for v in valid_levels], dtype=bool)
	low_sum = float(np.sum(w[low_mask])) if np.any(low_mask) else 0.0
	mid_sum = float(np.sum(w[mid_mask])) if np.any(mid_mask) else 0.0
	high_sum = float(np.sum(w[high_mask])) if np.any(high_mask) else 0.0
	out[f"{pfx}_width_sum"] = float(np.sum(w))
	out[f"{pfx}_width_mean"] = float(np.mean(w))
	out[f"{pfx}_width_median"] = float(np.median(w))
	out[f"{pfx}_width_max"] = float(np.max(w))
	out[f"{pfx}_width_min"] = float(np.min(w))
	out[f"{pfx}_width_valid_n"] = float(w.size)
	out[f"{pfx}_width_auc_norm"] = float(np.trapezoid(w, lv)) if w.size >= 2 else 0.0
	out[f"{pfx}_width_weighted_mean"] = float(np.average(w, weights=np.maximum(weights, 1.0)))
	out[f"{pfx}_width_low_sum"] = low_sum
	out[f"{pfx}_width_mid_sum"] = mid_sum
	out[f"{pfx}_width_high_sum"] = high_sum
	out[f"{pfx}_width_low_high_ratio"] = float(low_sum / max(high_sum, 1e-12))
	out[f"{pfx}_width_mid_high_ratio"] = float(mid_sum / max(high_sum, 1e-12))
	out[f"{pfx}_debug"] = debug
	return out


def compute_ball_descent_metrics(
		signal: np.ndarray,
		*,
		apex_idx: int,
		detection_left: int,
		detection_right: int,
		prefix: str,
		bg_mad: float | None = None,
		ball_noise_k_mad: float = 1.0,
		ball_stop_rel_amp: float = 0.05,
		context_pad_pts: int = 20,
		context_min_pad_pts: int = 10,
		context_max_pad_pts: int | None = 80,
		prevent_crossing_neighbor_peak: bool = True,
) -> Dict[str, object]:
	"""Compute rolling-particle-inspired side descent proxies inside spike edges."""
	x = np.asarray(signal, dtype=float)
	pfx = str(prefix).strip().rstrip("_")
	keys = [
		"v_total_left", "v_total_right", "vx_left", "vx_right", "vy_left", "vy_right",
		"vy_over_vx_left", "vy_over_vx_right", "horizontal_run_left", "horizontal_run_right",
		"vertical_drop_left", "vertical_drop_right", "v_total_mean", "v_total_min",
		"v_total_max", "vx_mean", "vy_mean", "vy_over_vx_mean", "horizontal_run_mean",
		"vertical_drop_mean", "slide_time_left", "slide_time_right", "slide_time_mean",
	]
	out: Dict[str, object] = {f"{pfx}_{k}": np.nan for k in keys}
	debug: Dict[str, object] = {"reason": "ok"}
	if x.ndim != 1 or x.size == 0:
		debug["reason"] = "empty_signal"
		out[f"{pfx}_debug"] = debug
		return out
	n = int(x.size)
	left = int(np.clip(detection_left, 0, n - 1))
	right = int(np.clip(detection_right, 0, n - 1))
	if right < left:
		left, right = right, left
	pad = max(int(context_pad_pts), int(context_min_pad_pts))
	if context_max_pad_pts is not None:
		pad = min(pad, int(context_max_pad_pts))
	ctx_left = max(0, left - max(0, pad))
	ctx_right = min(n - 1, right + max(0, pad))
	apex = int(np.clip(apex_idx, ctx_left, ctx_right))
	seg = np.asarray(x[ctx_left:ctx_right + 1], dtype=float)
	if seg.size < 2 or not np.all(np.isfinite(seg)):
		debug.update({"reason": "invalid_segment", "measurement_left": int(left), "measurement_right": int(right), "ball_context_left": int(ctx_left), "ball_context_right": int(ctx_right), "apex": int(apex)})
		out[f"{pfx}_debug"] = debug
		return out
	baseline = float(np.min(seg))
	apex_y = float(x[apex])
	amp = float(np.max(seg) - baseline)
	if not np.isfinite(amp) or amp <= 1e-12:
		debug["reason"] = "flat_segment"
		out[f"{pfx}_debug"] = debug
		return out
	noise_floor = float(ball_noise_k_mad) * max(float(bg_mad), 1e-12) if bg_mad is not None else float(ball_stop_rel_amp) * amp
	stop_delta = max(noise_floor, float(ball_stop_rel_amp) * amp)
	y_stop = float(baseline + stop_delta)
	debug.update({
		"measurement_left": int(left),
		"measurement_right": int(right),
		"ball_context_left": int(ctx_left),
		"ball_context_right": int(ctx_right),
		"apex": int(apex),
		"baseline": baseline,
		"apex_y": apex_y,
		"amp": amp,
		"y_stop": y_stop,
		"noise_floor": float(noise_floor),
	})

	def _side(side: str) -> Dict[str, float | bool | List[float]]:
		if side == "left":
			indices = list(range(apex, ctx_left - 1, -1))
		else:
			indices = list(range(apex, ctx_right + 1))
		stop = indices[-1]
		stop_reason = "context_edge"
		threshold_hits = 0
		valley_idx = int(apex)
		valley_y = float(x[apex])
		for idx in indices[1:]:
			yv = float(x[idx])
			if yv < valley_y:
				valley_idx = int(idx)
				valley_y = yv
			if yv <= y_stop:
				threshold_hits += 1
			else:
				threshold_hits = 0
			if threshold_hits >= 2:
				stop = int(idx)
				stop_reason = "noise_level"
				break
			if bool(prevent_crossing_neighbor_peak) and valley_idx != apex and (yv - valley_y) > max(float(noise_floor), 1e-12):
				stop = int(valley_idx)
				stop_reason = "neighbor_valley"
				break
		if side == "left":
			path_idx = np.arange(stop, apex + 1, dtype=int)
		else:
			path_idx = np.arange(apex, stop + 1, dtype=int)
		stop_y_actual = float(x[stop])
		dy_total = float(max(apex_y - stop_y_actual, 0.0))
		dx_total = float(abs(stop - apex))
		if path_idx.size >= 2:
			if side == "left":
				i0, i1 = int(path_idx[0]), int(path_idx[min(1, path_idx.size - 1)])
			else:
				i0, i1 = int(path_idx[max(path_idx.size - 2, 0)]), int(path_idx[-1])
			m = float(abs((float(x[i1]) - float(x[i0])) / max(abs(i1 - i0), 1)))
		else:
			m = 0.0
		v_total = float(np.sqrt(max(2.0 * dy_total, 0.0)))
		den = float(np.sqrt(1.0 + m * m))
		vx = float(v_total / den)
		vy = float(v_total * m / den)
		vy_over_vx = float(vy / max(vx, 1e-12))
		t = 0.0
		if path_idx.size >= 2:
			vals = np.asarray(x[path_idx], dtype=float)
			for j in range(vals.size - 1):
				ds = float(np.sqrt(1.0 + (float(vals[j + 1]) - float(vals[j])) ** 2))
				segment_y = min(float(vals[j]), float(vals[j + 1]))
				v = float(np.sqrt(max(2.0 * (apex_y - segment_y), 1e-12)))
				t += ds / v
		return {
			"stop_index": float(stop),
			"stop_y": stop_y_actual,
			"stop_found": bool(stop_reason != "context_edge"),
			"stop_reason": stop_reason,
			"hit_context_edge": bool(stop_reason == "context_edge"),
			"hit_neighbor_valley": bool(stop_reason == "neighbor_valley"),
			"local_slope": m,
			"v_total": v_total,
			"vx": vx,
			"vy": vy,
			"vy_over_vx": vy_over_vx,
			"horizontal_run": dx_total,
			"vertical_drop": dy_total,
			"slide_time": float(t),
			"path_indices": [int(v) for v in path_idx.tolist()],
		}

	left_info = _side("left")
	right_info = _side("right")
	for side, info in (("left", left_info), ("right", right_info)):
		for key in ("v_total", "vx", "vy", "vy_over_vx", "horizontal_run", "vertical_drop", "slide_time"):
			out[f"{pfx}_{key}_{side}"] = float(info[key])
		debug[f"{side}_stop_index"] = int(info["stop_index"])
		debug[f"stop_index_{side}"] = int(info["stop_index"])
		debug[f"stop_reason_{side}"] = str(info["stop_reason"])
		debug[f"hit_context_edge_{side}"] = bool(info["hit_context_edge"])
		debug[f"hit_neighbor_valley_{side}"] = bool(info["hit_neighbor_valley"])
		debug[f"{side}_stop_y"] = float(info["stop_y"])
		debug[f"{side}_stop_found"] = bool(info["stop_found"])
		debug[f"{side}_local_slope"] = float(info["local_slope"])
		debug[f"{side}_path_indices"] = list(info["path_indices"])  # type: ignore[arg-type]
	for key in ("v_total", "vx", "vy", "vy_over_vx", "horizontal_run", "vertical_drop", "slide_time"):
		vals = np.asarray([float(left_info[key]), float(right_info[key])], dtype=float)
		out[f"{pfx}_{key}_mean"] = float(np.mean(vals))
	out[f"{pfx}_v_total_min"] = float(min(float(left_info["v_total"]), float(right_info["v_total"])))
	out[f"{pfx}_v_total_max"] = float(max(float(left_info["v_total"]), float(right_info["v_total"])))
	out[f"{pfx}_debug"] = debug
	return out


def compute_exponential_decay_metrics(
		signal: np.ndarray,
		*,
		apex_idx: int,
		detection_left: int,
		detection_right: int,
		prefix: str,
		bg_mad: float | None = None,
		exp_fit_min_rel_amp: float = 0.05,
		exp_fit_noise_k_mad: float = 1.0,
		context_pad_pts: int = 20,
		exp_foot_low_rel: float = 0.05,
		exp_foot_high_rel: float = 0.45,
		exp_min_points: int = 3,
		exp_prevent_apex_region: bool = True,
) -> Dict[str, object]:
	"""Fit log-linear exponential side decays inside spike edges."""
	x = np.asarray(signal, dtype=float)
	pfx = str(prefix).strip().rstrip("_")
	keys = [
		"decay_k_left", "decay_k_right", "decay_r2_left", "decay_r2_right",
		"decay_n_left", "decay_n_right", "decay_k_mean", "decay_k_min",
		"decay_k_max", "decay_k_asymmetry", "decay_r2_mean", "decay_valid_sides",
	]
	out: Dict[str, object] = {f"{pfx}_{k}": np.nan for k in keys}
	debug: Dict[str, object] = {"reason": "ok"}
	if x.ndim != 1 or x.size == 0:
		debug["reason"] = "empty_signal"
		out[f"{pfx}_debug"] = debug
		return out
	n = int(x.size)
	left = int(np.clip(detection_left, 0, n - 1))
	right = int(np.clip(detection_right, 0, n - 1))
	if right < left:
		left, right = right, left
	pad = max(0, int(context_pad_pts))
	ctx_left = max(0, left - pad)
	ctx_right = min(n - 1, right + pad)
	apex = int(np.clip(apex_idx, ctx_left, ctx_right))
	seg = np.asarray(x[ctx_left:ctx_right + 1], dtype=float)
	if seg.size < 3 or not np.all(np.isfinite(seg)):
		debug.update({"reason": "invalid_segment", "measurement_left": int(left), "measurement_right": int(right), "exp_context_left": int(ctx_left), "exp_context_right": int(ctx_right), "apex": int(apex)})
		out[f"{pfx}_debug"] = debug
		return out
	baseline = float(np.percentile(seg, 5.0))
	apex_y = float(x[apex])
	amp = float(apex_y - baseline)
	if not np.isfinite(amp) or amp <= 1e-12:
		debug["reason"] = "flat_segment"
		out[f"{pfx}_debug"] = debug
		return out
	noise_abs = float(exp_fit_noise_k_mad) * max(float(bg_mad), 1e-12) if bg_mad is not None else 0.0
	y_low = float(baseline + max(float(exp_foot_low_rel) * amp, noise_abs, 1e-12))
	y_high = float(baseline + float(exp_foot_high_rel) * amp)
	debug.update({
		"measurement_left": int(left),
		"measurement_right": int(right),
		"exp_context_left": int(ctx_left),
		"exp_context_right": int(ctx_right),
		"fit_mode": "foot",
		"apex": int(apex),
		"baseline": baseline,
		"amp": amp,
		"fit_low_threshold": y_low,
		"fit_high_threshold": y_high,
	})

	def _fit(side: str) -> Dict[str, object]:
		if side == "left":
			idx = np.arange(apex, ctx_left - 1, -1, dtype=int)
		else:
			idx = np.arange(apex, ctx_right + 1, dtype=int)
		y_side = np.asarray(x[idx], dtype=float)
		y_rel = np.asarray(y_side - baseline, dtype=float)
		valid = np.isfinite(y_side) & (y_side >= y_low) & (y_side <= y_high) & (y_rel > 1e-12)
		if bool(exp_prevent_apex_region):
			valid = valid & (np.abs(idx - apex) > 0)
		if np.count_nonzero(valid) < int(exp_min_points):
			return {"valid": False, "reason": "too_few_points", "n": int(np.count_nonzero(valid)), "indices": [int(v) for v in idx[valid].tolist()]}
		d = np.arange(idx.size, dtype=float)[valid]
		log_y = np.log(y_rel[valid])
		A = np.vstack([np.ones_like(d), d]).T
		coef, *_ = np.linalg.lstsq(A, log_y, rcond=None)
		intercept = float(coef[0])
		slope = float(coef[1])
		pred = A @ coef
		ss_res = float(np.sum((log_y - pred) ** 2))
		ss_tot = float(np.sum((log_y - float(np.mean(log_y))) ** 2))
		r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
		k = float(max(-slope, 0.0))
		fit_domain = np.flatnonzero(valid)
		full_d = np.arange(idx.size, dtype=float)
		fit_y = baseline + np.exp(intercept - k * full_d)
		return {
			"valid": True,
			"reason": "ok",
			"n": int(np.count_nonzero(valid)),
			"k": k,
			"r2": r2,
			"intercept": intercept,
			"slope": slope,
			"indices": [int(v) for v in idx[valid].tolist()],
			"fit_indices": [int(v) for v in idx[fit_domain].tolist()],
			"fit_values": [float(v) for v in fit_y[fit_domain].tolist()],
			"fit_start": int(idx[fit_domain[0]]) if fit_domain.size else -1,
			"fit_end": int(idx[fit_domain[-1]]) if fit_domain.size else -1,
		}

	left_fit = _fit("left")
	right_fit = _fit("right")
	valid_ks: List[float] = []
	valid_r2: List[float] = []
	for side, fit in (("left", left_fit), ("right", right_fit)):
		out[f"{pfx}_decay_n_{side}"] = float(fit.get("n", 0))
		debug[f"{side}_reason"] = str(fit.get("reason", ""))
		debug[f"{side}_invalid_reason"] = str(fit.get("reason", ""))
		debug[f"{side}_indices"] = list(fit.get("indices", []))
		debug[f"fit_{side}_start"] = int(fit.get("fit_start", -1))
		debug[f"fit_{side}_end"] = int(fit.get("fit_end", -1))
		if bool(fit.get("valid", False)):
			k = float(fit.get("k", np.nan))
			r2 = float(fit.get("r2", np.nan))
			out[f"{pfx}_decay_k_{side}"] = k
			out[f"{pfx}_decay_r2_{side}"] = r2
			debug[f"{side}_intercept"] = float(fit.get("intercept", np.nan))
			debug[f"{side}_slope"] = float(fit.get("slope", np.nan))
			debug[f"{side}_fit_indices"] = list(fit.get("fit_indices", []))
			debug[f"{side}_fit_values"] = list(fit.get("fit_values", []))
			if np.isfinite(k):
				valid_ks.append(k)
			if np.isfinite(r2):
				valid_r2.append(r2)
	if valid_ks:
		arr = np.asarray(valid_ks, dtype=float)
		out[f"{pfx}_decay_k_mean"] = float(np.mean(arr))
		out[f"{pfx}_decay_k_min"] = float(np.min(arr))
		out[f"{pfx}_decay_k_max"] = float(np.max(arr))
		if np.isfinite(float(out[f"{pfx}_decay_k_left"])) and np.isfinite(float(out[f"{pfx}_decay_k_right"])):
			kl = float(out[f"{pfx}_decay_k_left"])
			kr = float(out[f"{pfx}_decay_k_right"])
			out[f"{pfx}_decay_k_asymmetry"] = float(abs(kl - kr) / max(abs(kl) + abs(kr), 1e-12))
	out[f"{pfx}_decay_valid_sides"] = float(len(valid_ks))
	out[f"{pfx}_decay_r2_mean"] = float(np.mean(np.asarray(valid_r2, dtype=float))) if valid_r2 else np.nan
	out[f"{pfx}_debug"] = debug
	return out


def compute_shape_top_hat_signal(raw_signal: np.ndarray, window: int) -> np.ndarray:
	"""Compute a larger top-hat representation for candidate broadening analysis."""
	x = np.asarray(raw_signal, dtype=float)
	if x.ndim != 1 or x.size == 0:
		return np.asarray([], dtype=float)
	w = max(3, int(window))
	if w % 2 == 0:
		w += 1
	opn = ndimage.grey_opening(x, size=int(w)).astype(float)
	return np.maximum(x - opn, 0.0).astype(float, copy=False)


def compute_candidate_tophat_shape_metrics(
		top_hat_signal: np.ndarray,
		*,
		apex_idx: int,
		detection_left: int,
		detection_right: int,
		prefix: str = "th",
) -> Dict[str, object]:
	"""Measure broadening and compactness on the original candidate-detection top-hat signal."""
	pfx = str(prefix).strip() or "th"
	def _k(name: str) -> str:
		return f"{pfx}_{name}"
	x = np.asarray(top_hat_signal, dtype=float)
	zero = {
		_k("width_at_80"): np.nan,
		_k("width_at_50"): np.nan,
		_k("width_at_20"): np.nan,
		_k("width_50_over_80"): np.nan,
		_k("width_20_over_80"): np.nan,
		_k("width_20_minus_80"): np.nan,
		_k("width_50_minus_80"): np.nan,
		_k("base_expansion_rate"): np.nan,
		_k("area_total"): np.nan,
		_k("area_above_80"): np.nan,
		_k("area_above_50"): np.nan,
		_k("area_above_20"): np.nan,
		_k("area_above_80_fraction"): np.nan,
		_k("area_above_50_fraction"): np.nan,
		_k("area_above_20_fraction"): np.nan,
		_k("area_core_fraction_1pt"): np.nan,
		_k("area_core_fraction_2pt"): np.nan,
		_k("area_core_fraction_3pt"): np.nan,
		_k("points_to_50_area"): np.nan,
		_k("points_to_80_area"): np.nan,
		_k("points_to_50_area_norm"): np.nan,
		_k("points_to_80_area_norm"): np.nan,
		f"{pfx}_debug": {
			"apex_value": np.nan,
			"component_left": None,
			"component_right": None,
			"apex_inside_component": 0,
			"nearest_component_used": 0,
			"distance_to_selected_component": np.nan,
			"width_bounds_80": None,
			"width_bounds_50": None,
			"width_bounds_20": None,
		},
	}
	if x.ndim != 1 or x.size < 3:
		return zero
	n = int(x.size)
	p = int(np.clip(apex_idx, 0, n - 1))
	a = int(np.clip(detection_left, 0, n - 1))
	b = int(np.clip(detection_right, 0, n - 1))
	if b < a:
		a, b = b, a

	pos = np.maximum(x, 0.0)
	sel = _select_component_from_mask(pos > 1e-12, p)
	left = sel["selected_left"]
	right = sel["selected_right"]
	if not np.isfinite(float(left)) or not np.isfinite(float(right)):
		return zero
	left = int(left)
	right = int(right)
	comp = pos[left:right + 1]
	comp_width = max(1, int(right - left + 1))
	comp_area = float(np.sum(comp))
	apex_val = float(np.max(comp)) if comp.size else float("nan")
	if not np.isfinite(apex_val) or apex_val <= 0.0 or comp_area <= 0.0:
		return zero
	if comp_area <= 0.0:
		return zero
	eps = 1e-12

	def _width_bounds(frac: float) -> Tuple[int, Optional[Tuple[int, int]]]:
		thr = float(frac * apex_val)
		mask = comp >= thr
		if not np.any(mask):
			return 0, None
		idx = np.where(mask)[0].astype(int)
		lb = int(left + idx[0])
		rb = int(left + idx[-1])
		return int(rb - lb + 1), (lb, rb)

	def _area_above(frac: float) -> float:
		thr = float(frac * apex_val)
		return float(np.sum(comp[comp >= thr]))

	def _core_fraction(radius: int) -> float:
		lc = max(left, p - int(radius))
		rc = min(right, p + int(radius))
		return float(np.sum(pos[lc:rc + 1]) / max(comp_area, eps))

	def _points_to_area_fraction(target_fraction: float) -> float:
		target = float(target_fraction) * comp_area
		acc = float(pos[p])
		if acc >= target:
			return 0.0
		for radius in range(1, comp_width + 1):
			lc = max(left, p - radius)
			rc = min(right, p + radius)
			acc = float(np.sum(pos[lc:rc + 1]))
			if acc >= target:
				return float(radius)
		return float(comp_width - 1)

	w80, b80 = _width_bounds(0.80)
	w50, b50 = _width_bounds(0.50)
	w20, b20 = _width_bounds(0.20)
	p50 = _points_to_area_fraction(0.50)
	p80 = _points_to_area_fraction(0.80)
	out = {
		_k("width_at_80"): float(w80),
		_k("width_at_50"): float(w50),
		_k("width_at_20"): float(w20),
		_k("width_50_over_80"): float(w50 / max(w80, 1)),
		_k("width_20_over_80"): float(w20 / max(w80, 1)),
		_k("width_20_minus_80"): float(w20 - w80),
		_k("width_50_minus_80"): float(w50 - w80),
		_k("base_expansion_rate"): float((w20 - w80) / max(float(w80), 1.0)),
		_k("area_total"): float(comp_area),
		_k("area_above_80"): float(_area_above(0.80)),
		_k("area_above_50"): float(_area_above(0.50)),
		_k("area_above_20"): float(_area_above(0.20)),
		_k("area_core_fraction_1pt"): float(_core_fraction(1)),
		_k("area_core_fraction_2pt"): float(_core_fraction(2)),
		_k("area_core_fraction_3pt"): float(_core_fraction(3)),
		_k("points_to_50_area"): float(p50),
		_k("points_to_80_area"): float(p80),
		_k("points_to_50_area_norm"): float(p50 / max(float(comp_width), 1.0)),
		_k("points_to_80_area_norm"): float(p80 / max(float(comp_width), 1.0)),
	}
	out[_k("area_above_80_fraction")] = float(out[_k("area_above_80")] / max(comp_area, eps))
	out[_k("area_above_50_fraction")] = float(out[_k("area_above_50")] / max(comp_area, eps))
	out[_k("area_above_20_fraction")] = float(out[_k("area_above_20")] / max(comp_area, eps))
	out[f"{pfx}_debug"] = {
		"apex_value": float(apex_val),
		"component_left": int(left),
		"component_right": int(right),
		"apex_inside_component": int(float(sel["apex_inside_support"]) > 0.5),
		"nearest_component_used": int(float(sel["nearest_component_used"]) > 0.5),
		"distance_to_selected_component": float(sel["distance_to_selected_component"]),
		"width_bounds_80": b80,
		"width_bounds_50": b50,
		"width_bounds_20": b20,
		"detection_left": int(a),
		"detection_right": int(b),
		"component_width": int(comp_width),
		"component_area_total": float(comp_area),
	}
	return out


def compute_compact_gws_metrics(
		diag: Dict[str, object],
		*,
		ctx_suffix: str = "ctx10",
		prefix: Optional[str] = None,
) -> Dict[str, float]:
	"""Build a compact, interpretable GWS metric set from one shared diagnostic payload."""
	bundle = compute_gws_feature_bundle_from_diagnostics(diag)
	bundle.update(
		_gws_support_fraction_metrics(
			bundle,
			float(diag.get("gws_context_adjusted_width", np.nan)),
		)
	)
	out = {
		f"width_soft_slope_{ctx_suffix}": float(diag.get("width_soft_slope", 0.0)),
		f"width_soft_d1_mean_{ctx_suffix}": float(diag.get("width_soft_d1_mean", 0.0)),
		f"support01_{ctx_suffix}": float(diag.get("gws_support01", 0.0)),
		f"evidence_signed_{ctx_suffix}": float(diag.get("gws_evidence_signed", -1.0)),
		"support_initial_fraction": float(bundle.get("gws_support_initial_fraction", np.nan)),
		"support_total_increase": float(bundle.get("gws_support_total_increase", np.nan)),
		"support_total_increase_fraction": float(bundle.get("gws_support_total_increase_fraction", np.nan)),
		"support_final": float(bundle.get("gws_support_final", np.nan)),
		"support_final_fraction": float(bundle.get("gws_support_final_fraction", np.nan)),
		"support_final_minus_initial": float(bundle.get("gws_support_final_minus_initial", np.nan)),
		"support_final_minus_initial_fraction": float(bundle.get("gws_support_final_minus_initial_fraction", np.nan)),
		"support_max": float(bundle.get("gws_support_max", np.nan)),
		"support_max_fraction": float(bundle.get("gws_support_max_fraction", np.nan)),
		"support_total_abs_change": float(bundle.get("gws_support_total_abs_change", np.nan)),
		"support_total_abs_change_fraction": float(bundle.get("gws_support_total_abs_change_fraction", np.nan)),
		"support_max_abs_change": float(bundle.get("gws_support_max_abs_change", np.nan)),
		"support_num_increases": float(bundle.get("gws_support_num_increases", np.nan)),
		"support_num_changes": float(bundle.get("gws_support_num_changes", np.nan)),
		"support_change_density": float(bundle.get("gws_support_change_density", np.nan)),
		"support_longest_constant_run_norm": float(bundle.get("gws_support_longest_constant_run_norm", np.nan)),
		"comp_support_initial": float(bundle.get("gws_comp_support_initial", np.nan)),
		"comp_support_initial_fraction": float(bundle.get("gws_comp_support_initial_fraction", np.nan)),
		"comp_support_final": float(bundle.get("gws_comp_support_final", np.nan)),
		"comp_support_final_fraction": float(bundle.get("gws_comp_support_final_fraction", np.nan)),
		"comp_support_min": float(bundle.get("gws_comp_support_min", np.nan)),
		"comp_support_max": float(bundle.get("gws_comp_support_max", np.nan)),
		"comp_support_max_fraction": float(bundle.get("gws_comp_support_max_fraction", np.nan)),
		"comp_support_final_minus_initial": float(bundle.get("gws_comp_support_final_minus_initial", np.nan)),
		"comp_support_final_minus_initial_fraction": float(bundle.get("gws_comp_support_final_minus_initial_fraction", np.nan)),
		"comp_support_total_increase": float(bundle.get("gws_comp_support_total_increase", np.nan)),
		"comp_support_total_increase_fraction": float(bundle.get("gws_comp_support_total_increase_fraction", np.nan)),
		"comp_support_total_decrease": float(bundle.get("gws_comp_support_total_decrease", np.nan)),
		"comp_support_total_abs_change": float(bundle.get("gws_comp_support_total_abs_change", np.nan)),
		"comp_support_total_abs_change_fraction": float(bundle.get("gws_comp_support_total_abs_change_fraction", np.nan)),
		"comp_support_num_increases": float(bundle.get("gws_comp_support_num_increases", np.nan)),
		"comp_support_num_decreases": float(bundle.get("gws_comp_support_num_decreases", np.nan)),
		"comp_support_num_changes": float(bundle.get("gws_comp_support_num_changes", np.nan)),
		"comp_support_change_density": float(bundle.get("gws_comp_support_change_density", np.nan)),
		"comp_support_max_abs_change": float(bundle.get("gws_comp_support_max_abs_change", np.nan)),
		"comp_support_longest_constant_run": float(bundle.get("gws_comp_support_longest_constant_run", np.nan)),
		"comp_support_longest_constant_run_norm": float(bundle.get("gws_comp_support_longest_constant_run_norm", np.nan)),
		"comp_area_initial": float(bundle.get("gws_comp_area_initial", np.nan)),
		"comp_area_final": float(bundle.get("gws_comp_area_final", np.nan)),
		"comp_area_total_increase": float(bundle.get("gws_comp_area_total_increase", np.nan)),
		"comp_width_initial": float(bundle.get("gws_comp_width_initial", np.nan)),
		"comp_width_final": float(bundle.get("gws_comp_width_final", np.nan)),
		"comp_width_total_abs_change": float(bundle.get("gws_comp_width_total_abs_change", np.nan)),
		"comp_apex_inside_support": float(bundle.get("gws_comp_apex_inside_support", np.nan)),
		"comp_nearest_component_used": float(bundle.get("gws_comp_nearest_component_used", np.nan)),
		"comp_distance_to_selected_component": float(bundle.get("gws_comp_distance_to_selected_component", np.nan)),
		"comp_n_components": float(bundle.get("gws_comp_n_components", np.nan)),
		"comp_selected_left": float(bundle.get("gws_comp_selected_left", np.nan)),
		"comp_selected_right": float(bundle.get("gws_comp_selected_right", np.nan)),
	}
	pfx = "" if prefix is None else str(prefix).strip()
	if not pfx:
		return {
			f"gmt_{name}" if name.startswith("width_soft_") else f"gws_{name}": float(value)
			for name, value in out.items()
		}
	prefixed = {
		f"{pfx}_{name}": float(value)
		for name, value in out.items()
	}
	if pfx == "gws_mg":
		# Normalize by the final adjusted GWS context width to make the metric
		# more portable across datasets with different segment/context sizes.
		context_width = float(diag.get("gws_context_adjusted_width", np.nan))
		support_final = float(bundle.get("gws_support_final", np.nan))
		if np.isfinite(context_width) and context_width > 0.0 and np.isfinite(support_final):
			support_final_frac = float(np.clip(support_final / context_width, 0.0, 1.0))
			prefixed[f"{pfx}_support_final_fraction"] = support_final_frac
			prefixed[f"{pfx}_support_final_evidence_signed"] = _signed_evidence_low_is_muon(support_final_frac)
		else:
			prefixed[f"{pfx}_support_final_fraction"] = float("nan")
			prefixed[f"{pfx}_support_final_evidence_signed"] = float("nan")
	return prefixed


def _gws_width_trace_stability_metrics(width_trace: np.ndarray) -> Dict[str, float]:
	"""Summarize plateau/stability properties of the final GWS width trace."""
	out = {
		"gws_width_longest_constant_run": np.nan,
		"gws_width_longest_constant_run_norm": np.nan,
		"gws_width_num_changes": 0.0,
		"gws_width_change_density": np.nan,
		"gws_width_total_variation": np.nan,
		"gws_width_total_variation_norm": np.nan,
		"gws_width_longest_stable_run_tol1": np.nan,
		"gws_width_longest_stable_run_tol1_norm": np.nan,
		"gws_width_num_changes_tol1": 0.0,
		"gws_width_change_density_tol1": np.nan,
		"gws_width_stable_suffix_len_tol1": np.nan,
		"gws_width_stable_suffix_len_tol1_norm": np.nan,
		"gws_width_last_change_index_tol1": np.nan,
		"gws_width_last_change_scale_norm_tol1": np.nan,
	}
	w = np.asarray(width_trace, dtype=float)
	if w.ndim != 1:
		return out
	mask = np.isfinite(w)
	if not np.any(mask):
		return out
	w = w[mask]
	n = int(w.size)
	if n <= 0:
		return out
	if n == 1:
		return {
			**out,
			"gws_width_longest_constant_run": 1.0,
			"gws_width_longest_constant_run_norm": 1.0,
			"gws_width_change_density": 0.0,
			"gws_width_total_variation": 0.0,
			"gws_width_total_variation_norm": 0.0,
			"gws_width_longest_stable_run_tol1": 1.0,
			"gws_width_longest_stable_run_tol1_norm": 1.0,
			"gws_width_change_density_tol1": 0.0,
			"gws_width_stable_suffix_len_tol1": 1.0,
			"gws_width_stable_suffix_len_tol1_norm": 1.0,
			"gws_width_last_change_index_tol1": -1.0,
			"gws_width_last_change_scale_norm_tol1": 0.0,
		}
	dw = np.diff(w)
	abs_dw = np.abs(dw)

	def _longest_run(diff_mask: np.ndarray) -> int:
		best = 1
		cur = 1
		for ok in diff_mask:
			if bool(ok):
				cur += 1
			else:
				best = max(best, cur)
				cur = 1
		return max(best, cur)

	const_mask = abs_dw <= 0.0
	stable1_mask = abs_dw <= 1.0
	longest_const = int(_longest_run(const_mask))
	longest_stable1 = int(_longest_run(stable1_mask))
	num_changes = int(np.count_nonzero(abs_dw > 0.0))
	num_changes_tol1 = int(np.count_nonzero(abs_dw > 1.0))
	total_variation = float(np.sum(abs_dw))
	w_range = float(np.nanmax(w) - np.nanmin(w))
	last_change_idx = int(np.max(np.where(abs_dw > 1.0)[0])) if np.any(abs_dw > 1.0) else -1
	suffix_len = 1
	for diff_val in abs_dw[::-1]:
		if diff_val <= 1.0:
			suffix_len += 1
		else:
			break
	return {
		"gws_width_longest_constant_run": float(longest_const),
		"gws_width_longest_constant_run_norm": float(longest_const / max(n, 1)),
		"gws_width_num_changes": float(num_changes),
		"gws_width_change_density": float(num_changes / max(n - 1, 1)),
		"gws_width_total_variation": float(total_variation),
		"gws_width_total_variation_norm": float(total_variation / max(w_range, 1e-12)) if w_range > 0.0 else 0.0,
		"gws_width_longest_stable_run_tol1": float(longest_stable1),
		"gws_width_longest_stable_run_tol1_norm": float(longest_stable1 / max(n, 1)),
		"gws_width_num_changes_tol1": float(num_changes_tol1),
		"gws_width_change_density_tol1": float(num_changes_tol1 / max(n - 1, 1)),
		"gws_width_stable_suffix_len_tol1": float(suffix_len),
		"gws_width_stable_suffix_len_tol1_norm": float(suffix_len / max(n, 1)),
		"gws_width_last_change_index_tol1": float(last_change_idx),
		"gws_width_last_change_scale_norm_tol1": 0.0 if last_change_idx < 0 else float((last_change_idx + 1) / max(n - 1, 1)),
	}


def _compute_gws_shape_features_from_curves(
		scale_arr: np.ndarray,
		width_soft_arr: np.ndarray,
		area_arr: np.ndarray,
		area_above_thr_arr: np.ndarray,
) -> Dict[str, float]:
	"""Compute GWS curve-shape features from already-shared diagnostic traces."""
	out: Dict[str, float] = {
		"gws_width_auc_norm": 0.0,
		"gws_width_scale_at_50": 1.0,
		"gws_width_scale_at_80": 1.0,
		"gws_width_stepiness": 0.0,
		"gws_width_max_d1_norm": 0.0,
		"gws_area_auc_norm": 0.0,
		"gws_area_scale_at_50": 1.0,
		"gws_area_scale_at_80": 1.0,
		"gws_area_stepiness": 0.0,
		"gws_area_max_d1_norm": 0.0,
		"gws_area_above_thr_auc_norm": 0.0,
		"gws_area_above_thr_scale_at_50": 1.0,
		"gws_area_above_thr_stepiness": 0.0,
	}
	scale_use = np.asarray(scale_arr, dtype=float)
	if scale_use.size == 0:
		return out
	x_norm = _scale_axis_norm(scale_use)
	width_metrics = _curve_shape_metrics(np.asarray(width_soft_arr, dtype=float), x_norm, include_scale80=True, include_max_d1=True)
	area_metrics = _curve_shape_metrics(np.asarray(area_arr, dtype=float), x_norm, include_scale80=True, include_max_d1=True)
	area_thr_metrics = _curve_shape_metrics(np.asarray(area_above_thr_arr, dtype=float), x_norm, include_scale80=False, include_max_d1=False)
	out.update({
		"gws_width_auc_norm": float(width_metrics["auc_norm"]),
		"gws_width_scale_at_50": float(width_metrics["scale_at_50"]),
		"gws_width_scale_at_80": float(width_metrics["scale_at_80"]),
		"gws_width_stepiness": float(width_metrics["stepiness"]),
		"gws_width_max_d1_norm": float(width_metrics["max_d1_norm"]),
		"gws_area_auc_norm": float(area_metrics["auc_norm"]),
		"gws_area_scale_at_50": float(area_metrics["scale_at_50"]),
		"gws_area_scale_at_80": float(area_metrics["scale_at_80"]),
		"gws_area_stepiness": float(area_metrics["stepiness"]),
		"gws_area_max_d1_norm": float(area_metrics["max_d1_norm"]),
		"gws_area_above_thr_auc_norm": float(area_thr_metrics["auc_norm"]),
		"gws_area_above_thr_scale_at_50": float(area_thr_metrics["scale_at_50"]),
		"gws_area_above_thr_stepiness": float(area_thr_metrics["stepiness"]),
	})
	return out


def compute_gws_context_infos(
		gradient_signal: np.ndarray,
		*,
		apex_indices: Sequence[int],
		detection_lefts: Sequence[int],
		detection_rights: Sequence[int],
		context_mode: str = "segment_width",
		context_pad_factor: float = 1.0,
		context_fixed_pad: Optional[int] = None,
		context_min_pad: Optional[int] = None,
		context_max_pad: Optional[int] = None,
		split_overlapping_contexts: bool = GWS_SPLIT_OVERLAPPING_CONTEXTS,
		split_source: str = GWS_SPLIT_SOURCE,
		split_smooth_pts: int = GWS_SPLIT_SMOOTH_PTS,
		split_valley_alpha: float = GWS_SPLIT_VALLEY_ALPHA,
		split_min_distance_from_apex: int = GWS_SPLIT_MIN_DISTANCE_FROM_APEX,
		split_min_context_width: int = GWS_SPLIT_MIN_CONTEXT_WIDTH,
		split_debug: bool = GWS_SPLIT_DEBUG,
) -> List[Dict[str, object]]:
	"""Build shared per-candidate GWS context windows with optional overlap splitting."""
	grad = np.asarray(gradient_signal, dtype=float)
	n = int(grad.size)
	m = min(len(apex_indices), len(detection_lefts), len(detection_rights))
	if n <= 0 or m <= 0:
		return []

	source_name = str(split_source).strip().lower()
	if source_name != "gradient":
		source_name = "gradient"
	source_signal = grad
	source_smoothed = _smooth_gws_split_source(source_signal, int(split_smooth_pts))
	infos: List[Dict[str, object]] = []
	for idx in range(m):
		p = int(np.clip(apex_indices[idx], 0, n - 1))
		a = int(np.clip(detection_lefts[idx], 0, n - 1))
		b = int(np.clip(detection_rights[idx], 0, n - 1))
		if b < a:
			a, b = b, a
		ctx_l, ctx_r, pad = _compute_gws_context_bounds(
			n,
			a,
			b,
			context_mode=context_mode,
			context_pad_factor=context_pad_factor,
			context_fixed_pad=context_fixed_pad,
			context_min_pad=context_min_pad,
			context_max_pad=context_max_pad,
		)
		infos.append(
			{
				"candidate_index": int(idx),
				"apex_index": int(p),
				"detection_left": int(a),
				"detection_right": int(b),
				"original_context_left": int(ctx_l),
				"original_context_right": int(ctx_r),
				"context_left": int(ctx_l),
				"context_right": int(ctx_r),
				"context_pad": int(pad),
				"context_mode": str(context_mode).strip().lower(),
				"context_pad_factor": float(context_pad_factor),
				"context_fixed_pad": context_fixed_pad,
				"gws_context_overlap_count": 0.0,
				"gws_context_overlap_points": 0.0,
				"gws_context_overlap_fraction": 0.0,
				"gws_context_split_applied": 0.0,
				"gws_context_split_left_applied": 0.0,
				"gws_context_split_right_applied": 0.0,
				"gws_context_valley_depth_ratio_min": np.nan,
				"gws_context_valley_depth_ratio_left": np.nan,
				"gws_context_valley_depth_ratio_right": np.nan,
				"gws_context_original_width": float(max(0, ctx_r - ctx_l + 1)),
				"gws_context_adjusted_width": float(max(0, ctx_r - ctx_l + 1)),
				"gws_context_width_reduction_fraction": 0.0,
				"gws_split_left_index": None,
				"gws_split_right_index": None,
			}
		)

	if m == 1:
		return infos

	order = sorted(range(m), key=lambda i: (int(infos[i]["apex_index"]), int(i)))
	proposed_split: List[Optional[int]] = [None] * (m - 1)
	proposed_ratio: List[float] = [np.nan] * (m - 1)

	for pos in range(len(order) - 1):
		i = order[pos]
		j = order[pos + 1]
		info_i = infos[i]
		info_j = infos[j]
		orig_li = int(info_i["original_context_left"])
		orig_ri = int(info_i["original_context_right"])
		orig_lj = int(info_j["original_context_left"])
		orig_rj = int(info_j["original_context_right"])
		overlap_points = max(0, min(orig_ri, orig_rj) - max(orig_li, orig_lj) + 1)
		if overlap_points > 0:
			info_i["gws_context_overlap_count"] = float(info_i["gws_context_overlap_count"]) + 1.0
			info_j["gws_context_overlap_count"] = float(info_j["gws_context_overlap_count"]) + 1.0
			info_i["gws_context_overlap_points"] = float(info_i["gws_context_overlap_points"]) + float(overlap_points)
			info_j["gws_context_overlap_points"] = float(info_j["gws_context_overlap_points"]) + float(overlap_points)
		if not bool(split_overlapping_contexts) or overlap_points <= 0:
			continue
		apex_i = int(info_i["apex_index"])
		apex_j = int(info_j["apex_index"])
		lo = max(0, min(apex_i, apex_j) + int(split_min_distance_from_apex))
		hi = min(n - 1, max(apex_i, apex_j) - int(split_min_distance_from_apex))
		if hi < lo:
			if bool(split_debug):
				info_i["gws_context_valley_depth_ratio_right"] = np.nan
				info_j["gws_context_valley_depth_ratio_left"] = np.nan
			continue
		interval = np.asarray(source_smoothed[lo:hi + 1], dtype=float)
		if interval.size == 0 or not np.any(np.isfinite(interval)):
			continue
		local_min_rel = int(np.nanargmin(interval))
		split_idx = int(lo + local_min_rel)
		peak_win_i = source_signal[max(0, apex_i - 2):min(n, apex_i + 3)]
		peak_win_j = source_signal[max(0, apex_j - 2):min(n, apex_j + 3)]
		local_peak_i = float(np.nanmax(peak_win_i)) if peak_win_i.size and np.any(np.isfinite(peak_win_i)) else np.nan
		local_peak_j = float(np.nanmax(peak_win_j)) if peak_win_j.size and np.any(np.isfinite(peak_win_j)) else np.nan
		valley_val = float(source_smoothed[split_idx]) if np.isfinite(source_smoothed[split_idx]) else np.nan
		den = max(1e-12, min(local_peak_i, local_peak_j)) if np.isfinite(local_peak_i) and np.isfinite(local_peak_j) else np.nan
		ratio = float(valley_val / den) if np.isfinite(valley_val) and np.isfinite(den) else np.nan
		info_i["gws_context_valley_depth_ratio_right"] = ratio
		info_j["gws_context_valley_depth_ratio_left"] = ratio
		new_right_i = min(orig_ri, split_idx)
		new_left_j = max(orig_lj, split_idx + 1)
		width_i = int(new_right_i - orig_li + 1)
		width_j = int(orig_rj - new_left_j + 1)
		if np.isfinite(ratio) and ratio <= float(split_valley_alpha) and width_i >= int(split_min_context_width) and width_j >= int(split_min_context_width):
			proposed_split[pos] = int(split_idx)
			proposed_ratio[pos] = float(ratio)

	adjusted_left = [int(info["original_context_left"]) for info in infos]
	adjusted_right = [int(info["original_context_right"]) for info in infos]
	accepted_ratio_by_candidate: Dict[int, List[float]] = {i: [] for i in range(m)}
	for pos in range(len(order) - 1):
		split_idx = proposed_split[pos]
		if split_idx is None:
			continue
		i = order[pos]
		j = order[pos + 1]
		new_right_i = min(adjusted_right[i], int(split_idx))
		new_left_j = max(adjusted_left[j], int(split_idx) + 1)
		if (new_right_i - adjusted_left[i] + 1) < int(split_min_context_width):
			continue
		if (adjusted_right[j] - new_left_j + 1) < int(split_min_context_width):
			continue
		adjusted_right[i] = int(new_right_i)
		adjusted_left[j] = int(new_left_j)
		infos[i]["gws_context_split_applied"] = 1.0
		infos[j]["gws_context_split_applied"] = 1.0
		infos[i]["gws_context_split_right_applied"] = 1.0
		infos[j]["gws_context_split_left_applied"] = 1.0
		infos[i]["gws_split_right_index"] = int(split_idx)
		infos[j]["gws_split_left_index"] = int(split_idx)
		if np.isfinite(proposed_ratio[pos]):
			accepted_ratio_by_candidate[i].append(float(proposed_ratio[pos]))
			accepted_ratio_by_candidate[j].append(float(proposed_ratio[pos]))

	for idx, info in enumerate(infos):
		info["context_left"] = int(adjusted_left[idx])
		info["context_right"] = int(adjusted_right[idx])
		orig_width = float(info["gws_context_original_width"])
		adj_width = float(max(0, adjusted_right[idx] - adjusted_left[idx] + 1))
		info["gws_context_adjusted_width"] = float(adj_width)
		info["gws_context_width_reduction_fraction"] = 0.0 if orig_width <= 0.0 else float(1.0 - (adj_width / orig_width))
		if accepted_ratio_by_candidate[idx]:
			info["gws_context_valley_depth_ratio_min"] = float(min(accepted_ratio_by_candidate[idx]))
		overlap_points = float(info["gws_context_overlap_points"])
		info["gws_context_overlap_fraction"] = 0.0 if orig_width <= 0.0 else float(overlap_points / orig_width)
	return infos


def compute_gws_feature_bundle_from_diagnostics(diag: Dict[str, object]) -> Dict[str, float]:
	"""Export scalar GWS features from one shared diagnostic payload."""
	scale_arr = np.asarray(diag.get("scales", []), dtype=float)
	width_soft_arr = np.asarray(diag.get("width_soft", []), dtype=float)
	support_counts_arr = np.asarray(diag.get("support_counts", []), dtype=float)
	comp_support_counts_arr = np.asarray(diag.get("comp_support_counts", []), dtype=float)
	area_arr = np.asarray(diag.get("area_trace", []), dtype=float)
	area_above_thr_arr = np.asarray(diag.get("area_above_thr_trace", []), dtype=float)
	comp_area_arr = np.asarray(diag.get("comp_area_trace", []), dtype=float)
	comp_width_arr = np.asarray(diag.get("comp_width_trace", []), dtype=float)
	out: Dict[str, float] = {
		"gmt_width_soft_d1_mean": float(diag.get("width_soft_d1_mean", 0.0)),
		"gmt_width_soft_slope": float(diag.get("width_soft_slope", 0.0)),
		"gws_support01": float(diag.get("gws_support01", 0.0)),
		"gws_evidence_signed": float(diag.get("gws_evidence_signed", -1.0)),
	}
	out.update(_compute_gws_shape_features_from_curves(scale_arr, width_soft_arr, area_arr, area_above_thr_arr))
	out.update(_gws_width_trace_stability_metrics(width_soft_arr))
	out.update(_gws_support_count_metrics(support_counts_arr))
	out.update(_gws_support_fraction_metrics(out, float(diag.get("gws_context_adjusted_width", np.nan))))
	comp_metrics = _gws_support_count_metrics(comp_support_counts_arr)
	comp_metrics = {key.replace("gws_support_", "gws_comp_support_", 1): float(value) for key, value in comp_metrics.items()}
	out.update(comp_metrics)
	comp_fraction_den = float(
		diag.get(
			"measure_width",
			diag.get("gws_context_adjusted_width", np.nan),
		)
	) if str(diag.get("gws_measure_region", "mask")).strip().lower() == "spike_edges" else float(diag.get("gws_context_adjusted_width", np.nan))
	comp_fraction_metrics = _gws_support_fraction_metrics(
		{key.replace("gws_comp_support_", "gws_support_", 1): value for key, value in comp_metrics.items()},
		comp_fraction_den,
	)
	for src, dst in (
		("gws_support_initial_fraction", "gws_comp_support_initial_fraction"),
		("gws_support_final_fraction", "gws_comp_support_final_fraction"),
		("gws_support_max_fraction", "gws_comp_support_max_fraction"),
		("gws_support_final_minus_initial_fraction", "gws_comp_support_final_minus_initial_fraction"),
		("gws_support_total_increase_fraction", "gws_comp_support_total_increase_fraction"),
		("gws_support_total_abs_change_fraction", "gws_comp_support_total_abs_change_fraction"),
	):
		out[dst] = float(comp_fraction_metrics.get(src, np.nan))
	out.update(_gws_simple_trace_metrics(comp_area_arr, "gws_comp_area"))
	out.update(_gws_simple_trace_metrics(comp_width_arr, "gws_comp_width"))
	for key in (
		"gws_comp_apex_inside_support",
		"gws_comp_nearest_component_used",
		"gws_comp_distance_to_selected_component",
		"gws_comp_n_components",
		"gws_comp_selected_left",
		"gws_comp_selected_right",
	):
		out[key] = float(diag.get(key, np.nan))
	for key in (
		"gws_context_overlap_count",
		"gws_context_overlap_points",
		"gws_context_overlap_fraction",
		"gws_context_split_applied",
		"gws_context_split_left_applied",
		"gws_context_split_right_applied",
		"gws_context_valley_depth_ratio_min",
		"gws_context_valley_depth_ratio_left",
		"gws_context_valley_depth_ratio_right",
		"gws_context_adjusted_width",
		"gws_context_original_width",
		"gws_context_width_reduction_fraction",
	):
		out[key] = float(diag.get(key, np.nan))
	out["gws_width_auc_norm_evidence_signed"] = _signed_evidence_high_is_muon(out["gws_width_auc_norm"])
	out["gws_width_scale_at_50_evidence_signed"] = _signed_evidence_low_is_muon(out["gws_width_scale_at_50"])
	out["gws_width_scale_at_80_evidence_signed"] = _signed_evidence_low_is_muon(out["gws_width_scale_at_80"])
	out["gws_width_stepiness_evidence_signed"] = _signed_evidence_high_is_muon(out["gws_width_stepiness"])
	out["gws_width_max_d1_norm_evidence_signed"] = _signed_evidence_high_is_muon(out["gws_width_max_d1_norm"])
	out["gws_area_auc_norm_evidence_signed"] = _signed_evidence_high_is_muon(out["gws_area_auc_norm"])
	out["gws_area_scale_at_50_evidence_signed"] = _signed_evidence_low_is_muon(out["gws_area_scale_at_50"])
	out["gws_area_scale_at_80_evidence_signed"] = _signed_evidence_low_is_muon(out["gws_area_scale_at_80"])
	out["gws_area_stepiness_evidence_signed"] = _signed_evidence_high_is_muon(out["gws_area_stepiness"])
	out["gws_area_max_d1_norm_evidence_signed"] = _signed_evidence_high_is_muon(out["gws_area_max_d1_norm"])
	out["gws_area_above_thr_auc_norm_evidence_signed"] = _signed_evidence_high_is_muon(out["gws_area_above_thr_auc_norm"])
	out["gws_area_above_thr_scale_at_50_evidence_signed"] = _signed_evidence_low_is_muon(out["gws_area_above_thr_scale_at_50"])
	out["gws_area_above_thr_stepiness_evidence_signed"] = _signed_evidence_high_is_muon(out["gws_area_above_thr_stepiness"])
	return {key: float(value) for key, value in out.items()}


def compute_gws_diagnostics(
		gradient_signal: np.ndarray,
		*,
		raw_signal: Optional[np.ndarray] = None,
		apex_idx: int,
		detection_left: int,
		detection_right: int,
		bg_mad: Optional[float] = None,
		scales: Sequence[int] = GWS_GRANULO_SCALES,
		context_mode: str = "segment_width",
		context_pad_factor: float = 1.0,
		context_fixed_pad: Optional[int] = None,
		context_min_pad: Optional[int] = None,
		context_max_pad: Optional[int] = None,
		context_left_override: Optional[int] = None,
		context_right_override: Optional[int] = None,
		context_info: Optional[Dict[str, object]] = None,
		support_fraction: float = 0.15,
		bg_fraction: float = 0.5,
		source_mode: str = GWS_SOURCE,
		include_scale_zero: bool = False,
		measure_region: str = GWS_MEASURE_REGION,
		threshold_region: str = GWS_THRESHOLD_REGION,
) -> Dict[str, object]:
	"""Compute viewer-style GWS diagnostics on a shared implementation."""
	grad = np.asarray(gradient_signal, dtype=float)
	n = int(grad.size)
	raw_ctx: Optional[np.ndarray] = None
	if n <= 0:
		return {
			"context_left": 0,
			"context_right": 0,
			"apex_index": 0,
			"gradient_context": np.array([], dtype=float),
			"raw_context": None,
			"bg_mad": float(max(float(bg_mad) if bg_mad is not None else 0.0, 1e-12)),
			"rows": [],
			"width_soft": np.array([], dtype=float),
			"width_soft_d1": np.array([], dtype=float),
			"width_soft_d1_mean": 0.0,
			"width_soft_slope": 0.0,
			"gws_support01": 0.0,
			"gws_evidence_signed": -1.0,
			"source_mode": _normalize_gws_source_mode(source_mode),
			"gws_measure_region": _normalize_gws_measure_region(measure_region),
			"gws_threshold_region": _normalize_gws_threshold_region(threshold_region),
			"gws_threshold_region_effective": _normalize_gws_threshold_region(threshold_region),
			"smoothing_type": "none",
			"smoothing_window": 1,
			"context_mode": str(context_mode),
			"context_pad_factor": float(context_pad_factor),
			"context_fixed_pad": context_fixed_pad,
			"scales": tuple(int(v) for v in scales),
			"include_scale_zero": bool(include_scale_zero),
			"measure_left": 0,
			"measure_right": 0,
			"measure_width": 0,
			"gws_threshold_left": 0,
			"gws_threshold_right": 0,
			"gws_threshold_width": 0,
			"gws_context_r_max": np.asarray([], dtype=float),
			"gws_threshold_region_r_max": np.asarray([], dtype=float),
			"gws_support_threshold": np.asarray([], dtype=float),
			"gws_support_rel_threshold": float(support_fraction),
			"detection_left": 0,
			"detection_right": 0,
		}

	a = int(np.clip(detection_left, 0, n - 1))
	b = int(np.clip(detection_right, 0, n - 1))
	p = int(np.clip(apex_idx, 0, n - 1))
	if b < a:
		a, b = b, a
	mode = str(context_mode).strip().lower()
	ctx_l_base, ctx_r_base, pad = _compute_gws_context_bounds(
		n,
		a,
		b,
		context_mode=mode,
		context_pad_factor=context_pad_factor,
		context_fixed_pad=context_fixed_pad,
		context_min_pad=context_min_pad,
		context_max_pad=context_max_pad,
	)
	if context_info is not None:
		ctx_l = int(np.clip(int(context_info.get("context_left", ctx_l_base)), 0, n - 1))
		ctx_r = int(np.clip(int(context_info.get("context_right", ctx_r_base)), 0, n - 1))
	else:
		ctx_l = int(np.clip(ctx_l_base if context_left_override is None else int(context_left_override), 0, n - 1))
		ctx_r = int(np.clip(ctx_r_base if context_right_override is None else int(context_right_override), 0, n - 1))
	if ctx_r < ctx_l:
		ctx_l, ctx_r = ctx_r, ctx_l
	grad_ctx = grad[ctx_l:ctx_r + 1]
	if raw_signal is not None:
		raw_arr = np.asarray(raw_signal, dtype=float)
		if raw_arr.size == n:
			raw_ctx = raw_arr[ctx_l:ctx_r + 1]
	source_mode_norm = _normalize_gws_source_mode(source_mode)
	source_ctx = _compute_gws_source_context(grad_ctx, raw_ctx, source_mode=source_mode_norm)
	measure_region_norm = _normalize_gws_measure_region(measure_region)
	measure_left = int(np.clip(a - ctx_l, 0, max(ctx_r - ctx_l, 0)))
	measure_right = int(np.clip(b - ctx_l, 0, max(ctx_r - ctx_l, 0)))
	threshold_region_norm = _normalize_gws_threshold_region(threshold_region)
	threshold_region_effective = threshold_region_norm
	if threshold_region_norm == "context":
		threshold_left = 0
		threshold_right = int(max(ctx_r - ctx_l, 0))
	elif threshold_region_norm == "spike_edges":
		threshold_left = int(measure_left)
		threshold_right = int(measure_right)
	else:
		if measure_region_norm == "spike_edges":
			threshold_left = int(measure_left)
			threshold_right = int(measure_right)
			threshold_region_effective = "spike_edges"
		else:
			threshold_left = int(measure_left)
			threshold_right = int(measure_right)
			threshold_region_effective = "spike_edges"

	bg_source_for_mad = source_ctx if source_ctx.size == grad_ctx.size and source_ctx.size > 0 else grad_ctx
	bg_mad_val = float(bg_mad) if bg_mad is not None else estimate_background_mad(bg_source_for_mad, a - ctx_l, b - ctx_l)
	bg_mad_val = max(bg_mad_val, 1e-12)
	curves = _compute_multiscale_tophat_residual_curves(
		source_ctx,
		bg_mad_val,
		scales=scales,
		support_fraction=float(support_fraction),
		bg_fraction=float(bg_fraction),
		hard_bg_fraction=1.0,
		include_scale_zero=bool(include_scale_zero),
		threshold_left_idx=int(threshold_left),
		threshold_right_idx=int(threshold_right),
	)
	if curves is None:
		scale_rows: List[Dict[str, object]] = []
		scale_arr = np.array([], dtype=float)
		area_arr = np.array([], dtype=float)
		area_above_thr_arr = np.array([], dtype=float)
		width_soft_arr = np.array([], dtype=float)
		width_soft_d1 = np.array([], dtype=float)
		width_soft_d1_mean = 0.0
		width_soft_slope = 0.0
	else:
		scale_rows = [dict(row) for row in list(curves["rows"])]
		scale_arr = np.asarray(curves["scales"], dtype=float)
		area_arr = np.asarray(curves["areas"], dtype=float)
		area_above_thr_arr = np.asarray(curves["areas_above_thr"], dtype=float)
		width_soft_arr = np.asarray(curves["widths_soft"], dtype=float)
		if scale_arr.size >= 2:
			width_soft_d1 = np.diff(width_soft_arr) / np.diff(scale_arr)
			width_soft_d1_mean = float(np.mean(width_soft_d1)) if width_soft_d1.size else 0.0
			width_soft_slope = float((width_soft_arr[-1] - width_soft_arr[0]) / max(scale_arr[-1] - scale_arr[0], 1e-12))
		else:
			width_soft_d1 = np.array([], dtype=float)
			width_soft_d1_mean = 0.0
			width_soft_slope = 0.0
			area_arr = np.asarray([], dtype=float)
			area_above_thr_arr = np.asarray([], dtype=float)
	support_trace = compute_gws_support_trace(scale_rows)
	support_counts_arr = np.asarray(support_trace["support_counts"], dtype=int)
	support_diffs_arr = np.asarray(support_trace["diffs"], dtype=int)
	apex_rel_ctx = int(np.clip(p - ctx_l, 0, max(ctx_r - ctx_l, 0)))
	component_trace = compute_gws_measure_trace(
		scale_rows,
		apex_rel_idx=apex_rel_ctx,
		measure_left_idx=measure_left,
		measure_right_idx=measure_right,
		measure_region=measure_region_norm,
	)
	gws_support01 = float(
		interval_membership(
			width_soft_d1_mean,
			GWS_LEFT_OUTER,
			GWS_LEFT_INNER,
			GWS_RIGHT_INNER,
			GWS_RIGHT_OUTER,
		)
	)
	gws_evidence_signed = float(np.clip(2.0 * gws_support01 - 1.0, -1.0, 1.0))
	context_meta = dict(context_info) if context_info is not None else {}
	return {
		"context_left": int(ctx_l),
		"context_right": int(ctx_r),
		"apex_index": int(p),
		"gradient_context": grad_ctx,
		"raw_context": raw_ctx,
		"source_context": source_ctx,
		"source_mode": source_mode_norm,
		"gws_measure_region": measure_region_norm,
		"gws_threshold_region": threshold_region_norm,
		"gws_threshold_region_effective": threshold_region_effective,
		"smoothing_type": (
			"median" if "med" in source_mode_norm else ("mean" if "mean" in source_mode_norm else "none")
		),
		"smoothing_window": (
			3 if source_mode_norm.endswith("3") else (5 if source_mode_norm.endswith("5") else 1)
		),
		"bg_mad": float(bg_mad_val),
		"rows": scale_rows,
		"scales": tuple(int(v) for v in scale_arr),
		"include_scale_zero": bool(include_scale_zero),
		"detection_left": int(a),
		"detection_right": int(b),
		"measure_left": int(ctx_l + measure_left),
		"measure_right": int(ctx_l + measure_right),
		"measure_width": int(max(0, measure_right - measure_left + 1)),
		"gws_threshold_left": int(ctx_l + threshold_left),
		"gws_threshold_right": int(ctx_l + threshold_right),
		"gws_threshold_width": int(max(0, threshold_right - threshold_left + 1)),
		"area_trace": area_arr,
		"area_above_thr_trace": area_above_thr_arr,
		"width_soft": width_soft_arr,
		"support_counts": support_counts_arr,
		"support_count_diffs": support_diffs_arr,
		"support_masks": list(support_trace["support_masks"]),
		"support_indices": list(support_trace["support_indices"]),
		"gws_context_r_max": np.asarray([float(row.get("r_max", np.nan)) for row in scale_rows], dtype=float),
		"gws_threshold_region_r_max": np.asarray([float(row.get("r_max_threshold_region", np.nan)) for row in scale_rows], dtype=float),
		"gws_support_threshold": np.asarray([float(row.get("r_thr_soft", np.nan)) for row in scale_rows], dtype=float),
		"gws_support_rel_threshold": float(support_fraction),
		"comp_support_counts": np.asarray(component_trace["support_counts"], dtype=int),
		"comp_support_count_diffs": np.asarray(component_trace["diffs"], dtype=int),
		"comp_area_trace": np.asarray(component_trace["areas"], dtype=float),
		"comp_width_trace": np.asarray(component_trace["widths"], dtype=float),
		"comp_selected_masks": list(component_trace["selected_masks"]),
		"gws_comp_apex_inside_support": float(
			0.0 if np.asarray(component_trace["apex_inside_flags"], dtype=float).size == 0
			else np.min(np.asarray(component_trace["apex_inside_flags"], dtype=float))
		),
		"gws_comp_nearest_component_used": float(
			0.0 if np.asarray(component_trace["nearest_component_flags"], dtype=float).size == 0
			else np.max(np.asarray(component_trace["nearest_component_flags"], dtype=float))
		),
		"gws_comp_distance_to_selected_component": float(
			0.0
			if (
				np.asarray(component_trace["distance_to_selected_component"], dtype=float).size == 0
				or not np.any(np.isfinite(np.asarray(component_trace["distance_to_selected_component"], dtype=float)))
			)
			else np.nanmax(np.asarray(component_trace["distance_to_selected_component"], dtype=float))
		),
		"gws_comp_n_components": float(
			0.0
			if (
				np.asarray(component_trace["n_components"], dtype=float).size == 0
				or not np.any(np.isfinite(np.asarray(component_trace["n_components"], dtype=float)))
			)
			else np.nanmax(np.asarray(component_trace["n_components"], dtype=float))
		),
		"gws_comp_selected_left": float(
			np.nan if np.asarray(component_trace["selected_lefts"], dtype=float).size == 0
			else np.asarray(component_trace["selected_lefts"], dtype=float)[-1]
		),
		"gws_comp_selected_right": float(
			np.nan if np.asarray(component_trace["selected_rights"], dtype=float).size == 0
			else np.asarray(component_trace["selected_rights"], dtype=float)[-1]
		),
		"width_soft_d1": width_soft_d1,
		"width_soft_d1_mean": float(width_soft_d1_mean),
		"width_soft_slope": float(width_soft_slope),
		"gws_support01": float(gws_support01),
		"gws_evidence_signed": float(gws_evidence_signed),
		"context_mode": mode,
		"context_pad_factor": float(context_pad_factor),
		"context_fixed_pad": context_fixed_pad,
		"gws_context_overlap_count": float(context_meta.get("gws_context_overlap_count", 0.0)),
		"gws_context_overlap_points": float(context_meta.get("gws_context_overlap_points", 0.0)),
		"gws_context_overlap_fraction": float(context_meta.get("gws_context_overlap_fraction", 0.0)),
		"gws_context_split_applied": float(context_meta.get("gws_context_split_applied", 0.0)),
		"gws_context_split_left_applied": float(context_meta.get("gws_context_split_left_applied", 0.0)),
		"gws_context_split_right_applied": float(context_meta.get("gws_context_split_right_applied", 0.0)),
		"gws_context_valley_depth_ratio_min": float(context_meta.get("gws_context_valley_depth_ratio_min", np.nan)),
		"gws_context_valley_depth_ratio_left": float(context_meta.get("gws_context_valley_depth_ratio_left", np.nan)),
		"gws_context_valley_depth_ratio_right": float(context_meta.get("gws_context_valley_depth_ratio_right", np.nan)),
		"gws_context_adjusted_width": float(context_meta.get("gws_context_adjusted_width", max(0, ctx_r - ctx_l + 1))),
		"gws_context_original_width": float(context_meta.get("gws_context_original_width", max(0, ctx_r_base - ctx_l_base + 1))),
		"gws_context_width_reduction_fraction": float(context_meta.get("gws_context_width_reduction_fraction", 0.0)),
		"gws_split_left_index": context_meta.get("gws_split_left_index"),
		"gws_split_right_index": context_meta.get("gws_split_right_index"),
	}


def compute_gws_granulometric_shape_features(
	gws_signal: np.ndarray,
	bg_mad: float,
	scales: Sequence[int] = GWS_GRANULO_SCALES,
) -> Dict[str, float]:
	"""
	Compute GWS granulometric curve-shape descriptors from multiscale top-hat residuals.
	"""
	zero_out: Dict[str, float] = {
		"gws_width_auc_norm": 0.0,
		"gws_width_scale_at_50": 1.0,
		"gws_width_scale_at_80": 1.0,
		"gws_width_stepiness": 0.0,
		"gws_width_max_d1_norm": 0.0,
		"gws_area_auc_norm": 0.0,
		"gws_area_scale_at_50": 1.0,
		"gws_area_scale_at_80": 1.0,
		"gws_area_stepiness": 0.0,
		"gws_area_max_d1_norm": 0.0,
		"gws_area_above_thr_auc_norm": 0.0,
		"gws_area_above_thr_scale_at_50": 1.0,
		"gws_area_above_thr_stepiness": 0.0,
	}
	curves = _compute_multiscale_tophat_residual_curves(gws_signal, float(bg_mad), scales=scales)
	if curves is None:
		out = dict(zero_out)
	else:
		out = dict(zero_out)
		out.update(
			_compute_gws_shape_features_from_curves(
				np.asarray(curves["scales"], dtype=float),
				np.asarray(curves["widths_soft"], dtype=float),
				np.asarray(curves["areas"], dtype=float),
				np.asarray(curves["areas_above_thr"], dtype=float),
			)
		)

	out["gws_width_auc_norm_evidence_signed"] = _signed_evidence_high_is_muon(out["gws_width_auc_norm"])
	out["gws_width_scale_at_50_evidence_signed"] = _signed_evidence_low_is_muon(out["gws_width_scale_at_50"])
	out["gws_width_scale_at_80_evidence_signed"] = _signed_evidence_low_is_muon(out["gws_width_scale_at_80"])
	out["gws_width_stepiness_evidence_signed"] = _signed_evidence_high_is_muon(out["gws_width_stepiness"])
	out["gws_width_max_d1_norm_evidence_signed"] = _signed_evidence_high_is_muon(out["gws_width_max_d1_norm"])
	out["gws_area_auc_norm_evidence_signed"] = _signed_evidence_high_is_muon(out["gws_area_auc_norm"])
	out["gws_area_scale_at_50_evidence_signed"] = _signed_evidence_low_is_muon(out["gws_area_scale_at_50"])
	out["gws_area_scale_at_80_evidence_signed"] = _signed_evidence_low_is_muon(out["gws_area_scale_at_80"])
	out["gws_area_stepiness_evidence_signed"] = _signed_evidence_high_is_muon(out["gws_area_stepiness"])
	out["gws_area_max_d1_norm_evidence_signed"] = _signed_evidence_high_is_muon(out["gws_area_max_d1_norm"])
	out["gws_area_above_thr_auc_norm_evidence_signed"] = _signed_evidence_high_is_muon(out["gws_area_above_thr_auc_norm"])
	out["gws_area_above_thr_scale_at_50_evidence_signed"] = _signed_evidence_low_is_muon(out["gws_area_above_thr_scale_at_50"])
	out["gws_area_above_thr_stepiness_evidence_signed"] = _signed_evidence_high_is_muon(out["gws_area_above_thr_stepiness"])

	return {key: float(value) for key, value in out.items()}


def filter_multiscale_tophat_export_features(features: Dict[str, float], prefix: str = "mt") -> Dict[str, float]:
	"""Keep only the whitelisted multiscale top-hat export features for the requested prefix."""
	pfx = str(prefix).strip().lower()
	if pfx not in {"mt", "gmt"}:
		raise ValueError(f"Unsupported top-hat feature prefix: {prefix}")

	base_keep = TOPHAT_RAW_EXPORT_FEATURES if pfx == "mt" else TOPHAT_GRADIENT_EXPORT_FEATURES
	out: Dict[str, float] = {
		name: float(features.get(name, 0.0))
		for name in base_keep
	}
	for out_name, src_name in TOPHAT_LOG1P_EXPORT_MAP.items():
		if not out_name.startswith(f"{pfx}_"):
			continue
		out[out_name] = float(np.log1p(max(float(features.get(src_name, 0.0)), 0.0)))
	return out


def compute_residual_component_features(raw_wide: np.ndarray, gradient_seg: np.ndarray, bg_mad: float) -> Dict[str, float]:
	"""Measure fragmentation of residual and gradient supports:"""
	x = np.asarray(raw_wide, dtype=float)
	g = np.asarray(gradient_seg, dtype=float)
	if x.size < 5:
		return {
			'residual_component_count': 0.0,
			'residual_component_spacing': 0.0,
			'gradient_component_count': 0.0,
			'gradient_support_width': 0.0,
		}

	opn = ndimage.grey_opening(x, size=min(9, max(3, x.size // 2 * 2 - 1)))
	res = np.maximum(x - opn, 0.0)
	r_thr = max(float(np.median(res) + 2.0 * np.median(np.abs(res - np.median(res)))), 1.5 * float(bg_mad))
	r_mask = res >= r_thr
	r_comps = _components_from_mask(r_mask)

	gabs = np.abs(np.diff(g)) if g.size >= 3 else np.array([], dtype=float)
	if gabs.size:
		g_thr = float(np.median(gabs) + 2.0 * np.median(np.abs(gabs - np.median(gabs))))
		g_mask = gabs >= max(g_thr, float(bg_mad))
		g_comps = _components_from_mask(g_mask)
		g_width = float(np.count_nonzero(g_mask))
	else:
		g_comps = []
		g_width = 0.0

	return {
		'residual_component_count': float(len(r_comps)),
		'residual_component_spacing': _component_spacing(r_comps),
		'gradient_component_count': float(len(g_comps)),
		'gradient_support_width': g_width,
	}


def _curvature_second_difference(
		segment: np.ndarray,
		peak_rel: Optional[int] = None,
) -> Tuple[np.ndarray, int, int]:
	"""Return discrete second difference plus apex indices in signal and d2 coordinates."""
	seg = np.asarray(segment, dtype=float)
	d2 = seg[:-2] - 2.0 * seg[1:-1] + seg[2:]
	if peak_rel is None:
		peak_signal_idx = int(np.argmax(seg))
	else:
		peak_signal_idx = int(peak_rel)
	peak_signal_idx = int(np.clip(peak_signal_idx, 1, seg.size - 2))
	peak_d2_idx = peak_signal_idx - 1
	return d2, peak_signal_idx, peak_d2_idx


def _argmax_abs_idx(values: np.ndarray) -> Optional[int]:
	"""Return index of the largest absolute value, or None for empty input."""
	x = np.asarray(values, dtype=float)
	if x.size == 0:
		return None
	return int(np.argmax(np.abs(x)))


def _signed_extreme_idx(values: np.ndarray, sign: int, candidate_idx: Optional[np.ndarray] = None) -> Optional[int]:
	"""Return index of strongest positive/negative value in an optional index subset."""
	x = np.asarray(values, dtype=float)
	if x.size == 0:
		return None
	if candidate_idx is None:
		idx = np.arange(x.size, dtype=int)
	else:
		idx = np.asarray(candidate_idx, dtype=int)
		idx = idx[(idx >= 0) & (idx < x.size)]
	if idx.size == 0:
		return None
	if sign < 0:
		valid = idx[x[idx] < 0.0]
		if valid.size == 0:
			return None
		return int(valid[np.argmin(x[valid])])
	valid = idx[x[idx] > 0.0]
	if valid.size == 0:
		return None
	return int(valid[np.argmax(x[valid])])


def _local_curvature_indices(d2_size: int, peak_d2_idx: int, radius: int) -> np.ndarray:
	"""Return local curvature indices around the apex in d2 coordinates."""
	if d2_size <= 0:
		return np.array([], dtype=int)
	rad = max(1, int(radius))
	left = max(0, int(peak_d2_idx) - rad)
	right = min(int(d2_size) - 1, int(peak_d2_idx) + rad)
	return np.arange(left, right + 1, dtype=int)


def _apply_positive_to_negative_correction(
		d2: np.ndarray,
		base_idx: Optional[int],
		*,
		tolerance: float,
		negative_search_idx: Optional[np.ndarray] = None,
) -> Tuple[float, int]:
	"""Apply one-way positive->negative correction if a close-enough negative exists."""
	x = np.asarray(d2, dtype=float)
	if base_idx is None or x.size == 0:
		return 0.0, 0
	base_val = float(x[int(base_idx)])
	if base_val <= 0.0:
		return base_val, 0
	neg_idx = _signed_extreme_idx(x, sign=-1, candidate_idx=negative_search_idx)
	if neg_idx is None:
		return base_val, 0
	neg_val = float(x[int(neg_idx)])
	if abs(neg_val) >= float(tolerance) * max(abs(base_val), 1e-12):
		return neg_val, 1
	return base_val, 0


def compute_curvature_negpref_diagnostics(
		signal_segment: np.ndarray,
		*,
		peak_rel: Optional[int],
		tolerance: float,
		local: bool = False,
		local_radius: int = CURVATURE_NEGPREF_LOCAL_RADIUS,
) -> Dict[str, object]:
	"""Return shared curvature negpref diagnostics preserving the viewer behavior."""
	seg = np.asarray(signal_segment, dtype=float)
	if seg.size < 3:
		return {
			"d2": np.array([], dtype=float),
			"peak_signal_idx": 0,
			"apex_d2_idx": 0,
			"base_idx": 0,
			"base_value": 0.0,
			"negative_idx": None,
			"negative_value": 0.0,
			"chosen_idx": 0,
			"chosen_value": 0.0,
			"switched": 0,
			"tolerance": float(tolerance),
			"local": bool(local),
			"local_left_idx": 0,
			"local_right_idx": 0,
		}
	d2, peak_signal_idx, apex_d2_idx = _curvature_second_difference(seg, peak_rel=peak_rel)
	base_idx = _argmax_abs_idx(d2)
	base_idx = int(base_idx) if base_idx is not None else 0
	base_value = float(d2[base_idx]) if d2.size else 0.0
	local_idx = None
	local_left_idx = 0
	local_right_idx = max(0, int(d2.size) - 1)
	if bool(local) and d2.size:
		local_idx = _local_curvature_indices(d2.size, apex_d2_idx, int(local_radius))
		if local_idx.size:
			local_left_idx = int(local_idx[0])
			local_right_idx = int(local_idx[-1])
		else:
			local_left_idx = int(apex_d2_idx)
			local_right_idx = int(apex_d2_idx)
	negative_idx = _signed_extreme_idx(d2, sign=-1, candidate_idx=local_idx)
	negative_value = float(d2[int(negative_idx)]) if negative_idx is not None else 0.0
	chosen_value, switched = _apply_positive_to_negative_correction(
		d2,
		base_idx,
		tolerance=float(tolerance),
		negative_search_idx=local_idx,
	)
	chosen_idx = int(base_idx)
	if int(switched) and negative_idx is not None:
		chosen_idx = int(negative_idx)
	return {
		"d2": d2,
		"peak_signal_idx": int(peak_signal_idx),
		"apex_d2_idx": int(apex_d2_idx),
		"base_idx": int(base_idx),
		"base_value": float(base_value),
		"negative_idx": (int(negative_idx) if negative_idx is not None else None),
		"negative_value": float(negative_value),
		"chosen_idx": int(chosen_idx),
		"chosen_value": float(chosen_value),
		"switched": int(switched),
		"tolerance": float(tolerance),
		"local": bool(local),
		"local_left_idx": int(local_left_idx),
		"local_right_idx": int(local_right_idx),
	}


def compute_curvature_features(
		segment: np.ndarray,
		*,
		peak_rel: Optional[int] = None,
		negpref_local_radius: int = CURVATURE_NEGPREF_LOCAL_RADIUS,
) -> Dict[str, float]:
	"""Compute legacy and negative-preferred curvature descriptors from a 1D segment."""
	seg = np.asarray(segment, dtype=float)
	if seg.size < 3:
		out = {
			'peak_curvature_apex': 0.0,
			'peak_curvature_extreme': 0.0,
			'peak_curvature_apex_at_peak': 0.0,
			'peak_curvature_negative_extreme': 0.0,
			'peak_curvature_positive_extreme': 0.0,
			'peak_curvature_neg_to_abs_ratio': 0.0,
			'peak_curvature_extreme_sign': 0.0,
			'peak_curvature_local_negative_extreme': 0.0,
			'peak_curvature_local_positive_extreme': 0.0,
			'peak_curvature_local_neg_to_abs_ratio': 0.0,
			'peak_curvature_local_extreme_sign': 0.0,
			'peak_curvature_mean_abs': 0.0,
		}
		for tol in CURVATURE_NEGPREF_TOLERANCES:
			tag = _tolerance_tag(tol)
			out[f'peak_curvature_extreme_negpref_{tag}'] = 0.0
			out[f'peak_curvature_extreme_negpref_local_{tag}'] = 0.0
			out[f'peak_curvature_negpref_switched_{tag}'] = 0.0
			out[f'peak_curvature_negpref_local_switched_{tag}'] = 0.0
		return out

	d2, peak_signal_idx, apex_idx = _curvature_second_difference(seg, peak_rel=peak_rel)
	i_extreme = _argmax_abs_idx(d2)
	negative_idx = _signed_extreme_idx(d2, sign=-1)
	positive_idx = _signed_extreme_idx(d2, sign=1)
	local_idx = _local_curvature_indices(d2.size, apex_idx, int(negpref_local_radius))
	local_abs_idx = _argmax_abs_idx(d2[local_idx]) if local_idx.size else None
	local_abs_global_idx = int(local_idx[int(local_abs_idx)]) if local_abs_idx is not None and local_idx.size else None
	local_negative_idx = _signed_extreme_idx(d2, sign=-1, candidate_idx=local_idx)
	local_positive_idx = _signed_extreme_idx(d2, sign=1, candidate_idx=local_idx)

	extreme_val = float(d2[int(i_extreme)]) if i_extreme is not None else 0.0
	negative_val = float(d2[int(negative_idx)]) if negative_idx is not None else 0.0
	positive_val = float(d2[int(positive_idx)]) if positive_idx is not None else 0.0
	local_negative_val = float(d2[int(local_negative_idx)]) if local_negative_idx is not None else 0.0
	local_positive_val = float(d2[int(local_positive_idx)]) if local_positive_idx is not None else 0.0
	local_extreme_val = float(d2[int(local_abs_global_idx)]) if local_abs_global_idx is not None else 0.0
	extreme_abs = max(abs(extreme_val), 1e-12)

	out = {
		'peak_curvature_apex': float(np.min(d2)),
		'peak_curvature_extreme': extreme_val,
		'peak_curvature_apex_at_peak': float(d2[apex_idx]),
		'peak_curvature_negative_extreme': negative_val,
		'peak_curvature_positive_extreme': positive_val,
		'peak_curvature_neg_to_abs_ratio': float(abs(negative_val) / extreme_abs) if negative_idx is not None else 0.0,
		'peak_curvature_extreme_sign': float(np.sign(extreme_val)),
		'peak_curvature_local_negative_extreme': local_negative_val,
		'peak_curvature_local_positive_extreme': local_positive_val,
		'peak_curvature_local_neg_to_abs_ratio': float(abs(local_negative_val) / extreme_abs) if local_negative_idx is not None else 0.0,
		'peak_curvature_local_extreme_sign': float(np.sign(local_extreme_val)),
		'peak_curvature_mean_abs': float(np.mean(np.abs(d2))),
	}
	for tol in CURVATURE_NEGPREF_TOLERANCES:
		tag = _tolerance_tag(tol)
		negpref_val, switched_val = _apply_positive_to_negative_correction(
			d2, i_extreme, tolerance=float(tol), negative_search_idx=None
		)
		negpref_local_val, switched_local_val = _apply_positive_to_negative_correction(
			d2, i_extreme, tolerance=float(tol), negative_search_idx=local_idx
		)
		out[f'peak_curvature_extreme_negpref_{tag}'] = float(negpref_val)
		out[f'peak_curvature_extreme_negpref_local_{tag}'] = float(negpref_local_val)
		out[f'peak_curvature_negpref_switched_{tag}'] = float(switched_val)
		out[f'peak_curvature_negpref_local_switched_{tag}'] = float(switched_local_val)
	return out


def compute_third_derivative_features(segment: np.ndarray) -> Dict[str, float]:
	"""Compute extreme and mean-abs descriptors of the discrete third derivative."""
	seg = np.asarray(segment, dtype=float)
	if seg.size < 4:
		return {
			'peak_curvature_d3_extreme': 0.0,
			'peak_curvature_d3_mean_abs': 0.0,
		}

	d3 = np.diff(seg, n=3)
	i_extreme = int(np.argmax(np.abs(d3)))
	return {
		'peak_curvature_d3_extreme': float(d3[i_extreme]),
		'peak_curvature_d3_mean_abs': float(np.mean(np.abs(d3))),
	}


def compute_peak_curvature_features(
		signal_segment: np.ndarray,
		bg_mad: float,
		*,
		peak_rel: Optional[int] = None,
		negpref_local_radius: int = CURVATURE_NEGPREF_LOCAL_RADIUS,
) -> Dict[str, float]:
	"""Compute backward-compatible and explicit curvature features from one segment."""
	curv = compute_curvature_features(
		signal_segment,
		peak_rel=peak_rel,
		negpref_local_radius=negpref_local_radius,
	)
	bg = max(float(bg_mad), 1e-12)
	extreme = float(curv['peak_curvature_extreme'])
	apex = float(curv['peak_curvature_apex'])
	apex_at_peak = float(curv['peak_curvature_apex_at_peak'])
	negative_extreme = float(curv['peak_curvature_negative_extreme'])
	positive_extreme = float(curv['peak_curvature_positive_extreme'])
	mean_abs = float(curv['peak_curvature_mean_abs'])
	extreme_abs = float(abs(extreme))
	apex_abs = float(abs(apex))
	apex_at_peak_abs = float(abs(apex_at_peak))
	negative_extreme_abs = float(abs(negative_extreme))
	positive_extreme_abs = float(abs(positive_extreme))
	out = {
		# Backward-compatible legacy keys.
		'peak_curvature': extreme_abs,
		'peak_curvature_z': float(extreme_abs / bg),
		# New explicit variants.
		'peak_curvature_apex': apex,
		'peak_curvature_apex_z': float(apex_abs / bg),
		'peak_curvature_extreme': extreme,
		'peak_curvature_extreme_z': float(extreme_abs / bg),
		'peak_curvature_apex_at_peak': apex_at_peak,
		'peak_curvature_apex_at_peak_z': float(apex_at_peak_abs / bg),
		'peak_curvature_negative_extreme': negative_extreme,
		'peak_curvature_negative_extreme_z': float(negative_extreme_abs / bg),
		'peak_curvature_positive_extreme': positive_extreme,
		'peak_curvature_positive_extreme_z': float(positive_extreme_abs / bg),
		'peak_curvature_neg_to_abs_ratio': float(curv['peak_curvature_neg_to_abs_ratio']),
		'peak_curvature_extreme_sign': float(curv['peak_curvature_extreme_sign']),
		'peak_curvature_local_negative_extreme': float(curv['peak_curvature_local_negative_extreme']),
		'peak_curvature_local_negative_extreme_z': float(abs(float(curv['peak_curvature_local_negative_extreme'])) / bg),
		'peak_curvature_local_positive_extreme': float(curv['peak_curvature_local_positive_extreme']),
		'peak_curvature_local_positive_extreme_z': float(abs(float(curv['peak_curvature_local_positive_extreme'])) / bg),
		'peak_curvature_local_neg_to_abs_ratio': float(curv['peak_curvature_local_neg_to_abs_ratio']),
		'peak_curvature_local_extreme_sign': float(curv['peak_curvature_local_extreme_sign']),
		'peak_curvature_mean_abs': mean_abs,
		'peak_curvature_mean_abs_z': float(mean_abs / bg),
	}
	for tol in CURVATURE_NEGPREF_TOLERANCES:
		tag = _tolerance_tag(tol)
		raw_key = f'peak_curvature_extreme_negpref_{tag}'
		local_key = f'peak_curvature_extreme_negpref_local_{tag}'
		switched_key = f'peak_curvature_negpref_switched_{tag}'
		local_switched_key = f'peak_curvature_negpref_local_switched_{tag}'
		raw_diag = compute_curvature_negpref_diagnostics(
			signal_segment,
			peak_rel=peak_rel,
			tolerance=float(tol),
			local=False,
			local_radius=int(negpref_local_radius),
		)
		local_diag = compute_curvature_negpref_diagnostics(
			signal_segment,
			peak_rel=peak_rel,
			tolerance=float(tol),
			local=True,
			local_radius=int(negpref_local_radius),
		)
		raw_val = float(raw_diag["chosen_value"])
		local_val = float(local_diag["chosen_value"])
		out[raw_key] = raw_val
		out[f'{raw_key}_z'] = float(abs(raw_val) / bg)
		out[local_key] = local_val
		out[f'{local_key}_z'] = float(abs(local_val) / bg)
		out[switched_key] = float(raw_diag["switched"])
		out[local_switched_key] = float(local_diag["switched"])
	return out


def compute_peak_curvature_d3_features(signal_segment: np.ndarray, bg_mad: float) -> Dict[str, float]:
	"""Compute extreme and mean-abs descriptors of the discrete third derivative."""
	curv3 = compute_third_derivative_features(signal_segment)
	bg = max(float(bg_mad), 1e-12)
	extreme = float(curv3['peak_curvature_d3_extreme'])
	mean_abs = float(curv3['peak_curvature_d3_mean_abs'])
	return {
		'peak_curvature_d3_extreme': extreme,
		'peak_curvature_d3_extreme_z': float(abs(extreme) / bg),
		'peak_curvature_d3_mean_abs': mean_abs,
		'peak_curvature_d3_mean_abs_z': float(mean_abs / bg),
	}


def compute_optional_foot_shape_features(raw_segment: np.ndarray, peak_rel: int) -> Dict[str, float]:
	"""Optional low-cost foot-shape descriptors based on level width."""
	x = np.asarray(raw_segment, dtype=float)
	if x.size < 5:
		return {
			'foot_width_ratio_left_30_70': 0.0,
			'foot_width_ratio_right_30_70': 0.0,
		}
	peak_rel = int(np.clip(peak_rel, 0, x.size - 1))
	base = float(min(np.min(x[: peak_rel + 1]), np.min(x[peak_rel:])))
	amp = float(max(x[peak_rel] - base, 1e-12))

	def _dist_to_level(level_frac: float, side: str) -> float:
		thr = base + level_frac * amp
		if side == "left":
			idx = np.where(x[: peak_rel + 1] <= thr)[0]
			if idx.size == 0:
				return float(peak_rel)
			return float(peak_rel - idx[-1])
		idx = np.where(x[: peak_rel + 1] <= thr)[0]
		if idx.size == 0:
			return float(x.size - 1 - peak_rel)
		return float(idx[0])

	l30 = _dist_to_level(0.30, "left")
	l70 = _dist_to_level(0.70, "left")
	r30 = _dist_to_level(0.30, "right")
	r70 = _dist_to_level(0.70, "right")

	return {
		'foot_width_ratio_left_30_70': float(l30 / max(l70, 1e-12)),
		'foot_width_ratio_right_30_70': float(r30 / max(r70, 1e-12))
	}
