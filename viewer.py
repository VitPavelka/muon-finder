# viewer.py
from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Literal

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import ndimage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.widgets import TextBox, Button, CheckButtons

from muon_pipeline import SpikeSegment
from spike_merge import merge_spike_segments_by_signal_foot
from feature_window import (
	expand_interval_to_signal_foot,
	enforce_shared_boundaries_by_minima
)
from feature_discrimination import (
	CURVATURE_NEGPREF_LOCAL_RADIUS,
	CURVATURE_NEGPREF_TOLERANCES,
	GWS_GRANULO_SCALES,
	GWS_MEASURE_REGION,
	GWS_THRESHOLD_REGION,
	GWS_SPLIT_DEBUG,
	GWS_SPLIT_MIN_CONTEXT_WIDTH,
	GWS_SPLIT_MIN_DISTANCE_FROM_APEX,
	GWS_SPLIT_OVERLAPPING_CONTEXTS,
	GWS_SPLIT_SMOOTH_PTS,
	GWS_SPLIT_SOURCE,
	GWS_SPLIT_VALLEY_ALPHA,
	MDWS510_SUPPORT_FULL,
	MDWS510_SUPPORT_ZERO,
	MDWS510_VETO_FULL,
	MDWS510_VETO_ZERO,
	compute_curvature_negpref_diagnostics,
	compute_edge_width_metrics,
	compute_edge_dense_width_metrics,
	compute_raw_upper_component_dense_width_metrics,
	compute_ball_descent_metrics,
	compute_exponential_decay_metrics,
	compute_gws_context_infos,
	compute_gws_diagnostics,
	compute_peak_curvature_features,
	compute_multiscale_tophat_features,
	compute_shape_top_hat_signal,
	compute_curvature_support_features,
	compute_local_shape_features,
	compute_median_residual_features,
	compute_opening_residual_features,
	compute_spike_score_v2_features,
	ramp_down,
	ramp_up,
	to_signed_evidence,
	_tolerance_tag,
)
from muon_decision import (
	MUON_RULE_V3_COLOR_BY_DECISION,
	annotate_feature_dict_with_muon_rule_v3,
	classify_segment_with_muon_rule_v3,
)


def show_hover_map(
		x_axis: np.ndarray,
		spectra: np.ndarray,              # (H,W,N)
		score_map: np.ndarray,            # (H,W)
		candidate_mask: np.ndarray,       # (H,W)
		spikes_by_pixel: Dict[Tuple[int, int], List[SpikeSegment]],
		overlays: Dict[str, np.ndarray],  # erosion/dilation/opening/top_hat
		source_coords_map: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,  # compact(y,x) -> source(y,x)
		plot_raw: bool = True,
		plot_opening: bool = False,
		plot_erosion: bool = False,
		plot_dilation: bool = False,
		plot_top_hat: bool = True,
		plot_corrected_spectra: bool = True,
		initial_checked: Optional[Dict[str, bool]] = None,
		corrected_spectra: Optional[np.ndarray] = None,
		map_central_mass: float = 0.95,
		highlight_detected_pixels: bool = True,
		hover_fps: float = 10.0,
		merge_duplicate_segments: bool = False,
		feature_expand_to_gradient_foot: bool = False,
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
		gws_measure_region: str = GWS_MEASURE_REGION,
		gws_threshold_region: str = GWS_THRESHOLD_REGION,
		edge_dense_levels: Tuple[int, ...] = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95),
		edge_dense_min_snr: float = 1.0,
		edge_dense_context_pad_pts: int = 20,
		edge_dense_context_min_pad_pts: int = 10,
		edge_dense_context_max_pad_pts: int = 80,
		edge_use_enhanced_spike_mapping: bool = False,
		edge_mapping_levels_desc: Tuple[int, ...] = (95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5),
		edge_mapping_refine_step_percent: int = 1,
		edge_mapping_min_level_percent: int = 1,
		edge_mapping_require_closed_interval: bool = True,
		edge_mapping_use_apex_component: bool = True,
		edge_mapping_enable_merge_guard: bool = True,
		edge_mapping_max_width_jump_factor: float = 2.5,
		edge_mapping_max_width_jump_points: int = 8,
		edge_mapping_fallback_to_old: bool = False,
		edge_mapping_noise_guard_enabled: bool = False,
		rucdw_enabled: bool = True,
		rucdw_context_pad_pts: int = 20,
		rucdw_context_max_pad_pts: int = 80,
		rucdw_levels: Tuple[int, ...] = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90),
		rucdw_min_snr: float = 1.0,
		rucdw_noise_fallback_rel_amp: float = 0.05,
		rucdw_anchor_mode: str = "max_in_candidate",
		rucdw_baseline_mode: str = "context_low_percentile",
		rucdw_baseline_percentile: float = 5.0,
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
		despike_diagnostics: Optional[List[Dict[str, object]]] = None,
		despike_contact_analysis: Optional[Dict[str, object]] = None,
		ss4_candidate_metrics_by_pixel: Optional[Dict[Tuple[int, int], List[Dict[str, object]]]] = None,
		despike_contact_candidates_enabled: bool = True,
		despike_contact_context_pad_pts: int = 4,
		merge_max_width_pts: Optional[int] = None,
) -> None:
	view_spikes_by_pixel = {
		pix: list(segs)
		for pix, segs in spikes_by_pixel.items()
	}
	despike_diag_by_pixel: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
	contact_parent_by_pixel: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
	def _num_from(row: Dict[str, object], *keys: str) -> float:
		for key in keys:
			if key in row and row.get(key) is not None:
				try:
					return float(row.get(key))
				except Exception:
					continue
		return float("nan")

	def _dedupe_ss4_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
		by_id: Dict[object, Dict[str, object]] = {}

		def _row_rank(row: Dict[str, object]) -> Tuple[float, float, float]:
			ss4_v = _num_from(row, "primary_ss4", "ss4")
			ss1_v = _num_from(row, "primary_spike_score_v1", "spike_score_v1")
			pce_v = _num_from(row, "primary_pce_negpref_t098_evidence_signed", "pce_negpref_t098_evidence_signed")
			return (
				ss4_v if np.isfinite(ss4_v) else -1.0,
				ss1_v if np.isfinite(ss1_v) else -1.0,
				pce_v if np.isfinite(pce_v) else -1.0,
			)

		for row in rows:
			try:
				row_id: object = row.get("candidate_id") or (
					int(row.get("peak_index")),
					int(row.get("start")),
					int(row.get("end")),
				)
			except Exception:
				continue
			prev = by_id.get(row_id)
			if prev is None or _row_rank(row) > _row_rank(prev):
				by_id[row_id] = row
		return sorted(by_id.values(), key=lambda r: (int(r.get("peak_index", -1)), int(r.get("start", -1)), int(r.get("end", -1))))

	def _dedupe_contact_parents(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
		out: List[Dict[str, object]] = []
		seen: set[Tuple[int, int, int]] = set()
		for row in rows:
			try:
				key = (int(row.get("parent_apex")), int(row.get("parent_start")), int(row.get("parent_end")))
			except Exception:
				key = (len(out), -1, -1)
			if key in seen:
				continue
			seen.add(key)
			out.append(row)
		return out

	ss4_metrics_by_pixel: Dict[Tuple[int, int], List[Dict[str, object]]] = {
		(int(k[0]), int(k[1])): _dedupe_ss4_rows(list(v))
		for k, v in (ss4_candidate_metrics_by_pixel or {}).items()
	}
	source_to_compact: Dict[Tuple[int, int], Tuple[int, int]] = {}
	if source_coords_map is not None:
		for compact_pix, source_pix in source_coords_map.items():
			try:
				source_to_compact[(int(source_pix[0]), int(source_pix[1]))] = (int(compact_pix[0]), int(compact_pix[1]))
			except Exception:
				continue
	for diag in despike_diagnostics or []:
		try:
			dy = int(diag.get("y"))  # type: ignore[arg-type]
			dx = int(diag.get("x"))  # type: ignore[arg-type]
		except Exception:
			continue
		despike_diag_by_pixel.setdefault((dy, dx), []).append(diag)
		compact_pix = source_to_compact.get((dy, dx))
		if compact_pix is not None:
			despike_diag_by_pixel.setdefault(compact_pix, []).append(diag)
	if isinstance(despike_contact_analysis, dict):
		for parent in despike_contact_analysis.get("parents", []) or []:
			if not isinstance(parent, dict):
				continue
			try:
				py = int(parent.get("y"))  # type: ignore[arg-type]
				px = int(parent.get("x"))  # type: ignore[arg-type]
			except Exception:
				continue
			contact_parent_by_pixel.setdefault((py, px), []).append(parent)
			primary_row = {
				"candidate_id": parent.get("candidate_id"),
				"parent_id": parent.get("parent_id"),
				"y": py,
				"x": px,
				"peak_index": parent.get("parent_apex"),
				"start": parent.get("parent_start"),
				"end": parent.get("parent_end"),
				"primary_ss4": parent.get("primary_ss4", parent.get("parent_ss4_value")),
				"primary_ss4_decision": parent.get("primary_ss4_decision"),
				"primary_ss4_reason": parent.get("primary_ss4_reason", parent.get("parent_ss4_reason")),
				"primary_ss4_rve_feature": parent.get("primary_ss4_rve_feature", parent.get("parent_edge_feature")),
				"primary_spike_score_v1": parent.get("primary_spike_score_v1", parent.get("parent_ss1")),
				"primary_pce_negpref_t098_evidence_signed": parent.get("primary_pce_negpref_t098_evidence_signed", parent.get("parent_pce")),
				"primary_recdw_sum_0_90_raman_veto_evidence_signed": parent.get("primary_recdw_sum_0_90_raman_veto_evidence_signed", parent.get("parent_edge")),
			}
			ss4_metrics_by_pixel.setdefault((py, px), []).append(primary_row)
			compact_pix = source_to_compact.get((py, px))
			if compact_pix is not None and compact_pix != (py, px):
				contact_parent_by_pixel.setdefault(compact_pix, []).append(parent)
				ss4_metrics_by_pixel.setdefault(compact_pix, []).append(primary_row)
		for pix, parents in list(contact_parent_by_pixel.items()):
			contact_parent_by_pixel[pix] = _dedupe_contact_parents(parents)
		for pix, rows in list(ss4_metrics_by_pixel.items()):
			ss4_metrics_by_pixel[pix] = _dedupe_ss4_rows(rows)
	for (sy, sx), vals in list(ss4_metrics_by_pixel.items()):
		compact_pix = source_to_compact.get((int(sy), int(sx)))
		if compact_pix is not None and compact_pix != (int(sy), int(sx)):
			ss4_metrics_by_pixel.setdefault(compact_pix, []).extend(vals)
			ss4_metrics_by_pixel[compact_pix] = _dedupe_ss4_rows(ss4_metrics_by_pixel[compact_pix])
	GWS_SCALES = tuple(int(v) for v in GWS_GRANULO_SCALES)
	GWS_DRAW_SCALES = tuple(int(v) for v in GWS_SCALES[::max(1, len(GWS_SCALES) // 5)]) or tuple(int(v) for v in GWS_SCALES)
	GWS_DEFAULT_SCALE = 7
	MDWS_CONTEXT_PAD = 10
	MDWS_MEDIAN_WINDOW = 5

	H, W, N = spectra.shape

	fig, (ax_map, ax_spec) = plt.subplots(1, 2, figsize=(16.0, 9.0))
	plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.25, wspace=0.18)
	fig.canvas.manager.set_window_title("Muon finder - hover viewer")
	median_window_choices = (5, 9, 15, 31, 61, 81, 101, 125)
	mean_window_choices = (5, 9, 15, 31, 61, 81, 101, 125)
	opening_window_choices = (81, 101, 125)
	top_hat_window_choices = (3, 7, 9, 11)
	diag_state = {"median_window": 81, "mean_window": 81, "opening_window": 81}
	top_hat_state = {"window": 3}

	v = score_map.astype(float)
	v = v[np.isfinite(v)]
	if v.size and 0.0 < map_central_mass < 1.0:
		tail = 0.5 * (1.0 - float(map_central_mass))
		vmin = float(np.quantile(v, tail))
		vmax = float(np.quantile(v, 1.0 - tail))
		if not np.isfinite(vmin) or not np.isfinite(vmax):
			vmin = None
			vmax = None
	else:
		vmin = None
		vmax = None

	im = ax_map.imshow(score_map, origin="upper", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
	ax_map.set_title("score map (hover)")
	ax_map.set_xlabel("x (pixel)")
	ax_map.set_ylabel("y (pixel)")
	ax_map.set_box_aspect(1.0)
	cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
	cbar.set_label("z (score intensity)")

	# candidates' overlay (visually)
	# ys, xs = np.where(candidate_mask)
	# ax_map.scatter(xs, ys, s=8, marker="o", linewidths=0.5, facecolors="none")

	# # detected spikes overlay
	# if highlight_detected_pixels and merged_spikes_by_pixel:
	# 	sp_y = []
	# 	sp_x = []
	# 	for (py, px), segs in merged_spikes_by_pixel.items():
	# 		if not segs:
	# 			continue
	# 		sp_y.append(py)
	# 		sp_x.append(px)
	# 	if sp_x:
	# 		ax_map.scatter(
	# 			sp_x, sp_y,
	# 			s=42,
	# 			marker="s",
	# 			facecolors="none",
	# 			edgecolors="red",
	# 			linewidth=1.2
	# 		)

	# marker for actual pixel
	marker = ax_map.scatter([0], [0], s=80, marker="s", facecolors="none", edgecolors="white", linewidths=2)
	located_muon_pixels = sorted(
		{
			(int(py), int(px))
			for py, px in contact_parent_by_pixel.keys()
			if 0 <= int(py) < int(score_map.shape[0]) and 0 <= int(px) < int(score_map.shape[1])
		}
	)
	if located_muon_pixels:
		loc_y = [p[0] for p in located_muon_pixels]
		loc_x = [p[1] for p in located_muon_pixels]
	else:
		loc_y = []
		loc_x = []
	located_muon_artist = ax_map.scatter(
		loc_x,
		loc_y,
		s=34,
		marker="s",
		facecolors="#d62728",
		edgecolors="#ffffff",
		linewidths=0.35,
		alpha=0.88,
		visible=False,
		zorder=4,
	)

	# stable line colors + checkbox-driven visibility
	line_colors = {
		"raw": "#1f77b4",
		"opening": "#9467bd",
		"erosion": "#7f7f7f",
		"dilation": "#8c564b",
		"top_hat": "#ff7f0e",
		"gradient": "#e377c2",
		"raw_d2": "#17becf",
		"raw_d3": "#8c564b",
		"median": "#4daf4a",
		"median_residual": "#00a7c7",
		"mean": "#377eb8",
		"mean_residual": "#984ea3",
		"opening_diag": "#ff7f0e",
		"opening_residual": "#d95f02",
		"dilation_minus_opening": "#bcbd22",
		"corrected": "#2ca02c",
		"gws_residual": "#7b3294",
		"gws_residual_all": "#c2a5cf",
		"gws_opening": "#008080",
		"mdws510_residual": "#1b9e77",
	}
	checked = {
		"raw": True,
		"opening": False,
		"erosion": False,
		"dilation": False,
		"top_hat": False,
		"gradient": True,
		"raw_d2": False,
		"raw_d3": False,
		"median": False,
		"median_residual": False,
		"mean": False,
		"mean_residual": False,
		"opening_diag": False,
		"opening_residual": False,
		"dilation_minus_opening": False,
		"corrected": False,
	}
	if initial_checked:
		for k, v in initial_checked.items():
			if k in checked:
				checked[k] = bool(v)

	# spectra lines
	(ln_raw,) = ax_spec.plot([], [], label="raw", color=line_colors['raw'])
	(ln_open,) = ax_spec.plot([], [], label="opening", color=line_colors['opening'])
	(ln_ero,) = ax_spec.plot([], [], label="erosion", color=line_colors['erosion'])
	(ln_dil,) = ax_spec.plot([], [], label="dilation", color=line_colors['dilation'])
	(ln_th,) = ax_spec.plot([], [], label="top_hat", color=line_colors['top_hat'])
	(ln_grad,) = ax_spec.plot([], [], label="gradient", color=line_colors['gradient'])
	(ln_raw_d2,) = ax_spec.plot([], [], label="raw_d2", color=line_colors['raw_d2'])
	(ln_raw_d3,) = ax_spec.plot([], [], label="raw_d3", color=line_colors['raw_d3'])
	(ln_median,) = ax_spec.plot([], [], label="median", color=line_colors['median'], linewidth=1.8)
	(ln_median_res,) = ax_spec.plot([], [], label="median_residual", color=line_colors['median_residual'], linewidth=1.4)
	(ln_mean,) = ax_spec.plot([], [], label="mean", color=line_colors['mean'], linewidth=1.8)
	(ln_mean_res,) = ax_spec.plot([], [], label="mean_residual", color=line_colors['mean_residual'], linewidth=1.4)
	(ln_open_diag,) = ax_spec.plot([], [], label="opening_diag", color=line_colors['opening_diag'], linewidth=1.8)
	(ln_open_res,) = ax_spec.plot([], [], label="opening_residual", color=line_colors['opening_residual'], linewidth=1.4)
	(ln_dmo,) = ax_spec.plot([], [], label="dilation_minus_opening", color=line_colors['dilation_minus_opening'])
	(ln_corr,) = ax_spec.plot([], [], label="corrected", color=line_colors['corrected'])
	(ln_gws_residual,) = ax_spec.plot([], [], label="gws_residual", color=line_colors['gws_residual'], linewidth=1.5)
	(ln_gws_residual_all,) = ax_spec.plot([], [], label="gws_residual_all", color=line_colors['gws_residual_all'], linewidth=1.0, alpha=0.55)
	(ln_gws_opening,) = ax_spec.plot([], [], label="gws_opening", color=line_colors['gws_opening'], linewidth=1.2, linestyle="--")
	(ln_mdws_residual,) = ax_spec.plot([], [], label="mdws510_residual", color=line_colors['mdws510_residual'], linewidth=1.5)
	lines = {
		"raw": ln_raw,
		"opening": ln_open,
		"erosion": ln_ero,
		"dilation": ln_dil,
		"top_hat": ln_th,
		"gradient": ln_grad,
		"raw_d2": ln_raw_d2,
		"raw_d3": ln_raw_d3,
		"median": ln_median,
		"median_residual": ln_median_res,
		"mean": ln_mean,
		"mean_residual": ln_mean_res,
		"opening_diag": ln_open_diag,
		"opening_residual": ln_open_res,
		"dilation_minus_opening": ln_dmo,
		"corrected": ln_corr,
		"gws_residual": ln_gws_residual,
		"gws_residual_all": ln_gws_residual_all,
		"gws_opening": ln_gws_opening,
		"mdws510_residual": ln_mdws_residual,
	}
	(ln_ls_anatomy,) = ax_spec.plot([], [], label="ls_anatomy", color="#6a3d9a", linestyle="--", linewidth=1.5)
	(ln_ls_support,) = ax_spec.plot([], [], label="ls_support", color="#2ca25f", linewidth=2.0)
	(ln_ls_curvature,) = ax_spec.plot([], [], label="ls_curvature", color="#d62728", linewidth=1.6)
	(ln_mth_1st,) = ax_spec.plot([], [], label="mth_1st", color="#9467bd")
	(ln_mth_2nd,) = ax_spec.plot([], [], label="mth_2nd", color="#8c564b")
	(ln_mth_3rd,) = ax_spec.plot([], [], label="mth_3rd", color="#e377c2")
	(ln_mth_decay,) = ax_spec.plot([], [], label="mth_decay", color="#7f7f7f")
	(ln_c2_abs,) = ax_spec.plot([], [], label="c2_abs", color="#bcbd22")
	(ln_c2_mask,) = ax_spec.plot([], [], label="c2_mask", color="#17becf")
	(ln_c2_core,) = ax_spec.plot([], [], label="c2_core", color="#4b0082")
	(ln_pce_negpref,) = ax_spec.plot([], [], label="pce_negpref", color="#111111", linewidth=1.5)
	(ln_pce_negpref_local,) = ax_spec.plot([], [], label="pce_negpref_local", color="#444444", linewidth=1.3, linestyle="--")
	metric_lines = {
		"ls_anatomy": ln_ls_anatomy,
		"ls_support": ln_ls_support,
		"ls_curvature": ln_ls_curvature,
		"mth_1st": ln_mth_1st,
		"mth_2nd": ln_mth_2nd,
		"mth_3rd": ln_mth_3rd,
		"mth_decay": ln_mth_decay,
		"c2_abs": ln_c2_abs,
		"c2_mask": ln_c2_mask,
		"c2_core": ln_c2_core,
		"pce_negpref": ln_pce_negpref,
		"pce_negpref_local": ln_pce_negpref_local,
	}

	spike_peak_lines: List = []
	spike_edge_lines: List = []
	spike_bands: List = []
	metric_guide_artists: List = []
	interest_metric_cache: Dict[object, object] = {}
	derived_signal_cache: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
	prepared_segment_cache: Dict[object, Tuple[List[SpikeSegment], List[int], List[int]]] = {}
	prepared_decision_cache: Dict[object, List[Dict[str, object]]] = {}
	gws_diag_cache: Dict[object, Dict[str, object]] = {}
	gws_context_cache: Dict[object, List[Dict[str, object]]] = {}
	mdws_diag_cache: Dict[object, Dict[str, object]] = {}
	pce_warning_cache: set[Tuple[object, ...]] = set()
	gws_warning_cache: set[Tuple[object, ...]] = set()
	metric_checked = {
		"ls_anatomy": False,
		"ls_support": False,
		"ls_curvature": False,
		"mth_1st": False,
		"mth_2nd": False,
		"mth_3rd": False,
		"mth_decay": False,
		"c2_abs": False,
		"c2_mask": False,
		"c2_core": False,
		"pce_negpref": False,
		"pce_negpref_local": False,
		"show_raw_edge_dense": False,
		"show_mg_edge_dense": False,
		"show_raw_edge_dense_ctx": False,
		"show_rucdw_components": False,
		"show_dilation_edge_dense_ctx": False,
		"show_mg_edge_dense_ctx": False,
		"show_raw_ball_descent": False,
		"show_mg_ball_descent": False,
		"show_raw_exp_fit": False,
		"show_mg_exp_fit": False,
		"show_despike_patch": False,
		"show_erosion_contacts": False,
		"show_dilation_contacts": False,
		"show_contact_cells": False,
		"show_cell_salience_labels": False,
		"show_contact_cell_chords": False,
		"secondary_spike_peaks": False,
		"show_secondary_ss4_labels": False,
		"despike_chords": False,
		"located_muon": False,
		"primal_metrics": False,
	}
	spike_overlay_checked = {
		"spike_peaks": False,
		"spike_edges": False,
		"spike_bands": False,
	}

	ax_spec.set_xlabel("Raman shift/cm$^{-1}$")
	ax_spec.set_ylabel("Intensity")
	ax_spec.set_xlim(float(np.min(x_axis)), float(np.max(x_axis)))
	ax_spec.set_box_aspect(1.0)
	spec_info_head = ax_spec.text(0.0, 1.11, "", transform=ax_spec.transAxes, ha="left", va="bottom", clip_on=False, fontsize=10, fontweight="bold")
	spec_info_diag = ax_spec.text(0.0, 1.075, "", transform=ax_spec.transAxes, ha="left", va="bottom", clip_on=False, fontsize=9)
	spec_info_metric = ax_spec.text(0.0, 1.04, "", transform=ax_spec.transAxes, ha="left", va="bottom", clip_on=False, fontsize=9)
	use_blit = bool(getattr(fig.canvas, "supports_blit", True))
	blit_state = {"bg": None, "bbox": None}

	curv_tolerance_tags = [_tolerance_tag(tol) for tol in CURVATURE_NEGPREF_TOLERANCES]
	curv_variant_state = {"i": 0}
	gws_scale_choices: Tuple[int, ...] = tuple(int(v) for v in GWS_SCALES)
	gws_default_scale = int(min(gws_scale_choices, key=lambda v: (abs(int(v) - int(GWS_DEFAULT_SCALE)), int(v))))
	gws_state = {"scale_index": gws_scale_choices.index(int(gws_default_scale))}

	frozen = {"state": False}  # right click = freeze/unfreeze
	current = {"y": 0, "x": 0}
	focus = {"which": "", "replace_x": False, "replace_y": False}
	hover_state = {"last_t": 0.0, "last_xy": (-1, -1)}
	input_buffers = {"x": "0", "y": "0"}

	def _warn_once(cache: set[Tuple[object, ...]], key: Tuple[object, ...], message: str) -> None:
		if key in cache:
			return
		cache.add(key)
		print(message)

	def _fmt_num(value: object, digits: int = 3) -> str:
		try:
			v = float(value)
		except Exception:
			return "nan"
		if not np.isfinite(v):
			return "nan"
		return f"{v:.{int(digits)}g}"

	def _refresh_legend() -> None:
		handles = []
		labels = []
		for nm, ln in lines.items():
			if ln.get_visible():
				handles.append(ln)
				labels.append(nm)
		for nm, ln in metric_lines.items():
			if ln.get_visible():
				handles.append(ln)
				labels.append(nm)
		if metric_checked.get("pce_negpref", False):
			handles.extend([
				Line2D([], [], linestyle="None", marker="x", color="#d62728", markersize=7),
				Line2D([], [], linestyle="None", marker="+", color="#ff7f0e", markersize=8),
				Line2D([], [], linestyle="None", marker="X", color="#1f77b4", markersize=7),
				Line2D([], [], linestyle="None", marker=(5, 2, 0), color="#111111", markersize=10),
			])
			labels.extend([
				"pce apex",
				"pce abs-extreme",
				"pce negative candidate",
				f"pce chosen {curv_tolerance_tags[int(curv_variant_state['i'])]}",
			])
		if metric_checked.get("pce_negpref_local", False):
			handles.extend([
				Line2D([], [], linestyle="None", marker=(5, 2, 90), color="#444444", markersize=10),
				Patch(facecolor="#444444", edgecolor="#444444", alpha=0.08),
			])
			labels.extend([
				f"pce local chosen {curv_tolerance_tags[int(curv_variant_state['i'])]}",
				"pce local search window",
			])
		if metric_checked.get("show_raw_edge_dense", False):
			handles.append(Line2D([], [], color="#d62728", linewidth=1.2, linestyle="-"))
			labels.append("raw edge dense")
		if metric_checked.get("show_mg_edge_dense", False):
			handles.append(Line2D([], [], color="#1f77b4", linewidth=1.2, linestyle="--"))
			labels.append("mg edge dense")
		if metric_checked.get("show_raw_edge_dense_ctx", False):
			handles.append(Line2D([], [], color="#fb6a4a", linewidth=1.1, linestyle="-"))
			labels.append("recdw")
		if metric_checked.get("show_rucdw_components", False):
			handles.append(Line2D([], [], color="#2ca25f", linewidth=1.1, linestyle="-"))
			labels.append("rucdw components")
		if metric_checked.get("show_dilation_edge_dense_ctx", False):
			handles.append(Line2D([], [], color="#8c564b", linewidth=1.1, linestyle="-."))
			labels.append("decdw")
		if metric_checked.get("show_mg_edge_dense_ctx", False):
			handles.append(Line2D([], [], color="#6baed6", linewidth=1.1, linestyle="--"))
			labels.append("mg edge dense ctx")
		if metric_checked.get("show_raw_ball_descent", False):
			handles.append(Line2D([], [], color="#e6550d", linewidth=1.5, marker="o", markersize=4))
			labels.append("raw ball descent")
		if metric_checked.get("show_mg_ball_descent", False):
			handles.append(Line2D([], [], color="#3182bd", linewidth=1.5, marker="o", markersize=4))
			labels.append("mg ball descent")
		if metric_checked.get("show_raw_exp_fit", False):
			handles.append(Line2D([], [], color="#a63603", linewidth=1.5, linestyle=":"))
			labels.append("raw exp fit")
		if metric_checked.get("show_mg_exp_fit", False):
			handles.append(Line2D([], [], color="#08519c", linewidth=1.5, linestyle=":"))
			labels.append("mg exp fit")
		if metric_checked.get("show_erosion_contacts", False):
			handles.append(Line2D([], [], linestyle="None", marker="o", color="#000000", markerfacecolor="#17becf", markersize=5))
			labels.append("erosion contacts")
		if metric_checked.get("show_dilation_contacts", False):
			handles.append(Line2D([], [], linestyle="None", marker="^", color="#000000", markerfacecolor="#ff7f0e", markersize=5))
			labels.append("dilation contacts")
		if metric_checked.get("show_contact_cells", False):
			handles.append(Patch(facecolor="#17becf", edgecolor="#17becf", alpha=0.08))
			labels.append("contact cells")
		if metric_checked.get("show_contact_cell_chords", False):
			handles.append(Line2D([], [], color="#6a3d9a", linewidth=1.0, linestyle="-"))
			labels.append("cell chords")
		if metric_checked.get("secondary_spike_peaks", False):
			handles.extend([
				Line2D([], [], color="#d62728", linewidth=1.0, linestyle="--"),
				Line2D([], [], color="#1f77b4", linewidth=1.0, linestyle="--"),
			])
			labels.extend(["secondary cell spike", "secondary cell non-spike"])
		if metric_checked.get("despike_chords", False):
			handles.extend([
				Line2D([], [], color="#2ca02c", linewidth=1.8, linestyle="-"),
				Line2D([], [], color="#9467bd", linewidth=1.8, linestyle="--"),
			])
			labels.extend(["ordinary despike chord", "supporting tangent chord"])
		if metric_checked.get("gws_support", False):
			handles.append(Line2D([], [], linestyle="None", marker="o", color="#7b3294", markersize=5))
			labels.append("gws support points")
		if metric_checked.get("gws_context", False):
			handles.append(Patch(facecolor="#7b3294", edgecolor="#7b3294", alpha=0.05))
			labels.append("gws context")
		if metric_checked.get("gws_width_trace", False):
			handles.append(Line2D([], [], color="#7b3294", linewidth=1.4, marker="s", markersize=4))
			labels.append("gws width trace")
		if metric_checked.get("gws_area_trace", False):
			handles.append(Line2D([], [], color="#008080", linewidth=1.4, marker="D", markersize=4))
			labels.append("gws area trace")
		if metric_checked.get("mdws510_support", False):
			handles.append(Line2D([], [], linestyle="None", marker="^", color="#1b9e77", markersize=5))
			labels.append("mdws510 support")
		if metric_checked.get("mdws510_context", False):
			handles.append(Patch(facecolor="#1b9e77", edgecolor="#1b9e77", alpha=0.05))
			labels.append("mdws510 context")
		leg = ax_spec.get_legend()
		if leg is not None:
			leg.remove()
		if handles:
			ax_spec.legend(handles, labels, loc="best")
		blit_state["bg"] = None
		blit_state["bbox"] = None

	def _redraw_dynamic(use_fast: bool = True) -> None:
		if not use_blit or not use_fast:
			fig.canvas.draw()
			blit_state["bg"] = fig.canvas.copy_from_bbox(fig.bbox)
			blit_state["bbox"] = fig.bbox.frozen()
			return
		if blit_state["bg"] is None or blit_state["bbox"] is None:
			fig.canvas.draw()
			blit_state["bg"] = fig.canvas.copy_from_bbox(fig.bbox)
			blit_state["bbox"] = fig.bbox.frozen()
			return
		renderer = fig.canvas.get_renderer()
		if renderer is None:
			fig.canvas.draw()
			blit_state["bg"] = fig.canvas.copy_from_bbox(fig.bbox)
			blit_state["bbox"] = fig.bbox.frozen()
			return
		fig.canvas.restore_region(blit_state["bg"])
		ax_map.draw_artist(ax_map.patch)
		ax_map.draw_artist(im)
		if located_muon_artist.get_visible():
			ax_map.draw_artist(located_muon_artist)
		ax_map.draw_artist(marker)
		ax_spec.draw_artist(ax_spec.patch)
		ax_spec.draw_artist(spec_info_head)
		ax_spec.draw_artist(spec_info_diag)
		ax_spec.draw_artist(spec_info_metric)
		for ln in lines.values():
			if ln.get_visible():
				ax_spec.draw_artist(ln)
		for ln in metric_lines.values():
			if ln.get_visible():
				ax_spec.draw_artist(ln)
		for artist in spike_bands:
			ax_spec.draw_artist(artist)
		for artist in spike_edge_lines:
			ax_spec.draw_artist(artist)
		for artist in spike_peak_lines:
			ax_spec.draw_artist(artist)
		for artist in metric_guide_artists:
			ax_spec.draw_artist(artist)
		leg = ax_spec.get_legend()
		if leg is not None:
			ax_spec.draw_artist(leg)
		fig.canvas.blit(fig.bbox)
		fig.canvas.flush_events()

	def _apply_signal_visibility() -> None:
		for nm, ln in lines.items():
			if nm == "corrected" and corrected_spectra is None:
				ln.set_visible(False)
			elif nm == "gradient" and "gradient" not in overlays:
				ln.set_visible(False)
			elif nm == "dilation_minus_opening" and "dilation_minus_opening" not in overlays:
				ln.set_visible(False)
			else:
				ln.set_visible(bool(checked.get(nm, False)))

	def _apply_metric_visibility() -> None:
		for nm, ln in metric_lines.items():
			ln.set_visible(bool(metric_checked.get(nm, False)))
		located_muon_artist.set_visible(bool(metric_checked.get("located_muon", False)) and bool(located_muon_pixels))

	def _is_primary_ss4_parent(y: int, x: int, peak: int, left: int, right: int) -> bool:
		parents = contact_parent_by_pixel.get((int(y), int(x)), [])
		if not parents:
			return False
		for parent in parents:
			try:
				pp = int(parent.get("parent_apex"))
				pl = int(parent.get("parent_start"))
				pr = int(parent.get("parent_end"))
			except Exception:
				continue
			if int(pp) == int(peak) and int(pl) == int(left) and int(pr) == int(right):
				return True
		for parent in parents:
			try:
				pp = int(parent.get("parent_apex"))
				pl = int(parent.get("parent_start"))
				pr = int(parent.get("parent_end"))
			except Exception:
				continue
			if abs(int(pp) - int(peak)) <= 1 and int(pl) <= int(peak) <= int(pr):
				return True
		return False

	def _primary_value(row: Dict[str, object], primary_key: str, legacy_key: str) -> object:
		return row.get(primary_key, row.get(legacy_key))

	def _pixel_ss4_metrics(y: int, x: int) -> List[Dict[str, object]]:
		return list(ss4_metrics_by_pixel.get((int(y), int(x)), []))

	def _primary_ss4_metric_for_candidate(y: int, x: int, peak: int, left: int, right: int) -> Optional[Dict[str, object]]:
		rows = _pixel_ss4_metrics(int(y), int(x))
		if not rows:
			return None
		peak_i = int(peak)
		left_i = int(left)
		right_i = int(right)
		for row in rows:
			try:
				if int(row.get("peak_index")) == peak_i and int(row.get("start")) == left_i and int(row.get("end")) == right_i:
					return row
			except Exception:
				continue
		for row in rows:
			try:
				if int(row.get("peak_index")) == peak_i:
					return row
			except Exception:
				continue
		best: Optional[Dict[str, object]] = None
		best_dist: Optional[int] = None
		for row in rows:
			try:
				pp = int(row.get("peak_index"))
			except Exception:
				continue
			dist = abs(pp - peak_i)
			if dist <= 1 and (best_dist is None or dist < best_dist):
				best = row
				best_dist = dist
		return best

	def _is_primary_ss4_candidate(y: int, x: int, peak: int, left: int, right: int) -> bool:
		row = _primary_ss4_metric_for_candidate(y, x, peak, left, right)
		if row is None:
			return False
		try:
			decision = str(_primary_value(row, "primary_ss4_decision", "ss4_decision") or "")
			if decision:
				return decision == "spike"
			return bool(float(_primary_value(row, "primary_ss4", "ss4")) >= 0.5)
		except Exception:
			return False

	def _draw_curvature_scatter(
			y: int,
			x: int,
			segs: List[SpikeSegment],
			lefts: List[int],
			rights: List[int],
	) -> None:
		return

	def _render_inputs() -> None:
		txt_x.text_disp.set_text(input_buffers["x"])
		txt_y.text_disp.set_text(input_buffers["y"])
		try:
			txt_x.cursor.set_visible(False)
			txt_y.cursor.set_visible(False)
			if focus["which"] == "x":
				txt_x.cursor_index = len(input_buffers["x"])
				txt_x._rendercursor()
			elif focus["which"] == "y":
				txt_y.cursor_index = len(input_buffers["y"])
				txt_y._rendercursor()
		except Exception:
			pass
		fig.canvas.draw_idle()

	def _set_focus(which: str) -> None:
		focus['which'] = which
		try:
			txt_x.stop_typing()
			txt_y.stop_typing()
		except Exception:
			pass
		_render_inputs()

		ax_txt_x.set_facecolor("#ffffff" if which == "x" else "#f0f0f0")
		ax_txt_y.set_facecolor("#ffffff" if which == "y" else "#f0f0f0")
		ax_btn_go.set_facecolor("#dfefff" if which == "go" else "#f0f0f0")
		fig.canvas.draw_idle()

	def _nearest_valid_window(size: int, choices: Tuple[int, ...], fallback: int) -> int:
		valid = [int(w) for w in choices if int(w) > 1 and int(w) % 2 == 1 and int(w) <= int(size)]
		if not valid:
			return int(min(max(3, fallback), max(3, size if size % 2 == 1 else size - 1)))
		best = min(valid, key=lambda w: (abs(int(w) - int(fallback)), int(w)))
		return int(best)

	def _cycle_choice(current: int, choices: Tuple[int, ...], step: int, size: int) -> int:
		valid = [int(w) for w in choices if int(w) > 1 and int(w) % 2 == 1 and int(w) <= int(size)]
		if not valid:
			return int(current)
		if int(current) not in valid:
			return valid[0]
		i = valid.index(int(current))
		return valid[(i + int(step)) % len(valid)]

	def _nearest_choice(target: int, choices: Tuple[int, ...]) -> int:
		if not choices:
			return int(target)
		return int(min((int(v) for v in choices), key=lambda v: (abs(int(v) - int(target)), int(v))))

	def _current_gws_scale() -> int:
		return int(gws_scale_choices[int(gws_state["scale_index"]) % len(gws_scale_choices)])

	def _estimate_bg_mad(sig: np.ndarray, left: int, right: int) -> float:
		x = np.asarray(sig, dtype=float)
		n = int(x.size)
		a = int(np.clip(left, 0, n - 1))
		b = int(np.clip(right, 0, n - 1))
		context = max(10, 3 * max(1, b - a + 1))
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

	def _scale_overlay_to_raw_panel(
			curve: np.ndarray,
			raw_context: np.ndarray,
			*,
			band_low: float = 0.10,
			band_high: float = 0.35,
	) -> np.ndarray:
		raw_ctx = np.asarray(raw_context, dtype=float)
		cur = np.asarray(curve, dtype=float)
		out = np.full(cur.shape, np.nan, dtype=float)
		mask = np.isfinite(cur)
		if not np.any(mask):
			return out
		raw_min = float(np.min(raw_ctx)) if raw_ctx.size else 0.0
		raw_max = float(np.max(raw_ctx)) if raw_ctx.size else 1.0
		raw_range = raw_max - raw_min
		if not np.isfinite(raw_range) or raw_range <= 0.0:
			raw_range = max(abs(raw_max), 1.0)
		lo = raw_min + float(band_low) * raw_range
		hi = raw_min + float(band_high) * raw_range
		cur_valid = cur[mask]
		cmin = float(np.min(cur_valid))
		cmax = float(np.max(cur_valid))
		if not np.isfinite(cmin) or not np.isfinite(cmax):
			return out
		if cmax <= cmin + 1e-12:
			out[mask] = 0.5 * (lo + hi)
			return out
		out[mask] = lo + (cur_valid - cmin) * (hi - lo) / (cmax - cmin)
		return out

	def _scale_overlay_above_signal(
			curve: np.ndarray,
			anchor_signal: np.ndarray,
			*,
			band_low: float = 0.08,
			band_high: float = 0.24,
	) -> np.ndarray:
		"""Map a diagnostic trace into a compact band above the local anchor signal."""
		anchor = np.asarray(anchor_signal, dtype=float)
		cur = np.asarray(curve, dtype=float)
		out = np.full(cur.shape, np.nan, dtype=float)
		mask = np.isfinite(cur)
		if not np.any(mask):
			return out
		a_min = float(np.nanmin(anchor)) if anchor.size and np.any(np.isfinite(anchor)) else 0.0
		a_max = float(np.nanmax(anchor)) if anchor.size and np.any(np.isfinite(anchor)) else 1.0
		a_range = a_max - a_min
		if not np.isfinite(a_range) or a_range <= 0.0:
			a_range = max(abs(a_max), 1.0)
		lo = a_max + float(band_low) * a_range
		hi = a_max + float(band_high) * a_range
		cur_valid = cur[mask]
		cmin = float(np.min(cur_valid))
		cmax = float(np.max(cur_valid))
		if not np.isfinite(cmin) or not np.isfinite(cmax):
			return out
		if cmax <= cmin + 1e-12:
			out[mask] = 0.5 * (lo + hi)
			return out
		out[mask] = lo + (cur_valid - cmin) * (hi - lo) / (cmax - cmin)
		return out

	def _x_from_float_index(idx_float: float) -> float:
		"""Map a fractional sample index to the plotted x-axis coordinate."""
		if not np.isfinite(idx_float):
			return float("nan")
		if x_axis.size <= 0:
			return float(idx_float)
		if x_axis.size == 1:
			return float(x_axis[0])
		pos = float(np.clip(idx_float, 0.0, float(x_axis.size - 1)))
		i0 = int(np.floor(pos))
		i1 = int(np.ceil(pos))
		if i0 == i1:
			return float(x_axis[i0])
		t = float(pos - i0)
		return float((1.0 - t) * float(x_axis[i0]) + t * float(x_axis[i1]))

	def _compute_gws_diagnostics(
			y: int,
			x: int,
			seg: SpikeSegment,
			context_info: Optional[Dict[str, object]] = None,
			context_mode: str = "segment_width",
			context_pad_factor: float = 1.0,
			context_fixed_pad: Optional[int] = None,
	) -> Dict[str, object]:
		cache_key = (
			int(y), int(x), int(seg.peak_index), int(seg.start), int(seg.end),
			str(context_mode), float(context_pad_factor), None if context_fixed_pad is None else int(context_fixed_pad),
			str(gws_measure_region),
			str(gws_threshold_region),
			None if context_info is None else int(context_info.get("context_left", -1)),
			None if context_info is None else int(context_info.get("context_right", -1)),
		)
		if cache_key in gws_diag_cache:
			return gws_diag_cache[cache_key]
		raw_sig = spectra[int(y), int(x), :].astype(float)
		grad_sig = overlays["gradient"][int(y), int(x), :].astype(float) if "gradient" in overlays else raw_sig
		out = compute_gws_diagnostics(
			grad_sig,
			raw_signal=raw_sig,
			apex_idx=int(seg.peak_index),
			detection_left=int(seg.start),
			detection_right=int(seg.end),
			scales=GWS_SCALES,
			context_mode=str(context_mode),
			context_pad_factor=float(context_pad_factor),
			context_fixed_pad=context_fixed_pad,
			context_info=context_info,
			measure_region=str(gws_measure_region),
			threshold_region=str(gws_threshold_region),
		)
		gws_diag_cache[cache_key] = out
		return out

	def _get_gws_context_infos(
			y: int,
			x: int,
			segs: List[SpikeSegment],
			*,
			context_mode: str,
			context_pad_factor: float = 1.0,
			context_fixed_pad: Optional[int] = None,
	) -> List[Dict[str, object]]:
		if not segs or "gradient" not in overlays:
			return []
		cache_key = (
			int(y),
			int(x),
			str(context_mode),
			float(context_pad_factor),
			None if context_fixed_pad is None else int(context_fixed_pad),
			bool(gws_split_overlapping_contexts),
			str(gws_split_source),
			int(gws_split_smooth_pts),
			float(gws_split_valley_alpha),
			int(gws_split_min_distance_from_apex),
			int(gws_split_min_context_width),
			tuple((int(s.peak_index), int(s.start), int(s.end)) for s in segs),
		)
		if cache_key in gws_context_cache:
			return gws_context_cache[cache_key]
		grad_sig = overlays["gradient"][int(y), int(x), :].astype(float)
		out = compute_gws_context_infos(
			grad_sig,
			apex_indices=[int(s.peak_index) for s in segs],
			detection_lefts=[int(s.start) for s in segs],
			detection_rights=[int(s.end) for s in segs],
			context_mode=str(context_mode),
			context_pad_factor=float(context_pad_factor),
			context_fixed_pad=context_fixed_pad,
			split_overlapping_contexts=bool(gws_split_overlapping_contexts),
			split_source=str(gws_split_source),
			split_smooth_pts=int(gws_split_smooth_pts),
			split_valley_alpha=float(gws_split_valley_alpha),
			split_min_distance_from_apex=int(gws_split_min_distance_from_apex),
			split_min_context_width=int(gws_split_min_context_width),
			split_debug=bool(gws_split_debug),
		)
		gws_context_cache[cache_key] = out
		return out

	def _compute_mdws510_diagnostics(
			y: int,
			x: int,
			seg: SpikeSegment,
	) -> Dict[str, object]:
		cache_key = (int(y), int(x), int(seg.peak_index), int(seg.start), int(seg.end))
		if cache_key in mdws_diag_cache:
			return mdws_diag_cache[cache_key]
		raw_sig = spectra[int(y), int(x), :].astype(float)
		a = int(np.clip(seg.start, 0, N - 1))
		b = int(np.clip(seg.end, 0, N - 1))
		p = int(np.clip(seg.peak_index, 0, N - 1))
		ctx_l = max(0, a - MDWS_CONTEXT_PAD)
		ctx_r = min(N - 1, b + MDWS_CONTEXT_PAD)
		raw_ctx = raw_sig[ctx_l:ctx_r + 1]
		apex_rel = int(np.clip(p - ctx_l, 0, raw_ctx.size - 1))
		bg_mad = _estimate_bg_mad(raw_sig, ctx_l, ctx_r)
		med = compute_median_residual_features(raw_ctx, median_windows=(MDWS_MEDIAN_WINDOW,))
		residual = np.asarray(med.get(f"median_residual_w{MDWS_MEDIAN_WINDOW}", np.zeros_like(raw_ctx)), dtype=float)
		pos = np.maximum(residual, 0.0)
		apex_pos = float(max(pos[apex_rel], 0.0)) if pos.size else 0.0
		thr_soft = float(max(0.08 * apex_pos, 0.5 * bg_mad))
		support_mask = pos >= thr_soft if pos.size else np.zeros(0, dtype=bool)
		width_soft = float(np.count_nonzero(support_mask)) if apex_pos > 0.0 else 0.0
		mdws_support01 = float(ramp_down(width_soft, MDWS510_SUPPORT_FULL, MDWS510_SUPPORT_ZERO))
		mdws_veto01 = float(ramp_up(width_soft, MDWS510_VETO_ZERO, MDWS510_VETO_FULL))
		mdws_evidence_signed = float(to_signed_evidence(mdws_support01, mdws_veto01))
		out = {
			"context_left": int(ctx_l),
			"context_right": int(ctx_r),
			"apex_index": int(p),
			"raw_context": raw_ctx,
			"median_residual": residual,
			"support_mask": support_mask,
			"width_soft": float(width_soft),
			"bg_mad": float(bg_mad),
			"mdws510_evidence_signed": float(mdws_evidence_signed),
		}
		mdws_diag_cache[cache_key] = out
		return out

	def _prepare_pixel_segments(
			y: int,
			x: int,
			segs: List[SpikeSegment],
	) -> Tuple[List[SpikeSegment], List[int], List[int]]:
		if not segs:
			return [], [], []
		cache_key = (
			int(y),
			int(x),
			tuple(
				(
					int(s.peak_index),
					int(s.start),
					int(s.end),
					float(getattr(s, "peak_height", 0.0)),
					float(getattr(s, "area", 0.0)),
				)
				for s in segs
			),
		)
		if cache_key in prepared_segment_cache:
			return prepared_segment_cache[cache_key]

		peaks: List[int] = []
		lefts: List[int] = []
		rights: List[int] = []
		for s in segs:
			pi = int(np.clip(s.peak_index, 0, N - 1))
			si = int(np.clip(s.start, 0, N - 1))
			ei = int(np.clip(s.end, 0, N - 1))

			if bool(feature_expand_to_gradient_foot) and "gradient" in overlays:
				grad_sig = overlays["gradient"][y, x, :].astype(float)
				si, ei = expand_interval_to_signal_foot(
					sig=grad_sig,
					left=si,
					right=ei,
					peak=pi,
					enabled=True,
					k_mad=float(feature_foot_k_mad),
					min_run=int(feature_foot_min_run),
					method=feature_window_method,
					erosion_se_size=int(feature_erosion_se_size),
				)

			peaks.append(pi)
			lefts.append(int(si))
			rights.append(int(ei))

		bsrc = str(boundary_minimum_source).strip().lower()
		if bsrc == "gradient" and "gradient" in overlays:
			boundary_sig = overlays['gradient'][y, x, :].astype(float)
		else:
			boundary_sig = spectra[y, x, :].astype(float)
		lefts, rights = enforce_shared_boundaries_by_minima(
			peaks=peaks,
			lefts=lefts,
			rights=rights,
			signal=boundary_sig,
		)

		prepared = [
			SpikeSegment(
				y=int(y),
				x=int(x),
				peak_index=int(peaks[i]),
				start=int(lefts[i]),
				end=int(rights[i]),
				peak_height=float(segs[i].peak_height),
				area=float(segs[i].area),
			)
			for i in range(len(segs))
		]

		if bool(merge_duplicate_segments) and "gradient" in overlays:
			prepared = merge_spike_segments_by_signal_foot(
				prepared,
				signal=overlays["gradient"][y, x, :].astype(float),
				k_mad=float(feature_foot_k_mad),
				min_run=int(feature_foot_min_run),
				max_width_pts=(None if merge_max_width_pts is None else int(merge_max_width_pts)),
				merge_adjacent=True,
				peak_distance_max=None,
			)

		out_lefts = [int(s.start) for s in prepared]
		out_rights = [int(s.end) for s in prepared]
		out = (prepared, out_lefts, out_rights)
		prepared_segment_cache[cache_key] = out
		return out

	def _get_pixel_signals(y: int, x: int) -> Dict[str, np.ndarray]:
		key = (int(y), int(x))
		if key not in derived_signal_cache:
			raw_sig = spectra[y, x, :].astype(float)
			out: Dict[str, np.ndarray] = {
				'raw': raw_sig,
				'opening': overlays['opening'][y, x, :].astype(float),
				'erosion': overlays['erosion'][y, x, :].astype(float),
				'dilation': overlays['dilation'][y, x, :].astype(float),
				'top_hat': overlays['top_hat'][y, x, :].astype(float),
			}
			if "gradient" in overlays:
				out['gradient'] = overlays['gradient'][y, x, :].astype(float)
			if "dilation_minus_opening" in overlays:
				out['dilation_minus_opening'] = overlays['dilation_minus_opening'][y, x, :].astype(float)
			if corrected_spectra is not None:
				out['corrected'] = corrected_spectra[y, x, :].astype(float)
			derived_signal_cache[key] = out

		out = derived_signal_cache[key]
		raw_sig = np.asarray(out['raw'], dtype=float)
		def _centered_mean_signal(sig: np.ndarray, window: int) -> np.ndarray:
			w = int(max(1, window))
			if w % 2 == 0:
				w += 1
			if w <= 1:
				return np.asarray(sig, dtype=float).copy()
			pad = w // 2
			padded = np.pad(np.asarray(sig, dtype=float), (pad, pad), mode="reflect")
			kernel = np.ones(w, dtype=float) / float(w)
			return np.convolve(padded, kernel, mode="valid")

		if checked.get("top_hat", False):
			for window in top_hat_window_choices:
				if int(window) <= 3:
					continue
				key_name = f"top_hat_w{int(window)}"
				if key_name not in out:
					out[key_name] = compute_shape_top_hat_signal(raw_sig, int(window))
		if (checked.get("raw_d2", False) or checked.get("raw_d3", False)) and 'raw_d2' not in out:
			out['raw_d2'] = np.gradient(np.gradient(raw_sig))
		if checked.get("raw_d3", False) and 'raw_d3' not in out:
			raw_d1 = np.gradient(raw_sig)
			out['raw_d3'] = np.gradient(np.gradient(raw_d1))
		if (checked.get("median", False) or checked.get("median_residual", False)):
			needed_median_keys = [f"median_w{int(w)}" for w in median_window_choices] + [f"median_residual_w{int(w)}" for w in median_window_choices]
			if any(k not in out for k in needed_median_keys):
				out.update(
					compute_median_residual_features(
						raw_sig,
						median_windows=tuple(sorted(set(median_window_choices))),
					)
				)
		if (checked.get("opening_diag", False) or checked.get("opening_residual", False)):
			needed_open_keys = [f"opening_w{int(w)}" for w in opening_window_choices] + [f"opening_residual_w{int(w)}" for w in opening_window_choices]
			if any(k not in out for k in needed_open_keys):
				out.update(
					compute_opening_residual_features(
						raw_sig,
						opening_windows=tuple(sorted(set(opening_window_choices))),
					)
				)
		if checked.get("mean", False) or checked.get("mean_residual", False):
			w = int(_nearest_valid_window(raw_sig.size, mean_window_choices, int(diag_state["mean_window"])))
			mean_key = f"mean_w{w}"
			res_key = f"mean_residual_w{w}"
			if mean_key not in out or res_key not in out:
				mean_sig = _centered_mean_signal(raw_sig, w)
				out[mean_key] = mean_sig
				out[res_key] = np.maximum(raw_sig - mean_sig, 0.0)
		return out

	def _classify_prepared_segments(y: int, x: int, segs: List[SpikeSegment]) -> List[Dict[str, object]]:
		if not segs:
			return []
		cache_key = (
			int(y),
			int(x),
			tuple((int(s.peak_index), int(s.start), int(s.end)) for s in segs),
		)
		if cache_key in prepared_decision_cache:
			return prepared_decision_cache[cache_key]
		raw_sig = spectra[int(y), int(x), :].astype(float)
		grad_sig = overlays["gradient"][int(y), int(x), :].astype(float) if "gradient" in overlays else None
		src_sig = grad_sig if grad_sig is not None and grad_sig.size == raw_sig.size else raw_sig
		out: List[Dict[str, object]] = []
		for s in segs:
			a = int(np.clip(s.start, 0, max(0, src_sig.size - 1)))
			b = int(np.clip(s.end, 0, max(0, src_sig.size - 1)))
			p = int(np.clip(s.peak_index, 0, max(0, src_sig.size - 1)))
			if b < a or src_sig.size < 3:
				out.append({})
				continue
			seg = np.asarray(src_sig[a:b + 1], dtype=float)
			peak_rel = int(np.clip(p - a, 0, max(0, seg.size - 1)))
			d = np.diff(seg)
			rise = d[: max(1, p - a)]
			fall = d[max(1, p - a) :]
			context = max(10, 3 * max(1, b - a + 1))
			l0 = max(0, a - context)
			r1 = min(src_sig.size, b + context + 1)
			bg = np.concatenate([src_sig[l0:a], src_sig[b + 1:r1]])
			if bg.size < 5:
				bg = np.concatenate([src_sig[:a], src_sig[b + 1:]])
			if bg.size < 5:
				bg = src_sig
			bg_med = float(np.median(bg))
			bg_mad = max(float(np.median(np.abs(bg - bg_med))), 1e-12)
			rise_slope = float(np.max(rise)) if rise.size else 0.0
			fall_slope = float(np.min(fall)) if fall.size else 0.0
			features: Dict[str, object] = {
				"spike_score_v1": float(
					0.5 * np.tanh(float(rise_slope / bg_mad) / 6.0)
					+ 0.5 * np.tanh(float(abs(fall_slope / bg_mad)) / 6.0)
				),
				"rise_slope_z": float(rise_slope / bg_mad),
				"fall_slope_z": float(abs(fall_slope / bg_mad)),
				"gws_evidence_signed": float("nan"),
			}
			features.update(compute_peak_curvature_features(seg, bg_mad, peak_rel=peak_rel))
			features.update(compute_spike_score_v2_features(features))  # PCE evidence only; GWS remains NaN.
			features["gws_support01"] = float("nan")
			features["gws_evidence_signed"] = float("nan")
			features.update(annotate_feature_dict_with_muon_rule_v3(features))
			out.append(features)
		prepared_decision_cache[cache_key] = out
		return out

	def _current_top_hat_window() -> int:
		return int(_nearest_choice(int(top_hat_state["window"]), top_hat_window_choices))

	def _current_top_hat_key() -> str:
		window = int(_current_top_hat_window())
		return "top_hat" if window <= 3 else f"top_hat_w{window}"

	def _apply_dynamic_ylim() -> None:
		vals: List[np.ndarray] = []
		for ln in lines.values():
			if not ln.get_visible():
				continue
			ydata = np.asarray(ln.get_ydata(), dtype=float)
			ydata = ydata[np.isfinite(ydata)]
			if ydata.size:
				vals.append(ydata)
		for ln in metric_lines.values():
			if not ln.get_visible():
				continue
			ydata = np.asarray(ln.get_ydata(), dtype=float)
			ydata = ydata[np.isfinite(ydata)]
			if ydata.size:
				vals.append(ydata)
		if not vals:
			if tuple(map(float, ax_spec.get_ylim())) != (0.0, 1.0):
				ax_spec.set_ylim(0.0, 1.0)
				blit_state["bg"] = None
				blit_state["bbox"] = None
			return
		stacked = np.concatenate(vals)
		ymin = float(np.min(stacked))
		ymax = float(np.max(stacked))
		yrange = ymax - ymin
		margin = 0.06 * (yrange if yrange > 0.0 else max(abs(ymax), 1.0))
		bottom = ymin - margin
		top = ymax + margin
		if not any(checked.get(k, False) for k in ("raw_d2", "raw_d3")) and not any(metric_checked.get(k, False) for k in ("ls_curvature", "c2_abs", "c2_mask", "c2_core")) and bottom > 0.0:
			bottom = 0.0
		if not np.isfinite(bottom) or not np.isfinite(top) or bottom == top:
			bottom, top = 0.0, 1.0
		cur_ylim = tuple(map(float, ax_spec.get_ylim()))
		new_ylim = (float(bottom), float(top))
		if any(abs(a - b) > 1e-12 for a, b in zip(cur_ylim, new_ylim)):
			ax_spec.set_ylim(*new_ylim)
			blit_state["bg"] = None
			blit_state["bbox"] = None

	def _build_metric_overlays(
			y: int,
			x: int,
			segs: List[SpikeSegment],
			lefts: List[int],
			rights: List[int],
			segment_decisions: Optional[List[Dict[str, object]]] = None,
	) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]]]:
		cache_key = (
			int(y),
			int(x),
			int(curv_variant_state["i"]),
			str(_current_gws_scale()),
			tuple(sorted((str(k), bool(v)) for k, v in checked.items() if str(k).startswith("gws_") or str(k).startswith("mdws510_"))),
			tuple(sorted((str(k), bool(v)) for k, v in metric_checked.items() if str(k).startswith("gws_") or str(k).startswith("mdws510_") or str(k).startswith("pce_") or str(k).startswith("ls_") or str(k).startswith("mth_") or str(k).startswith("c2_") or str(k).startswith("show_") or str(k) in {"secondary_spike_peaks", "despike_chords", "primal_metrics"})),
			tuple((int(s.peak_index), int(lefts[i]), int(rights[i])) for i, s in enumerate(segs)),
		)
		if cache_key in interest_metric_cache:
			return interest_metric_cache[cache_key]  # type: ignore[return-value]

		overlay_keys = tuple(metric_checked.keys()) + (
			"gws_residual",
			"gws_residual_all",
			"gws_opening",
			"mdws510_residual",
		)
		out = {k: np.full(N, np.nan, dtype=float) for k in overlay_keys}
		guide_specs: List[Dict[str, object]] = []
		raw_sig = spectra[y, x, :].astype(float)
		curv_sig = overlays["gradient"][y, x, :].astype(float) if "gradient" in overlays else raw_sig
		active_gws_scale = int(_current_gws_scale())
		need_contact_overlay = bool(despike_contact_candidates_enabled) and any(
			bool(metric_checked.get(k, False))
			for k in (
				"show_erosion_contacts",
				"show_dilation_contacts",
				"show_contact_cells",
				"show_cell_salience_labels",
				"show_contact_cell_chords",
				"secondary_spike_peaks",
				"show_secondary_ss4_labels",
				"despike_chords",
				"primal_metrics",
			)
		)
		if need_contact_overlay:
			for parent in contact_parent_by_pixel.get((int(y), int(x)), []):
				try:
					cl = int(np.clip(int(parent.get("context_left")), 0, N - 1))
					cr = int(np.clip(int(parent.get("context_right")), 0, N - 1))
				except Exception:
					continue
				if cl > cr:
					cl, cr = cr, cl
				context_metric = (
					"show_contact_cells" if bool(metric_checked.get("show_contact_cells", False))
					else ("show_erosion_contacts" if bool(metric_checked.get("show_erosion_contacts", False))
					else ("show_dilation_contacts" if bool(metric_checked.get("show_dilation_contacts", False))
					else ("secondary_spike_peaks" if bool(metric_checked.get("secondary_spike_peaks", False))
					else ("despike_chords" if bool(metric_checked.get("despike_chords", False)) else "show_cell_salience_labels"))))
				)
				guide_specs.append({"metric": context_metric, "kind": "span", "x0": float(x_axis[cl]), "x1": float(x_axis[cr]), "color": "#17becf", "alpha": 0.055})
				guide_specs.append({"metric": context_metric, "kind": "vline", "x": float(x_axis[cl]), "color": "#17becf", "alpha": 0.65, "linestyle": ":"})
				guide_specs.append({"metric": context_metric, "kind": "vline", "x": float(x_axis[cr]), "color": "#17becf", "alpha": 0.65, "linestyle": ":"})
				if bool(metric_checked.get("show_contact_cells", False)) or bool(metric_checked.get("show_cell_salience_labels", False)):
					try:
						pa = int(np.clip(int(parent.get("parent_apex")), 0, N - 1))
						guide_specs.append({
							"metric": "show_contact_cells",
							"kind": "vline",
							"x": float(x_axis[pa]),
							"color": "#d62728",
							"alpha": 0.85,
							"linestyle": "-",
						})
					except Exception:
						pass
				if bool(metric_checked.get("show_erosion_contacts", False)):
					contacts = [int(v) for v in parent.get("erosion_contacts", []) or []]
					for ci in contacts:
						if 0 <= ci < N:
							guide_specs.append({"metric": "show_erosion_contacts", "kind": "point", "x": float(x_axis[ci]), "y": float(raw_sig[ci]), "color": "#000000", "marker": "o", "markersize": 6.0, "alpha": 0.98})
					if not contacts:
						y_text = float(np.nanmax(raw_sig[cl:cr + 1])) if np.any(np.isfinite(raw_sig[cl:cr + 1])) else float(raw_sig[cl])
						guide_specs.append({"metric": "show_erosion_contacts", "kind": "text", "x": float(x_axis[cl]), "y": y_text, "text": "no erosion contacts", "color": "#17becf", "fontsize": 7.0, "alpha": 0.85})
				if bool(metric_checked.get("show_dilation_contacts", False)):
					for ci in [int(v) for v in parent.get("dilation_contacts", []) or []]:
						if 0 <= ci < N:
							guide_specs.append({"metric": "show_dilation_contacts", "kind": "point", "x": float(x_axis[ci]), "y": float(raw_sig[ci]), "color": "#ff7f0e", "marker": "^", "markersize": 6.5, "alpha": 0.96})
				cells = parent.get("cells", []) or []
				island = set(int(v) for v in (parent.get("summary", {}) or {}).get("preliminary_island_cell_indices", []) or [])
				for cell in cells:
					if not isinstance(cell, dict):
						continue
					try:
						cell_idx = int(cell.get("cell_index"))
						left = int(np.clip(int(cell.get("cell_left")), 0, N - 1))
						right = int(np.clip(int(cell.get("cell_right")), 0, N - 1))
						sal = float(cell.get("salience", np.nan))
					except Exception:
						continue
					if left > right:
						left, right = right, left
					if bool(metric_checked.get("show_contact_cells", False)):
						color = "#d62728" if cell_idx in island else ("#ff7f0e" if bool(cell.get("contains_parent_apex", False)) else "#17becf")
						alpha = 0.12 if cell_idx in island else (0.10 if bool(cell.get("overlaps_parent_segment", False)) else 0.045)
						guide_specs.append({"metric": "show_contact_cells", "kind": "span", "x0": float(x_axis[left]), "x1": float(x_axis[right]), "color": color, "alpha": alpha})
						guide_specs.append({"metric": "show_contact_cells", "kind": "vline", "x": float(x_axis[left]), "color": color, "alpha": 0.40, "linestyle": "--"})
						guide_specs.append({"metric": "show_contact_cells", "kind": "vline", "x": float(x_axis[right]), "color": color, "alpha": 0.40, "linestyle": "--"})
					if bool(metric_checked.get("secondary_spike_peaks", False)):
						is_spike = bool(cell.get("secondary_final_is_spike", False))
						color = "#d62728" if is_spike else "#1f77b4"
						try:
							if bool(cell.get("contains_parent_apex", False)) and parent.get("parent_apex") is not None:
								xi = int(parent.get("parent_apex"))
							else:
								xi = int(cell.get("secondary_anchor_index", int((left + right) // 2)))
						except Exception:
							xi = int((left + right) // 2)
						xi = int(np.clip(xi, left, right))
						guide_specs.append({"metric": "secondary_spike_peaks", "kind": "vline", "x": float(x_axis[xi]), "color": color, "alpha": 0.88, "linestyle": "--"})
					if bool(metric_checked.get("show_contact_cell_chords", False)):
						try:
							cx = np.asarray(cell.get("chord_x", []), dtype=float)
							cy = np.asarray(cell.get("chord_y", []), dtype=float)
							if cx.size >= 2 and cx.size == cy.size and np.all(np.isfinite(cx)) and np.all(np.isfinite(cy)):
								guide_specs.append({"metric": "show_contact_cell_chords", "kind": "polyline", "x": cx, "y": cy, "color": "#6a3d9a", "alpha": 0.70, "linewidth": 1.0})
						except Exception:
							pass
					try:
						sal_n = float(cell.get("salience_norm", np.nan))
						sal_total = float(cell.get("salience_total_norm", np.nan))
						sal_density = float(cell.get("salience_density_norm", np.nan))
					except Exception:
						sal_n = sal_total = sal_density = float("nan")
					if bool(metric_checked.get("show_cell_salience_labels", False)) and np.isfinite(sal_n):
						yy = float(np.nanmax(raw_sig[left:right + 1])) if np.any(np.isfinite(raw_sig[left:right + 1])) else float(raw_sig[left])
						guide_specs.append({
							"metric": "show_cell_salience_labels",
							"kind": "text",
							"x": float(x_axis[left]),
							"y": yy,
							"text": f"S={sal_n:.2g}\nT={sal_total:.2g}\nD={sal_density:.2g}",
							"color": "#000000",
							"fontsize": 7.0,
							"alpha": 0.85,
						})
					if (
							bool(metric_checked.get("show_secondary_ss4_labels", False))
							and bool(cell.get("secondary_ss4_ran", False))
							and str(cell.get("secondary_final_source", "")) == "secondary_ss4"
					):
						yy = float(np.nanmin(raw_sig[left:right + 1])) if np.any(np.isfinite(raw_sig[left:right + 1])) else float(raw_sig[left])
						edge_value = cell.get("secondary_recdw_sum_0_90_raman_veto_evidence_signed")
						edge_label = "edge"
						try:
							if not np.isfinite(float(edge_value)):
								edge_value = cell.get("secondary_rve_value_used_for_ss4", cell.get("secondary_edge_rve_proxy"))
								edge_label = "edge*"
						except Exception:
							edge_value = cell.get("secondary_rve_value_used_for_ss4", cell.get("secondary_edge_rve_proxy"))
							edge_label = "edge*"
						guide_specs.append({
							"metric": "show_secondary_ss4_labels",
							"kind": "text",
							"x": float(x_axis[left]),
							"y": yy,
							"text": (
								f"secondary\n"
								f"ss1={_fmt_num(cell.get('secondary_spike_score_v1', cell.get('secondary_ss1')), 2)}\n"
								f"pce={_fmt_num(cell.get('secondary_pce_negpref_t098_evidence_signed', cell.get('secondary_pce')), 2)}\n"
								f"{edge_label}={_fmt_num(edge_value, 2)}"
							),
							"color": "#333333",
							"fontsize": 7.0,
							"alpha": 0.90,
						})
				if bool(metric_checked.get("despike_chords", False)):
					cells_by_index = {
						int(c.get("cell_index")): c
						for c in cells
						if isinstance(c, dict) and c.get("cell_index") is not None
					}
					for chord in (parent.get("summary", {}) or {}).get("final_despike_chords", []) or []:
						try:
							cell_indices = [int(v) for v in chord.get("cell_indices", []) or []]
							if not cell_indices:
								continue
							if any(not bool(cells_by_index.get(int(ci), {}).get("secondary_final_is_spike", False)) for ci in cell_indices):
								continue
							cx = np.asarray(chord.get("chord_x", []), dtype=float)
							cy = np.asarray(chord.get("chord_y", []), dtype=float)
							if cx.size < 2 or cx.size != cy.size or not np.all(np.isfinite(cx)) or not np.all(np.isfinite(cy)):
								continue
							method = str(chord.get("chord_method", "ordinary"))
							color = "#2ca02c" if method == "ordinary" else "#9467bd"
							linestyle = "-" if method == "ordinary" else "--"
							guide_specs.append({"metric": "despike_chords", "kind": "polyline", "x": cx, "y": cy, "color": color, "alpha": 0.95, "linewidth": 1.8, "linestyle": linestyle})
							fl = int(np.clip(int(chord.get("final_left_edge")), 0, N - 1))
							fr = int(np.clip(int(chord.get("final_right_edge")), 0, N - 1))
							guide_specs.append({"metric": "despike_chords", "kind": "vline", "x": float(x_axis[fl]), "color": color, "alpha": 0.75, "linestyle": ":"})
							guide_specs.append({"metric": "despike_chords", "kind": "vline", "x": float(x_axis[fr]), "color": color, "alpha": 0.75, "linestyle": ":"})
						except Exception:
							continue
		if bool(metric_checked.get("primal_metrics", False)):
			primal_rows = sorted(_pixel_ss4_metrics(int(y), int(x)), key=lambda c: int(c.get("peak_index", -1)))
			label_counts_by_peak: Dict[int, int] = {}
			for cand in primal_rows:
				try:
					ap = int(np.clip(int(cand.get("peak_index")), 0, N - 1))
				except Exception:
					continue
				try:
					primary_decision = str(_primary_value(cand, "primary_ss4_decision", "ss4_decision") or "")
					is_spike = primary_decision == "spike" if primary_decision else bool(float(_primary_value(cand, "primary_ss4", "ss4")) >= 0.5)
				except Exception:
					is_spike = False
				color = "#d62728" if is_spike else "#1f77b4"
				offset_i = int(label_counts_by_peak.get(ap, 0))
				label_counts_by_peak[ap] = offset_i + 1
				local_left = max(0, ap - 4)
				local_right = min(N - 1, ap + 4)
				local_seg = raw_sig[local_left:local_right + 1]
				local_range = float(np.nanmax(local_seg) - np.nanmin(local_seg)) if np.any(np.isfinite(local_seg)) else 0.0
				yy = float(raw_sig[ap]) + float(offset_i) * max(local_range * 0.18, 1.0)
				guide_specs.append({
					"metric": "primal_metrics",
					"kind": "text",
					"x": float(x_axis[ap]),
					"y": yy,
					"text": (
						f"primary i={ap}\nss4={_fmt_num(_primary_value(cand, 'primary_ss4', 'ss4'))}\n"
						f"ss1={_fmt_num(_primary_value(cand, 'primary_spike_score_v1', 'spike_score_v1'))}\n"
						f"pce={_fmt_num(_primary_value(cand, 'primary_pce_negpref_t098_evidence_signed', 'pce_negpref_t098_evidence_signed'))}\n"
						f"edge={_fmt_num(_primary_value(cand, 'primary_recdw_sum_0_90_raman_veto_evidence_signed', 'recdw_sum_0_90_raman_veto_evidence_signed'))}\n"
						f"{_primary_value(cand, 'primary_ss4_reason', 'ss4_reason') or ''}"
					),
					"color": color,
					"fontsize": 7.5,
					"alpha": 0.92,
				})
				guide_specs.append({"metric": "primal_metrics", "kind": "vline", "x": float(x_axis[ap]), "color": color, "alpha": 0.75, "linestyle": ":"})
		if bool(metric_checked.get("show_despike_patch", False)):
			for diag in despike_diag_by_pixel.get((int(y), int(x)), []):
				reason = str(diag.get("skipped_reason", ""))
				color = "#d62728" if not reason else "#ff7f0e"
				try:
					osl = int(np.clip(int(diag.get("original_start")), 0, N - 1))
					osr = int(np.clip(int(diag.get("original_end")), 0, N - 1))
					if osl <= osr:
						guide_specs.append({"metric": "show_despike_patch", "kind": "span", "x0": float(x_axis[osl]), "x1": float(x_axis[osr]), "color": "#9467bd", "alpha": 0.07})
				except Exception:
					pass
				try:
					sl = int(diag.get("support_left"))
					sr = int(diag.get("support_right"))
					sl = int(np.clip(sl, 0, N - 1))
					sr = int(np.clip(sr, 0, N - 1))
					if sl <= sr:
						guide_specs.append({"metric": "show_despike_patch", "kind": "span", "x0": float(x_axis[sl]), "x1": float(x_axis[sr]), "color": color, "alpha": 0.12})
						guide_specs.append({"metric": "show_despike_patch", "kind": "vline", "x": float(x_axis[sl]), "color": color, "alpha": 0.85, "linestyle": "--"})
						guide_specs.append({"metric": "show_despike_patch", "kind": "vline", "x": float(x_axis[sr]), "color": color, "alpha": 0.85, "linestyle": "--"})
				except Exception:
					pass
				for zone_key, zone_color in (("left_anchor_zone", "#2ca02c"), ("right_anchor_zone", "#2ca02c")):
					try:
						z = diag.get(zone_key)
						if not isinstance(z, (list, tuple)) or len(z) < 2:
							continue
						zl = int(np.clip(int(z[0]), 0, N - 1))
						zr = int(np.clip(int(z[1]), 0, N - 1))
						if zl <= zr:
							guide_specs.append({"metric": "show_despike_patch", "kind": "span", "x0": float(x_axis[zl]), "x1": float(x_axis[zr]), "color": zone_color, "alpha": 0.16})
					except Exception:
						continue
				try:
					for sup in diag.get("individual_supports", []) or []:
						if not isinstance(sup, dict):
							continue
						il = int(np.clip(int(sup.get("left")), 0, N - 1))
						ir = int(np.clip(int(sup.get("right")), 0, N - 1))
						if il <= ir:
							guide_specs.append({"metric": "show_despike_patch", "kind": "span", "x0": float(x_axis[il]), "x1": float(x_axis[ir]), "color": "#ff9896", "alpha": 0.16})
				except Exception:
					pass
				try:
					pi = int(np.clip(int(diag.get("peak_index")), 0, N - 1))
					guide_specs.append({"metric": "show_despike_patch", "kind": "point", "x": float(x_axis[pi]), "y": float(raw_sig[pi]), "color": color, "marker": "x", "markersize": 7.0})
				except Exception:
					pass
				try:
					rx = np.asarray(diag.get("replacement_x"), dtype=float)
					ry = np.asarray(diag.get("replacement_y"), dtype=float)
					if rx.size >= 2 and rx.size == ry.size and np.all(np.isfinite(rx)) and np.all(np.isfinite(ry)):
						guide_specs.append({"metric": "show_despike_patch", "kind": "polyline", "x": rx, "y": ry, "color": "#d62728", "alpha": 0.45, "linewidth": 1.1})
				except Exception:
					pass
				try:
					fx = np.asarray(diag.get("final_patch_x"), dtype=float)
					fy = np.asarray(diag.get("final_patch_y"), dtype=float)
					if fx.size >= 1 and fx.size == fy.size and np.all(np.isfinite(fx)) and np.all(np.isfinite(fy)):
						guide_specs.append({"metric": "show_despike_patch", "kind": "polyline", "x": fx, "y": fy, "color": "#d62728", "alpha": 1.0, "linewidth": 2.0})
				except Exception:
					pass
		need_ls = any(bool(metric_checked.get(k, False)) for k in ("ls_anatomy", "ls_support", "ls_curvature"))
		need_pce = any(bool(metric_checked.get(k, False)) for k in ("pce_negpref", "pce_negpref_local"))
		need_mth = any(bool(metric_checked.get(k, False)) for k in ("mth_1st", "mth_2nd", "mth_3rd", "mth_decay"))
		need_c2 = any(bool(metric_checked.get(k, False)) for k in ("c2_abs", "c2_mask", "c2_core"))
		need_gws = any(
			(
				bool(checked.get("gws_residual", False)),
				bool(checked.get("gws_residual_all", False)),
				bool(checked.get("gws_opening", False)),
				bool(metric_checked.get("gws_support", False)),
				bool(metric_checked.get("gws_context", False)),
				bool(metric_checked.get("gws_width_trace", False)),
				bool(metric_checked.get("gws_area_trace", False)),
			)
		)
		need_edge_levels = any(
			(
				bool(metric_checked.get("show_raw_edge_dense", False)),
				bool(metric_checked.get("show_mg_edge_dense", False)),
				bool(metric_checked.get("show_raw_edge_dense_ctx", False)),
				bool(metric_checked.get("show_rucdw_components", False)),
				bool(metric_checked.get("show_dilation_edge_dense_ctx", False)),
				bool(metric_checked.get("show_mg_edge_dense_ctx", False)),
			)
		)
		need_ball = any(bool(metric_checked.get(k, False)) for k in ("show_raw_ball_descent", "show_mg_ball_descent"))
		need_exp = any(bool(metric_checked.get(k, False)) for k in ("show_raw_exp_fit", "show_mg_exp_fit"))
		need_mdws = any(
			(
				bool(checked.get("mdws510_residual", False)),
				bool(metric_checked.get("mdws510_support", False)),
				bool(metric_checked.get("mdws510_context", False)),
			)
		)

		decisions = list(segment_decisions) if segment_decisions is not None else []
		gws_context_infos = _get_gws_context_infos(y, x, segs, context_mode="segment_width", context_pad_factor=1.0) if need_gws else []
		gws_ctx10_infos = _get_gws_context_infos(y, x, segs, context_mode="fixed_pad", context_fixed_pad=10) if need_gws else []
		for idx, s in enumerate(segs):
			pi = int(np.clip(s.peak_index, 0, N - 1))
			si = int(np.clip(lefts[idx], 0, N - 1))
			ei = int(np.clip(rights[idx], 0, N - 1))
			if ei <= si:
				continue

			raw_seg = raw_sig[si:ei + 1]
			p_rel = int(np.clip(pi - si, 0, raw_seg.size - 1))
			shape_feat: Optional[Dict[str, object]] = None
			shoulder_median: Optional[float] = None

			if need_ls or need_pce:
				shape_feat = compute_local_shape_features(
					raw_seg,
					p_rel,
					core_radius=1,
					center_radius=2,
					shoulder_inner=3,
					shoulder_outer=8,
					curvature_search_radius=3,
				)

				left_sh_start = max(si, pi - 8)
				left_sh_end = max(si, pi - 3 + 1)
				right_sh_start = min(N, pi + 3)
				right_sh_end = min(N, pi + 8 + 1)
				left_sh = raw_sig[left_sh_start:left_sh_end] if left_sh_start < left_sh_end else np.array([], dtype=float)
				right_sh = raw_sig[right_sh_start:right_sh_end] if right_sh_start < right_sh_end else np.array([], dtype=float)
				shoulder_vals = np.concatenate([left_sh, right_sh]) if (left_sh.size or right_sh.size) else np.array([], dtype=float)
				shoulder_median = float(np.median(shoulder_vals)) if shoulder_vals.size >= 2 else float(np.median(raw_seg))

				if need_ls:
					left_anchor_start = max(si, pi - 8)
					left_anchor_end = max(si, pi - 1)
					right_anchor_start = min(N, pi + 2)
					right_anchor_end = min(N, pi + 8 + 1)
					left_anchor = raw_sig[left_anchor_start:left_anchor_end] if left_anchor_start < left_anchor_end else np.array([], dtype=float)
					right_anchor = raw_sig[right_anchor_start:right_anchor_end] if right_anchor_start < right_anchor_end else np.array([], dtype=float)
					if left_anchor.size >= 1 and right_anchor.size >= 1:
						left_idx = int(round(0.5 * (left_anchor_start + left_anchor_end - 1)))
						right_idx = int(round(0.5 * (right_anchor_start + right_anchor_end - 1)))
						if right_idx > left_idx:
							xi = np.arange(left_idx, right_idx + 1, dtype=int)
							t = (xi - left_idx) / max(right_idx - left_idx, 1)
							line_vals = (1.0 - t) * float(np.median(left_anchor)) + t * float(np.median(right_anchor))
							out["ls_anatomy"][xi] = line_vals
							guide_specs.append({"metric": "ls_anatomy", "kind": "span", "x0": float(x_axis[left_sh_start]), "x1": float(x_axis[max(left_sh_start, left_sh_end - 1)]), "color": "#6a3d9a", "alpha": 0.08})
							guide_specs.append({"metric": "ls_anatomy", "kind": "span", "x0": float(x_axis[min(right_sh_start, N - 1)]), "x1": float(x_axis[max(min(right_sh_end - 1, N - 1), min(right_sh_start, N - 1))]), "color": "#6a3d9a", "alpha": 0.08})
							guide_specs.append({"metric": "ls_anatomy", "kind": "hline", "x0": float(x_axis[si]), "x1": float(x_axis[ei]), "y": float(shoulder_median), "color": "#6a3d9a", "alpha": 0.45, "linestyle": ":"})
							guide_specs.append({"metric": "ls_anatomy", "kind": "point", "x": float(x_axis[pi]), "y": float(raw_sig[pi]), "color": "#6a3d9a", "marker": "o", "markersize": 5.0})

					win_l = max(si, pi - 8)
					win_r = min(ei, pi + 8)
					window = raw_sig[win_l:win_r + 1]
					full_pos = np.maximum(window - float(shoulder_median), 0.0)
					if full_pos.size:
						rem_l = max(win_l, pi - 2)
						rem_r = min(win_r, pi + 2)
						keep_mask = np.ones(window.size, dtype=bool)
						keep_mask[(rem_l - win_l):(rem_r - win_l + 1)] = False
						support_line = np.full(window.size, np.nan, dtype=float)
						support_line[keep_mask] = float(shoulder_median) + full_pos[keep_mask]
						out["ls_support"][win_l:win_r + 1] = support_line
						guide_specs.append({"metric": "ls_support", "kind": "span", "x0": float(x_axis[rem_l]), "x1": float(x_axis[rem_r]), "color": "#2ca25f", "alpha": 0.10})
						guide_specs.append({"metric": "ls_support", "kind": "hline", "x0": float(x_axis[win_l]), "x1": float(x_axis[win_r]), "y": float(shoulder_median), "color": "#2ca25f", "alpha": 0.40, "linestyle": ":"})

			if need_ls or need_pce:
				curv_seg = curv_sig[si:ei + 1]
				if curv_seg.size >= 3 and shape_feat is not None:
					d2 = np.diff(curv_seg, n=2)
					seg_slice = slice(si + 1, ei)
					if metric_checked.get("ls_curvature", False):
						out["ls_curvature"][seg_slice] = d2
					if need_pce:
						out["pce_negpref"][seg_slice] = d2
						out["pce_negpref_local"][seg_slice] = d2
					apex_d2_idx = int(np.clip(pi - si - 1, 0, d2.size - 1))
					if metric_checked.get("ls_curvature", False):
						ext_signal_idx = int(np.clip(pi + int(round(float(shape_feat["curvature_extreme_offset"]))), si + 1, ei - 1))
						guide_specs.append({"metric": "ls_curvature", "kind": "point", "x": float(x_axis[pi]), "y": float(shape_feat["curvature_at_apex"]), "color": "#d62728", "marker": "o", "markersize": 5.0})
						guide_specs.append({"metric": "ls_curvature", "kind": "point", "x": float(x_axis[ext_signal_idx]), "y": float(shape_feat["curvature_extreme_centered"]), "color": "#ff7f0e", "marker": "D", "markersize": 4.5})
						search_l = max(si + 1, pi - 3)
						search_r = min(ei - 1, pi + 3)
						guide_specs.append({"metric": "ls_curvature", "kind": "span", "x0": float(x_axis[search_l]), "x1": float(x_axis[search_r]), "color": "#d62728", "alpha": 0.06})
					if need_pce:
						curv_tol = float(CURVATURE_NEGPREF_TOLERANCES[int(curv_variant_state["i"])])
						global_sel = compute_curvature_negpref_diagnostics(curv_seg, peak_rel=pi - si, tolerance=curv_tol, local=False, local_radius=CURVATURE_NEGPREF_LOCAL_RADIUS)
						local_sel = compute_curvature_negpref_diagnostics(curv_seg, peak_rel=pi - si, tolerance=curv_tol, local=True, local_radius=CURVATURE_NEGPREF_LOCAL_RADIUS)
						curv_tag = _tolerance_tag(curv_tol)
						curv_feat = compute_peak_curvature_features(curv_seg, 1.0, peak_rel=pi - si, negpref_local_radius=CURVATURE_NEGPREF_LOCAL_RADIUS)
						for sel_name, sel_diag, metric_key in (
							("global", global_sel, f"peak_curvature_extreme_negpref_{curv_tag}"),
							("local", local_sel, f"peak_curvature_extreme_negpref_local_{curv_tag}"),
						):
							feature_val = float(curv_feat.get(metric_key, np.nan))
							diag_val = float(sel_diag["chosen_value"])
							if np.isfinite(feature_val) and np.isfinite(diag_val) and abs(feature_val - diag_val) > 1e-9:
								src_y, src_x = source_coords_map.get((int(y), int(x)), (int(y), int(x))) if source_coords_map is not None else (int(y), int(x))
								warn_key = ("pce", int(y), int(x), int(pi), int(si), int(ei), curv_tag, sel_name)
								_warn_once(
									pce_warning_cache,
									warn_key,
									f"[viewer:pce-mismatch] compact(y={int(y)}, x={int(x)}) source(y={int(src_y)}, x={int(src_x)}) "
									f"peak_index={int(pi)} seg=[{int(si)},{int(ei)}] tol={curv_tag} curvature_source=gradient "
									f"feature_value={feature_val:.12g} diagnostic_chosen_value={diag_val:.12g}",
								)
						guide_specs.append({"metric": "pce_negpref", "kind": "point", "x": float(x_axis[pi]), "y": float(d2[apex_d2_idx]), "color": "#d62728", "marker": "x", "markersize": 7.0})
						guide_specs.append({"metric": "pce_negpref", "kind": "point", "x": float(x_axis[si + 1 + int(global_sel['base_idx'])]), "y": float(global_sel["base_value"]), "color": "#ff7f0e", "marker": "+", "markersize": 8.0})
						if global_sel["negative_idx"] is not None:
							guide_specs.append({"metric": "pce_negpref", "kind": "point", "x": float(x_axis[si + 1 + int(global_sel['negative_idx'])]), "y": float(global_sel["negative_value"]), "color": "#1f77b4", "marker": "X", "markersize": 7.0})
						guide_specs.append({"metric": "pce_negpref", "kind": "point", "x": float(x_axis[si + 1 + int(global_sel['chosen_idx'])]), "y": float(global_sel["chosen_value"]), "color": "#111111", "marker": (5, 2, 0), "markersize": 10.0})
						guide_specs.append({"metric": "pce_negpref_local", "kind": "point", "x": float(x_axis[pi]), "y": float(d2[apex_d2_idx]), "color": "#d62728", "marker": "x", "markersize": 7.0})
						guide_specs.append({"metric": "pce_negpref_local", "kind": "point", "x": float(x_axis[si + 1 + int(local_sel['base_idx'])]), "y": float(local_sel["base_value"]), "color": "#ff7f0e", "marker": "+", "markersize": 8.0})
						if local_sel["negative_idx"] is not None:
							guide_specs.append({"metric": "pce_negpref_local", "kind": "point", "x": float(x_axis[si + 1 + int(local_sel['negative_idx'])]), "y": float(local_sel["negative_value"]), "color": "#1f77b4", "marker": "X", "markersize": 7.0})
						guide_specs.append({"metric": "pce_negpref_local", "kind": "point", "x": float(x_axis[si + 1 + int(local_sel['chosen_idx'])]), "y": float(local_sel["chosen_value"]), "color": "#444444", "marker": (5, 2, 90), "markersize": 10.0})
						guide_specs.append({"metric": "pce_negpref_local", "kind": "span", "x0": float(x_axis[si + 1 + int(local_sel['local_left_idx'])]), "x1": float(x_axis[si + 1 + int(local_sel['local_right_idx'])]), "color": "#444444", "alpha": 0.08})

			if need_mth or need_c2:
				w = max(3, int(ei - si + 1))
				l0 = max(0, si - w)
				r1 = min(N - 1, ei + w)
				raw_wide = raw_sig[l0:r1 + 1]
				if raw_wide.size >= 5 and need_mth:
					tophat_by_size: Dict[str, np.ndarray] = {}
					for label, size in (("mth_1st", 3), ("mth_2nd", 5), ("mth_3rd", 7)):
						if not metric_checked.get(label, False) and not (label == "mth_1st" and metric_checked.get("mth_decay", False)) and not (label == "mth_3rd" and metric_checked.get("mth_decay", False)):
							continue
						s_eff = int(min(size, max(3, raw_wide.size - (1 - raw_wide.size % 2))))
						if s_eff % 2 == 0:
							s_eff -= 1
						opn = ndimage.grey_opening(raw_wide, size=s_eff)
						res = np.maximum(raw_wide - opn, 0.0)
						tophat_by_size[label] = res
						out[label][l0:r1 + 1] = res
					if metric_checked.get("mth_decay", False) and "mth_1st" in tophat_by_size and "mth_3rd" in tophat_by_size:
						out["mth_decay"][l0:r1 + 1] = tophat_by_size["mth_1st"] / np.maximum(tophat_by_size["mth_3rd"], 1e-12)

				if raw_wide.size >= 7 and need_c2:
					d2w = np.diff(raw_wide, n=2)
					ad2 = np.abs(d2w)
					wide_slice = slice(l0 + 1, r1)
					if metric_checked.get("c2_abs", False):
						out["c2_abs"][wide_slice] = ad2
					med = float(np.median(ad2))
					mad = float(np.median(np.abs(d2w - med)))
					thr = med + max(1.5 * mad, 1e-12)
					if metric_checked.get("c2_mask", False):
						mask = (ad2 >= thr).astype(float)
						out["c2_mask"][wide_slice] = mask
					if metric_checked.get("c2_core", False):
						pk = int(np.clip((pi - l0) - 1, 0, ad2.size - 1))
						core_hw = max(1, int(round(ad2.size * 0.08)))
						la = max(0, pk - core_hw)
						ra = min(ad2.size - 1, pk + core_hw)
						core = np.full(ad2.size, np.nan, dtype=float)
						core[la:ra + 1] = np.nanmax(ad2) if np.isfinite(np.nanmax(ad2)) else 1.0
						out["c2_core"][wide_slice] = core

			seg_for_diag = SpikeSegment(
				y=int(y),
				x=int(x),
				peak_index=int(s.peak_index),
				start=int(lefts[idx]),
				end=int(rights[idx]),
				peak_height=float(s.peak_height),
				area=float(s.area),
			)
			bg_left = raw_sig[max(0, si - max(10, 3 * (ei - si + 1))):si]
			bg_right = raw_sig[ei + 1:min(N, ei + 1 + max(10, 3 * (ei - si + 1)))]
			bg_vals = np.concatenate([bg_left, bg_right]) if (bg_left.size or bg_right.size) else raw_sig
			bg_med = float(np.median(bg_vals)) if bg_vals.size else 0.0
			bg_mad = max(float(np.median(np.abs(bg_vals - bg_med))) if bg_vals.size else 0.0, 1e-12)

			if need_gws:
				gws_context_info = gws_context_infos[idx] if idx < len(gws_context_infos) else None
				gws_diag = _compute_gws_diagnostics(
					int(y), int(x), seg_for_diag,
					context_info=gws_context_info,
					context_mode="segment_width",
					context_pad_factor=1.0,
				)
				seg_decision = decisions[idx] if idx < len(decisions) else {}
				ctx10_context_info = gws_ctx10_infos[idx] if idx < len(gws_ctx10_infos) else None
				ctx10_diag = compute_gws_diagnostics(
					overlays["gradient"][int(y), int(x), :].astype(float) if "gradient" in overlays else raw_sig,
					raw_signal=raw_sig,
					apex_idx=int(seg_for_diag.peak_index),
					detection_left=int(seg_for_diag.start),
					detection_right=int(seg_for_diag.end),
					context_mode="fixed_pad",
					context_fixed_pad=10,
					scales=GWS_SCALES,
					context_info=ctx10_context_info,
					measure_region=str(gws_measure_region),
					threshold_region=str(gws_threshold_region),
				)
				split_applied = float(gws_diag.get("gws_context_split_applied", 0.0)) > 0.0 or float(ctx10_diag.get("gws_context_split_applied", 0.0)) > 0.0
				if (not split_applied) and str(gws_measure_region).strip().lower() == "mask":
					for metric_name, diag_key in (
						("gmt_width_soft_d1_mean", "width_soft_d1_mean"),
						("gws_support01", "gws_support01"),
						("gws_evidence_signed", "gws_evidence_signed"),
					):
						feature_val = float(seg_decision.get(metric_name, np.nan))
						diag_val = float(gws_diag.get(diag_key, np.nan))
						if np.isfinite(feature_val) and np.isfinite(diag_val) and abs(feature_val - diag_val) > 1e-9:
							src_y, src_x = source_coords_map.get((int(y), int(x)), (int(y), int(x))) if source_coords_map is not None else (int(y), int(x))
							_warn_once(
								gws_warning_cache,
								("gws", metric_name, int(y), int(x), int(seg_for_diag.peak_index), int(seg_for_diag.start), int(seg_for_diag.end)),
								f"[viewer:gws-mismatch] compact(y={int(y)}, x={int(x)}) source(y={int(src_y)}, x={int(src_x)}) "
								f"peak_index={int(seg_for_diag.peak_index)} seg=[{int(seg_for_diag.start)},{int(seg_for_diag.end)}] "
								f"scales={list(GWS_SCALES)} diag_{diag_key}={diag_val:.12g} feature_{metric_name}={feature_val:.12g}",
							)
					for metric_name, diag_key in (
						("gmt_width_soft_d1_mean_ctx10", "width_soft_d1_mean"),
						("gmt_width_soft_slope_ctx10", "width_soft_slope"),
						("gws_support01_ctx10", "gws_support01"),
						("gws_evidence_signed_ctx10", "gws_evidence_signed"),
					):
						feature_val = float(seg_decision.get(metric_name, np.nan))
						diag_val = float(ctx10_diag.get(diag_key, np.nan))
						if np.isfinite(feature_val) and np.isfinite(diag_val) and abs(feature_val - diag_val) > 1e-9:
							src_y, src_x = source_coords_map.get((int(y), int(x)), (int(y), int(x))) if source_coords_map is not None else (int(y), int(x))
							_warn_once(
								gws_warning_cache,
								("gws-ctx10", metric_name, int(y), int(x), int(seg_for_diag.peak_index), int(seg_for_diag.start), int(seg_for_diag.end)),
								f"[viewer:gws-ctx10-mismatch] compact(y={int(y)}, x={int(x)}) source(y={int(src_y)}, x={int(src_x)}) "
								f"peak_index={int(seg_for_diag.peak_index)} seg=[{int(seg_for_diag.start)},{int(seg_for_diag.end)}] "
								f"batch_ctx=[{int(ctx10_diag['context_left'])},{int(ctx10_diag['context_right'])}] scales={list(GWS_SCALES)} "
								f"diag_{diag_key}={diag_val:.12g} feature_{metric_name}={feature_val:.12g}",
							)
				ctx_l = int(gws_diag["context_left"])
				ctx_r = int(gws_diag["context_right"])
				grad_ctx = np.asarray(gws_diag["gradient_context"], dtype=float)
				raw_ctx = np.asarray(gws_diag["raw_context"], dtype=float)
				gws_rows = {int(row["scale"]): row for row in gws_diag["rows"]}  # type: ignore[index]
				row = gws_rows.get(active_gws_scale)
				if checked.get("gws_residual", False) and row is not None:
					residual = np.asarray(row["residual"], dtype=float)
					out["gws_residual"][ctx_l:ctx_r + 1] = residual
				if checked.get("gws_opening", False) and row is not None:
					opening = np.asarray(row["opening"], dtype=float)
					out["gws_opening"][ctx_l:ctx_r + 1] = opening
				if checked.get("gws_residual_all", False):
					for sc in GWS_DRAW_SCALES:
						row_all = gws_rows.get(int(sc))
						if row_all is None:
							continue
						residual = np.asarray(row_all["residual"], dtype=float)
						guide_specs.append(
							{
								"metric": "gws_residual_all",
								"kind": "polyline",
								"x": np.asarray(x_axis[ctx_l:ctx_r + 1], dtype=float),
								"y": residual,
								"color": "#9b59b6",
								"alpha": 0.58,
								"linewidth": 1.1,
							}
						)
				if metric_checked.get("gws_context", False):
					guide_specs.append({"metric": "gws_context", "kind": "span", "x0": float(x_axis[ctx_l]), "x1": float(x_axis[ctx_r]), "color": "#7b3294", "alpha": 0.05})
					for split_key in ("gws_split_left_index", "gws_split_right_index"):
						split_idx = gws_diag.get(split_key)
						if split_idx is not None:
							split_idx_int = int(split_idx)
							if 0 <= split_idx_int < len(x_axis):
								guide_specs.append({"metric": "gws_context", "kind": "vline", "x": float(x_axis[split_idx_int]), "color": "#7b3294", "alpha": 0.45, "linestyle": ":"})
				if metric_checked.get("gws_support", False) and row is not None:
					mask = np.asarray(row["support_mask"], dtype=bool)
					if np.any(mask):
						residual = np.asarray(row["residual"], dtype=float)
						idx_local = np.where(mask)[0] + ctx_l
						for rel_idx, p_idx in zip(np.where(mask)[0], idx_local):
							guide_specs.append({"metric": "gws_support", "kind": "point", "x": float(x_axis[int(p_idx)]), "y": float(residual[int(rel_idx)]), "color": "#7b3294", "marker": "o", "markersize": 4.2, "alpha": 0.9})
				if metric_checked.get("gws_width_trace", False):
					scale_arr = np.array([int(row0["scale"]) for row0 in gws_diag["rows"]], dtype=float)  # type: ignore[index]
					width_arr = np.asarray(gws_diag.get("support_counts", gws_diag["width_soft"]), dtype=float)
					if scale_arr.size and width_arr.size:
						x_trace = np.linspace(float(x_axis[ctx_l]), float(x_axis[ctx_r]), scale_arr.size)
						y_trace = _scale_overlay_above_signal(width_arr, grad_ctx if grad_ctx.size else raw_ctx, band_low=0.06, band_high=0.18)
						guide_specs.append({"metric": "gws_width_trace", "kind": "polyline", "x": x_trace, "y": y_trace, "color": "#7b3294", "alpha": 0.95, "linewidth": 1.4})
						for xt, yt, sc in zip(x_trace, y_trace, scale_arr):
							guide_specs.append({"metric": "gws_width_trace", "kind": "point", "x": float(xt), "y": float(yt), "color": "#9b59b6" if int(sc) == active_gws_scale else "#7b3294", "marker": "s", "markersize": 4.0 if int(sc) == active_gws_scale else 3.4})
				if metric_checked.get("gws_area_trace", False):
					scale_arr = np.array([int(row0["scale"]) for row0 in gws_diag["rows"]], dtype=float)  # type: ignore[index]
					area_arr = np.array(
						[float(np.sum(np.asarray(row0["residual"], dtype=float))) for row0 in gws_diag["rows"]],  # type: ignore[index]
						dtype=float,
					)
					if scale_arr.size and area_arr.size:
						x_trace = np.linspace(float(x_axis[ctx_l]), float(x_axis[ctx_r]), scale_arr.size)
						y_trace = _scale_overlay_above_signal(area_arr, grad_ctx if grad_ctx.size else raw_ctx, band_low=0.20, band_high=0.32)
						guide_specs.append({"metric": "gws_area_trace", "kind": "polyline", "x": x_trace, "y": y_trace, "color": "#008080", "alpha": 0.95, "linewidth": 1.4})
						for xt, yt, sc in zip(x_trace, y_trace, scale_arr):
							guide_specs.append({"metric": "gws_area_trace", "kind": "point", "x": float(xt), "y": float(yt), "color": "#16a6a6" if int(sc) == active_gws_scale else "#008080", "marker": "D", "markersize": 4.0 if int(sc) == active_gws_scale else 3.4})

			if need_edge_levels:
				def _draw_dense(metric_name: str, source_sig: np.ndarray, prefix: str, color: str, ctx: bool, linestyle: str) -> None:
					if not metric_checked.get(metric_name, False):
						return
					if bool(edge_use_enhanced_spike_mapping) and (str(prefix).startswith("raw_") or str(prefix).startswith("dilation_")):
						edge_left = int(seg_for_diag.start)
						edge_right = int(seg_for_diag.end)
						edge_prefix = "dilation_edge" if str(prefix).startswith("dilation_") else "raw_edge"
						if bool(ctx):
							edge_pad = max(int(edge_dense_context_pad_pts), int(edge_dense_context_min_pad_pts))
							edge_pad = min(edge_pad, int(edge_dense_context_max_pad_pts))
							edge_left = max(0, edge_left - edge_pad)
							edge_right = min(N - 1, edge_right + edge_pad)
							edge_prefix = "dilation_edge_ctx" if str(prefix).startswith("dilation_") else "raw_edge_ctx"
						edge_result = compute_edge_width_metrics(
							source_sig,
							detection_left=int(edge_left),
							detection_right=int(edge_right),
							prefix=edge_prefix,
							apex_idx=int(getattr(seg_for_diag, "peak_index", getattr(seg_for_diag, "peak", seg_for_diag.start))),
							bg_mad=bg_mad,
							include_low_root_metrics=True,
							low_root_noise_k_mad=float(edge_dense_min_snr),
							use_enhanced_spike_mapping=True,
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
						diag = edge_result.get(f"{edge_prefix}_debug", {})
						if not isinstance(diag, dict):
							return
						mapping = diag.get("edge_mapping", {})
						if not isinstance(mapping, dict):
							return
						ml = int(diag.get("measurement_left", seg_for_diag.start))
						mr = int(diag.get("measurement_right", seg_for_diag.end))
						if 0 <= ml < N and 0 <= mr < N:
							guide_specs.append({"metric": metric_name, "kind": "span", "x0": float(x_axis[ml]), "x1": float(x_axis[mr]), "color": color, "alpha": 0.035})
							ap = int(diag.get("apex", ml))
							if 0 <= ap < N:
								guide_specs.append({"metric": metric_name, "kind": "point", "x": float(x_axis[ap]), "y": float(source_sig[ap]), "color": color, "marker": "*", "markersize": 7.0, "alpha": 0.9})
							local_zero = float(mapping.get("local_zero_level_value", np.nan))
							if np.isfinite(local_zero):
								guide_specs.append({"metric": metric_name, "kind": "hline", "x0": float(x_axis[ml]), "x1": float(x_axis[mr]), "y": local_zero, "color": color, "alpha": 0.45, "linestyle": ":"})
						dense_diag = diag.get("dense_width_0_90", {})
						dense_rows = dense_diag.get("widths", {}) if isinstance(dense_diag, dict) else {}
						dense_levels = dense_diag.get("levels", ()) if isinstance(dense_diag, dict) else ()
						if not dense_levels:
							dense_levels = tuple(range(0, 95, 5))
						dense_reason = str(dense_diag.get("reason", "ok")) if isinstance(dense_diag, dict) else "no_debug"
						dense_valid_n = float(edge_result.get(edge_prefix + "_dense_width_valid_n_0_90", np.nan))
						dense_missing_n = float(edge_result.get(edge_prefix + "_dense_width_missing_n_0_90", np.nan))
						dense_root_snr = float(dense_diag.get("root_snr", np.nan)) if isinstance(dense_diag, dict) else np.nan
						if 0 <= ml < N and 0 <= mr < N:
							reason_y = float(np.nanmax(source_sig[ml:mr + 1])) if np.any(np.isfinite(source_sig[ml:mr + 1])) else float(source_sig[ml])
							short_name = "decdw" if str(prefix).startswith("dilation_") else "recdw"
							guide_specs.append({
								"metric": metric_name,
								"kind": "text",
								"x": float(x_axis[ml]),
								"y": reason_y,
								"color": color,
								"fontsize": 7.0,
								"alpha": 0.9,
								"text": (
									f"{short_name} sum={float(edge_result.get(edge_prefix + '_dense_width_sum_0_90', np.nan)):.3g} "
									f"wlow={float(edge_result.get(edge_prefix + '_dense_width_weighted_low_0_90', np.nan)):.3g} "
									f"rsum={float(edge_result.get(edge_prefix + '_dense_width_ratio_sum_0_90', np.nan)):.3g}\n"
									f"{dense_reason} n={dense_valid_n:.0f}/19 miss={dense_missing_n:.0f} snr={dense_root_snr:.2g}"
								),
							})
						for level_pct in dense_levels:
							level_pct = int(level_pct)
							row = dense_rows.get(str(level_pct), {}) if isinstance(dense_rows, dict) else {}
							if isinstance(row, dict):
								level_y = float(row.get("level", diag.get(f"level_{level_pct}", np.nan)))
								left_rel = float(row.get("left_cross", diag.get(f"left_cross_{level_pct}", np.nan)))
								right_rel = float(row.get("right_cross", diag.get(f"right_cross_{level_pct}", np.nan)))
							else:
								level_y = float(diag.get(f"level_{level_pct}", np.nan))
								left_rel = float(diag.get(f"left_cross_{level_pct}", np.nan))
								right_rel = float(diag.get(f"right_cross_{level_pct}", np.nan))
							if not (np.isfinite(level_y) and np.isfinite(left_rel) and np.isfinite(right_rel)):
								continue
							left_x = _x_from_float_index(left_rel)
							right_x = _x_from_float_index(right_rel)
							alpha = 0.20 + 0.45 * (float(level_pct) / 100.0)
							guide_specs.append({"metric": metric_name, "kind": "hline", "x0": float(min(left_x, right_x)), "x1": float(max(left_x, right_x)), "y": float(level_y), "color": color, "alpha": alpha, "linestyle": linestyle})
							guide_specs.append({"metric": metric_name, "kind": "point", "x": float(left_x), "y": float(level_y), "color": color, "marker": "|", "markersize": 7.0, "alpha": alpha})
							guide_specs.append({"metric": metric_name, "kind": "point", "x": float(right_x), "y": float(level_y), "color": color, "marker": "|", "markersize": 7.0, "alpha": alpha})
						return
					dense_result = compute_edge_dense_width_metrics(
						source_sig,
						detection_left=int(seg_for_diag.start),
						detection_right=int(seg_for_diag.end),
						prefix=prefix,
						bg_mad=bg_mad,
						levels=tuple(int(v) for v in edge_dense_levels),
						context_pad_pts=(int(edge_dense_context_pad_pts) if ctx else 0),
						context_min_pad_pts=(int(edge_dense_context_min_pad_pts) if ctx else 0),
						context_max_pad_pts=int(edge_dense_context_max_pad_pts),
						edge_dense_min_snr=float(edge_dense_min_snr),
					)
					diag = dense_result.get(f"{prefix}_debug", {})
					if not isinstance(diag, dict):
						return
					ml = int(diag.get("measurement_left", seg_for_diag.start))
					mr = int(diag.get("measurement_right", seg_for_diag.end))
					if 0 <= ml < N and 0 <= mr < N:
						guide_specs.append({"metric": metric_name, "kind": "span", "x0": float(x_axis[ml]), "x1": float(x_axis[mr]), "color": color, "alpha": 0.035 if ctx else 0.025})
						label_y = float(np.nanmax(source_sig[ml:mr + 1])) if np.any(np.isfinite(source_sig[ml:mr + 1])) else float(source_sig[ml])
						guide_specs.append({
							"metric": metric_name,
							"kind": "text",
							"x": float(x_axis[ml]),
							"y": label_y,
							"color": color,
							"fontsize": 7.5,
							"alpha": 0.9,
							"text": (
								f"sum={float(dense_result.get(prefix + '_width_sum', np.nan)):.3g} "
								f"L/M/H={float(dense_result.get(prefix + '_width_low_sum', np.nan)):.3g}/"
								f"{float(dense_result.get(prefix + '_width_mid_sum', np.nan)):.3g}/"
								f"{float(dense_result.get(prefix + '_width_high_sum', np.nan)):.3g}"
							),
						})
					level_debug = diag.get("level_debug", {})
					valid_levels = [int(v) for v in diag.get("valid_levels", [])]
					for level_pct in valid_levels[::2]:
						row = level_debug.get(str(level_pct), {}) if isinstance(level_debug, dict) else {}
						try:
							level_y = float(row.get("level_value", np.nan))
							left_cross = _x_from_float_index(float(row.get("left_cross", np.nan)))
							right_cross = _x_from_float_index(float(row.get("right_cross", np.nan)))
						except Exception:
							continue
						if np.isfinite(level_y) and np.isfinite(left_cross) and np.isfinite(right_cross):
							alpha = 0.32 + 0.50 * (float(level_pct) / 100.0)
							guide_specs.append({"metric": metric_name, "kind": "hline", "x0": float(min(left_cross, right_cross)), "x1": float(max(left_cross, right_cross)), "y": float(level_y), "color": color, "alpha": alpha, "linestyle": linestyle})
							guide_specs.append({"metric": metric_name, "kind": "point", "x": float(left_cross), "y": float(level_y), "color": color, "marker": "|", "markersize": 7.0, "alpha": alpha})
							guide_specs.append({"metric": metric_name, "kind": "point", "x": float(right_cross), "y": float(level_y), "color": color, "marker": "|", "markersize": 7.0, "alpha": alpha})

				_draw_dense("show_raw_edge_dense", raw_sig, "raw_edge_dense", "#d62728", False, "-")
				if "gradient" in overlays:
					_draw_dense("show_mg_edge_dense", curv_sig, "mg_edge_dense", "#1f77b4", False, "--")
				_draw_dense("show_raw_edge_dense_ctx", raw_sig, "raw_edge_dense_ctx", "#fb6a4a", True, "-")
				if bool(metric_checked.get("show_rucdw_components", False)) and bool(rucdw_enabled):
					ruc = compute_raw_upper_component_dense_width_metrics(
						raw_sig,
						detection_left=int(seg_for_diag.start),
						detection_right=int(seg_for_diag.end),
						prefix="rucdw",
						bg_mad=bg_mad,
						context_pad_pts=int(rucdw_context_pad_pts),
						context_max_pad_pts=int(rucdw_context_max_pad_pts),
						levels=tuple(int(v) for v in rucdw_levels),
						min_snr=float(rucdw_min_snr),
						noise_fallback_rel_amp=float(rucdw_noise_fallback_rel_amp),
						anchor_mode=str(rucdw_anchor_mode),
						baseline_mode=str(rucdw_baseline_mode),
						baseline_percentile=float(rucdw_baseline_percentile),
					)
					diag = ruc.get("rucdw_debug", {})
					if isinstance(diag, dict):
						cl = int(diag.get("context_left", seg_for_diag.start))
						cr = int(diag.get("context_right", seg_for_diag.end))
						dl = int(diag.get("detection_left", seg_for_diag.start))
						dr = int(diag.get("detection_right", seg_for_diag.end))
						if 0 <= cl < N and 0 <= cr < N:
							guide_specs.append({"metric": "show_rucdw_components", "kind": "span", "x0": float(x_axis[cl]), "x1": float(x_axis[cr]), "color": "#2ca25f", "alpha": 0.035})
						if 0 <= dl < N and 0 <= dr < N:
							guide_specs.append({"metric": "show_rucdw_components", "kind": "span", "x0": float(x_axis[dl]), "x1": float(x_axis[dr]), "color": "#2ca25f", "alpha": 0.055})
						anchor = int(diag.get("anchor_index", int(seg_for_diag.peak_index)))
						if 0 <= anchor < N:
							guide_specs.append({"metric": "show_rucdw_components", "kind": "point", "x": float(x_axis[anchor]), "y": float(raw_sig[anchor]), "color": "#2ca25f", "marker": "*", "markersize": 7.0, "alpha": 0.95})
						comps = diag.get("components", {})
						if isinstance(comps, dict):
							for level_key in sorted(comps.keys(), key=lambda s: int(s)):
								pct = int(level_key)
								if pct % 10 != 0 and pct not in {5, 25, 75, 85}:
									continue
								row = comps.get(level_key, {})
								if not isinstance(row, dict):
									continue
								level_y = float(row.get("level_value", np.nan))
								c_l = int(row.get("component_left", -1))
								c_r = int(row.get("component_right", -1))
								if not (np.isfinite(level_y) and 0 <= c_l < N and 0 <= c_r < N):
									continue
								alpha = 0.22 + 0.45 * (float(pct) / 100.0)
								guide_specs.append({"metric": "show_rucdw_components", "kind": "hline", "x0": float(x_axis[c_l]), "x1": float(x_axis[c_r]), "y": float(level_y), "color": "#2ca25f", "alpha": alpha, "linestyle": "-"})
								guide_specs.append({"metric": "show_rucdw_components", "kind": "point", "x": float(x_axis[c_l]), "y": float(level_y), "color": "#2ca25f", "marker": "|", "markersize": 7.0, "alpha": alpha})
								guide_specs.append({"metric": "show_rucdw_components", "kind": "point", "x": float(x_axis[c_r]), "y": float(level_y), "color": "#2ca25f", "marker": "|", "markersize": 7.0, "alpha": alpha})
						if 0 <= cl < N and 0 <= cr < N:
							label_y = float(np.nanmax(raw_sig[cl:cr + 1])) if np.any(np.isfinite(raw_sig[cl:cr + 1])) else float(raw_sig[cl])
							guide_specs.append({
								"metric": "show_rucdw_components",
								"kind": "text",
								"x": float(x_axis[cl]),
								"y": label_y,
								"color": "#2ca25f",
								"fontsize": 7.0,
								"alpha": 0.9,
								"text": f"rucdw sum={float(ruc.get('rucdw_sum_0_90', np.nan)):.3g} n={float(ruc.get('rucdw_valid_n_0_90', np.nan)):.0f}",
							})
				if "dilation" in overlays:
					dil_sig = overlays["dilation"][y, x, :].astype(float)
					_draw_dense("show_dilation_edge_dense_ctx", dil_sig, "dilation_edge_dense_ctx", "#8c564b", True, "-.")
				if "gradient" in overlays:
					_draw_dense("show_mg_edge_dense_ctx", curv_sig, "mg_edge_dense_ctx", "#6baed6", True, "--")

			if need_ball:
				for metric_name, source_sig, prefix, color in (
					("show_raw_ball_descent", raw_sig, "raw_ball", "#e6550d"),
					("show_mg_ball_descent", curv_sig, "mg_ball", "#3182bd"),
				):
					if not metric_checked.get(metric_name, False):
						continue
					diag = compute_ball_descent_metrics(
						source_sig,
						apex_idx=int(seg_for_diag.peak_index),
						detection_left=int(seg_for_diag.start),
						detection_right=int(seg_for_diag.end),
						prefix=prefix,
						bg_mad=bg_mad,
						ball_noise_k_mad=float(ball_stop_k_mad),
						ball_stop_rel_amp=float(ball_stop_rel_amp),
						context_pad_pts=int(ball_context_pad_pts),
						context_min_pad_pts=int(ball_context_min_pad_pts),
						context_max_pad_pts=int(ball_context_max_pad_pts),
						prevent_crossing_neighbor_peak=bool(ball_prevent_crossing_neighbor_peak),
					).get(f"{prefix}_debug", {})
					if not isinstance(diag, dict):
						continue
					ctx_l = int(diag.get("ball_context_left", seg_for_diag.start))
					ctx_r = int(diag.get("ball_context_right", seg_for_diag.end))
					if 0 <= ctx_l < N and 0 <= ctx_r < N:
						guide_specs.append({"metric": metric_name, "kind": "span", "x0": float(x_axis[ctx_l]), "x1": float(x_axis[ctx_r]), "color": color, "alpha": 0.035})
					apex_i = int(diag.get("apex", seg_for_diag.peak_index))
					if 0 <= apex_i < N:
						guide_specs.append({"metric": metric_name, "kind": "point", "x": float(x_axis[apex_i]), "y": float(source_sig[apex_i]), "color": color, "marker": "o", "markersize": 5.0, "alpha": 0.95})
					y_stop = float(diag.get("y_stop", np.nan))
					if np.isfinite(y_stop):
						guide_specs.append({"metric": metric_name, "kind": "hline", "x0": float(x_axis[si]), "x1": float(x_axis[ei]), "y": y_stop, "color": color, "alpha": 0.35, "linestyle": ":"})
					for side in ("left", "right"):
						path = [int(v) for v in diag.get(f"{side}_path_indices", []) if 0 <= int(v) < N]
						if path:
							guide_specs.append({"metric": metric_name, "kind": "polyline", "x": np.asarray(x_axis[path], dtype=float), "y": np.asarray(source_sig[path], dtype=float), "color": color, "alpha": 0.90, "linewidth": 1.5})
						stop_i = int(diag.get(f"{side}_stop_index", -1))
						if 0 <= stop_i < N:
							guide_specs.append({"metric": metric_name, "kind": "point", "x": float(x_axis[stop_i]), "y": float(source_sig[stop_i]), "color": color, "marker": "s", "markersize": 4.5, "alpha": 0.95})

			if need_exp:
				for metric_name, source_sig, prefix, color in (
					("show_raw_exp_fit", raw_sig, "raw_exp", "#a63603"),
					("show_mg_exp_fit", curv_sig, "mg_exp", "#08519c"),
				):
					if not metric_checked.get(metric_name, False):
						continue
					diag = compute_exponential_decay_metrics(
						source_sig,
						apex_idx=int(seg_for_diag.peak_index),
						detection_left=int(seg_for_diag.start),
						detection_right=int(seg_for_diag.end),
						prefix=prefix,
						bg_mad=bg_mad,
						exp_fit_noise_k_mad=float(exp_foot_noise_k_mad),
						context_pad_pts=int(exp_context_pad_pts),
						exp_foot_low_rel=float(exp_foot_low_rel),
						exp_foot_high_rel=float(exp_foot_high_rel),
						exp_min_points=int(exp_min_points),
						exp_prevent_apex_region=bool(exp_prevent_apex_region),
					).get(f"{prefix}_debug", {})
					if not isinstance(diag, dict):
						continue
					ctx_l = int(diag.get("exp_context_left", seg_for_diag.start))
					ctx_r = int(diag.get("exp_context_right", seg_for_diag.end))
					if 0 <= ctx_l < N and 0 <= ctx_r < N:
						guide_specs.append({"metric": metric_name, "kind": "span", "x0": float(x_axis[ctx_l]), "x1": float(x_axis[ctx_r]), "color": color, "alpha": 0.028})
					for side in ("left", "right"):
						fit_idx = [int(v) for v in diag.get(f"{side}_fit_indices", []) if 0 <= int(v) < N]
						fit_vals = [float(v) for v in diag.get(f"{side}_fit_values", [])]
						if fit_idx and len(fit_idx) == len(fit_vals):
							guide_specs.append({"metric": metric_name, "kind": "polyline", "x": np.asarray(x_axis[fit_idx], dtype=float), "y": np.asarray(fit_vals, dtype=float), "color": color, "alpha": 0.95, "linewidth": 1.5})
						used_idx = [int(v) for v in diag.get(f"{side}_indices", []) if 0 <= int(v) < N]
						for used_i in used_idx:
							guide_specs.append({"metric": metric_name, "kind": "point", "x": float(x_axis[used_i]), "y": float(source_sig[used_i]), "color": color, "marker": ".", "markersize": 4.0, "alpha": 0.60})

			if need_mdws:
				mdws_diag = _compute_mdws510_diagnostics(int(y), int(x), seg_for_diag)
				md_ctx_l = int(mdws_diag["context_left"])
				md_ctx_r = int(mdws_diag["context_right"])
				md_raw_ctx = np.asarray(mdws_diag["raw_context"], dtype=float)
				md_residual = np.asarray(mdws_diag["median_residual"], dtype=float)
				if checked.get("mdws510_residual", False):
					out["mdws510_residual"][md_ctx_l:md_ctx_r + 1] = _scale_overlay_to_raw_panel(md_residual, md_raw_ctx, band_low=0.56, band_high=0.76)
				if metric_checked.get("mdws510_context", False):
					guide_specs.append({"metric": "mdws510_context", "kind": "span", "x0": float(x_axis[md_ctx_l]), "x1": float(x_axis[md_ctx_r]), "color": "#1b9e77", "alpha": 0.05})
				if metric_checked.get("mdws510_support", False):
					mask = np.asarray(mdws_diag["support_mask"], dtype=bool)
					if np.any(mask):
						idx_local = np.where(mask)[0] + md_ctx_l
						for p_idx in idx_local:
							guide_specs.append({"metric": "mdws510_support", "kind": "point", "x": float(x_axis[int(p_idx)]), "y": float(raw_sig[int(p_idx)]), "color": "#1b9e77", "marker": "^", "markersize": 4.0})

		interest_metric_cache[cache_key] = (out, guide_specs)  # type: ignore[assignment]
		return out, guide_specs

	def _update(y: int, x: int) -> None:
		y = int(np.clip(y, 0, H - 1))
		x = int(np.clip(x, 0, W - 1))
		current["y"] = y
		current["x"] = x

		marker.set_offsets([[x, y]])

		pixel_signals = _get_pixel_signals(y, x)
		raw = pixel_signals['raw']
		diag_state["median_window"] = _nearest_valid_window(raw.size, median_window_choices, int(diag_state["median_window"]))
		diag_state["mean_window"] = _nearest_valid_window(raw.size, mean_window_choices, int(diag_state["mean_window"]))
		diag_state["opening_window"] = _nearest_valid_window(raw.size, opening_window_choices, int(diag_state["opening_window"]))
		top_hat_state["window"] = _nearest_valid_window(raw.size, top_hat_window_choices, int(top_hat_state["window"]))
		ln_raw.set_data(x_axis, pixel_signals['raw'])
		ln_open.set_data(x_axis, pixel_signals['opening'])
		ln_ero.set_data(x_axis, pixel_signals['erosion'])
		ln_dil.set_data(x_axis, pixel_signals['dilation'])
		top_hat_key = _current_top_hat_key()
		ln_th.set_data(x_axis, pixel_signals.get(top_hat_key, pixel_signals['top_hat']))
		ln_th.set_label(f"top_hat_w{_current_top_hat_window()}")
		if "gradient" in pixel_signals:
			ln_grad.set_data(x_axis, pixel_signals['gradient'])
		else:
			ln_grad.set_data([], [])
		ln_raw_d2.set_data(x_axis, pixel_signals.get('raw_d2', np.full(N, np.nan, dtype=float)))
		ln_raw_d3.set_data(x_axis, pixel_signals.get('raw_d3', np.full(N, np.nan, dtype=float)))
		ln_median.set_data(x_axis, pixel_signals.get(f"median_w{int(diag_state['median_window'])}", np.full(N, np.nan, dtype=float)))
		ln_median_res.set_data(x_axis, pixel_signals.get(f"median_residual_w{int(diag_state['median_window'])}", np.full(N, np.nan, dtype=float)))
		ln_mean.set_data(x_axis, pixel_signals.get(f"mean_w{int(diag_state['mean_window'])}", np.full(N, np.nan, dtype=float)))
		ln_mean_res.set_data(x_axis, pixel_signals.get(f"mean_residual_w{int(diag_state['mean_window'])}", np.full(N, np.nan, dtype=float)))
		ln_open_diag.set_data(x_axis, pixel_signals.get(f"opening_w{int(diag_state['opening_window'])}", np.full(N, np.nan, dtype=float)))
		ln_open_res.set_data(x_axis, pixel_signals.get(f"opening_residual_w{int(diag_state['opening_window'])}", np.full(N, np.nan, dtype=float)))
		if "dilation_minus_opening" in pixel_signals:
			ln_dmo.set_data(x_axis, pixel_signals["dilation_minus_opening"])
		else:
			ln_dmo.set_data([], [])
		if "corrected" in pixel_signals:
			ln_corr.set_data(x_axis, pixel_signals["corrected"])
		else:
			ln_corr.set_data([], [])

		# clear old spike overlays
		for col in (spike_peak_lines, spike_edge_lines, spike_bands):
			for artist in col:
				try:
					artist.remove()
				except Exception:
					pass
			col.clear()
		for artist in metric_guide_artists:
			try:
				artist.remove()
			except Exception:
				pass
		metric_guide_artists.clear()

		raw_segs = view_spikes_by_pixel.get((y, x), [])
		need_spike_overlays = any(bool(v) for v in spike_overlay_checked.values())
		metric_active = any(bool(metric_checked[k]) for k in metric_checked)
		need_segment_diagnostics = bool(raw_segs) and (need_spike_overlays or metric_active)
		segs: List[SpikeSegment]
		lefts: List[int]
		rights: List[int]
		if need_segment_diagnostics:
			segs, lefts, rights = _prepare_pixel_segments(y, x, raw_segs)
			segment_decisions = []
		else:
			segs, lefts, rights = list(raw_segs), [], []
			segment_decisions = []
		need_segments = bool(segs)
		prepared_candidate_count = len(segs) if need_segment_diagnostics else (len(_prepare_pixel_segments(y, x, raw_segs)[0]) if raw_segs else 0)

		if need_spike_overlays and ((need_segments and lefts and rights) or contact_parent_by_pixel.get((int(y), int(x)))):
			drawn_primary_peaks: set[int] = set()
			for idx, s in enumerate(segs if (need_segments and lefts and rights) else []):
				pi = int(np.clip(s.peak_index, 0, len(x_axis) - 1))
				si = int(np.clip(lefts[idx], 0, len(x_axis) - 1))
				ei = int(np.clip(rights[idx], 0, len(x_axis) - 1))
				x_peak = x_axis[pi]
				x_start = x_axis[si]
				x_end = x_axis[ei]
				is_primary_ss4 = _is_primary_ss4_candidate(int(y), int(x), int(pi), int(si), int(ei))
				decision_color = "#d62728" if is_primary_ss4 else "#1f77b4"
				boundary_color = "#2ca02c"

				if spike_overlay_checked['spike_bands']:
					x0 = min(float(x_start), float(x_end))
					x1 = max(float(x_start), float(x_end))
					band = ax_spec.axvspan(x0, x1, color=boundary_color, alpha=0.12, zorder=0)
					spike_bands.append(band)

				if spike_overlay_checked['spike_edges']:
					le = ax_spec.axvline(x_start, linestyle="--", linewidth=1, color=boundary_color, alpha=0.9)
					re = ax_spec.axvline(x_end, linestyle="--", linewidth=1, color=boundary_color, alpha=0.9)
					spike_edge_lines.extend([le, re])

				if spike_overlay_checked['spike_peaks']:
					lp = ax_spec.axvline(x_peak, linestyle="--", linewidth=1, color=decision_color, alpha=0.9)
					spike_peak_lines.append(lp)
					if is_primary_ss4:
						drawn_primary_peaks.add(int(pi))
			if spike_overlay_checked['spike_peaks']:
				for parent in contact_parent_by_pixel.get((int(y), int(x)), []):
					try:
						pi = int(np.clip(int(parent.get("parent_apex")), 0, len(x_axis) - 1))
					except Exception:
						continue
					if pi in drawn_primary_peaks:
						continue
					lp = ax_spec.axvline(float(x_axis[pi]), linestyle="--", linewidth=1.25, color="#d62728", alpha=0.95)
					spike_peak_lines.append(lp)
					drawn_primary_peaks.add(pi)

		has_contact_diag = bool(contact_parent_by_pixel.get((int(y), int(x)), []))
		has_primary_metrics = bool(ss4_metrics_by_pixel.get((int(y), int(x)), []))
		metric_overlay, metric_guides = _build_metric_overlays(y, x, segs, lefts, rights, segment_decisions) if (metric_active and (segs or has_contact_diag or has_primary_metrics)) else (None, [])
		if metric_overlay is None:
			ln_gws_residual.set_data([], [])
			ln_gws_residual_all.set_data([], [])
			ln_gws_opening.set_data([], [])
			ln_mdws_residual.set_data([], [])
		else:
			ln_gws_residual.set_data(x_axis, metric_overlay.get("gws_residual", np.full(N, np.nan, dtype=float)))
			ln_gws_residual_all.set_data(x_axis, metric_overlay.get("gws_residual_all", np.full(N, np.nan, dtype=float)))
			ln_gws_opening.set_data(x_axis, metric_overlay.get("gws_opening", np.full(N, np.nan, dtype=float)))
			ln_mdws_residual.set_data(x_axis, metric_overlay.get("mdws510_residual", np.full(N, np.nan, dtype=float)))
		for nm, ln in metric_lines.items():
			if metric_overlay is None:
				ln.set_data([], [])
			else:
				ln.set_data(x_axis, metric_overlay.get(nm, np.full(N, np.nan, dtype=float)))
		for spec in metric_guides:
			metric_name = str(spec.get("metric", ""))
			if not metric_checked.get(metric_name, False) and not checked.get(metric_name, False):
				continue
			kind = str(spec.get("kind", ""))
			if kind == "span":
				artist = ax_spec.axvspan(float(spec["x0"]), float(spec["x1"]), color=str(spec["color"]), alpha=float(spec["alpha"]), zorder=0)
			elif kind == "hline":
				artist = ax_spec.hlines(
					y=float(spec["y"]),
					xmin=float(spec["x0"]),
					xmax=float(spec["x1"]),
					colors=str(spec["color"]),
					alpha=float(spec["alpha"]),
					linestyles=str(spec["linestyle"]),
					linewidth=1.2,
				)
			elif kind == "point":
				artist = ax_spec.plot(
					[float(spec["x"])],
					[float(spec["y"])],
					linestyle="None",
					marker=spec["marker"],
					markersize=float(spec["markersize"]),
					color=str(spec["color"]),
					alpha=float(spec.get("alpha", 1.0)),
				)[0]
			elif kind == "vline":
				artist = ax_spec.axvline(
					float(spec["x"]),
					color=str(spec["color"]),
					alpha=float(spec["alpha"]),
					linestyle=str(spec["linestyle"]),
					linewidth=1.1,
				)
			elif kind == "polyline":
				artist = ax_spec.plot(
					np.asarray(spec["x"], dtype=float),
					np.asarray(spec["y"], dtype=float),
					color=str(spec["color"]),
					alpha=float(spec.get("alpha", 1.0)),
					linewidth=float(spec.get("linewidth", 1.0)),
					linestyle=str(spec.get("linestyle", "-")),
				)[0]
			elif kind == "text":
				artist = ax_spec.text(
					float(spec["x"]),
					float(spec["y"]),
					str(spec.get("text", "")),
					color=str(spec.get("color", "#000000")),
					fontsize=float(spec.get("fontsize", 8.0)),
					alpha=float(spec.get("alpha", 0.9)),
				)
			else:
				continue
			metric_guide_artists.append(artist)

		header = ""
		diag_info = ""
		metric_info = ""
		if metric_active and segs:
			parents_here = contact_parent_by_pixel.get((int(y), int(x)), [])
			if parents_here:
				lead_parent = max(parents_here, key=lambda p: float(p.get("parent_peak_height", 0.0) or 0.0))
				metric_info = (
					f"primary_ss4={_fmt_num(lead_parent.get('parent_ss4_value'))} "
					f"reason={lead_parent.get('parent_ss4_reason', '')} | "
					f"ss1={_fmt_num(lead_parent.get('parent_ss1'))} "
					f"pce={_fmt_num(lead_parent.get('parent_pce'))} "
					f"edge={_fmt_num(lead_parent.get('parent_edge'))}"
				)
			else:
				lead = max(segs, key=lambda sp: float(getattr(sp, "peak_height", 0.0)))
				lead_idx = int(segs.index(lead))
				metric_info = (
					f"cand={lead_idx} primary_ss4=not_selected | "
					"pipeline primary record unavailable"
				)
		active_diag = [
			name for name in (
				"pce_negpref", "pce_negpref_local",
				"show_raw_edge_dense_ctx",
				"show_erosion_contacts", "show_dilation_contacts", "show_contact_cells", "show_cell_salience_labels", "show_contact_cell_chords",
				"secondary_spike_peaks", "show_secondary_ss4_labels",
				"despike_chords", "located_muon", "primal_metrics",
			)
			if bool(checked.get(name, False)) or bool(metric_checked.get(name, False))
		]
		if active_diag:
			diag_info = f"diag={','.join(active_diag)}"
		curv_label = curv_tolerance_tags[int(curv_variant_state['i'])]
		if source_coords_map is not None:
			src_y, src_x = source_coords_map.get((y, x), (y, x))
			header = f"spectrum @ compact(y={y}, x={x}) -> source(y={src_y}, x={src_x}) | candidates={prepared_candidate_count}"
		else:
			header = f"spectrum @ (y={y}, x={x}) | candidates={prepared_candidate_count}"
		spec_info_head.set_text(header)
		if need_segment_diagnostics and segs:
			lead = max(segs, key=lambda sp: float(getattr(sp, "peak_height", 0.0)))
			lead_idx = int(segs.index(lead))
			lead_left = int(lefts[lead_idx]) if lead_idx < len(lefts) else int(lead.start)
			lead_right = int(rights[lead_idx]) if lead_idx < len(rights) else int(lead.end)
			is_primary_ss4 = _is_primary_ss4_candidate(int(y), int(x), int(lead.peak_index), lead_left, lead_right)
			spec_info_head.set_color("#d62728" if is_primary_ss4 else "#1f77b4")
		else:
			spec_info_head.set_color("#000000")
		spec_info_diag.set_text(
			f"curv={curv_label} | th_w={_current_top_hat_window()} | med_w={int(diag_state['median_window'])} | mean_w={int(diag_state['mean_window'])}"
			+ (f" | {diag_info}" if diag_info else "")
		)
		spec_info_metric.set_text(metric_info)
		_apply_dynamic_ylim()

		_redraw_dynamic(use_fast=True)

	def on_move(event) -> None:
		if frozen["state"]:
			return
		if event.inaxes != ax_map:
			return
		if event.xdata is None or event.ydata is None:
			return
		x = int(round(event.xdata))
		y = int(round(event.ydata))

		if (x, y) == hover_state["last_xy"]:
			return
		now = time.perf_counter()
		if hover_fps > 0:
			min_dt = 1.0 / float(hover_fps)
			if (now - hover_state['last_t']) < min_dt:
				return
		hover_state['last_t'] = now
		hover_state['last_xy'] = (x, y)
		_update(y, x)

	def on_click(event) -> None:
		if event.inaxes == ax_txt_x:
			_set_focus("x")
			if getattr(event, "dblclick", False):
				focus['replace_x'] = True
			return
		if event.inaxes == ax_txt_y:
			_set_focus("y")
			if getattr(event, "dblclick", False):
				focus['replace_y'] = True
			return
		if event.inaxes == ax_btn_go:
			_set_focus("go")
			return

		if event.inaxes != ax_map:
			return
		if event.button != 3:
			return
		frozen["state"] = not frozen["state"]

	def _toggle_line(label: str) -> None:
		if label in checked:
			checked[label] = not checked[label]
		elif label in spike_overlay_checked:
			spike_overlay_checked[label] = not spike_overlay_checked[label]
		elif label in metric_checked:
			metric_checked[label] = not metric_checked[label]
		interest_metric_cache.clear()
		_apply_signal_visibility()
		_apply_metric_visibility()
		_refresh_legend()
		_update(current["y"], current["x"])
	ax_txt_y = fig.add_axes((0.08, 0.08, 0.08, 0.045))
	ax_txt_x = fig.add_axes((0.18, 0.08, 0.08, 0.045))
	ax_btn_go = fig.add_axes((0.28, 0.08, 0.10, 0.045))
	ax_chk_1 = fig.add_axes((0.42, 0.03, 0.105, 0.20))
	ax_chk_2 = fig.add_axes((0.535, 0.03, 0.105, 0.20))
	ax_chk_3 = fig.add_axes((0.650, 0.03, 0.105, 0.20))
	ax_chk_4 = fig.add_axes((0.765, 0.03, 0.105, 0.20))
	ax_chk_5 = fig.add_axes((0.880, 0.03, 0.105, 0.20))
	txt_y = TextBox(ax_txt_y, "y", initial="0")
	txt_x = TextBox(ax_txt_x, "x", initial="0")
	btn_go = Button(ax_btn_go, "Go to (y,x)")
	txt_y.set_active(False)
	txt_x.set_active(False)

	all_labels = [
		"raw", "raw_d3", "opening", "erosion", "dilation", "top_hat", "gradient", "corrected",
		"median", "mean",
		"spike_peaks", "spike_edges", "spike_bands",
		"pce_negpref", "pce_negpref_local",
		"show_raw_edge_dense_ctx",
		"show_erosion_contacts", "show_dilation_contacts", "show_contact_cells", "show_cell_salience_labels", "show_contact_cell_chords",
		"secondary_spike_peaks", "show_secondary_ss4_labels",
		"despike_chords", "located_muon", "primal_metrics",
	]
	all_actives = [
		checked['raw'], checked['raw_d3'], checked['opening'], checked['erosion'], checked['dilation'], checked['top_hat'], checked['gradient'], checked['corrected'],
		checked['median'], checked['mean'],
		spike_overlay_checked['spike_peaks'], spike_overlay_checked['spike_edges'], spike_overlay_checked['spike_bands'],
		metric_checked['pce_negpref'], metric_checked['pce_negpref_local'],
		metric_checked['show_raw_edge_dense_ctx'],
		metric_checked['show_erosion_contacts'], metric_checked['show_dilation_contacts'], metric_checked['show_contact_cells'], metric_checked['show_cell_salience_labels'], metric_checked['show_contact_cell_chords'],
		metric_checked['secondary_spike_peaks'], metric_checked['show_secondary_ss4_labels'],
		metric_checked['despike_chords'], metric_checked['located_muon'], metric_checked['primal_metrics'],
	]
	chunk = int(np.ceil(len(all_labels) / 5))
	check_axes = [ax_chk_1, ax_chk_2, ax_chk_3, ax_chk_4, ax_chk_5]
	check_widgets = []
	for idx, ax_chk in enumerate(check_axes):
		start = idx * chunk
		stop = min(len(all_labels), (idx + 1) * chunk)
		chk = CheckButtons(ax_chk, labels=all_labels[start:stop], actives=all_actives[start:stop])
		chk.on_clicked(_toggle_line)
		ax_chk.set_facecolor("#fafafa")
		check_widgets.append(chk)
	ax_chk_1.set_title("signals / overlays", fontsize=10)
	for chk in check_widgets:
		for lbl in list(chk.labels):
			lbl.set_fontsize(7)
			lbl.set_wrap(True)

	def _go_to_xy(_event=None) -> None:
		try:
			y = int(float(input_buffers["y"].strip()))
			x = int(float(input_buffers["x"].strip()))
		except Exception:
			return

		x = int(np.clip(x, 0, W - 1))
		y = int(np.clip(y, 0, H - 1))
		input_buffers["x"] = str(x)
		input_buffers["y"] = str(y)
		hover_state["last_xy"] = (x, y)
		_render_inputs()
		_update(y, x)

	def on_key(event) -> None:
		key_raw = str(event.key or "")
		key = key_raw.lower()
		shift_pressed = ("shift+" in key) or (len(key_raw) == 1 and key_raw.isalpha() and key_raw != key_raw.lower())
		if key == "tab":
			order = ["y", "x", "go"]
			i = order.index(focus['which']) if focus['which'] in order else -1
			_set_focus(order[(i + 1) % len(order)])
			return
		if key == "ctrl+a":
			if focus['which'] == "x":
				focus['replace_x'] = True
			elif focus['which'] == "y":
				focus['replace_y'] = True
			return
		if key in ("backspace", "delete"):
			if focus['which'] == "x" and focus['replace_x']:
				input_buffers["x"] = ""
				_render_inputs()
				focus['replace_x'] = False
				return
			if focus['which'] == "y" and focus['replace_y']:
				input_buffers["y"] = ""
				_render_inputs()
				focus['replace_y'] = False
				return
		if key in ("enter", "return") and focus['which'] == "go":
			_go_to_xy()
			return

		if focus['which'] == "x" and focus['replace_x'] and len(key) == 1 and key.isprintable():
			input_buffers["x"] = ""
			_render_inputs()
			focus['replace_x'] = False
		if focus['which'] == "y" and focus['replace_y'] and len(key) == 1 and key.isprintable():
			input_buffers["y"] = ""
			_render_inputs()
			focus['replace_y'] = False

		if focus['which'] in ("x", "y"):
			if len(key) == 1 and key.isprintable():
				input_buffers[focus['which']] += key
				_render_inputs()
				return
			if key == "backspace":
				input_buffers[focus['which']] = input_buffers[focus['which']][:-1]
				_render_inputs()
				return
		if key in ("m", "shift+m"):
			step = -1 if shift_pressed else 1
			diag_state["median_window"] = _cycle_choice(int(diag_state["median_window"]), median_window_choices, step, N)
			_update(current["y"], current["x"])
			return
		if key in ("n", "shift+n"):
			step = -1 if shift_pressed else 1
			diag_state["mean_window"] = _cycle_choice(int(diag_state["mean_window"]), mean_window_choices, step, N)
			_update(current["y"], current["x"])
			return
		if key in ("o", "shift+o"):
			step = -1 if shift_pressed else 1
			diag_state["opening_window"] = _cycle_choice(int(diag_state["opening_window"]), opening_window_choices, step, N)
			_update(current["y"], current["x"])
			return
		if key in ("t", "shift+t"):
			step = -1 if shift_pressed else 1
			top_hat_state["window"] = _cycle_choice(int(top_hat_state["window"]), top_hat_window_choices, step, N)
			_refresh_legend()
			_update(current["y"], current["x"])
			return
		if key in ("c", "shift+c"):
			step = -1 if shift_pressed else 1
			curv_variant_state["i"] = (int(curv_variant_state["i"]) + step) % len(curv_tolerance_tags)
			_refresh_legend()
			_update(current["y"], current["x"])
			return

	def _sync_inputs() -> None:
		input_buffers["x"] = str(current["x"])
		input_buffers["y"] = str(current["y"])
		_render_inputs()

	btn_go.on_clicked(_go_to_xy)

	fig.canvas.mpl_connect("motion_notify_event", on_move)
	fig.canvas.mpl_connect("button_press_event", on_click)
	fig.canvas.mpl_connect("key_press_event", on_key)

	# init
	_apply_signal_visibility()
	_apply_metric_visibility()
	_refresh_legend()
	_update(0, 0)
	_sync_inputs()
	plt.show()
