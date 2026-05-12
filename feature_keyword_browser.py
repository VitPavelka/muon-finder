from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from candidate_labels import LABEL_CLASS_COLORS, TERNARY_CLASSES, load_binary_labels, load_label_classes
from feature_discrimination import GWS_SOURCE_PREFIX_BY_MODE
from feature_discrimination import TH_SHAPE_ACTIVE_METRICS
from feature_discrimination import TOPHAT_GRADIENT_EXPORT_FEATURES


GWS_KEYWORD_METRICS = {
	name
	for name in TOPHAT_GRADIENT_EXPORT_FEATURES
	if str(name).startswith("gws_")
}
GWS_KEYWORD_METRICS.update(
	{
		"gmt_width_soft_d1_mean",
		"gmt_width_soft_slope",
		"gws_support01",
		"gws_evidence_signed",
	}
)
GWS_IMP_KEYWORD_METRICS = {
	"gws_context_overlap_count",
	"gws_context_overlap_points",
	"gws_context_overlap_fraction",
	"gws_context_split_applied",
	"gws_context_split_left_applied",
	"gws_context_split_right_applied",
	"gws_context_valley_depth_ratio_min",
	"gws_context_valley_depth_ratio_left",
	"gws_context_valley_depth_ratio_right",
	"gws_context_original_width",
	"gws_context_adjusted_width",
	"gws_context_width_reduction_fraction",
	"gws_width_longest_constant_run",
	"gws_width_longest_constant_run_norm",
	"gws_width_num_changes",
	"gws_width_change_density",
	"gws_width_total_variation",
	"gws_width_total_variation_norm",
	"gws_width_longest_stable_run_tol1",
	"gws_width_longest_stable_run_tol1_norm",
	"gws_width_num_changes_tol1",
	"gws_width_change_density_tol1",
	"gws_width_stable_suffix_len_tol1",
	"gws_width_stable_suffix_len_tol1_norm",
	"gws_width_last_change_index_tol1",
	"gws_width_last_change_scale_norm_tol1",
}
GWS_KEYWORD_PREFIXES = (
	"gws_",
	"gmt_width_soft_",
)
GWS_SUPPORT_KEYWORD_PREFIXES = (
	"gws_support",
)
GWS_SUPPORT_KEYWORD_METRICS = {
	"gws_support01",
	"gws_evidence_signed",
	"gws_support_num_changes",
	"gws_support_change_density",
	"gws_support_total_abs_change",
	"gws_support_max_abs_change",
	"gws_support_num_increases",
	"gws_support_num_decreases",
	"gws_support_total_increase",
	"gws_support_total_decrease",
	"gws_support_longest_constant_run",
	"gws_support_longest_constant_run_norm",
	"gws_support_initial",
	"gws_support_final",
	"gws_support_final_minus_initial",
	"gws_support_max",
	"gws_support_min",
}
GWS_WIDTH_KEYWORD_PREFIXES = (
	"gws_width_",
	"gmt_width_soft_",
)
GWS_AREA_KEYWORD_PREFIXES = (
	"gws_area_",
)
GMT_KEYWORD_PREFIXES = (
	"gmt_",
	"gradient_multiscale_tophat_",
)
MT_KEYWORD_PREFIXES = (
	"mt_",
	"multiscale_tophat_",
)
GWS_SOURCE_KEYWORD_PREFIXES = tuple(
	sorted(
		{
			str(prefix).lower()
			for prefix in GWS_SOURCE_PREFIX_BY_MODE.values()
			if str(prefix).strip()
		}
	)
)
GWS_COMPONENT_KEYWORD_PREFIXES = tuple(f"{prefix}_comp" for prefix in GWS_SOURCE_KEYWORD_PREFIXES)
TH_SHAPE_KEYWORD_PREFIXES = ("th_w7", "th_w9", "th_w11")
TH_KEYWORD_METRICS = {
	str(name).lower()
	for name in TH_SHAPE_ACTIVE_METRICS
	if str(name).strip()
}
RAW_EDGE_KEYWORD_PREFIXES = ("raw_edge_",)
MG_EDGE_KEYWORD_PREFIXES = ("mg_edge_",)
EDGE_CTX_KEYWORD_PREFIXES = ("raw_edge_ctx_", "raw_edge_dense_ctx_")
EDGE_DENSE_KEYWORD_PREFIXES = ("recdw_", "rucdw_", "decdw_", "raw_edge_dense_", "raw_edge_dense_ctx_", "raw_edge_ctx_dense_", "mg_edge_dense_")
EDGE_SIMPLE_KEYWORD_PREFIXES = (
	"width_",
	"width_ratio_",
	"raw_edge_width_",
	"raw_edge_ctx_width_",
	"raw_edge_base_expansion_rate",
	"raw_edge_ctx_base_expansion_rate",
	"raw_edge_intensity_per_width_loss_",
	"raw_edge_broadening_per_intensity_",
	"raw_edge_profile_",
	"mg_edge_width_",
	"mg_edge_base_expansion_rate",
	"mg_edge_intensity_per_width_loss_",
	"mg_edge_broadening_per_intensity_",
	"mg_edge_profile_",
	"mg_edge_root_width_",
)
EDGE_ROOT_KEYWORD_PREFIXES = (
	"raw_edge_width_at_0",
	"raw_edge_width_at_5",
	"raw_edge_width_at_10",
	"raw_edge_width_at_15",
	"raw_edge_width_at_20",
	"raw_edge_width_at_25",
	"raw_edge_root_width_",
	"raw_edge_ctx_width_at_0",
	"raw_edge_ctx_width_at_5",
	"raw_edge_ctx_width_at_10",
	"raw_edge_ctx_width_at_15",
	"raw_edge_ctx_width_at_20",
	"raw_edge_ctx_width_at_25",
	"raw_edge_ctx_root_width_",
	"mg_edge_width_at_5",
	"mg_edge_width_at_10",
	"mg_edge_width_at_15",
	"mg_edge_width_at_20",
	"mg_edge_width_at_25",
	"mg_edge_root_width_",
)
EDGE_ROOT_CTX_KEYWORD_PREFIXES = (
	"recdw_at_5",
	"recdw_at_10",
	"recdw_at_15",
	"recdw_at_20",
	"recdw_at_25",
	"decdw_at_5",
	"decdw_at_10",
	"decdw_at_15",
	"decdw_at_20",
	"decdw_at_25",
	"raw_edge_ctx_width_at_0",
	"raw_edge_ctx_width_at_5",
	"raw_edge_ctx_width_at_10",
	"raw_edge_ctx_width_at_15",
	"raw_edge_ctx_width_at_20",
	"raw_edge_ctx_width_at_25",
	"raw_edge_ctx_root_width_",
)
EDGE_TH_KEYWORD_PREFIXES = (
	"th_width_",
	"th_base_expansion_rate",
	"th_points_to_",
	"th_w7_width_",
	"th_w7_base_expansion_rate",
	"th_w7_points_to_",
	"th_w9_width_",
	"th_w9_base_expansion_rate",
	"th_w9_points_to_",
	"th_w11_width_",
	"th_w11_base_expansion_rate",
	"th_w11_points_to_",
)
EDGE_WIDTH_KEYWORD_PREFIXES = RAW_EDGE_KEYWORD_PREFIXES + MG_EDGE_KEYWORD_PREFIXES + EDGE_TH_KEYWORD_PREFIXES
RAW_BALL_KEYWORD_PREFIXES = ("raw_ball_",)
MG_BALL_KEYWORD_PREFIXES = ("mg_ball_",)
RAW_EXP_KEYWORD_PREFIXES = ("raw_exp_",)
MG_EXP_KEYWORD_PREFIXES = ("mg_exp_",)


def _display_feature_name(name: str, max_len: int = 52) -> str:
	short = str(name)
	if short.startswith("gradient_multiscale_tophat_"):
		short = "gmt_" + short[len("gradient_multiscale_tophat_"):]
	elif short.startswith("multiscale_tophat_"):
		short = "mt_" + short[len("multiscale_tophat_"):]
	return short if len(short) <= max_len else (short[: max_len - 1] + "…")


def _matches_keyword(name: str, keyword: str) -> bool:
	nm = str(name).strip().lower()
	kw = str(keyword).strip().lower()
	if kw == "gws-support":
		return nm in GWS_SUPPORT_KEYWORD_METRICS or any(nm.startswith(prefix) for prefix in GWS_SUPPORT_KEYWORD_PREFIXES)
	if kw == "gws-width":
		return any(nm.startswith(prefix) for prefix in GWS_WIDTH_KEYWORD_PREFIXES)
	if kw == "gws-area":
		return any(nm.startswith(prefix) for prefix in GWS_AREA_KEYWORD_PREFIXES)
	if kw == "gmt":
		return any(nm.startswith(prefix) for prefix in GMT_KEYWORD_PREFIXES)
	if kw == "mt":
		return any(nm.startswith(prefix) for prefix in MT_KEYWORD_PREFIXES)
	if kw in {"edge-dense", "edge_dense"}:
		return any(nm.startswith(prefix) for prefix in EDGE_DENSE_KEYWORD_PREFIXES)
	if kw in {"ss4", "v4", "spike-v4", "spike_v4", "three-friends", "three_friends"}:
		return nm == "ss4" or nm.startswith("ss4_")
	if kw in {"edge-ctx", "edge_ctx", "raw-edge-ctx", "raw_edge_ctx", "recdw", "rucdw", "decdw"}:
		if kw == "recdw":
			return nm.startswith("recdw_")
		if kw == "rucdw":
			return nm.startswith("rucdw_")
		if kw == "decdw":
			return nm.startswith("decdw_")
		return any(nm.startswith(prefix) for prefix in EDGE_CTX_KEYWORD_PREFIXES + ("recdw_", "rucdw_", "decdw_"))
	if kw in {"edge-simple", "edge_simple", "edge-level", "edge_level"}:
		return any(nm.startswith(prefix) for prefix in EDGE_SIMPLE_KEYWORD_PREFIXES)
	if kw in {"edge-root-ctx", "edge_root_ctx", "raw-edge-root-ctx", "raw_edge_root_ctx"}:
		return any(nm.startswith(prefix) for prefix in EDGE_ROOT_CTX_KEYWORD_PREFIXES)
	if kw in {"edge-root", "edge_root", "raw-edge-root", "raw_edge_root", "mg-edge-root", "mg_edge_root", "raman-veto-edge", "raman_veto_edge"}:
		return any(nm.startswith(prefix) for prefix in EDGE_ROOT_KEYWORD_PREFIXES)
	if kw in {"edge-th", "edge_th", "th-edge", "th_edge"}:
		return any(nm.startswith(prefix) for prefix in EDGE_TH_KEYWORD_PREFIXES)
	if kw in {"edge-th-w7", "edge_th_w7", "th-w7", "th_w7"}:
		return nm.startswith(("th_w7_width_", "th_w7_base_expansion_rate", "th_w7_points_to_"))
	if kw in {"edge-th-w9", "edge_th_w9", "th-w9", "th_w9"}:
		return nm.startswith(("th_w9_width_", "th_w9_base_expansion_rate", "th_w9_points_to_"))
	if kw in {"edge-th-w11", "edge_th_w11", "th-w11", "th_w11"}:
		return nm.startswith(("th_w11_width_", "th_w11_base_expansion_rate", "th_w11_points_to_"))
	if kw in {"raw-edge", "raw_edge"}:
		return any(nm.startswith(prefix) for prefix in RAW_EDGE_KEYWORD_PREFIXES)
	if kw in {"mg-edge", "mg_edge"}:
		return any(nm.startswith(prefix) for prefix in MG_EDGE_KEYWORD_PREFIXES)
	if kw in {"raw-ball", "raw_ball"}:
		return any(nm.startswith(prefix) for prefix in RAW_BALL_KEYWORD_PREFIXES)
	if kw in {"mg-ball", "mg_ball"}:
		return any(nm.startswith(prefix) for prefix in MG_BALL_KEYWORD_PREFIXES)
	if kw in {"raw-exp", "raw_exp"}:
		return any(nm.startswith(prefix) for prefix in RAW_EXP_KEYWORD_PREFIXES)
	if kw in {"mg-exp", "mg_exp"}:
		return any(nm.startswith(prefix) for prefix in MG_EXP_KEYWORD_PREFIXES)
	if kw == "edge":
		return any(nm.startswith(prefix) for prefix in EDGE_WIDTH_KEYWORD_PREFIXES)
	if kw == "ball":
		return any(nm.startswith(prefix) for prefix in RAW_BALL_KEYWORD_PREFIXES + MG_BALL_KEYWORD_PREFIXES)
	if kw == "exp":
		return any(nm.startswith(prefix) for prefix in RAW_EXP_KEYWORD_PREFIXES + MG_EXP_KEYWORD_PREFIXES)
	if kw == "gws-imp":
		return nm in GWS_IMP_KEYWORD_METRICS
	if kw == "th":
		return nm in TH_KEYWORD_METRICS or any(nm.startswith(prefix + "_") for prefix in TH_SHAPE_KEYWORD_PREFIXES)
	if kw == "gws":
		if nm in GWS_KEYWORD_METRICS:
			return True
		if any(nm.startswith(prefix) for prefix in GWS_KEYWORD_PREFIXES):
			return True
	if kw.endswith("_*"):
		prefix = kw[:-2]
		if prefix in GWS_SOURCE_KEYWORD_PREFIXES:
			return nm.startswith(prefix + "_")
		if prefix in GWS_COMPONENT_KEYWORD_PREFIXES or prefix in TH_SHAPE_KEYWORD_PREFIXES:
			return nm.startswith(prefix + "_")
		if prefix in {"ss4", "recdw", "rucdw", "decdw", "raw_edge", "raw_edge_ctx", "mg_edge", "raw_ball", "mg_ball", "raw_exp", "mg_exp"}:
			return nm.startswith(prefix + "_")
	if kw in GWS_SOURCE_KEYWORD_PREFIXES:
		return nm.startswith(kw + "_")
	if kw in GWS_COMPONENT_KEYWORD_PREFIXES:
		return nm.startswith(kw + "_")
	if kw in TH_SHAPE_KEYWORD_PREFIXES:
		return nm.startswith(kw + "_")
	if kw in nm:
		return True
	expanded_names = {nm}
	if nm.startswith("mt_"):
		expanded_names.add("multiscale_tophat_" + nm[len("mt_"):])
	elif nm.startswith("gmt_"):
		expanded_names.add("gradient_multiscale_tophat_" + nm[len("gmt_"):])
	elif nm.startswith("medres_"):
		expanded_names.add("median_residual_" + nm[len("medres_"):])
	elif nm.startswith("opres_"):
		expanded_names.add("opening_residual_" + nm[len("opres_"):])
	if "gmt_width_soft_d1_mean" in nm or "gws_support01" in nm or "gws_evidence_signed" in nm:
		expanded_names.update({"gws", "gradient_width_support"})
	if "peak_curvature" in nm or nm.startswith("pce_") or nm.startswith("curv2_") or "curvature_extreme" in nm:
		expanded_names.update({"pce", "peak_curvature", "curv2", "curvature"})
	if nm.startswith("pce_negpref_"):
		expanded_names.update({"pce_negpref", "negpref"})
	if "negpref" in nm:
		expanded_names.update({"negpref", "negative_preferred"})
	if "apex_at_peak" in nm or nm.startswith("pce_apex_"):
		expanded_names.update({"pce_apex", "apex_curvature"})

	expanded_keywords = {kw}
	if kw.startswith("mt_"):
		expanded_keywords.add("multiscale_tophat_" + kw[len("mt_"):])
	elif kw.startswith("gmt_"):
		expanded_keywords.add("gradient_multiscale_tophat_" + kw[len("gmt_"):])
	elif kw == "multiscale_tophat":
		expanded_keywords.update({"mt_", "gmt_", "gradient_multiscale_tophat"})
	elif kw.startswith("medres_"):
		expanded_keywords.add("median_residual_" + kw[len("medres_"):])
	elif kw == "medres":
		expanded_keywords.add("median_residual")
	elif kw.startswith("median_residual_"):
		expanded_keywords.add("medres_" + kw[len("median_residual_"):])
	elif kw == "median_residual":
		expanded_keywords.add("medres")
	elif kw.startswith("opres_"):
		expanded_keywords.add("opening_residual_" + kw[len("opres_"):])
	elif kw == "opres":
		expanded_keywords.add("opening_residual")
	elif kw.startswith("opening_residual_"):
		expanded_keywords.add("opres_" + kw[len("opening_residual_"):])
	elif kw == "opening_residual":
		expanded_keywords.add("opres")
	elif kw == "gws":
		expanded_keywords.update({"gmt_width_soft_d1_mean", "gws_support01", "gws_evidence_signed", "gradient_width_support"})
	elif kw == "gws-support":
		expanded_keywords.update(GWS_SUPPORT_KEYWORD_METRICS)
		expanded_keywords.update(GWS_SUPPORT_KEYWORD_PREFIXES)
	elif kw == "gws-width":
		expanded_keywords.update(GWS_WIDTH_KEYWORD_PREFIXES)
	elif kw == "gws-area":
		expanded_keywords.update(GWS_AREA_KEYWORD_PREFIXES)
	elif kw == "gmt":
		expanded_keywords.update(GMT_KEYWORD_PREFIXES)
	elif kw == "mt":
		expanded_keywords.update(MT_KEYWORD_PREFIXES)
	elif kw in {"edge-dense", "edge_dense"}:
		expanded_keywords.update(EDGE_DENSE_KEYWORD_PREFIXES)
	elif kw in {"ss4", "v4", "spike-v4", "spike_v4", "three-friends", "three_friends"}:
		expanded_keywords.add("ss4")
	elif kw in {"edge-ctx", "edge_ctx", "raw-edge-ctx", "raw_edge_ctx", "recdw", "rucdw", "decdw"}:
		if kw == "recdw":
			expanded_keywords.add("recdw_")
		elif kw == "rucdw":
			expanded_keywords.add("rucdw_")
		elif kw == "decdw":
			expanded_keywords.add("decdw_")
		else:
			expanded_keywords.update(EDGE_CTX_KEYWORD_PREFIXES + ("recdw_", "rucdw_", "decdw_"))
	elif kw in {"edge-simple", "edge_simple", "edge-level", "edge_level"}:
		expanded_keywords.update(EDGE_SIMPLE_KEYWORD_PREFIXES)
	elif kw in {"edge-root-ctx", "edge_root_ctx", "raw-edge-root-ctx", "raw_edge_root_ctx"}:
		expanded_keywords.update(EDGE_ROOT_CTX_KEYWORD_PREFIXES)
	elif kw in {"edge-root", "edge_root", "raw-edge-root", "raw_edge_root", "mg-edge-root", "mg_edge_root", "raman-veto-edge", "raman_veto_edge"}:
		expanded_keywords.update(EDGE_ROOT_KEYWORD_PREFIXES)
	elif kw in {"edge-th", "edge_th", "th-edge", "th_edge"}:
		expanded_keywords.update(EDGE_TH_KEYWORD_PREFIXES)
	elif kw in {"edge-th-w7", "edge_th_w7", "th-w7", "th_w7"}:
		expanded_keywords.update({"th_w7_width_", "th_w7_base_expansion_rate", "th_w7_points_to_"})
	elif kw in {"edge-th-w9", "edge_th_w9", "th-w9", "th_w9"}:
		expanded_keywords.update({"th_w9_width_", "th_w9_base_expansion_rate", "th_w9_points_to_"})
	elif kw in {"edge-th-w11", "edge_th_w11", "th-w11", "th_w11"}:
		expanded_keywords.update({"th_w11_width_", "th_w11_base_expansion_rate", "th_w11_points_to_"})
	elif kw in {"raw-edge", "raw_edge"}:
		expanded_keywords.update(RAW_EDGE_KEYWORD_PREFIXES)
	elif kw in {"mg-edge", "mg_edge"}:
		expanded_keywords.update(MG_EDGE_KEYWORD_PREFIXES)
	elif kw in {"raw-ball", "raw_ball"}:
		expanded_keywords.update(RAW_BALL_KEYWORD_PREFIXES)
	elif kw in {"mg-ball", "mg_ball"}:
		expanded_keywords.update(MG_BALL_KEYWORD_PREFIXES)
	elif kw in {"raw-exp", "raw_exp"}:
		expanded_keywords.update(RAW_EXP_KEYWORD_PREFIXES)
	elif kw in {"mg-exp", "mg_exp"}:
		expanded_keywords.update(MG_EXP_KEYWORD_PREFIXES)
	elif kw == "edge":
		expanded_keywords.update(EDGE_WIDTH_KEYWORD_PREFIXES)
	elif kw == "ball":
		expanded_keywords.update(RAW_BALL_KEYWORD_PREFIXES + MG_BALL_KEYWORD_PREFIXES)
	elif kw == "exp":
		expanded_keywords.update(RAW_EXP_KEYWORD_PREFIXES + MG_EXP_KEYWORD_PREFIXES)
	elif kw == "gws-imp":
		expanded_keywords.update(GWS_IMP_KEYWORD_METRICS)
	elif kw.endswith("_*"):
		prefix = kw[:-2]
		if prefix in GWS_SOURCE_KEYWORD_PREFIXES:
			expanded_keywords.add(prefix + "_")
		elif prefix in GWS_COMPONENT_KEYWORD_PREFIXES or prefix in TH_SHAPE_KEYWORD_PREFIXES:
			expanded_keywords.add(prefix + "_")
		elif prefix in {"ss4", "recdw", "rucdw", "decdw", "raw_edge", "raw_edge_ctx", "mg_edge", "raw_ball", "mg_ball", "raw_exp", "mg_exp"}:
			expanded_keywords.add(prefix + "_")
	elif kw in GWS_SOURCE_KEYWORD_PREFIXES:
		expanded_keywords.add(kw + "_")
	elif kw in GWS_COMPONENT_KEYWORD_PREFIXES:
		expanded_keywords.add(kw + "_")
	elif kw in TH_SHAPE_KEYWORD_PREFIXES:
		expanded_keywords.add(kw + "_")
	elif kw == "gradient_width_support":
		expanded_keywords.update({"gws", "gmt_width_soft_d1_mean", "gws_support01", "gws_evidence_signed"})
	elif kw == "th":
		expanded_keywords.update(TH_KEYWORD_METRICS)
		expanded_keywords.update({prefix + "_" for prefix in TH_SHAPE_KEYWORD_PREFIXES})
	elif kw == "pce":
		expanded_keywords.update({"peak_curvature", "pce_", "curv2", "curvature_extreme", "curvature"})
	elif kw == "pce_negpref":
		expanded_keywords.update({"pce_negpref_", "negpref", "peak_curvature_extreme_negpref"})
	elif kw in {"negpref", "negative_preferred"}:
		expanded_keywords.update({"negpref", "peak_curvature_extreme_negpref", "negative_preferred"})
	elif kw in {"pce_apex", "apex_curvature"}:
		expanded_keywords.update({"pce_apex_", "peak_curvature_apex_at_peak", "apex_curvature"})
	elif kw == "peak_curvature":
		expanded_keywords.update({"pce", "peak_curvature_", "curv2", "curvature_extreme", "curvature"})
	elif kw == "curv2":
		expanded_keywords.update({"pce", "peak_curvature", "curv2_", "curvature"})
	elif kw == "curvature":
		expanded_keywords.update({"pce", "peak_curvature", "curv2", "curvature_"})

	return any(ekw in enm for ekw in expanded_keywords for enm in expanded_names)


def _fmt(v: object) -> str:
	try:
		vf = float(v)
	except Exception:
		return str(v)
	return f"{vf:.4f}" if np.isfinite(vf) else "nan"


def _uniform_bin_edges(vals: np.ndarray, target_bins: int = 72) -> np.ndarray:
	x = np.asarray(vals[np.isfinite(vals)], dtype=float)
	if x.size == 0:
		return np.linspace(0.0, 1.0, max(2, int(target_bins) + 1))
	vmin = float(np.min(x))
	vmax = float(np.max(x))
	if not np.isfinite(vmin) or not np.isfinite(vmax):
		return np.linspace(0.0, 1.0, max(2, int(target_bins) + 1))
	if vmin == vmax:
		span = max(abs(vmin) * 0.05, 1.0)
		vmin -= 0.5 * span
		vmax += 0.5 * span
	return np.linspace(vmin, vmax, max(2, int(target_bins) + 1))


def _collect_metric_values(report: Dict, labels: Dict[Tuple[int, int, int], int], metric: str) -> Tuple[np.ndarray, np.ndarray]:
	per = list(report.get("per_spectrum", []))
	per_by_coord: Dict[Tuple[int, int], Dict] = {
		(int(spec.get("y", -1)), int(spec.get("x", -1))): spec
		for spec in per
	}
	mu_vals: List[float] = []
	non_vals: List[float] = []
	for (y, x, p), is_muon in labels.items():
		spec = per_by_coord.get((y, x))
		if spec is None:
			continue
		for sp in list(spec.get("spikes", [])):
			if int(sp.get("peak_index", -1)) != int(p):
				continue
			v = float(sp.get(metric, np.nan))
			if np.isfinite(v):
				(mu_vals if int(is_muon) == 1 else non_vals).append(v)
			break
	return np.asarray(mu_vals, dtype=float), np.asarray(non_vals, dtype=float)


def _collect_metric_values_by_class(report: Dict, labels: Dict[Tuple[int, int, int], str], metric: str) -> Dict[str, np.ndarray]:
	per = list(report.get("per_spectrum", []))
	per_by_coord: Dict[Tuple[int, int], Dict] = {
		(int(spec.get("y", -1)), int(spec.get("x", -1))): spec
		for spec in per
	}
	out: Dict[str, List[float]] = {cls: [] for cls in ("muon", "raman", "noise", "unknown")}
	for (y, x, p), label_class in labels.items():
		spec = per_by_coord.get((y, x))
		if spec is None:
			continue
		for sp in list(spec.get("spikes", [])):
			if int(sp.get("peak_index", -1)) != int(p):
				continue
			v = float(sp.get(metric, np.nan))
			if np.isfinite(v):
				out.setdefault(str(label_class), []).append(v)
			break
	return {cls: np.asarray(vals, dtype=float) for cls, vals in out.items()}


def main() -> None:
	parser = argparse.ArgumentParser(description="Browse labeled feature histograms filtered by keyword from corr.json.")
	parser.add_argument("--corr-json", type=Path, required=True, help="Path to corr.json from debug_stats.py")
	parser.add_argument("--report", type=Path, required=True, help="Path to debug_report.json")
	parser.add_argument("--labels-csv", type=Path, required=True, help="labels.csv with y,x,peak_index,is_muon")
	parser.add_argument("--keyword", type=str, default="multiscale_tophat", help="Case-insensitive substring filter for feature names.")
	parser.add_argument("--top-k", type=int, default=0, help="Keep only the top K matched metrics by oriented AUC. 0 = all.")
	parser.add_argument("--label-mode", choices=["binary", "ternary"], default="binary", help="Default binary keeps legacy muon/non-muon plots.")
	parser.add_argument("--show-classes", type=str, default="muon,raman,noise", help="Ternary mode: comma-separated classes to display.")
	parser.add_argument("--exclude-unknown", action="store_true", default=True, help="Ternary mode: exclude unknown labels.")
	parser.add_argument("--unlabeled-as-noise", action="store_true", help="Ternary mode: treat unknown labels as noise.")
	args = parser.parse_args()

	corr = json.loads(args.corr_json.read_text(encoding="utf-8"))
	report = json.loads(args.report.read_text(encoding="utf-8"))
	label_mode = str(args.label_mode).lower()
	if label_mode == "ternary":
		show_classes = [c.strip().lower() for c in str(args.show_classes).split(",") if c.strip()]
		labels_cls = load_label_classes(
			args.labels_csv,
			include_unknown=not bool(args.exclude_unknown),
			unlabeled_as_noise=bool(args.unlabeled_as_noise),
		)
		labels = {}
	else:
		show_classes = []
		labels_cls = {}
		labels = load_binary_labels(args.labels_csv)

	rows = [row for row in corr.get("correlations", []) if isinstance(row, dict)]
	keyword = str(args.keyword).strip().lower()
	matched = [row for row in rows if _matches_keyword(str(row.get("feature", "")), keyword)]
	if not matched:
		raise ValueError(f"No corr.json features matched keyword '{args.keyword}'.")
	matched.sort(key=lambda r: float(r.get("auc_feature_oriented", float("-inf"))), reverse=True)
	if int(args.top_k) > 0:
		matched = matched[: int(args.top_k)]

	metrics = [str(row.get("feature", "")).strip() for row in matched if str(row.get("feature", "")).strip()]
	if not metrics:
		raise ValueError("No valid feature names after filtering.")

	cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
	class_cache: Dict[str, Dict[str, np.ndarray]] = {}
	state = {"i": 0}

	fig = plt.figure(figsize=(12, 7))
	ax_hist = fig.add_axes((0.07, 0.14, 0.58, 0.76))
	ax_info = fig.add_axes((0.69, 0.14, 0.28, 0.76))
	ax_info.axis("off")
	info_txt = ax_info.text(0.01, 0.99, "", va="top", ha="left", family="monospace", fontsize=10)

	def _get_vals(metric: str) -> Tuple[np.ndarray, np.ndarray]:
		if metric not in cache:
			cache[metric] = _collect_metric_values(report, labels, metric)
		return cache[metric]

	def _get_class_vals(metric: str) -> Dict[str, np.ndarray]:
		if metric not in class_cache:
			class_cache[metric] = _collect_metric_values_by_class(report, labels_cls, metric)
		return class_cache[metric]

	def _draw() -> None:
		i = int(state["i"])
		row = matched[i]
		metric = metrics[i]
		if label_mode == "ternary":
			by_class = _get_class_vals(metric)
			class_arrays = [by_class.get(cls, np.array([], dtype=float)) for cls in show_classes]
			vals = np.concatenate([arr for arr in class_arrays if arr.size]) if any(arr.size for arr in class_arrays) else np.array([], dtype=float)
			mu_vals = by_class.get("muon", np.array([], dtype=float))
			non_vals = np.concatenate([by_class.get("raman", np.array([], dtype=float)), by_class.get("noise", np.array([], dtype=float))])
		else:
			mu_vals, non_vals = _get_vals(metric)
			vals = np.concatenate([mu_vals, non_vals]) if (mu_vals.size or non_vals.size) else np.array([], dtype=float)

		ax_hist.clear()
		bins = _uniform_bin_edges(vals, target_bins=72)
		if label_mode == "ternary":
			for cls in show_classes:
				arr = by_class.get(cls, np.array([], dtype=float))
				if arr.size:
					ax_hist.hist(arr, bins=bins, alpha=0.42, color=LABEL_CLASS_COLORS.get(cls, "#7f7f7f"), label=f"{cls} ({arr.size})")
		else:
			if non_vals.size:
				ax_hist.hist(non_vals, bins=bins, alpha=0.45, color="#1f77b4", label=f"no-muon ({non_vals.size})")
			if mu_vals.size:
				ax_hist.hist(mu_vals, bins=bins, alpha=0.45, color="#d62728", label=f"muon ({mu_vals.size})")
		ax_hist.grid(alpha=0.25)
		ax_hist.set_xlabel(metric)
		ax_hist.set_ylabel("Count")
		ax_hist.set_title(f"{_display_feature_name(metric)} | {i + 1}/{len(metrics)}")
		if mu_vals.size or non_vals.size:
			ax_hist.legend(loc="best")

		info = [
			f"metric: {_display_feature_name(metric, max_len=60)}",
			f"full: {metric}",
			"",
			f"auc_oriented: {_fmt(row.get('auc_feature_oriented'))}",
			f"auc_raw: {_fmt(row.get('auc_feature_raw'))}",
			f"auc_direction: {row.get('auc_direction', '')}",
			f"pearson_label: {_fmt(row.get('pearson_label'))}",
			f"spearman_label: {_fmt(row.get('spearman_label'))}",
			f"mutual_info_bits: {_fmt(row.get('mutual_info_bits'))}",
			f"transform: {row.get('transform', '')}",
			"",
			f"label_mode: {label_mode}",
			f"muon n: {mu_vals.size}",
			f"no-muon n: {non_vals.size}",
		]
		if label_mode == "ternary":
			info += [f"{cls} n: {by_class.get(cls, np.array([], dtype=float)).size}" for cls in show_classes]
		if vals.size:
			info += [
				f"all min/max: {_fmt(np.min(vals))} / {_fmt(np.max(vals))}",
				f"mu mean/std: {_fmt(np.mean(mu_vals))} / {_fmt(np.std(mu_vals))}" if mu_vals.size else "mu mean/std: nan / nan",
				f"no mean/std: {_fmt(np.mean(non_vals))} / {_fmt(np.std(non_vals))}" if non_vals.size else "no mean/std: nan / nan",
			]
		info += ["", "Controls:", " left/right or p/n = browse metrics", " q = quit"]
		info_txt.set_text("\n".join(info))
		fig.canvas.draw_idle()

	def _on_key(event) -> None:
		key = (event.key or "").lower()
		if key in ("right", "n"):
			state["i"] = min(len(metrics) - 1, int(state["i"]) + 1)
			_draw()
		elif key in ("left", "p"):
			state["i"] = max(0, int(state["i"]) - 1)
			_draw()
		elif key == "q":
			plt.close(fig)

	fig.canvas.mpl_connect("key_press_event", _on_key)
	print("Controls: Left/Right arrows (or p/n) to browse metrics, q to quit.")
	_draw()
	plt.show()


if __name__ == "__main__":
	main()
