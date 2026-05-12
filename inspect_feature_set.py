from __future__ import annotations

import argparse
import csv
import json
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from muon_decision import get_muon_rule_v3_metric_thresholds
from feature_discrimination import GWS_SOURCE_PREFIX_BY_MODE
from feature_discrimination import TH_SHAPE_ACTIVE_METRICS
from feature_discrimination import TOPHAT_GRADIENT_EXPORT_FEATURES


FeatureRow = Dict[str, Any]
PairRow = Dict[str, Any]


AUC_SORT_KEYS = (
	"auc",
	"auc_feature_oriented",
	"auc_feat_oriented",
	"auc_muon_vs_rest",
	"auc_raman_vs_rest",
	"auc_noise_vs_rest",
	"auc_muon_vs_raman",
	"auc_muon_vs_noise",
	"auc_raman_vs_noise",
	"macro_ovr_auc",
	"macro_pairwise_auc",
	"best_pairwise_auc",
	"worst_pairwise_auc",
	"raman_veto_auc",
	"noise_filter_auc",
	"muon_detector_auc",
)


GWS_KEYWORD_METRICS = {
	str(name)
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


def _safe_float(value: object, default: float = float("nan")) -> float:
	try:
		v = float(value)
	except Exception:
		return default
	return v if np.isfinite(v) else default


def _fmt(value: object, digits: int = 4) -> str:
	v = _safe_float(value)
	if np.isfinite(v):
		return f"{v:.{digits}f}"
	if value is None:
		return "nan"
	return str(value)


def _normalize_auc_key(key: str | None, *, label_mode: str = "binary") -> str:
	if key is None or str(key).strip() == "":
		return "auc_feature_oriented" if str(label_mode).lower() != "ternary" else "macro_pairwise_auc"
	k = str(key).strip()
	if k in {"auc", "auc_feat_oriented"}:
		return "auc_feature_oriented"
	if k not in AUC_SORT_KEYS:
		raise ValueError(
			f"Unsupported AUC key: {key}. Supported: "
			+ ", ".join(k for k in AUC_SORT_KEYS if k != "auc_feat_oriented")
		)
	return k


def _feature_auc_value(row: FeatureRow, key: str) -> float:
	k = _normalize_auc_key(key)
	return _safe_float(row.get(k))


def _display_name(name: str, max_len: int = 30) -> str:
	short = str(name)
	if short.startswith("gradient_multiscale_tophat_"):
		short = "gmt_" + short[len("gradient_multiscale_tophat_"):]
	elif short.startswith("multiscale_tophat_"):
		short = "mt_" + short[len("multiscale_tophat_"):]
	return short if len(short) <= max_len else short[: max_len - 1] + "…"


def _read_feature_file(path: Path) -> List[str]:
	features: List[str] = []
	for line in path.read_text(encoding="utf-8").splitlines():
		name = str(line).strip()
		if not name or name.startswith("#"):
			continue
		features.append(name)
	return features


def _split_feature_tokens(values: Sequence[str]) -> List[str]:
	out: List[str] = []
	for value in values:
		text = str(value).strip()
		if not text:
			continue
		parts = [part.strip() for part in text.split(",")]
		out.extend([part for part in parts if part])
	return out


def _matches_feature_keyword(name: str, keyword: str) -> bool:
	nm = str(name).strip().lower()
	kw = str(keyword).strip().lower()
	if not nm or not kw:
		return False
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
	if kw == "gws":
		return nm in GWS_KEYWORD_METRICS or any(nm.startswith(prefix) for prefix in GWS_KEYWORD_PREFIXES)
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
	if kw == "pce":
		return (
			"peak_curvature" in nm
			or nm.startswith("pce_")
			or nm.startswith("curv2_")
			or "curvature_extreme" in nm
		)
	if kw == "th":
		return nm in TH_KEYWORD_METRICS or any(nm.startswith(prefix + "_") for prefix in TH_SHAPE_KEYWORD_PREFIXES)
	return False


FEATURE_KEYWORD_ALIASES = {
	"pce", "gws", "gws-imp", "gws-support", "gws-width", "gws-area", "gmt", "mt", "th",
	"ss4", "v4", "spike-v4", "spike_v4", "three-friends", "three_friends",
	"edge-dense", "edge_dense", "edge-ctx", "edge_ctx", "raw-edge-ctx", "raw_edge_ctx", "recdw", "rucdw", "decdw",
	"edge-simple", "edge_simple", "edge-level", "edge_level",
	"edge-root", "edge_root", "edge-root-ctx", "edge_root_ctx",
	"raw-edge-root", "raw_edge_root", "raw-edge-root-ctx", "raw_edge_root_ctx",
	"mg-edge-root", "mg_edge_root", "raman-veto-edge", "raman_veto_edge",
	"edge-th", "edge_th", "th-edge", "th_edge",
	"edge-th-w7", "edge_th_w7", "th-w7", "th_w7",
	"edge-th-w9", "edge_th_w9", "th-w9", "th_w9",
	"edge-th-w11", "edge_th_w11", "th-w11", "th_w11",
	"raw-edge", "raw_edge", "mg-edge", "mg_edge", "edge",
	"raw-ball", "raw_ball", "mg-ball", "mg_ball", "ball",
	"raw-exp", "raw_exp", "mg-exp", "mg_exp", "exp",
}


def _expand_keyword_features(token: str, available_names: Sequence[str]) -> List[str]:
	kw = str(token).strip().lower()
	if kw not in FEATURE_KEYWORD_ALIASES:
		prefix = kw[:-2] if kw.endswith("_*") else ""
		if kw.endswith("_*") and prefix in GWS_SOURCE_KEYWORD_PREFIXES:
			return [name for name in available_names if _matches_feature_keyword(name, kw)]
		if kw.endswith("_*") and (prefix in GWS_COMPONENT_KEYWORD_PREFIXES or prefix in TH_SHAPE_KEYWORD_PREFIXES):
			return [name for name in available_names if _matches_feature_keyword(name, kw)]
		if kw.endswith("_*") and prefix in {"ss4", "recdw", "rucdw", "decdw", "raw_edge", "raw_edge_ctx", "mg_edge", "raw_ball", "mg_ball", "raw_exp", "mg_exp"}:
			return [name for name in available_names if _matches_feature_keyword(name, kw)]
		if kw in GWS_SOURCE_KEYWORD_PREFIXES:
			return [name for name in available_names if _matches_feature_keyword(name, kw)]
		if kw in GWS_COMPONENT_KEYWORD_PREFIXES or kw in TH_SHAPE_KEYWORD_PREFIXES:
			return [name for name in available_names if _matches_feature_keyword(name, kw)]
		return []
	return [name for name in available_names if _matches_feature_keyword(name, kw)]


def _load_corr_json(path: Path) -> Tuple[Dict[str, FeatureRow], List[str], np.ndarray, np.ndarray]:
	data = json.loads(path.read_text(encoding="utf-8"))
	feature_rows: Dict[str, FeatureRow] = {}
	for row in data.get("correlations", []):
		if not isinstance(row, dict):
			continue
		name = str(row.get("feature", "")).strip()
		if not name:
			continue
		feature_rows[name] = row

	fc = data.get("feature_correlation", {})
	if not isinstance(fc, dict):
		raise ValueError("corr.json missing 'feature_correlation' block.")

	names = [str(x).strip() for x in fc.get("features", []) if str(x).strip()]
	pear_mat = np.asarray(fc.get("pearson_matrix", []), dtype=float)
	spear_mat = np.asarray(fc.get("spearman_matrix", []), dtype=float)
	if pear_mat.ndim != 2 or spear_mat.ndim != 2:
		raise ValueError("feature_correlation matrices must be 2D.")
	if pear_mat.shape != spear_mat.shape:
		raise ValueError("Pearson and Spearman matrix shapes differ.")
	if pear_mat.shape != (len(names), len(names)):
		raise ValueError("feature_correlation matrix shape does not match feature count.")
	return feature_rows, names, pear_mat, spear_mat


def _resolve_features(
		requested_features: Sequence[str],
		features_file: Optional[Path],
		available_feature_rows: Dict[str, FeatureRow],
		matrix_names: Sequence[str],
) -> List[str]:
	names: List[str] = []
	if requested_features:
		names.extend(_split_feature_tokens(requested_features))
	if features_file is not None:
		names.extend(_read_feature_file(features_file))
	if not names:
		raise ValueError("Provide --features and/or --features-file.")

	seen = set()
	ordered: List[str] = []
	for name in names:
		expanded = _expand_keyword_features(name, matrix_names)
		if expanded:
			print(f"Expanded keyword '{name}' to {len(expanded)} features.")
			for feature_name in expanded:
				if feature_name in seen:
					continue
				seen.add(feature_name)
				ordered.append(feature_name)
			continue
		if name in seen:
			continue
		seen.add(name)
		ordered.append(name)

	matrix_set = set(matrix_names)
	missing = [name for name in ordered if name not in available_feature_rows]
	matrix_missing = [name for name in ordered if name not in matrix_set]
	if missing:
		print(f"Warning: {len(missing)} requested features are missing from correlations: {', '.join(missing)}")
		for name in missing:
			matches = get_close_matches(name, list(available_feature_rows.keys()), n=5, cutoff=0.55)
			if matches:
				print(f"  suggestions for {name}: {', '.join(matches)}")
	if matrix_missing:
		print(f"Warning: {len(matrix_missing)} requested features are missing from feature_correlation: {', '.join(matrix_missing)}")
		for name in matrix_missing:
			matches = get_close_matches(name, list(matrix_names), n=5, cutoff=0.55)
			if matches:
				print(f"  matrix suggestions for {name}: {', '.join(matches)}")
	resolved = [name for name in ordered if name in available_feature_rows]
	if not resolved:
		raise ValueError("None of the requested features were found in corr.json correlations.")
	return resolved


def _sort_feature_names(
		feature_names: Sequence[str],
		feature_rows: Dict[str, FeatureRow],
		sort_by: str,
) -> List[str]:
	names = list(feature_names)
	if sort_by == "input":
		return names
	if sort_by in AUC_SORT_KEYS:
		auc_key = _normalize_auc_key(sort_by)
		return sorted(
			names,
			key=lambda name: _feature_auc_value(feature_rows[name], auc_key),
			reverse=True,
		)
	if sort_by == "fp_at_100_recall":
		return sorted(
			names,
			key=lambda name: (
				np.isnan(_safe_float(feature_rows[name].get("operating_point_stats", {}).get("fp_at_100_recall"))),
				_safe_float(feature_rows[name].get("operating_point_stats", {}).get("fp_at_100_recall")),
			),
		)
	raise ValueError(f"Unsupported --sort-by value: {sort_by}")


def _feature_summary_row(index: int, name: str, row: FeatureRow) -> Dict[str, Any]:
	intervals_block = row.get("empirical_purity_intervals", {})
	intervals = intervals_block.get("intervals", []) if isinstance(intervals_block, dict) else []
	if not isinstance(intervals, list):
		intervals = []
	no_muon = 0.0
	maybe_muon = 0.0
	ok_muon = 0.0
	mixed_intervals = [iv for iv in intervals if isinstance(iv, dict) and str(iv.get("kind")) == "mixed"]
	for iv in intervals:
		if not isinstance(iv, dict):
			continue
		kind = str(iv.get("kind", ""))
		if kind == "nonmuon_pure":
			no_muon += _safe_float(iv.get("n_total"), 0.0)
		elif kind == "mixed":
			maybe_muon += _safe_float(iv.get("n_muon"), 0.0)
		elif kind == "muon_pure":
			ok_muon += _safe_float(iv.get("n_muon"), 0.0)
	if mixed_intervals:
		best_mixed = max(mixed_intervals, key=lambda iv: (_safe_float(iv.get("n_total"), 0.0), _safe_float(iv.get("right")) - _safe_float(iv.get("left"))))
		thr_low = _safe_float(best_mixed.get("left"))
		thr_high = _safe_float(best_mixed.get("right"))
	else:
		thr_low = float("nan")
		thr_high = float("nan")
		rule_thr = get_muon_rule_v3_metric_thresholds(name)
		if rule_thr is not None:
			thr_low = float(rule_thr[0])
			thr_high = float(rule_thr[1])
	return {
		"index": int(index),
		"feature": name,
		"auc_feat_oriented": _safe_float(row.get("auc_feature_oriented")),
		"no-muon": no_muon,
		"maybe_muon": maybe_muon,
		"ok-muon": ok_muon,
		"thr-low": thr_low,
		"thr-high": thr_high,
	}


def _feature_summary_row_ternary(index: int, name: str, row: FeatureRow) -> Dict[str, Any]:
	return {
		"index": int(index),
		"feature": name,
		"n_muon": _safe_float(row.get("n_muon"), 0.0),
		"n_raman": _safe_float(row.get("n_raman"), 0.0),
		"n_noise": _safe_float(row.get("n_noise"), 0.0),
		"n_unknown_excluded": _safe_float(row.get("n_unknown_excluded"), 0.0),
		"auc_muon_vs_rest": _safe_float(row.get("auc_muon_vs_rest")),
		"auc_raman_vs_rest": _safe_float(row.get("auc_raman_vs_rest")),
		"auc_noise_vs_rest": _safe_float(row.get("auc_noise_vs_rest")),
		"auc_muon_vs_raman": _safe_float(row.get("auc_muon_vs_raman")),
		"auc_muon_vs_noise": _safe_float(row.get("auc_muon_vs_noise")),
		"auc_raman_vs_noise": _safe_float(row.get("auc_raman_vs_noise")),
		"macro_pairwise_auc": _safe_float(row.get("macro_pairwise_auc")),
		"raman_veto_auc": _safe_float(row.get("raman_veto_auc")),
		"noise_filter_auc": _safe_float(row.get("noise_filter_auc")),
		"muon_detector_auc": _safe_float(row.get("muon_detector_auc")),
		"class_order_by_median": str(row.get("class_order_by_median", "")),
		"auc_feat_oriented": _safe_float(row.get("auc_feature_oriented")),
		"pearson_label_binary": _safe_float(row.get("pearson_label")),
		"spearman_label_binary": _safe_float(row.get("spearman_label")),
	}


def _simple_pair_score(
		auc_a: float,
		auc_b: float,
		pearson_pair: float,
		spearman_pair: float,
		fp_a: float,
		fp_b: float,
) -> float:
	mean_auc = np.nanmean([auc_a, auc_b])
	mean_abs_corr = np.nanmean([abs(pearson_pair), abs(spearman_pair)])
	fp_penalty = np.nanmean([fp_a, fp_b])
	if not np.isfinite(fp_penalty):
		fp_penalty = 0.0
	return float(mean_auc - 0.5 * mean_abs_corr - 0.001 * fp_penalty)


def _build_pair_rows(
		feature_names: Sequence[str],
		feature_rows: Dict[str, FeatureRow],
		matrix_names: Sequence[str],
		pear_mat: np.ndarray,
		spear_mat: np.ndarray,
) -> List[PairRow]:
	index_by_name = {name: i for i, name in enumerate(matrix_names)}
	pairs: List[PairRow] = []
	for ia in range(len(feature_names)):
		for ib in range(ia + 1, len(feature_names)):
			name_a = feature_names[ia]
			name_b = feature_names[ib]
			row_a = feature_rows[name_a]
			row_b = feature_rows[name_b]
			idx_a = index_by_name.get(name_a)
			idx_b = index_by_name.get(name_b)
			if idx_a is None or idx_b is None:
				pearson_pair = float("nan")
				spearman_pair = float("nan")
			else:
				pearson_pair = _safe_float(pear_mat[idx_a, idx_b])
				spearman_pair = _safe_float(spear_mat[idx_a, idx_b])
			ops_a = row_a.get("operating_point_stats", {})
			ops_b = row_b.get("operating_point_stats", {})
			if not isinstance(ops_a, dict):
				ops_a = {}
			if not isinstance(ops_b, dict):
				ops_b = {}
			auc_a = _safe_float(row_a.get("auc_feature_oriented"))
			auc_b = _safe_float(row_b.get("auc_feature_oriented"))
			fp_a = _safe_float(ops_a.get("fp_at_100_recall"))
			fp_b = _safe_float(ops_b.get("fp_at_100_recall"))
			prec_high_a = _safe_float(ops_a.get("precision_high_zone"))
			prec_high_b = _safe_float(ops_b.get("precision_high_zone"))
			mean_auc = float(np.nanmean([auc_a, auc_b]))
			min_auc = float(np.nanmin([auc_a, auc_b])) if np.isfinite(auc_a) or np.isfinite(auc_b) else float("nan")
			mean_abs_corr = float(np.nanmean([abs(pearson_pair), abs(spearman_pair)]))
			max_abs_corr = float(np.nanmax([abs(pearson_pair), abs(spearman_pair)]))
			pairs.append(
				{
					"feature_a": name_a,
					"feature_b": name_b,
					"auc_a": auc_a,
					"auc_b": auc_b,
					"mean_auc": mean_auc,
					"min_auc": min_auc,
					"pearson_pair": pearson_pair,
					"spearman_pair": spearman_pair,
					"mean_abs_corr": mean_abs_corr,
					"max_abs_corr": max_abs_corr,
					"fp_at_100_recall_a": fp_a,
					"fp_at_100_recall_b": fp_b,
					"precision_high_zone_a": prec_high_a,
					"precision_high_zone_b": prec_high_b,
					"simple_pair_score": _simple_pair_score(auc_a, auc_b, pearson_pair, spearman_pair, fp_a, fp_b),
				}
			)
	return pairs


def _print_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str], *, title: str) -> None:
	print()
	print(title)
	if not rows:
		print("  <empty>")
		return
	int_like_cols = {"index", "no-muon", "maybe_muon", "ok-muon", "n_muon", "n_raman", "n_noise", "n_unknown_excluded"}
	widths = {col: len(col) for col in columns}
	for row in rows:
		for col in columns:
			if col in {"feature", "feature_a", "feature_b", "auc_direction", "class_order_by_median"}:
				txt = str(row.get(col, ""))
			elif col in int_like_cols:
				txt = str(int(round(_safe_float(row.get(col), 0.0))))
			else:
				txt = _fmt(row.get(col))
			widths[col] = max(widths[col], len(txt))
	header = "  ".join(col.ljust(widths[col]) for col in columns)
	print(header)
	print("  ".join("-" * widths[col] for col in columns))
	for row in rows:
		parts: List[str] = []
		for col in columns:
			value = row.get(col, "")
			if col in {"feature", "feature_a", "feature_b", "auc_direction", "class_order_by_median"}:
				txt = str(value)
			elif col in int_like_cols:
				txt = str(int(round(_safe_float(value, 0.0))))
			else:
				txt = _fmt(value)
			parts.append(txt.ljust(widths[col]))
		print("  ".join(parts))


def _save_json(path: Path, feature_rows: Sequence[FeatureRow], pair_rows: Sequence[PairRow], meta: Dict[str, Any]) -> None:
	payload = {
		"meta": meta,
		"feature_summary": list(feature_rows),
		"pairwise_summary": list(pair_rows),
	}
	path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_csv(path: Path, feature_rows: Sequence[FeatureRow], pair_rows: Sequence[PairRow]) -> None:
	fields = [
		"row_type",
		"index",
		"feature",
		"auc_feat_oriented",
		"no-muon",
		"maybe_muon",
		"ok-muon",
		"thr-low",
		"thr-high",
		"n_muon",
		"n_raman",
		"n_noise",
		"n_unknown_excluded",
		"auc_muon_vs_rest",
		"auc_raman_vs_rest",
		"auc_noise_vs_rest",
		"auc_muon_vs_raman",
		"auc_muon_vs_noise",
		"auc_raman_vs_noise",
		"macro_pairwise_auc",
		"raman_veto_auc",
		"noise_filter_auc",
		"muon_detector_auc",
		"class_order_by_median",
		"pearson_label_binary",
		"spearman_label_binary",
		"feature_a",
		"feature_b",
		"mean_auc",
		"pearson_pair",
		"spearman_pair",
		"simple_pair_score",
	]
	with path.open("w", encoding="utf-8", newline="") as f:
		w = csv.DictWriter(f, fieldnames=fields)
		w.writeheader()
		for row in feature_rows:
			out = {k: row.get(k) for k in fields}
			out["row_type"] = "feature"
			w.writerow(out)
		for row in pair_rows:
			out = {k: row.get(k) for k in fields}
			out["row_type"] = "pair"
			w.writerow(out)


def _tooltip_text(
		name_x: str,
		name_y: str,
		row_x: FeatureRow,
		row_y: FeatureRow,
		pearson_pair: float,
		spearman_pair: float,
) -> str:
	ops_x = row_x.get("operating_point_stats", {})
	ops_y = row_y.get("operating_point_stats", {})
	if not isinstance(ops_x, dict):
		ops_x = {}
	if not isinstance(ops_y, dict):
		ops_y = {}
	mean_abs_corr = np.nanmean([abs(pearson_pair), abs(spearman_pair)])
	max_abs_corr = np.nanmax([abs(pearson_pair), abs(spearman_pair)])
	return "\n".join(
		[
			f"X: {name_x}",
			f"Y: {name_y}",
			f"pearson_pair: {_fmt(pearson_pair)}",
			f"spearman_pair: {_fmt(spearman_pair)}",
			f"mean_abs_corr: {_fmt(mean_abs_corr)}",
			f"max_abs_corr: {_fmt(max_abs_corr)}",
			f"auc_X: {_fmt(row_x.get('auc_feature_oriented'))}",
			f"auc_Y: {_fmt(row_y.get('auc_feature_oriented'))}",
			f"macro_pair_X: {_fmt(row_x.get('macro_pairwise_auc'))}",
			f"macro_pair_Y: {_fmt(row_y.get('macro_pairwise_auc'))}",
			f"order_X: {row_x.get('class_order_by_median', '')}",
			f"order_Y: {row_y.get('class_order_by_median', '')}",
			f"fp@100_X: {_fmt(ops_x.get('fp_at_100_recall'))}",
			f"fp@100_Y: {_fmt(ops_y.get('fp_at_100_recall'))}",
			f"prec_high_X: {_fmt(ops_x.get('precision_high_zone'))}",
			f"prec_high_Y: {_fmt(ops_y.get('precision_high_zone'))}",
			f"hard_neg_X: {_fmt(ops_x.get('manual_hard_negative_error_count'))}",
			f"hard_neg_Y: {_fmt(ops_y.get('manual_hard_negative_error_count'))}",
		]
	)


def _prepare_axis(ax, mat: np.ndarray, names: Sequence[str], title: str):
	im = ax.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest", aspect="auto")
	labels = [_display_name(name, max_len=24) for name in names]
	ax.set_title(title)
	ax.set_xticks(np.arange(len(names)))
	ax.set_yticks(np.arange(len(names)))
	ax.set_xticklabels(labels, rotation=90, fontsize=8)
	ax.set_yticklabels(labels, fontsize=8)
	ax.set_xlim(-0.5, len(names) - 0.5)
	ax.set_ylim(len(names) - 0.5, -0.5)
	ax.set_xlabel("Feature")
	ax.set_ylabel("Feature")
	annot = ax.annotate(
		"",
		xy=(0, 0),
		xytext=(12, 12),
		textcoords="offset points",
		bbox={"boxstyle": "round", "fc": "w", "alpha": 0.95},
		fontsize=9,
		ha="left",
		va="bottom",
		annotation_clip=False,
		zorder=5,
	)
	annot.set_visible(False)
	return im, annot


def _build_small_matrices(
		selected_names: Sequence[str],
		matrix_names: Sequence[str],
		pear_mat: np.ndarray,
		spear_mat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
	index_by_name = {name: i for i, name in enumerate(matrix_names)}
	n = len(selected_names)
	pear_small = np.full((n, n), np.nan, dtype=float)
	spear_small = np.full((n, n), np.nan, dtype=float)
	for i, name_i in enumerate(selected_names):
		idx_i = index_by_name.get(name_i)
		for j, name_j in enumerate(selected_names):
			idx_j = index_by_name.get(name_j)
			if idx_i is None or idx_j is None:
				continue
			pear_small[i, j] = _safe_float(pear_mat[idx_i, idx_j])
			spear_small[i, j] = _safe_float(spear_mat[idx_i, idx_j])
	return pear_small, spear_small


def _show_gui(
		selected_names: Sequence[str],
		feature_rows: Dict[str, FeatureRow],
		pear_small: np.ndarray,
		spear_small: np.ndarray,
		corr_type: str,
) -> None:
	if corr_type == "both":
		fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=False)
		plt.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.90, wspace=0.22)
		ax_list = list(axes)
		matrix_specs = [("Pearson", pear_small), ("Spearman", spear_small)]
	else:
		fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=False)
		plt.subplots_adjust(left=0.12, right=0.94, bottom=0.22, top=0.90)
		ax_list = [ax]
		matrix_specs = [("Pearson", pear_small)] if corr_type == "pearson" else [("Spearman", spear_small)]

	axis_state: Dict[Any, Dict[str, Any]] = {}
	colorbars: Dict[Any, Any] = {}
	pinned: Dict[Tuple[int, int, int], Any] = {}

	def _rebuild_single_axis() -> None:
		ax = ax_list[0]
		ax.clear()
		for cb in colorbars.values():
			try:
				cb.remove()
			except Exception:
				pass
		colorbars.clear()
		title, mat = matrix_specs[0]
		im, annot = _prepare_axis(ax, mat, selected_names, f"{title} selected-feature correlation")
		colorbars[ax] = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		axis_state.clear()
		axis_state[ax] = {"title": title, "mat": mat, "annot": annot, "background": None}
		fig.canvas.draw()

	if corr_type == "both":
		for ax, (title, mat) in zip(ax_list, matrix_specs):
			im, annot = _prepare_axis(ax, mat, selected_names, f"{title} selected-feature correlation")
			colorbars[ax] = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
			axis_state[ax] = {"title": title, "mat": mat, "annot": annot, "background": None}
		fig.canvas.draw()
	else:
		_rebuild_single_axis()

	def _refresh_backgrounds(_event=None) -> None:
		for ax in ax_list:
			state = axis_state[ax]
			annot = state["annot"]
			was_visible = annot.get_visible()
			if was_visible:
				annot.set_visible(False)
			state["background"] = fig.canvas.copy_from_bbox(ax.bbox)
			if was_visible:
				annot.set_visible(True)

	def _place_annotation(ax, annot) -> None:
		renderer = fig.canvas.get_renderer()
		fig_box = fig.bbox
		candidates = [
			("left", "bottom", (12, 12)),
			("right", "bottom", (-12, 12)),
			("left", "top", (12, -12)),
			("right", "top", (-12, -12)),
		]
		best = None
		best_penalty = None
		for ha, va, pos in candidates:
			annot.set_ha(ha)
			annot.set_va(va)
			annot.set_position(pos)
			ext = annot.get_window_extent(renderer=renderer)
			penalty = 0.0
			penalty += max(0.0, fig_box.x0 - ext.x0)
			penalty += max(0.0, ext.x1 - fig_box.x1)
			penalty += max(0.0, fig_box.y0 - ext.y0)
			penalty += max(0.0, ext.y1 - fig_box.y1)
			if best_penalty is None or penalty < best_penalty:
				best_penalty = penalty
				best = (ha, va, pos)
				if penalty <= 0.0:
					break
		if best is not None:
			ha, va, pos = best
			annot.set_ha(ha)
			annot.set_va(va)
			annot.set_position(pos)

	def _hide_annotation(ax) -> None:
		state = axis_state[ax]
		annot = state["annot"]
		if not annot.get_visible():
			return
		background = state.get("background")
		if background is None:
			annot.set_visible(False)
			fig.canvas.draw_idle()
			return
		annot.set_visible(False)
		fig.canvas.restore_region(background)
		fig.canvas.blit(ax.bbox)

	def _show_annotation(ax, i: int, j: int) -> None:
		state = axis_state[ax]
		background = state.get("background")
		if background is None:
			_refresh_backgrounds()
			background = state.get("background")
			if background is None:
				return
		row_x = feature_rows[selected_names[j]]
		row_y = feature_rows[selected_names[i]]
		pearson_pair = _safe_float(pear_small[i, j])
		spearman_pair = _safe_float(spear_small[i, j])
		annot = state["annot"]
		annot.xy = (j, i)
		annot.set_text(_tooltip_text(selected_names[j], selected_names[i], row_x, row_y, pearson_pair, spearman_pair))
		_place_annotation(ax, annot)
		annot.set_visible(True)
		fig.canvas.restore_region(background)
		ax.draw_artist(annot)
		fig.canvas.blit(ax.bbox)

	def _on_motion(event) -> None:
		for ax in ax_list:
			if event.inaxes is not ax or event.xdata is None or event.ydata is None:
				_hide_annotation(ax)
				continue
			j = int(round(float(event.xdata)))
			i = int(round(float(event.ydata)))
			if i < 0 or j < 0 or i >= len(selected_names) or j >= len(selected_names):
				_hide_annotation(ax)
				continue
			if abs(float(event.xdata) - j) > 0.5 or abs(float(event.ydata) - i) > 0.5:
				_hide_annotation(ax)
				continue
			_show_annotation(ax, i, j)

	def _on_key(event) -> None:
		key = (event.key or "").lower()
		if key == "q":
			plt.close(fig)
		elif key == "p" and corr_type != "both":
			matrix_specs[0] = ("Spearman", spear_small) if matrix_specs[0][0] == "Pearson" else ("Pearson", pear_small)
			_rebuild_single_axis()
		elif key == "c":
			for ann in pinned.values():
				try:
					ann.remove()
				except Exception:
					pass
			pinned.clear()
			fig.canvas.draw_idle()

	def _on_click(event) -> None:
		if event.inaxes not in ax_list or event.xdata is None or event.ydata is None:
			return
		if int(getattr(event, "button", 0)) not in (1, 3):
			return
		ax = event.inaxes
		j = int(round(float(event.xdata)))
		i = int(round(float(event.ydata)))
		if i < 0 or j < 0 or i >= len(selected_names) or j >= len(selected_names):
			return
		key = (id(ax), i, j)
		if key in pinned:
			try:
				pinned[key].remove()
			except Exception:
				pass
			del pinned[key]
			fig.canvas.draw_idle()
			return
		row_x = feature_rows[selected_names[j]]
		row_y = feature_rows[selected_names[i]]
		pearson_pair = _safe_float(pear_small[i, j])
		spearman_pair = _safe_float(spear_small[i, j])
		ann = ax.annotate(
			_tooltip_text(selected_names[j], selected_names[i], row_x, row_y, pearson_pair, spearman_pair),
			xy=(j, i),
			xytext=(12, 12),
			textcoords="offset points",
			bbox={"boxstyle": "round", "fc": "w", "alpha": 0.95},
			fontsize=8,
			annotation_clip=False,
			zorder=4,
		)
		_place_annotation(ax, ann)
		pinned[key] = ann
		fig.canvas.draw_idle()

	fig.canvas.mpl_connect("draw_event", _refresh_backgrounds)
	fig.canvas.mpl_connect("resize_event", _refresh_backgrounds)
	fig.canvas.mpl_connect("motion_notify_event", _on_motion)
	fig.canvas.mpl_connect("key_press_event", _on_key)
	fig.canvas.mpl_connect("button_press_event", _on_click)
	if corr_type == "both":
		print("GUI controls: hover = tooltip, left/right click = pin/unpin, c = clear pins, q = quit")
	else:
		print("GUI controls: hover = tooltip, left/right click = pin/unpin, p = toggle Pearson/Spearman, c = clear pins, q = quit")
	plt.show()


def main() -> None:
	parser = argparse.ArgumentParser(description="Inspect a manually selected feature subset from corr.json.")
	parser.add_argument("--corr-json", type=Path, required=True, help="Path to corr.json from debug_stats.py")
	parser.add_argument("--features", nargs="*", default=[], help="Explicit feature names or supported keywords (currently: pce, gws).")
	parser.add_argument("--features-file", type=Path, default=None, help="Text file with one feature name or supported keyword per line.")
	parser.add_argument(
		"--corr-type",
		type=str,
		default="both",
		choices=["pearson", "spearman", "both"],
		help="Which correlation heatmap to display.",
	)
	parser.add_argument(
		"--sort-by",
		type=str,
		default="input",
		choices=["input", "fp_at_100_recall", *AUC_SORT_KEYS],
		help="How to order selected features. Ternary mode supports all printed AUC columns.",
	)
	parser.add_argument("--save-csv", type=Path, default=None, help="Optional flat CSV export.")
	parser.add_argument("--save-json", type=Path, default=None, help="Optional JSON export.")
	parser.add_argument(
		"--auc-min",
		type=float,
		default=None,
		help="Keep only selected features whose selected AUC key is >= this value.",
	)
	parser.add_argument(
		"--auc-min-key",
		type=str,
		default=None,
		choices=AUC_SORT_KEYS,
		help="AUC column used by --auc-min. Default: auc_feature_oriented in binary, macro_pairwise_auc in ternary.",
	)
	parser.add_argument(
		"--hide-pairwise-summary",
		action="store_true",
		help="Do not print the Pairwise Summary table in the console.",
	)
	parser.add_argument(
		"--label-mode",
		type=str,
		default="binary",
		choices=["binary", "ternary"],
		help="Read binary or ternary statistics from corr.json. Default binary preserves old output.",
	)
	args = parser.parse_args()

	feature_rows, matrix_names, pear_mat, spear_mat = _load_corr_json(args.corr_json)
	selected = _resolve_features(args.features, args.features_file, feature_rows, matrix_names)
	if args.auc_min is not None:
		auc_min = float(args.auc_min)
		auc_min_key = _normalize_auc_key(args.auc_min_key, label_mode=str(args.label_mode))
		before = len(selected)
		selected = [
			name for name in selected
			if np.isfinite(_feature_auc_value(feature_rows[name], auc_min_key))
			and _feature_auc_value(feature_rows[name], auc_min_key) >= auc_min
		]
		print(f"Applied --auc-min {auc_min:.4g} on {auc_min_key}: kept {len(selected)} of {before} selected features.")
		if not selected:
			raise ValueError(f"No selected features satisfy --auc-min {auc_min:.4g} on {auc_min_key}.")
	selected = _sort_feature_names(selected, feature_rows, args.sort_by)

	if str(args.label_mode).lower() == "ternary":
		feature_summary = [_feature_summary_row_ternary(i + 1, name, feature_rows[name]) for i, name in enumerate(selected)]
	else:
		feature_summary = [_feature_summary_row(i + 1, name, feature_rows[name]) for i, name in enumerate(selected)]
	pair_rows = _build_pair_rows(selected, feature_rows, matrix_names, pear_mat, spear_mat)
	pair_rows.sort(key=lambda row: _safe_float(row.get("simple_pair_score")), reverse=True)

	if str(args.label_mode).lower() == "ternary":
		feature_columns = [
			"index",
			"feature",
			"auc_muon_vs_rest",
			"auc_raman_vs_rest",
			"auc_noise_vs_rest",
			"auc_muon_vs_raman",
			"auc_muon_vs_noise",
			"auc_raman_vs_noise",
			"macro_pairwise_auc",
			"raman_veto_auc",
			"noise_filter_auc",
			"muon_detector_auc",
		]
	else:
		feature_columns = [
			"index",
			"feature",
			"auc_feat_oriented",
			"no-muon",
			"maybe_muon",
			"ok-muon",
			"thr-low",
			"thr-high",
		]
	pair_columns = [
		"feature_a",
		"feature_b",
		"mean_auc",
		"pearson_pair",
		"spearman_pair",
		"simple_pair_score",
	]
	_print_table(feature_summary, feature_columns, title="Feature Summary")
	if not bool(args.hide_pairwise_summary):
		_print_table(pair_rows, pair_columns, title="Pairwise Summary")

	if args.save_json is not None:
		_save_json(
			args.save_json,
			feature_summary,
			pair_rows,
			meta={
				"corr_json": str(args.corr_json),
				"selected_features": list(selected),
				"corr_type": str(args.corr_type),
				"sort_by": str(args.sort_by),
				"auc_min": (None if args.auc_min is None else float(args.auc_min)),
				"auc_min_key": (
					None
					if args.auc_min is None
					else _normalize_auc_key(args.auc_min_key, label_mode=str(args.label_mode))
				),
				"label_mode": str(args.label_mode),
			},
		)
		print(f"\nSaved JSON summary to {args.save_json}")
	if args.save_csv is not None:
		_save_csv(args.save_csv, feature_summary, pair_rows)
		print(f"Saved CSV summary to {args.save_csv}")

	pear_small, spear_small = _build_small_matrices(selected, matrix_names, pear_mat, spear_mat)
	_show_gui(selected, feature_rows, pear_small, spear_small, str(args.corr_type))


if __name__ == "__main__":
	main()
