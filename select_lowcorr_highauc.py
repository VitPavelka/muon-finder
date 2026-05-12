from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from correlation_hover_gui import _load_corr_payload
from muon_decision import get_muon_rule_v3_metric_thresholds


def _safe_float(v: Any, default: float = float("nan")) -> float:
	try:
		return float(v)
	except Exception:
		return default


def _safe_stat(row: Dict[str, Any], key: str, default: float = float("nan")) -> float:
	ops = row.get("operating_point_stats", {})
	if not isinstance(ops, dict):
		return default
	return _safe_float(ops.get(key), default)


def _purity_stats(row: Dict[str, Any]) -> Dict[str, float]:
	interval_block = row.get("empirical_purity_intervals", {})
	intervals = interval_block.get("intervals", []) if isinstance(interval_block, dict) else []
	if not isinstance(intervals, list):
		intervals = []
	no_muon = 0.0
	maybe_muon = 0.0
	ok_muon = 0.0
	mixed: List[Dict[str, Any]] = []
	for iv in intervals:
		if not isinstance(iv, dict):
			continue
		kind = str(iv.get("kind", ""))
		if kind == "nonmuon_pure":
			no_muon += _safe_float(iv.get("n_total"), 0.0)
		elif kind == "mixed":
			maybe_muon += _safe_float(iv.get("n_muon"), 0.0)
			mixed.append(iv)
		elif kind == "muon_pure":
			ok_muon += _safe_float(iv.get("n_muon"), 0.0)
	if mixed:
		best_mixed = max(mixed, key=lambda iv: (_safe_float(iv.get("n_total"), 0.0), _safe_float(iv.get("right")) - _safe_float(iv.get("left"))))
		thr_low = _safe_float(best_mixed.get("left"))
		thr_high = _safe_float(best_mixed.get("right"))
	else:
		thr_low = float("nan")
		thr_high = float("nan")
		rule_thr = get_muon_rule_v3_metric_thresholds(str(row.get("feature", "")))
		if rule_thr is not None:
			thr_low = float(rule_thr[0])
			thr_high = float(rule_thr[1])
	return {
		"no-muon": float(no_muon),
		"maybe_muon": float(maybe_muon),
		"ok-muon": float(ok_muon),
		"thr-low": float(thr_low),
		"thr-high": float(thr_high),
	}


def _passes_operating_filters(
		row: Dict[str, Any],
		use_operating_stats: bool,
		min_true_neg_at_100_recall: float | None,
		max_fp_at_100_recall: float | None,
		min_precision_high_zone: float | None,
		max_hard_negative_errors: float | None,
		min_ok_muon: float | None = None,
		min_maybe_muon: float | None = None,
		min_no_muon: float | None = None,
) -> bool:
	if not use_operating_stats:
		return True
	tn100 = _safe_float(row.get("tn_at_100_recall", float("nan")))
	fp100 = _safe_float(row.get("fp_at_100_recall", float("nan")))
	phz = _safe_float(row.get("precision_high_zone", float("nan")))
	hne = _safe_float(row.get("manual_hard_negative_error_count", float("nan")))
	ok_mu = _safe_float(row.get("ok-muon", float("nan")))
	maybe_mu = _safe_float(row.get("maybe_muon", float("nan")))
	no_mu = _safe_float(row.get("no-muon", float("nan")))
	if min_true_neg_at_100_recall is not None:
		if (not np.isfinite(tn100)) or tn100 < float(min_true_neg_at_100_recall):
			return False
	if max_fp_at_100_recall is not None and np.isfinite(fp100) and fp100 > float(max_fp_at_100_recall):
		return False
	if min_precision_high_zone is not None and np.isfinite(phz) and phz < float(min_precision_high_zone):
		return False
	if max_hard_negative_errors is not None and np.isfinite(hne) and hne > float(max_hard_negative_errors):
		return False
	if min_ok_muon is not None:
		if (not np.isfinite(ok_mu)) or ok_mu < float(min_ok_muon):
			return False
	if min_maybe_muon is not None:
		if (not np.isfinite(maybe_mu)) or maybe_mu < float(min_maybe_muon):
			return False
	if min_no_muon is not None:
		if (not np.isfinite(no_mu)) or no_mu < float(min_no_muon):
			return False
	return True


def _feature_operating_penalty_score(
		row: Dict[str, Any],
		use_operating_stats: bool,
		fp_penalty_weight: float,
		precision_weight: float,
) -> float:
	if not use_operating_stats:
		return 0.0
	precomputed = _safe_float(row.get("operating_score", float("nan")))
	if np.isfinite(precomputed):
		return float(precomputed)
	fp100 = _safe_float(row.get("fp_at_100_recall", float("nan")))
	phz = _safe_float(row.get("precision_high_zone", float("nan")))
	hne = _safe_float(row.get("manual_hard_negative_error_count", float("nan")))
	fp_term = 0.0 if not np.isfinite(fp100) else float(fp100)
	phz_term = 0.0 if not np.isfinite(phz) else float(phz)
	hne_term = 0.0 if not np.isfinite(hne) else float(hne)
	return float(-float(fp_penalty_weight) * (fp_term + hne_term) + float(precision_weight) * phz_term)


def _load_pairwise_corr_payload(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, float]]:
	"""Load the common correlation payload while tolerating extra returned metadata."""
	payload = _load_corr_payload(path)
	if not isinstance(payload, tuple) or len(payload) < 4:
		raise ValueError("_load_corr_payload returned an unexpected payload.")
	names, pear_mat, spear_mat, auc_by_feature = payload[:4]
	return names, pear_mat, spear_mat, auc_by_feature


def _prepare_operating_scores(
		rows: List[Dict[str, Any]],
		use_operating_stats: bool,
		true_neg_weight: float,
		precision_weight: float,
) -> None:
	"""Attach normalized operating scores so weighting has a stable effect."""
	if not use_operating_stats:
		for row in rows:
			row["tn_at_100_recall_norm"] = 0.0
			row["operating_score"] = 0.0
		return

	tn_vals = np.array(
		[
			_safe_float(row.get("tn_at_100_recall", float("nan")))
			for row in rows
			if np.isfinite(_safe_float(row.get("tn_at_100_recall", float("nan"))))
		],
		dtype=float,
	)
	if tn_vals.size:
		tn_min = float(np.min(tn_vals))
		tn_max = float(np.max(tn_vals))
	else:
		tn_min = 0.0
		tn_max = 0.0

	for row in rows:
		tn = _safe_float(row.get("tn_at_100_recall", float("nan")))
		if np.isfinite(tn):
			if tn_max > tn_min:
				tn_norm = float((tn - tn_min) / (tn_max - tn_min))
			else:
				tn_norm = 1.0
		else:
			tn_norm = 0.0
		phz = _safe_float(row.get("precision_high_zone", float("nan")))
		phz_term = float(np.clip(phz, 0.0, 1.0)) if np.isfinite(phz) else 0.0
		row["tn_at_100_recall_norm"] = float(tn_norm)
		row["operating_score"] = float(float(true_neg_weight) * tn_norm + float(precision_weight) * phz_term)


def _parse_required_features(text: str) -> List[str]:
	"""Parse a comma-separated list of mandatory feature names."""
	items = [part.strip() for part in str(text).split(",")]
	return [item for item in items if item]


def _pair_rows_from_corr_json(
		path: Path,
		use_operating_stats: bool = False,
		fp_penalty_weight: float = 0.0,
		precision_weight: float = 0.0,
) -> List[Dict[str, Any]]:
	data = json.loads(path.read_text(encoding="utf-8"))
	rows = data.get("correlations", [])
	if not isinstance(rows, list):
		raise ValueError("corr.json must contain list field 'correlations'")

	names, pear_mat, spear_mat, auc_by_feature = _load_pairwise_corr_payload(path)
	name_set = set(names)
	label_corr_by_feature: Dict[str, Dict[str, float]] = {}
	for row in rows:
		if not isinstance(row, dict):
			continue
		name = str(row.get("feature", "")).strip()
		if not name or name not in name_set:
			continue
		purity = _purity_stats(row)
		label_corr_by_feature[name] = {
			"auc": _safe_float(row.get("auc_feature_oriented", row.get("auc_feature_raw"))),
			"pearson_label": _safe_float(row.get("pearson_label", row.get("pearson"))),
			"spearman_label": _safe_float(row.get("spearman_label", row.get("spearman"))),
			"tn_at_100_recall": _safe_stat(row, "tn_at_100_recall"),
			"fp_at_100_recall": _safe_stat(row, "fp_at_100_recall"),
			"precision_at_100_recall": _safe_stat(row, "precision_at_100_recall"),
			"fpr_at_100_recall": _safe_stat(row, "fpr_at_100_recall"),
			"precision_high_zone": _safe_stat(row, "precision_high_zone"),
			"fp_high_zone": _safe_stat(row, "fp_high_zone"),
			"manual_hard_negative_error_count": _safe_stat(row, "manual_hard_negative_error_count"),
			"no-muon": float(purity["no-muon"]),
			"maybe_muon": float(purity["maybe_muon"]),
			"ok-muon": float(purity["ok-muon"]),
			"thr-low": float(purity["thr-low"]),
			"thr-high": float(purity["thr-high"]),
		}
	_prepare_operating_scores(list(label_corr_by_feature.values()), use_operating_stats, fp_penalty_weight, precision_weight)

	out: List[Dict[str, Any]] = []
	n = len(names)
	for i in range(n):
		f1 = names[i]
		auc1 = _safe_float(auc_by_feature.get(f1, label_corr_by_feature.get(f1, {}).get("auc")))
		if not np.isfinite(auc1):
			continue
		for j in range(i + 1, n):
			f2 = names[j]
			auc2 = _safe_float(auc_by_feature.get(f2, label_corr_by_feature.get(f2, {}).get("auc")))
			if not np.isfinite(auc2):
				continue
			pear = abs(_safe_float(pear_mat[i, j]))
			spear = abs(_safe_float(spear_mat[i, j]))
			if not np.isfinite(pear) and not np.isfinite(spear):
				continue
			valid = np.array([v for v in (pear, spear) if np.isfinite(v)], dtype=float)
			mean_abs_corr = float(np.mean(valid))
			max_abs_corr = float(np.max(valid))
			min_auc = float(min(auc1, auc2))
			mean_auc = float(0.5 * (auc1 + auc2))
			row1 = label_corr_by_feature.get(f1, {})
			row2 = label_corr_by_feature.get(f2, {})
			operating_score = float(
				0.5 * (
					_feature_operating_penalty_score(row1, use_operating_stats, fp_penalty_weight, precision_weight)
					+ _feature_operating_penalty_score(row2, use_operating_stats, fp_penalty_weight, precision_weight)
				)
			)
			pair_score = float(min_auc - 0.5 * mean_abs_corr + operating_score)
			out.append(
				{
					"feature_a": f1,
					"feature_b": f2,
					"auc_a": float(auc1),
					"auc_b": float(auc2),
					"min_auc": min_auc,
					"mean_auc": mean_auc,
					"pearson_pair": float(pear_mat[i, j]),
					"spearman_pair": float(spear_mat[i, j]),
					"abs_pearson_pair": pear,
					"abs_spearman_pair": spear,
					"mean_abs_corr": mean_abs_corr,
					"max_abs_corr": max_abs_corr,
					"operating_score": operating_score,
					"pair_score": pair_score,
					"pearson_label_a": _safe_float(label_corr_by_feature.get(f1, {}).get("pearson_label")),
					"spearman_label_a": _safe_float(label_corr_by_feature.get(f1, {}).get("spearman_label")),
					"pearson_label_b": _safe_float(label_corr_by_feature.get(f2, {}).get("pearson_label")),
					"spearman_label_b": _safe_float(label_corr_by_feature.get(f2, {}).get("spearman_label")),
					"tn_at_100_recall_a": _safe_float(label_corr_by_feature.get(f1, {}).get("tn_at_100_recall")),
					"tn_at_100_recall_norm_a": _safe_float(label_corr_by_feature.get(f1, {}).get("tn_at_100_recall_norm")),
					"fp_at_100_recall_a": _safe_float(label_corr_by_feature.get(f1, {}).get("fp_at_100_recall")),
					"precision_at_100_recall_a": _safe_float(label_corr_by_feature.get(f1, {}).get("precision_at_100_recall")),
					"fpr_at_100_recall_a": _safe_float(label_corr_by_feature.get(f1, {}).get("fpr_at_100_recall")),
					"precision_high_zone_a": _safe_float(label_corr_by_feature.get(f1, {}).get("precision_high_zone")),
					"fp_high_zone_a": _safe_float(label_corr_by_feature.get(f1, {}).get("fp_high_zone")),
					"manual_hard_negative_error_count_a": _safe_float(label_corr_by_feature.get(f1, {}).get("manual_hard_negative_error_count")),
					"no-muon_a": _safe_float(label_corr_by_feature.get(f1, {}).get("no-muon")),
					"maybe_muon_a": _safe_float(label_corr_by_feature.get(f1, {}).get("maybe_muon")),
					"ok-muon_a": _safe_float(label_corr_by_feature.get(f1, {}).get("ok-muon")),
					"thr-low_a": _safe_float(label_corr_by_feature.get(f1, {}).get("thr-low")),
					"thr-high_a": _safe_float(label_corr_by_feature.get(f1, {}).get("thr-high")),
					"tn_at_100_recall_b": _safe_float(label_corr_by_feature.get(f2, {}).get("tn_at_100_recall")),
					"tn_at_100_recall_norm_b": _safe_float(label_corr_by_feature.get(f2, {}).get("tn_at_100_recall_norm")),
					"fp_at_100_recall_b": _safe_float(label_corr_by_feature.get(f2, {}).get("fp_at_100_recall")),
					"precision_at_100_recall_b": _safe_float(label_corr_by_feature.get(f2, {}).get("precision_at_100_recall")),
					"fpr_at_100_recall_b": _safe_float(label_corr_by_feature.get(f2, {}).get("fpr_at_100_recall")),
					"precision_high_zone_b": _safe_float(label_corr_by_feature.get(f2, {}).get("precision_high_zone")),
					"fp_high_zone_b": _safe_float(label_corr_by_feature.get(f2, {}).get("fp_high_zone")),
					"manual_hard_negative_error_count_b": _safe_float(label_corr_by_feature.get(f2, {}).get("manual_hard_negative_error_count")),
					"no-muon_b": _safe_float(label_corr_by_feature.get(f2, {}).get("no-muon")),
					"maybe_muon_b": _safe_float(label_corr_by_feature.get(f2, {}).get("maybe_muon")),
					"ok-muon_b": _safe_float(label_corr_by_feature.get(f2, {}).get("ok-muon")),
					"thr-low_b": _safe_float(label_corr_by_feature.get(f2, {}).get("thr-low")),
					"thr-high_b": _safe_float(label_corr_by_feature.get(f2, {}).get("thr-high")),
				}
			)
	return out


def _feature_rows_from_corr_json(path: Path) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, List[str]]:
	data = json.loads(path.read_text(encoding="utf-8"))
	rows = data.get("correlations", [])
	if not isinstance(rows, list):
		raise ValueError("corr.json must contain list field 'correlations'")

	names, pear_mat, spear_mat, auc_by_feature = _load_pairwise_corr_payload(path)
	name_set = set(names)
	label_corr_by_feature: Dict[str, Dict[str, float]] = {}
	for row in rows:
		if not isinstance(row, dict):
			continue
		name = str(row.get("feature", "")).strip()
		if not name or name not in name_set:
			continue
		purity = _purity_stats(row)
		label_corr_by_feature[name] = {
			"auc": _safe_float(row.get("auc_feature_oriented", row.get("auc_feature_raw"))),
			"pearson_label": _safe_float(row.get("pearson_label", row.get("pearson"))),
			"spearman_label": _safe_float(row.get("spearman_label", row.get("spearman"))),
			"tn_at_100_recall": _safe_stat(row, "tn_at_100_recall"),
			"fp_at_100_recall": _safe_stat(row, "fp_at_100_recall"),
			"precision_at_100_recall": _safe_stat(row, "precision_at_100_recall"),
			"fpr_at_100_recall": _safe_stat(row, "fpr_at_100_recall"),
			"precision_high_zone": _safe_stat(row, "precision_high_zone"),
			"fp_high_zone": _safe_stat(row, "fp_high_zone"),
			"manual_hard_negative_error_count": _safe_stat(row, "manual_hard_negative_error_count"),
			"no-muon": float(purity["no-muon"]),
			"maybe_muon": float(purity["maybe_muon"]),
			"ok-muon": float(purity["ok-muon"]),
			"thr-low": float(purity["thr-low"]),
			"thr-high": float(purity["thr-high"]),
		}

	out: List[Dict[str, Any]] = []
	for i, name in enumerate(names):
		auc = _safe_float(auc_by_feature.get(name, label_corr_by_feature.get(name, {}).get("auc")))
		if not np.isfinite(auc):
			continue
		out.append(
			{
				"feature": name,
				"index": int(i),
				"auc": float(auc),
				"pearson_label": _safe_float(label_corr_by_feature.get(name, {}).get("pearson_label")),
				"spearman_label": _safe_float(label_corr_by_feature.get(name, {}).get("spearman_label")),
				"tn_at_100_recall": _safe_float(label_corr_by_feature.get(name, {}).get("tn_at_100_recall")),
				"fp_at_100_recall": _safe_float(label_corr_by_feature.get(name, {}).get("fp_at_100_recall")),
				"precision_at_100_recall": _safe_float(label_corr_by_feature.get(name, {}).get("precision_at_100_recall")),
				"fpr_at_100_recall": _safe_float(label_corr_by_feature.get(name, {}).get("fpr_at_100_recall")),
				"precision_high_zone": _safe_float(label_corr_by_feature.get(name, {}).get("precision_high_zone")),
				"fp_high_zone": _safe_float(label_corr_by_feature.get(name, {}).get("fp_high_zone")),
				"manual_hard_negative_error_count": _safe_float(label_corr_by_feature.get(name, {}).get("manual_hard_negative_error_count")),
				"no-muon": _safe_float(label_corr_by_feature.get(name, {}).get("no-muon")),
				"maybe_muon": _safe_float(label_corr_by_feature.get(name, {}).get("maybe_muon")),
				"ok-muon": _safe_float(label_corr_by_feature.get(name, {}).get("ok-muon")),
				"thr-low": _safe_float(label_corr_by_feature.get(name, {}).get("thr-low")),
				"thr-high": _safe_float(label_corr_by_feature.get(name, {}).get("thr-high")),
			}
		)
	return out, pear_mat, spear_mat, names


def _format_feature_operating_summary(row: Dict[str, Any]) -> str:
	"""Format a concise single-feature summary for no-result diagnostics."""
	return (
		f"{str(row.get('feature', ''))}: "
		f"auc={_safe_float(row.get('auc')):.5f}, "
		f"no-muon={_safe_float(row.get('no-muon')):.0f}, "
		f"maybe_muon={_safe_float(row.get('maybe_muon')):.0f}, "
		f"ok-muon={_safe_float(row.get('ok-muon')):.0f}, "
		f"tn@100={_safe_float(row.get('tn_at_100_recall')):.2f}, "
		f"fp@100={_safe_float(row.get('fp_at_100_recall')):.2f}, "
		f"prec_high={_safe_float(row.get('precision_high_zone')):.5f}, "
		f"hard_neg={_safe_float(row.get('manual_hard_negative_error_count')):.2f}"
	)


def _redundancy_to_selected(
		candidate_idx: int,
		selected_idx: List[int],
		pear_mat: np.ndarray,
		spear_mat: np.ndarray,
) -> Dict[str, float]:
	if not selected_idx:
		return {
			"mean_abs_corr_to_selected": 0.0,
			"max_abs_corr_to_selected": 0.0,
			"worst_with_index": -1,
			"worst_pair_pearson": float("nan"),
			"worst_pair_spearman": float("nan"),
		}

	agg_vals = []
	worst_score = -1.0
	worst_idx = -1
	worst_pear = float("nan")
	worst_spear = float("nan")
	for j in selected_idx:
		pear = _safe_float(pear_mat[candidate_idx, j])
		spear = _safe_float(spear_mat[candidate_idx, j])
		valid = np.array([abs(v) for v in (pear, spear) if np.isfinite(v)], dtype=float)
		if valid.size == 0:
			continue
		pair_mean = float(np.mean(valid))
		pair_max = float(np.max(valid))
		agg_vals.append(pair_mean)
		if pair_max > worst_score:
			worst_score = pair_max
			worst_idx = int(j)
			worst_pear = pear
			worst_spear = spear
	if not agg_vals:
		return {
			"mean_abs_corr_to_selected": 1.0,
			"max_abs_corr_to_selected": 1.0,
			"worst_with_index": -1,
			"worst_pair_pearson": float("nan"),
			"worst_pair_spearman": float("nan"),
		}
	return {
		"mean_abs_corr_to_selected": float(np.mean(np.asarray(agg_vals, dtype=float))),
		"max_abs_corr_to_selected": float(worst_score),
		"worst_with_index": int(worst_idx),
		"worst_pair_pearson": float(worst_pear),
		"worst_pair_spearman": float(worst_spear),
	}


def _pair_abs_corr_stats(
		i: int,
		j: int,
		pear_mat: np.ndarray,
		spear_mat: np.ndarray,
) -> Tuple[float, float]:
	pear = _safe_float(pear_mat[i, j])
	spear = _safe_float(spear_mat[i, j])
	valid = np.array([abs(v) for v in (pear, spear) if np.isfinite(v)], dtype=float)
	if valid.size == 0:
		return 1.0, 1.0
	return float(np.mean(valid)), float(np.max(valid))


def _subset_metrics(
		selected_idx: List[int],
		features_by_idx: Dict[int, Dict[str, Any]],
		pear_mat: np.ndarray,
		spear_mat: np.ndarray,
		redundancy_weight: float,
		use_operating_stats: bool = False,
		fp_penalty_weight: float = 0.0,
		precision_weight: float = 0.0,
) -> Dict[str, Any]:
	aucs = np.array([float(features_by_idx[i]["auc"]) for i in selected_idx], dtype=float)
	mean_auc = float(np.mean(aucs))
	min_auc = float(np.min(aucs))
	operating_scores = np.array(
		[
			_feature_operating_penalty_score(features_by_idx[i], use_operating_stats, fp_penalty_weight, precision_weight)
			for i in selected_idx
		],
		dtype=float,
	)
	mean_operating_score = float(np.mean(operating_scores)) if operating_scores.size else 0.0

	pair_means: List[float] = []
	pair_maxes: List[float] = []
	for pos, i in enumerate(selected_idx):
		for j in selected_idx[pos + 1:]:
			pair_mean, pair_max = _pair_abs_corr_stats(i, j, pear_mat, spear_mat)
			pair_means.append(pair_mean)
			pair_maxes.append(pair_max)

	mean_pair_corr = float(np.mean(np.asarray(pair_means, dtype=float))) if pair_means else 0.0
	max_pair_corr = float(np.max(np.asarray(pair_maxes, dtype=float))) if pair_maxes else 0.0
	subset_score = float(mean_auc - float(redundancy_weight) * mean_pair_corr + mean_operating_score)
	return {
		"size": int(len(selected_idx)),
		"mean_auc": mean_auc,
		"min_auc": min_auc,
		"mean_pair_corr": mean_pair_corr,
		"max_pair_corr": max_pair_corr,
		"mean_operating_score": mean_operating_score,
		"subset_score": subset_score,
	}


def _build_subset_output_rows(
		selected_idx: List[int],
		features_by_idx: Dict[int, Dict[str, Any]],
		pear_mat: np.ndarray,
		spear_mat: np.ndarray,
		names: List[str],
		redundancy_weight: float,
		use_operating_stats: bool,
		fp_penalty_weight: float,
		precision_weight: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
	remaining = list(selected_idx)
	ordered: List[int] = []
	while remaining:
		best_idx = None
		best_key = None
		for idx in remaining:
			redundancy = _redundancy_to_selected(idx, ordered, pear_mat, spear_mat)
			row = features_by_idx[idx]
			op_score = _feature_operating_penalty_score(row, use_operating_stats, fp_penalty_weight, precision_weight)
			step_score = float(row["auc"]) - float(redundancy_weight) * float(redundancy["mean_abs_corr_to_selected"]) + op_score
			key = (
				-step_score,
				-float(row["auc"]),
				float(redundancy["max_abs_corr_to_selected"]),
				str(row["feature"]),
			)
			if best_idx is None or key < best_key:
				best_idx = idx
				best_key = key
		assert best_idx is not None
		ordered.append(best_idx)
		remaining.remove(best_idx)

	rows: List[Dict[str, Any]] = []
	for step, idx in enumerate(ordered, start=1):
		row = features_by_idx[idx]
		redundancy = _redundancy_to_selected(idx, ordered[: step - 1], pear_mat, spear_mat)
		worst_idx = int(redundancy["worst_with_index"])
		op_score = _feature_operating_penalty_score(row, use_operating_stats, fp_penalty_weight, precision_weight)
		step_score = float(row["auc"]) - float(redundancy_weight) * float(redundancy["mean_abs_corr_to_selected"]) + op_score
		rows.append(
			{
				"step": int(step),
				"feature": str(row["feature"]),
				"auc": float(row["auc"]),
				"pearson_label": float(row["pearson_label"]),
				"spearman_label": float(row["spearman_label"]),
				"tn_at_100_recall": float(row.get("tn_at_100_recall", float("nan"))),
				"tn_at_100_recall_norm": float(row.get("tn_at_100_recall_norm", float("nan"))),
				"fp_at_100_recall": float(row["fp_at_100_recall"]),
				"precision_at_100_recall": float(row["precision_at_100_recall"]),
				"fpr_at_100_recall": float(row["fpr_at_100_recall"]),
				"precision_high_zone": float(row["precision_high_zone"]),
				"fp_high_zone": float(row["fp_high_zone"]),
				"manual_hard_negative_error_count": float(row["manual_hard_negative_error_count"]),
				"no-muon": float(row.get("no-muon", float("nan"))),
				"maybe_muon": float(row.get("maybe_muon", float("nan"))),
				"ok-muon": float(row.get("ok-muon", float("nan"))),
				"thr-low": float(row.get("thr-low", float("nan"))),
				"thr-high": float(row.get("thr-high", float("nan"))),
				"mean_abs_corr_to_selected": float(redundancy["mean_abs_corr_to_selected"]),
				"max_abs_corr_to_selected": float(redundancy["max_abs_corr_to_selected"]),
				"operating_score": float(op_score),
				"greedy_score": float(step_score),
				"worst_with_feature": (str(names[worst_idx]) if worst_idx >= 0 else ""),
				"worst_pair_pearson": float(redundancy["worst_pair_pearson"]),
				"worst_pair_spearman": float(redundancy["worst_pair_spearman"]),
			}
		)

	summary = _subset_metrics(
		ordered,
		features_by_idx,
		pear_mat,
		spear_mat,
		redundancy_weight,
		use_operating_stats=use_operating_stats,
		fp_penalty_weight=fp_penalty_weight,
		precision_weight=precision_weight,
	)
	summary["features"] = [str(features_by_idx[i]["feature"]) for i in ordered]
	return rows, summary


def _beam_search_subset(
		candidate_indices: List[int],
		required_indices: List[int],
		target_size: int,
		features_by_idx: Dict[int, Dict[str, Any]],
		pear_mat: np.ndarray,
		spear_mat: np.ndarray,
		max_corr: float,
		redundancy_weight: float,
		beam_width: int,
		use_operating_stats: bool,
		fp_penalty_weight: float,
		precision_weight: float,
) -> List[int]:
	beam: List[Tuple[int, ...]] = [tuple(sorted(required_indices))]
	best_full: List[Tuple[int, ...]] = []

	for _ in range(target_size):
		next_states: List[Tuple[Tuple[int, ...], Tuple[Any, ...]]] = []
		seen = set()
		for state in beam:
			start_pos = 0
			if state:
				start_pos = candidate_indices.index(state[-1]) + 1
			for pos in range(start_pos, len(candidate_indices)):
				idx = candidate_indices[pos]
				if idx in state:
					continue
				valid = True
				for prev in state:
					_, pair_max = _pair_abs_corr_stats(idx, prev, pear_mat, spear_mat)
					if pair_max > float(max_corr):
						valid = False
						break
				if not valid:
					continue
				new_state = tuple(sorted(state + (idx,)))
				if new_state in seen:
					continue
				seen.add(new_state)
				metrics = _subset_metrics(
					list(new_state),
					features_by_idx,
					pear_mat,
					spear_mat,
					redundancy_weight,
					use_operating_stats=use_operating_stats,
					fp_penalty_weight=fp_penalty_weight,
					precision_weight=precision_weight,
				)
				key = (
					-float(metrics["subset_score"]),
					-float(metrics["mean_auc"]),
					float(metrics["mean_pair_corr"]),
					float(metrics["max_pair_corr"]),
					-float(metrics["min_auc"]),
					tuple(str(features_by_idx[i]["feature"]) for i in new_state),
				)
				next_states.append((new_state, key))
				if len(new_state) == target_size:
					best_full.append(new_state)
		if not next_states:
			break
		next_states.sort(key=lambda item: item[1])
		beam = [state for state, _ in next_states[: max(1, int(beam_width))]]

	full_states = [state for state in best_full if len(state) == target_size]
	if full_states:
		full_states.sort(
			key=lambda state: (
				-float(_subset_metrics(
					list(state),
					features_by_idx,
					pear_mat,
					spear_mat,
					redundancy_weight,
					use_operating_stats=use_operating_stats,
					fp_penalty_weight=fp_penalty_weight,
					precision_weight=precision_weight,
				)["subset_score"]),
				-float(_subset_metrics(
					list(state),
					features_by_idx,
					pear_mat,
					spear_mat,
					redundancy_weight,
					use_operating_stats=use_operating_stats,
					fp_penalty_weight=fp_penalty_weight,
					precision_weight=precision_weight,
				)["mean_auc"]),
				float(_subset_metrics(
					list(state),
					features_by_idx,
					pear_mat,
					spear_mat,
					redundancy_weight,
					use_operating_stats=use_operating_stats,
					fp_penalty_weight=fp_penalty_weight,
					precision_weight=precision_weight,
				)["mean_pair_corr"]),
				float(_subset_metrics(
					list(state),
					features_by_idx,
					pear_mat,
					spear_mat,
					redundancy_weight,
					use_operating_stats=use_operating_stats,
					fp_penalty_weight=fp_penalty_weight,
					precision_weight=precision_weight,
				)["max_pair_corr"]),
				tuple(str(features_by_idx[i]["feature"]) for i in state),
			)
		)
		return list(full_states[0])

	beam.sort(
		key=lambda state: (
			-len(state),
			-float(_subset_metrics(
				list(state),
				features_by_idx,
				pear_mat,
				spear_mat,
				redundancy_weight,
				use_operating_stats=use_operating_stats,
				fp_penalty_weight=fp_penalty_weight,
				precision_weight=precision_weight,
			)["subset_score"]),
			-float(_subset_metrics(
				list(state),
				features_by_idx,
				pear_mat,
				spear_mat,
				redundancy_weight,
				use_operating_stats=use_operating_stats,
				fp_penalty_weight=fp_penalty_weight,
				precision_weight=precision_weight,
			)["mean_auc"]),
			float(_subset_metrics(
				list(state),
				features_by_idx,
				pear_mat,
				spear_mat,
				redundancy_weight,
				use_operating_stats=use_operating_stats,
				fp_penalty_weight=fp_penalty_weight,
				precision_weight=precision_weight,
			)["mean_pair_corr"]),
			float(_subset_metrics(
				list(state),
				features_by_idx,
				pear_mat,
				spear_mat,
				redundancy_weight,
				use_operating_stats=use_operating_stats,
				fp_penalty_weight=fp_penalty_weight,
				precision_weight=precision_weight,
			)["max_pair_corr"]),
			tuple(str(features_by_idx[i]["feature"]) for i in state),
		)
	)
	return list(beam[0]) if beam else []


def _refine_subset_by_swaps(
		selected_idx: List[int],
		required_indices: List[int],
		candidate_indices: List[int],
		features_by_idx: Dict[int, Dict[str, Any]],
		pear_mat: np.ndarray,
		spear_mat: np.ndarray,
		max_corr: float,
		redundancy_weight: float,
		use_operating_stats: bool,
		fp_penalty_weight: float,
		precision_weight: float,
) -> List[int]:
	current = sorted(selected_idx)
	required_set = set(int(i) for i in required_indices)
	if not current:
		return current

	while True:
		current_metrics = _subset_metrics(
			current,
			features_by_idx,
			pear_mat,
			spear_mat,
			redundancy_weight,
			use_operating_stats=use_operating_stats,
			fp_penalty_weight=fp_penalty_weight,
			precision_weight=precision_weight,
		)
		best_swap = None
		best_key = (
			-float(current_metrics["subset_score"]),
			-float(current_metrics["mean_auc"]),
			float(current_metrics["mean_pair_corr"]),
			float(current_metrics["max_pair_corr"]),
			-float(current_metrics["min_auc"]),
			tuple(str(features_by_idx[i]["feature"]) for i in current),
		)
		current_set = set(current)
		for out_idx in current:
			if out_idx in required_set:
				continue
			base = [i for i in current if i != out_idx]
			for in_idx in candidate_indices:
				if in_idx in current_set:
					continue
				valid = True
				for prev in base:
					_, pair_max = _pair_abs_corr_stats(in_idx, prev, pear_mat, spear_mat)
					if pair_max > float(max_corr):
						valid = False
						break
				if not valid:
					continue
				proposal = sorted(base + [in_idx])
				metrics = _subset_metrics(
					proposal,
					features_by_idx,
					pear_mat,
					spear_mat,
					redundancy_weight,
					use_operating_stats=use_operating_stats,
					fp_penalty_weight=fp_penalty_weight,
					precision_weight=precision_weight,
				)
				key = (
					-float(metrics["subset_score"]),
					-float(metrics["mean_auc"]),
					float(metrics["mean_pair_corr"]),
					float(metrics["max_pair_corr"]),
					-float(metrics["min_auc"]),
					tuple(str(features_by_idx[i]["feature"]) for i in proposal),
				)
				if key < best_key:
					best_key = key
					best_swap = proposal
		if best_swap is None:
			break
		current = best_swap
	return current


def _greedy_subset_from_corr_json(
		path: Path,
		subset_size: int,
		min_auc: float,
		max_corr: float,
		redundancy_weight: float,
		use_operating_stats: bool,
		min_true_neg_at_100_recall: float | None,
		max_fp_at_100_recall: float | None,
		min_precision_high_zone: float | None,
		max_hard_negative_errors: float | None,
		min_ok_muon: float | None,
		min_maybe_muon: float | None,
		min_no_muon: float | None,
		fp_penalty_weight: float,
		precision_weight: float,
		required_features: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
	features, pear_mat, spear_mat, names = _feature_rows_from_corr_json(path)
	required_names = [str(name).strip() for name in required_features if str(name).strip()]
	_prepare_operating_scores(features, use_operating_stats, fp_penalty_weight, precision_weight)
	feature_by_name = {str(r["feature"]): r for r in features}
	required_rows: List[Dict[str, Any]] = []
	if required_names:
		required_names = list(dict.fromkeys(required_names))
		target_size = max(1, int(subset_size))
		if len(required_names) > target_size:
			raise ValueError("Number of required features exceeds --subset-size.")
		missing = [name for name in required_names if name not in feature_by_name]
		if missing:
			raise ValueError(f"Required features not found in corr.json: {', '.join(missing)}")
		required_rows = [feature_by_name[name] for name in required_names]

	eligible_rows = [
		f for f in features
		if float(f["auc"]) >= float(min_auc)
		and _passes_operating_filters(
			f,
			use_operating_stats=use_operating_stats,
			min_true_neg_at_100_recall=min_true_neg_at_100_recall,
			max_fp_at_100_recall=max_fp_at_100_recall,
			min_precision_high_zone=min_precision_high_zone,
			max_hard_negative_errors=max_hard_negative_errors,
			min_ok_muon=min_ok_muon,
			min_maybe_muon=min_maybe_muon,
			min_no_muon=min_no_muon,
		)
	]

	candidate_map: Dict[int, Dict[str, Any]] = {}
	for row in required_rows:
		candidate_map[int(row["index"])] = row
	for row in eligible_rows:
		candidate_map[int(row["index"])] = row

	candidates = list(candidate_map.values())
	if not candidates:
		return [], {}

	candidates.sort(key=lambda r: (-float(r["auc"]), str(r["feature"])))
	target_size = max(1, int(subset_size))
	features_by_idx = {int(r["index"]): r for r in candidates}
	candidate_indices = [int(r["index"]) for r in candidates]
	beam_width = max(16, min(256, len(candidate_indices)))
	required_indices: List[int] = []
	if required_names:
		required_indices = [int(feature_by_name[name]["index"]) for name in required_names]
		for pos, i in enumerate(required_indices):
			for j in required_indices[pos + 1:]:
				_, pair_max = _pair_abs_corr_stats(i, j, pear_mat, spear_mat)
				if pair_max > float(max_corr):
					raise ValueError(
						f"Required features violate --max-corr: {features_by_idx[i]['feature']} vs {features_by_idx[j]['feature']}"
					)

	selected_idx = _beam_search_subset(
		candidate_indices=candidate_indices,
		required_indices=required_indices,
		target_size=target_size,
		features_by_idx=features_by_idx,
		pear_mat=pear_mat,
		spear_mat=spear_mat,
		max_corr=max_corr,
		redundancy_weight=redundancy_weight,
		beam_width=beam_width,
		use_operating_stats=use_operating_stats,
		fp_penalty_weight=fp_penalty_weight,
		precision_weight=precision_weight,
	)
	selected_idx = _refine_subset_by_swaps(
		selected_idx=selected_idx,
		required_indices=required_indices,
		candidate_indices=candidate_indices,
		features_by_idx=features_by_idx,
		pear_mat=pear_mat,
		spear_mat=spear_mat,
		max_corr=max_corr,
		redundancy_weight=redundancy_weight,
		use_operating_stats=use_operating_stats,
		fp_penalty_weight=fp_penalty_weight,
		precision_weight=precision_weight,
	)
	if not selected_idx:
		return [], {}
	return _build_subset_output_rows(
		selected_idx,
		features_by_idx,
		pear_mat,
		spear_mat,
		names,
		redundancy_weight,
		use_operating_stats=use_operating_stats,
		fp_penalty_weight=fp_penalty_weight,
		precision_weight=precision_weight,
	)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Rank orthogonal feature pairs by high AUC and low pairwise Pearson/Spearman correlation.",
	)
	parser.add_argument("--corr-json", type=Path, required=True, help="Path to corr.json from debug_stats.py")
	parser.add_argument("--top-k", type=int, default=20, help="How many pairs to print")
	parser.add_argument(
		"--mode",
		type=str,
		choices=["lexicographic", "composite", "greedy-subset"],
		default="composite",
		help="lexicographic/composite rank feature pairs; greedy-subset now searches for a high-AUC low-redundancy subset globally.",
	)
	parser.add_argument("--min-auc", type=float, default=0.0, help="Optional lower bound applied to both AUCs.")
	parser.add_argument("--max-corr", type=float, default=1.0, help="Optional upper bound on max(|Pearson|, |Spearman|).")
	parser.add_argument("--subset-size", type=int, default=6, help="Target size for --mode greedy-subset.")
	parser.add_argument("--redundancy-weight", type=float, default=0.5, help="Penalty multiplier for redundancy in greedy-subset mode.")
	parser.add_argument(
		"--use-operating-stats",
		action="store_true",
		help="Use operating-point false-positive statistics from corr.json in filtering and scoring.",
	)
	parser.add_argument(
		"--true-neg",
		type=float,
		default=None,
		help="Optional hard filter on tn_at_100_recall >= this threshold.",
	)
	parser.add_argument(
		"--ok-muon",
		type=float,
		default=None,
		help="Optional hard filter on muon-pure n_muon >= this threshold.",
	)
	parser.add_argument(
		"--maybe-muon",
		type=float,
		default=None,
		help="Optional hard filter on mixed n_muon >= this threshold.",
	)
	parser.add_argument(
		"--no-muon",
		type=float,
		default=None,
		help="Optional hard filter on nonmuon-pure n_total >= this threshold.",
	)
	parser.add_argument(
		"--max-fp-at-100-recall",
		type=float,
		default=None,
		help="Optional hard filter on fp_at_100_recall.",
	)
	parser.add_argument(
		"--min-precision-high-zone",
		type=float,
		default=None,
		help="Optional hard filter on precision_high_zone.",
	)
	parser.add_argument(
		"--max-hard-negative-errors",
		type=float,
		default=None,
		help="Optional hard filter on manual_hard_negative_error_count.",
	)
	parser.add_argument(
		"--required-features",
		type=str,
		default="",
		help="Comma-separated feature names that must be present in --mode greedy-subset.",
	)
	parser.add_argument(
		"--true-neg-weight",
		type=float,
		default=None,
		help="Reward weight applied to normalized tn_at_100_recall in operating-score mode.",
	)
	parser.add_argument(
		"--fp-penalty-weight",
		type=float,
		default=None,
		help="Deprecated alias for --true-neg-weight.",
	)
	parser.add_argument(
		"--precision-weight",
		type=float,
		default=0.0,
		help="Reward weight applied to precision_high_zone.",
	)
	parser.add_argument("--out-json", type=Path, default=None, help="Optional path to save ranked rows as JSON")
	args = parser.parse_args()
	subset_summary: Dict[str, Any] | None = None
	required_features = _parse_required_features(args.required_features)
	true_neg_weight = (
		float(args.true_neg_weight)
		if args.true_neg_weight is not None
		else float(args.fp_penalty_weight if args.fp_penalty_weight is not None else 0.0)
	)

	if args.mode == "greedy-subset":
		top, subset_summary = _greedy_subset_from_corr_json(
			path=args.corr_json,
			subset_size=int(args.subset_size),
			min_auc=float(args.min_auc),
			max_corr=float(args.max_corr),
			redundancy_weight=float(args.redundancy_weight),
			use_operating_stats=bool(args.use_operating_stats),
			min_true_neg_at_100_recall=args.true_neg,
			max_fp_at_100_recall=args.max_fp_at_100_recall,
			min_precision_high_zone=args.min_precision_high_zone,
			max_hard_negative_errors=args.max_hard_negative_errors,
			min_ok_muon=args.ok_muon,
			min_maybe_muon=args.maybe_muon,
			min_no_muon=args.no_muon,
			fp_penalty_weight=true_neg_weight,
			precision_weight=float(args.precision_weight),
			required_features=required_features,
		)
		if not top:
			raise ValueError("No features matched the selected thresholds for greedy-subset.")
		print(f"Ranking mode: {args.mode}")
		print(
			"Subset summary: "
			f"size={int(subset_summary['size'])} | "
			f"mean_auc={float(subset_summary['mean_auc']):.5f} | "
			f"min_auc={float(subset_summary['min_auc']):.5f} | "
			f"mean_pair_corr={float(subset_summary['mean_pair_corr']):.5f} | "
			f"max_pair_corr={float(subset_summary['max_pair_corr']):.5f} | "
			f"mean_operating_score={float(subset_summary['mean_operating_score']):.5f} | "
			f"subset_score={float(subset_summary['subset_score']):.5f}"
		)
		print(
			"Columns: step | feature | auc_feat_oriented | no-muon | maybe_muon | ok-muon | thr-low | thr-high | "
			"tn@100 | fp@100 | prec@100 | fpr@100 | prec_high | fp_high | "
			"hard_neg_err | pearson_label | spearman_label | mean|corr to selected| | max|corr to selected| | "
			"operating_score | worst_with | worst_pair_pearson | worst_pair_spearman | greedy_score"
		)
		for r in top:
			print(
				f"{int(r['step']):>4d} | "
				f"{r['feature']:<30.30s} | "
				f"{r['auc']:.5f} | "
				f"{r['no-muon']:.0f} | {r['maybe_muon']:.0f} | {r['ok-muon']:.0f} | "
				f"{r['thr-low']:.5g} | {r['thr-high']:.5g} | "
				f"{r['tn_at_100_recall']:.2f} | "
				f"{r['fp_at_100_recall']:.2f} | {r['precision_at_100_recall']:.5f} | {r['fpr_at_100_recall']:.5f} | "
				f"{r['precision_high_zone']:.5f} | {r['fp_high_zone']:.2f} | {r['manual_hard_negative_error_count']:.2f} | "
				f"{r['pearson_label']:+.5f} | {r['spearman_label']:+.5f} | "
				f"{r['mean_abs_corr_to_selected']:.5f} | {r['max_abs_corr_to_selected']:.5f} | "
				f"{r['operating_score']:.5f} | "
				f"{r['worst_with_feature']:<30.30s} | "
				f"{r['worst_pair_pearson']:+.5f} | {r['worst_pair_spearman']:+.5f} | "
				f"{r['greedy_score']:.5f}"
			)
	else:
		feature_rows, _, _, _ = _feature_rows_from_corr_json(args.corr_json)
		eligible_features = [
			r for r in feature_rows
			if float(r["auc"]) >= float(args.min_auc)
			and _passes_operating_filters(
				r,
				use_operating_stats=bool(args.use_operating_stats),
				min_true_neg_at_100_recall=args.true_neg,
				max_fp_at_100_recall=args.max_fp_at_100_recall,
				min_precision_high_zone=args.min_precision_high_zone,
				max_hard_negative_errors=args.max_hard_negative_errors,
				min_ok_muon=args.ok_muon,
				min_maybe_muon=args.maybe_muon,
				min_no_muon=args.no_muon,
			)
		]
		rows = _pair_rows_from_corr_json(
			args.corr_json,
			use_operating_stats=bool(args.use_operating_stats),
			fp_penalty_weight=true_neg_weight,
			precision_weight=float(args.precision_weight),
		)
		rows = [
			r for r in rows
			if float(r["auc_a"]) >= float(args.min_auc)
			and float(r["auc_b"]) >= float(args.min_auc)
			and float(r["max_abs_corr"]) <= float(args.max_corr)
			and _passes_operating_filters(
				{
					"tn_at_100_recall": r.get("tn_at_100_recall_a"),
					"fp_at_100_recall": r.get("fp_at_100_recall_a"),
					"precision_high_zone": r.get("precision_high_zone_a"),
					"manual_hard_negative_error_count": r.get("manual_hard_negative_error_count_a"),
					"ok-muon": r.get("ok-muon_a"),
					"maybe_muon": r.get("maybe_muon_a"),
					"no-muon": r.get("no-muon_a"),
				},
				use_operating_stats=bool(args.use_operating_stats),
				min_true_neg_at_100_recall=args.true_neg,
				max_fp_at_100_recall=args.max_fp_at_100_recall,
				min_precision_high_zone=args.min_precision_high_zone,
				max_hard_negative_errors=args.max_hard_negative_errors,
				min_ok_muon=args.ok_muon,
				min_maybe_muon=args.maybe_muon,
				min_no_muon=args.no_muon,
			)
			and _passes_operating_filters(
				{
					"tn_at_100_recall": r.get("tn_at_100_recall_b"),
					"fp_at_100_recall": r.get("fp_at_100_recall_b"),
					"precision_high_zone": r.get("precision_high_zone_b"),
					"manual_hard_negative_error_count": r.get("manual_hard_negative_error_count_b"),
					"ok-muon": r.get("ok-muon_b"),
					"maybe_muon": r.get("maybe_muon_b"),
					"no-muon": r.get("no-muon_b"),
				},
				use_operating_stats=bool(args.use_operating_stats),
				min_true_neg_at_100_recall=args.true_neg,
				max_fp_at_100_recall=args.max_fp_at_100_recall,
				min_precision_high_zone=args.min_precision_high_zone,
				max_hard_negative_errors=args.max_hard_negative_errors,
				min_ok_muon=args.ok_muon,
				min_maybe_muon=args.maybe_muon,
				min_no_muon=args.no_muon,
			)
		]
		if not rows:
			msg = [
				"No feature pairs matched the selected thresholds.",
				f"Eligible single features after AUC/operating filters: {len(eligible_features)}",
				f"Requested max_corr: {float(args.max_corr):.5f}",
			]
			if eligible_features:
				eligible_features.sort(key=lambda r: (-float(r["auc"]), str(r["feature"])))
				msg.append("Top surviving single features:")
				for row in eligible_features[: min(8, len(eligible_features))]:
					msg.append("  " + _format_feature_operating_summary(row))
			raise ValueError("\n".join(msg))

		if args.mode == "lexicographic":
			rows.sort(
				key=lambda r: (
					-float(r["min_auc"]),
					float(r["max_abs_corr"]),
					float(r["mean_abs_corr"]),
					-float(r["mean_auc"]),
					str(r["feature_a"]),
					str(r["feature_b"]),
				)
			)
		else:
			rows.sort(
				key=lambda r: (
					-float(r["pair_score"]),
					-float(r["min_auc"]),
					float(r["max_abs_corr"]),
					str(r["feature_a"]),
					str(r["feature_b"]),
				)
			)

		k = max(1, int(args.top_k))
		top = rows[:k]

		print(f"Ranking mode: {args.mode}")
		print(
			"Columns: rank | feature_a | feature_b | mean_auc | no-muon_a | maybe_muon_a | ok-muon_a | "
			"no-muon_b | maybe_muon_b | ok-muon_b | "
			"tn@100_a | tn@100_b | fp@100_a | fp@100_b | prec_high_a | prec_high_b | hard_neg_a | hard_neg_b | "
			"pearson_pair | spearman_pair | mean|corr| | max|corr| | operating_score | pair_score"
		)
		for i, r in enumerate(top, start=1):
			print(
				f"{i:>3d} | "
				f"{r['feature_a']:<30.30s} | "
				f"{r['feature_b']:<30.30s} | "
				f"{r['mean_auc']:.5f} | "
				f"{r['no-muon_a']:.0f} | {r['maybe_muon_a']:.0f} | {r['ok-muon_a']:.0f} | "
				f"{r['no-muon_b']:.0f} | {r['maybe_muon_b']:.0f} | {r['ok-muon_b']:.0f} | "
				f"{r['tn_at_100_recall_a']:.2f} | {r['tn_at_100_recall_b']:.2f} | "
				f"{r['fp_at_100_recall_a']:.2f} | {r['fp_at_100_recall_b']:.2f} | "
				f"{r['precision_high_zone_a']:.5f} | {r['precision_high_zone_b']:.5f} | "
				f"{r['manual_hard_negative_error_count_a']:.2f} | {r['manual_hard_negative_error_count_b']:.2f} | "
				f"{r['pearson_pair']:+.5f} | {r['spearman_pair']:+.5f} | "
				f"{r['mean_abs_corr']:.5f} | {r['max_abs_corr']:.5f} | {r['operating_score']:.5f} | {r['pair_score']:.5f}"
			)

	if args.out_json is not None:
		args.out_json.write_text(
			json.dumps(
				{
					"mode": args.mode,
					"min_auc": float(args.min_auc),
					"max_corr": float(args.max_corr),
					"subset_size": int(args.subset_size),
					"redundancy_weight": float(args.redundancy_weight),
					"use_operating_stats": bool(args.use_operating_stats),
					"required_features": required_features,
					"true_neg": args.true_neg,
					"max_fp_at_100_recall": args.max_fp_at_100_recall,
					"min_precision_high_zone": args.min_precision_high_zone,
					"max_hard_negative_errors": args.max_hard_negative_errors,
					"true_neg_weight": true_neg_weight,
					"fp_penalty_weight": args.fp_penalty_weight,
					"precision_weight": float(args.precision_weight),
					"subset_summary": subset_summary,
					"top": top,
				},
				ensure_ascii=False,
				indent=2,
			),
			encoding="utf-8",
		)
		print(f"Saved JSON: {args.out_json}")


if __name__ == "__main__":
	main()
