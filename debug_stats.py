from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata, norm
from candidate_labels import TERNARY_CLASSES, load_binary_labels, load_label_classes
from muon_decision import annotate_feature_dict_with_muon_rule_v3
from muon_decision import get_muon_rule_v3_metric_thresholds
try:
	from tqdm import tqdm
except Exception:
	tqdm = None


def _display_feature_name(name: str, max_len: int = 28) -> str:
	short = str(name)
	if short.startswith("gradient_multiscale_tophat_"):
		short = "gmt_" + short[len("gradient_multiscale_tophat_"):]
	elif short.startswith("multiscale_tophat_"):
		short = "mt_" + short[len("multiscale_tophat_"):]
	return short if len(short) <= max_len else (short[: max_len - 1] + "…")


def _progress(iterable, *, desc: str, enabled: bool = True, total: int | None = None):
	"""Wrap an iterable in tqdm when available and enabled."""
	if not enabled or tqdm is None:
		return iterable
	return tqdm(iterable, desc=desc, total=total, leave=False)


def _flatten_spikes(report: Dict[str, Any]) -> List[Dict[str, Any]]:
	per = report.get("per_spectrum", [])
	rows: List[Dict[str, Any]] = []
	for spec in per:
		y = int(spec.get("y", -1))
		x = int(spec.get("x", -1))
		for sp in spec.get("spikes", []):
			if not isinstance(sp, dict):
				continue
			row = {"y": y, "x": x}
			row.update(sp)
			if "muon_rule_v3_score" not in row:
				row.update(annotate_feature_dict_with_muon_rule_v3(row))
			rows.append(row)
	return rows


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
	if x.size < 3:
		return float("nan")
	xm = x - np.mean(x)
	ym = y - np.mean(y)
	den = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
	if den <= 0:
		return float("nan")
	return float(np.sum(xm * ym) / den)


def _rankdata(a: np.ndarray) -> np.ndarray:
	order = np.argsort(a)
	ranks = np.empty_like(order, dtype=float)
	ranks[order] = np.arange(1, len(a) + 1, dtype=float)
	return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
	return _pearson(_rankdata(x), _rankdata(y))


def _sigmoid(z: np.ndarray) -> np.ndarray:
	z_clip = np.clip(z, -50.0, 50.0)
	return 1.0 / (1.0 + np.exp(-z_clip))


def _roc_auc_binary(y_true: np.ndarray, score: np.ndarray) -> float:
	y = y_true.astype(int)
	if y.size < 3:
		return float("nan")
	n_pos = int(np.count_nonzero(y == 1))
	n_neg = int(np.count_nonzero(y == 0))
	if n_pos == 0 or n_neg == 0:
		return float("nan")
	ranks = rankdata(score, method="average")
	sum_pos = float(np.sum(ranks[y == 1]))
	u = sum_pos - (n_pos * (n_pos + 1) / 2.0)
	return float(u / (n_pos * n_neg))


def _class_distribution(values: np.ndarray) -> Dict[str, float]:
	x = np.asarray(values[np.isfinite(values)], dtype=float)
	if x.size == 0:
		return {
			"n": 0,
			"mean": float("nan"),
			"median": float("nan"),
			"std": float("nan"),
			"mad": float("nan"),
			"min": float("nan"),
			"max": float("nan"),
			"q10": float("nan"),
			"q25": float("nan"),
			"q75": float("nan"),
			"q90": float("nan"),
		}
	med = float(np.median(x))
	return {
		"n": int(x.size),
		"mean": float(np.mean(x)),
		"median": med,
		"std": float(np.std(x)),
		"mad": float(np.median(np.abs(x - med))),
		"min": float(np.min(x)),
		"max": float(np.max(x)),
		"q10": float(np.quantile(x, 0.10)),
		"q25": float(np.quantile(x, 0.25)),
		"q75": float(np.quantile(x, 0.75)),
		"q90": float(np.quantile(x, 0.90)),
	}


def _oriented_pairwise_auc(classes: np.ndarray, values: np.ndarray, a: str, b: str) -> Tuple[float, str]:
	mask = (classes == a) | (classes == b)
	if np.count_nonzero(mask) < 3:
		return float("nan"), ""
	y = (classes[mask] == a).astype(int)
	auc_raw = _roc_auc_binary(y, values[mask])
	if not np.isfinite(auc_raw):
		return float("nan"), ""
	if auc_raw >= 0.5:
		return float(auc_raw), f"higher_is_{a}"
	return float(1.0 - auc_raw), f"higher_is_{b}"


def _ternary_feature_stats(values: np.ndarray, classes: np.ndarray) -> Dict[str, Any]:
	"""Multiclass diagnostics for muon/Raman/noise feature behavior."""
	x = np.asarray(values, dtype=float)
	cls = np.asarray(classes, dtype=str)
	mask = np.isfinite(x) & np.isin(cls, list(TERNARY_CLASSES))
	x = x[mask]
	cls = cls[mask]
	out: Dict[str, Any] = {
		"n_muon": int(np.count_nonzero(cls == "muon")),
		"n_raman": int(np.count_nonzero(cls == "raman")),
		"n_noise": int(np.count_nonzero(cls == "noise")),
		"n_unknown_excluded": 0,
	}
	for c in TERNARY_CLASSES:
		dist = _class_distribution(x[cls == c])
		for key, val in dist.items():
			out[f"{c}_{key}"] = val
	ovr = []
	for c in TERNARY_CLASSES:
		if x.size < 3:
			auc = float("nan")
		else:
			auc = _roc_auc_binary((cls == c).astype(int), x)
		out[f"auc_{c}_vs_rest"] = float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")
		out[f"auc_{c}_vs_rest_direction"] = f"higher_is_{c}" if np.isfinite(auc) and auc >= 0.5 else f"lower_is_{c}"
		if np.isfinite(out[f"auc_{c}_vs_rest"]):
			ovr.append(float(out[f"auc_{c}_vs_rest"]))
	pairs = [
		("muon", "raman", "auc_muon_vs_raman"),
		("muon", "noise", "auc_muon_vs_noise"),
		("raman", "noise", "auc_raman_vs_noise"),
	]
	pair_vals = []
	for a, b, key in pairs:
		auc, direction = _oriented_pairwise_auc(cls, x, a, b)
		out[key] = auc
		out[f"{key}_direction"] = direction
		if np.isfinite(auc):
			pair_vals.append(float(auc))
	out["macro_ovr_auc"] = float(np.mean(ovr)) if ovr else float("nan")
	out["macro_pairwise_auc"] = float(np.mean(pair_vals)) if pair_vals else float("nan")
	out["best_pairwise_auc"] = float(np.max(pair_vals)) if pair_vals else float("nan")
	out["worst_pairwise_auc"] = float(np.min(pair_vals)) if pair_vals else float("nan")
	out["raman_veto_auc"] = out["auc_raman_vs_rest"]
	out["noise_filter_auc"] = out["auc_noise_vs_rest"]
	out["muon_detector_auc"] = out["auc_muon_vs_rest"]
	meds = [(c, out.get(f"{c}_median", float("nan"))) for c in TERNARY_CLASSES]
	if all(np.isfinite(float(v)) for _, v in meds):
		meds.sort(key=lambda item: float(item[1]))
		out["class_order_by_median"] = " < ".join(c for c, _ in meds)
	else:
		out["class_order_by_median"] = ""
	return out


def _fit_univariate_logreg(x: np.ndarray, y: np.ndarray, quadratic: bool = False) -> Dict[str, float]:
	def _nan_out(status: str) -> Dict[str, float]:
		return {
			"coef_linear_z": float("nan"),
			"coef_quadratic_z2": float("nan"),
			"intercept": float("nan"),
			"auc_logreg": float("nan"),
			"mcfadden_r2": float("nan"),
			"status": status,
		}

	if x.size < 4:
		return _nan_out("insufficient_samples")
	if np.unique(y).size < 2:
		return _nan_out("single_class")

	mu = float(np.mean(x))
	sd = float(np.std(x))
	if sd <= 0:
		return _nan_out("zero_variance")

	xz = np.clip((x - mu) / sd, -12.0, 12.0)
	cols = [np.ones_like(xz), xz]
	if quadratic:
		cols.append(xz * xz)
	X = np.vstack(cols).T

	l2 = 1e-6

	def nll(beta: np.ndarray) -> float:
		p = _sigmoid(X @ beta)
		eps = 1e-12
		base = float(-np.sum(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))
		reg = float(l2 * np.sum(beta[1:] * beta[1:])) if beta.size > 1 else 0.0
		return base + reg

	res = None
	for method in ("BFGS", "L-BFGS-B", "Powell"):
		cand = minimize(nll, x0=np.zeros(X.shape[1], dtype=float), method=method)
		if np.all(np.isfinite(cand.x)) and np.isfinite(cand.fun):
			res = cand
			if cand.success:
				break
	if res is None:
		# fallback: retry with linear-only model if quadratic fit is unstable
		if quadratic:
			out = _fit_univariate_logreg(x, y, quadratic=False)
			out['status'] = f"fallback_linear_after_failed_quadratic"
			return out
		return _nan_out("optimizer_failed")

	beta = res.x
	p_hat = _sigmoid(X @ beta)
	ll_model = -nll(beta)
	p0 = float(np.mean(y))
	eps = 1e-12
	ll_null = float(np.sum(y * np.log(p0 + eps) + (1.0 - y) * np.log(1.0 - p0 + eps)))
	mcfadden = float("nan") if ll_null == 0 else float(1.0 - (ll_model / ll_null))

	return {
		"intercept": float(beta[0]),
		"coef_linear_z": float(beta[1] if beta.size > 1 else float("nan")),
		"coef_quadratic_z2": float(beta[2] if beta.size > 2 else float("nan")),
		"auc_logreg": _roc_auc_binary(y, p_hat),
		"mcfadden_r2": mcfadden,
		"status": "ok" if bool(getattr(res, "success", False)) else "approximate_solution"
	}


def _mutual_info_binary_discrete(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
	if x.size < 4 or np.unique(y).size < 2:
		return float("nan")
	if np.allclose(np.std(x), 0.0):
		return float("nan")

	qs = np.linspace(0.0, 1.0, bins + 1)
	edges = np.quantile(x, qs)
	edges = np.unique(edges)
	if edges.size < 3:
		return float("nan")

	b = np.digitize(x, edges[1:-1], right=False)
	nx = int(np.max(b)) + 1
	ny = 2
	joint = np.zeros((nx, ny), dtype=float)
	for bi, yi in zip(b, y.astype(int)):
		joint[int(bi), int(yi)] += 1.0
	joint /= float(np.sum(joint))
	px = np.sum(joint, axis=1, keepdims=True)
	py = np.sum(joint, axis=0, keepdims=True)
	with np.errstate(divide="ignore", invalid="ignore"):
		ratio = np.where(joint > 0, joint / (px @ py), 1.0)
		log_term = np.where(joint > 0, np.log2(ratio), 0.0)
	mi = float(np.sum(joint * log_term))
	return mi


def _feature_transform(name: str, x: np.ndarray) -> Tuple[np.ndarray, str]:
	# fall slope is usually negative for sharp descending edge; abs() aligns
	# larger magnitude with "stronger muon-like evidence".
	if name == "fall_slope":
		return np.abs(x), "abs"
	return x, "identity"


def orient_feature_values(values: np.ndarray, auc_direction: str) -> np.ndarray:
	"""Orient feature values so larger score always means more muon-like."""
	x = np.asarray(values, dtype=float)
	return -x if str(auc_direction).strip().lower() == "negative" else x


def _safe_ratio(num: float, den: float) -> float | None:
	if den <= 0:
		return None
	return float(num / den)


def _confusion_at_threshold(y_true: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, int]:
	"""Compute confusion counts for the rule score >= threshold => positive."""
	y = np.asarray(y_true, dtype=int)
	s = np.asarray(score, dtype=float)
	mask = np.isfinite(s)
	if not np.any(mask):
		return {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
	y = y[mask]
	s = s[mask]
	pred = s >= float(threshold)
	pos = y == 1
	neg = y == 0
	return {
		"tp": int(np.count_nonzero(pred & pos)),
		"fn": int(np.count_nonzero((~pred) & pos)),
		"fp": int(np.count_nonzero(pred & neg)),
		"tn": int(np.count_nonzero((~pred) & neg)),
	}


def _build_conditional_subset_summary(
		paired: List[Tuple[Dict[str, Any], int]],
		results: List[Dict[str, Any]],
) -> Dict[str, Any]:
	"""Summarize the difficult ss-orange + pce-blue subset for experimental comparison."""
	subset = [
		(row, int(lbl))
		for row, lbl in paired
		if str(row.get("ss_category_v3", "")).strip().lower() == "orange"
		and str(row.get("pce_category_v3", "")).strip().lower() == "blue"
	]
	out: Dict[str, Any] = {
		"subset_name": "ss_orange_and_pce_blue",
		"n_candidates": int(len(subset)),
		"n_muon": int(sum(int(lbl) for _, lbl in subset)),
		"n_nonmuon": int(sum(1 for _, lbl in subset if not int(lbl))),
		"metrics": [],
		"most_borderline_examples": [],
	}
	if not subset:
		return out
	subset_rows = [row for row, _ in subset]
	subset_labels = np.asarray([int(lbl) for _, lbl in subset], dtype=float)
	for row in results:
		feat = str(row.get("feature", "")).strip()
		nm = feat.lower()
		if not nm.startswith(("recdw_", "rucdw_")):
			continue
		x_raw = np.asarray([float(r.get(feat, np.nan)) for r in subset_rows], dtype=float)
		mask = np.isfinite(x_raw)
		if np.count_nonzero(mask) < 3:
			continue
		x_eval, _ = _feature_transform(feat, x_raw[mask])
		y_eval = subset_labels[mask]
		auc_raw = _roc_auc_binary(y_eval, x_eval)
		auc_direction = str(row.get("auc_direction", "positive"))
		score_oriented = orient_feature_values(x_eval, auc_direction)
		threshold = row.get("operating_point_stats", {}).get("threshold_100_recall") if isinstance(row.get("operating_point_stats"), dict) else None
		try:
			thr_val = float(threshold)
		except Exception:
			thr_val = np.nan
		conf = _confusion_at_threshold(y_eval, score_oriented, thr_val) if np.isfinite(thr_val) else {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
		ss_ref = np.asarray([float(r.get("spike_score_v1", np.nan)) for r in subset_rows], dtype=float)[mask]
		pce_ref = np.asarray([float(r.get("pce_negpref_t098_evidence_signed", np.nan)) for r in subset_rows], dtype=float)[mask]
		out["metrics"].append({
			"feature": feat,
			"subset_auc_feature_oriented": float(max(auc_raw, 1.0 - auc_raw)) if np.isfinite(auc_raw) else np.nan,
			"subset_auc_direction": "positive" if (np.isfinite(auc_raw) and auc_raw >= 0.5) else "negative",
			"subset_corr_spike_score_v1": _pearson(score_oriented, ss_ref) if np.count_nonzero(np.isfinite(ss_ref)) >= 3 else np.nan,
			"subset_corr_pce_negpref_t098_evidence_signed": _pearson(score_oriented, pce_ref) if np.count_nonzero(np.isfinite(pce_ref)) >= 3 else np.nan,
			"threshold_100_recall": (float(thr_val) if np.isfinite(thr_val) else None),
			"tp": int(conf["tp"]),
			"fn": int(conf["fn"]),
			"fp": int(conf["fp"]),
			"tn": int(conf["tn"]),
		})
	out["metrics"].sort(key=lambda d: float(d.get("subset_auc_feature_oriented", float("-inf"))), reverse=True)
	out["metrics"] = out["metrics"][:20]
	if out["metrics"]:
		best_feat = str(out["metrics"][0]["feature"])
		best_row = next((r for r in results if str(r.get("feature", "")) == best_feat), None)
		if best_row is not None:
			x_raw = np.asarray([float(r.get(best_feat, np.nan)) for r in subset_rows], dtype=float)
			mask = np.isfinite(x_raw)
			if np.count_nonzero(mask):
				x_eval, _ = _feature_transform(best_feat, x_raw[mask])
				score_oriented = orient_feature_values(x_eval, str(best_row.get("auc_direction", "positive")))
				thr = best_row.get("operating_point_stats", {}).get("threshold_100_recall") if isinstance(best_row.get("operating_point_stats"), dict) else None
				try:
					thr_val = float(thr)
				except Exception:
					thr_val = np.nan
				items = []
				masked_rows = [subset_rows[i] for i, ok in enumerate(mask) if ok]
				masked_labels = [int(subset_labels[i]) for i, ok in enumerate(mask) if ok]
				for row_i, lbl_i, score_i in zip(masked_rows, masked_labels, score_oriented):
					delta = abs(float(score_i) - thr_val) if np.isfinite(thr_val) else abs(float(score_i))
					items.append({
						"y": int(row_i.get("y", -1)),
						"x": int(row_i.get("x", -1)),
						"peak_index": int(row_i.get("peak_index", -1)),
						"is_muon": int(lbl_i),
						"score": float(score_i),
						"distance_to_threshold": float(delta),
						"spike_score_v1": float(row_i.get("spike_score_v1", np.nan)),
						"pce_negpref_t098_evidence_signed": float(row_i.get("pce_negpref_t098_evidence_signed", np.nan)),
					})
				items.sort(key=lambda d: (d["distance_to_threshold"], d["y"], d["x"], d["peak_index"]))
				out["most_borderline_examples"] = items[:20]
	return out


def compute_operating_point_stats(
		values: np.ndarray,
		y_true: np.ndarray,
		auc_direction: str,
		high_zone_quantile: float,
) -> Dict[str, Any]:
	"""Compute threshold-based operating statistics for one oriented feature."""
	x = np.asarray(values, dtype=float)
	y = np.asarray(y_true, dtype=int)
	mask = np.isfinite(x) & np.isfinite(y)
	if np.count_nonzero(mask) == 0:
		return {
			"threshold_100_recall": None,
			"tp_at_100_recall": None,
			"fn_at_100_recall": None,
			"fp_at_100_recall": None,
			"tn_at_100_recall": None,
			"precision_at_100_recall": None,
			"recall_at_100_recall": None,
			"fpr_at_100_recall": None,
			"specificity_at_100_recall": None,
			"high_zone_threshold": None,
			"n_high_zone": None,
			"tp_high_zone": None,
			"fp_high_zone": None,
			"precision_high_zone": None,
			"recall_high_zone": None,
			"fpr_high_zone": None,
			"hard_negative_fp_high_zone": None,
			"hard_negative_fp_at_100_recall": None,
			"manual_hard_negative_error_count": None,
		}

	x = x[mask]
	y = y[mask]
	s = orient_feature_values(x, auc_direction)
	n_pos = int(np.count_nonzero(y == 1))
	n_neg = int(np.count_nonzero(y == 0))

	threshold_100 = None if n_pos == 0 else float(np.min(s[y == 1]))
	if threshold_100 is None:
		stats_100 = {"tp": None, "fn": None, "fp": None, "tn": None}
		precision_100 = None
		recall_100 = None
		fpr_100 = None
		spec_100 = None
		hard_neg_fp_100 = None
	else:
		stats_100 = _confusion_at_threshold(y, s, threshold_100)
		precision_100 = _safe_ratio(stats_100["tp"], stats_100["tp"] + stats_100["fp"])
		recall_100 = _safe_ratio(stats_100["tp"], n_pos)
		fpr_100 = _safe_ratio(stats_100["fp"], n_neg)
		spec_100 = _safe_ratio(stats_100["tn"], n_neg)
		hard_neg_fp_100 = int(stats_100["fp"])

	if s.size == 0:
		high_thr = None
	else:
		q = float(np.clip(high_zone_quantile, 0.0, 1.0))
		high_thr = float(np.quantile(s, q))

	if high_thr is None:
		n_high = None
		tp_high = None
		fp_high = None
		precision_high = None
		recall_high = None
		fpr_high = None
		hard_neg_fp_high = None
		manual_hard_negative_error_count = None
	else:
		stats_high = _confusion_at_threshold(y, s, high_thr)
		n_high = int(stats_high["tp"] + stats_high["fp"])
		tp_high = int(stats_high["tp"])
		fp_high = int(stats_high["fp"])
		precision_high = _safe_ratio(tp_high, n_high)
		recall_high = _safe_ratio(tp_high, n_pos)
		fpr_high = _safe_ratio(fp_high, n_neg)
		hard_neg_fp_high = int(fp_high)
		manual_hard_negative_error_count = int(fp_high)

	return {
		"threshold_100_recall": threshold_100,
		"tp_at_100_recall": stats_100["tp"],
		"fn_at_100_recall": stats_100["fn"],
		"fp_at_100_recall": stats_100["fp"],
		"tn_at_100_recall": stats_100["tn"],
		"precision_at_100_recall": precision_100,
		"recall_at_100_recall": recall_100,
		"fpr_at_100_recall": fpr_100,
		"specificity_at_100_recall": spec_100,
		"high_zone_threshold": high_thr,
		"n_high_zone": n_high,
		"tp_high_zone": tp_high,
		"fp_high_zone": fp_high,
		"precision_high_zone": precision_high,
		"recall_high_zone": recall_high,
		"fpr_high_zone": fpr_high,
		"hard_negative_fp_high_zone": hard_neg_fp_high,
		"hard_negative_fp_at_100_recall": hard_neg_fp_100,
		"manual_hard_negative_error_count": manual_hard_negative_error_count,
	}


def _empty_empirical_purity_payload(n_bins: int, purity_threshold: float, binning: str = "quantile") -> Dict[str, Any]:
	return {
		"empirical_purity_intervals": {
			"n_bins": int(max(1, n_bins)),
			"purity_threshold": float(purity_threshold),
			"binning": str(binning),
			"intervals": [],
		},
		"empirical_purity_summary": {
			"n_muon_pure_intervals": 0,
			"n_nonmuon_pure_intervals": 0,
			"best_muon_interval_left": None,
			"best_muon_interval_right": None,
			"best_muon_interval_precision": None,
			"best_muon_interval_recall_contribution": None,
			"total_muon_recall_in_muon_pure_intervals": 0.0,
			"total_fp_in_muon_pure_intervals": 0,
		},
	}


def compute_empirical_purity_intervals(
		values: np.ndarray,
		labels: np.ndarray,
		*,
		n_bins: int = 20,
		purity_threshold: float = 0.8,
		binning: str = "quantile",
) -> Dict[str, Any]:
	"""Estimate empirical muon/non-muon purity intervals from binned feature values."""
	x = np.asarray(values, dtype=float)
	y = np.asarray(labels, dtype=int)
	mask = np.isfinite(x) & np.isfinite(y)
	if np.count_nonzero(mask) < 3:
		return _empty_empirical_purity_payload(n_bins=n_bins, purity_threshold=purity_threshold, binning=binning)

	x = x[mask]
	y = y[mask]
	total_muons = int(np.count_nonzero(y == 1))
	if x.size < 3 or np.unique(x).size < 2:
		return _empty_empirical_purity_payload(n_bins=n_bins, purity_threshold=purity_threshold, binning=binning)

	nb = max(1, int(n_bins))
	if str(binning).strip().lower() == "quantile":
		edges = np.quantile(x, np.linspace(0.0, 1.0, nb + 1))
	else:
		edges = np.linspace(float(np.min(x)), float(np.max(x)), nb + 1)
	edges = np.unique(np.asarray(edges, dtype=float))
	if edges.size < 2:
		return _empty_empirical_purity_payload(n_bins=n_bins, purity_threshold=purity_threshold, binning=binning)

	bins: List[Dict[str, Any]] = []
	for i in range(edges.size - 1):
		left = float(edges[i])
		right = float(edges[i + 1])
		if i == edges.size - 2:
			in_bin = (x >= left) & (x <= right)
		else:
			in_bin = (x >= left) & (x < right)
		n_total = int(np.count_nonzero(in_bin))
		if n_total <= 0:
			continue
		n_muon = int(np.count_nonzero(y[in_bin] == 1))
		n_nonmuon = int(np.count_nonzero(y[in_bin] == 0))
		muon_fraction = float(n_muon / n_total)
		nonmuon_fraction = float(n_nonmuon / n_total)
		recall_contribution = float(n_muon / total_muons) if total_muons > 0 else 0.0
		fp_contribution = int(n_nonmuon)
		if muon_fraction >= float(purity_threshold):
			kind = "muon_pure"
		elif nonmuon_fraction >= float(purity_threshold):
			kind = "nonmuon_pure"
		else:
			kind = "mixed"
		bins.append(
			{
				"kind": kind,
				"left": left,
				"right": right,
				"n_total": n_total,
				"n_muon": n_muon,
				"n_nonmuon": n_nonmuon,
				"muon_fraction": muon_fraction,
				"nonmuon_fraction": nonmuon_fraction,
				"recall_contribution": recall_contribution,
				"fp_contribution": fp_contribution,
			}
		)

	if not bins:
		return _empty_empirical_purity_payload(n_bins=n_bins, purity_threshold=purity_threshold, binning=binning)

	intervals: List[Dict[str, Any]] = []
	for b in bins:
		if intervals and intervals[-1]["kind"] == b["kind"]:
			prev = intervals[-1]
			prev["right"] = float(b["right"])
			prev["n_total"] = int(prev["n_total"] + b["n_total"])
			prev["n_muon"] = int(prev["n_muon"] + b["n_muon"])
			prev["n_nonmuon"] = int(prev["n_nonmuon"] + b["n_nonmuon"])
			prev["fp_contribution"] = int(prev["fp_contribution"] + b["fp_contribution"])
			prev["muon_fraction"] = float(prev["n_muon"] / max(1, prev["n_total"]))
			prev["nonmuon_fraction"] = float(prev["n_nonmuon"] / max(1, prev["n_total"]))
			prev["recall_contribution"] = float(prev["n_muon"] / total_muons) if total_muons > 0 else 0.0
		else:
			intervals.append(dict(b))

	muon_intervals = [iv for iv in intervals if iv.get("kind") == "muon_pure"]
	nonmuon_intervals = [iv for iv in intervals if iv.get("kind") == "nonmuon_pure"]
	if muon_intervals:
		best_muon = max(
			muon_intervals,
			key=lambda iv: (float(iv.get("muon_fraction", 0.0)), float(iv.get("recall_contribution", 0.0)), -float(iv.get("fp_contribution", 0))),
		)
		best_left = float(best_muon.get("left"))
		best_right = float(best_muon.get("right"))
		best_precision = float(best_muon.get("muon_fraction"))
		best_recall = float(best_muon.get("recall_contribution"))
	else:
		best_left = None
		best_right = None
		best_precision = None
		best_recall = None

	return {
		"empirical_purity_intervals": {
			"n_bins": int(edges.size - 1),
			"purity_threshold": float(purity_threshold),
			"binning": str(binning),
			"intervals": intervals,
		},
		"empirical_purity_summary": {
			"n_muon_pure_intervals": int(len(muon_intervals)),
			"n_nonmuon_pure_intervals": int(len(nonmuon_intervals)),
			"best_muon_interval_left": best_left,
			"best_muon_interval_right": best_right,
			"best_muon_interval_precision": best_precision,
			"best_muon_interval_recall_contribution": best_recall,
			"total_muon_recall_in_muon_pure_intervals": float(sum(float(iv.get("recall_contribution", 0.0)) for iv in muon_intervals)),
			"total_fp_in_muon_pure_intervals": int(sum(int(iv.get("fp_contribution", 0)) for iv in muon_intervals)),
		},
	}


def _compute_fixed_rule_intervals(
		feature_name: str,
		values: np.ndarray,
		labels: np.ndarray,
) -> Dict[str, Any] | None:
	thr = get_muon_rule_v3_metric_thresholds(feature_name)
	if thr is None:
		return None
	low, high, _reverse = thr
	x = np.asarray(values, dtype=float)
	y = np.asarray(labels, dtype=int)
	mask = np.isfinite(x) & np.isfinite(y)
	if np.count_nonzero(mask) < 1:
		return None
	x = x[mask]
	y = y[mask]
	total_muons = int(np.count_nonzero(y == 1))
	zones = [
		("nonmuon_pure", float(np.min(x)) if x.size else float(low), float(low), x < float(low)),
		("mixed", float(low), float(high), (x >= float(low)) & (x < float(high))),
		("muon_pure", float(high), float(np.max(x)) if x.size else float(high), x >= float(high)),
	]
	intervals: List[Dict[str, Any]] = []
	for kind, left, right, zone_mask in zones:
		n_total = int(np.count_nonzero(zone_mask))
		n_muon = int(np.count_nonzero(y[zone_mask] == 1)) if n_total > 0 else 0
		n_nonmuon = int(np.count_nonzero(y[zone_mask] == 0)) if n_total > 0 else 0
		intervals.append(
			{
				"kind": kind,
				"left": float(left),
				"right": float(right),
				"n_total": int(n_total),
				"n_muon": int(n_muon),
				"n_nonmuon": int(n_nonmuon),
				"muon_fraction": float(n_muon / max(n_total, 1)),
				"nonmuon_fraction": float(n_nonmuon / max(n_total, 1)),
				"recall_contribution": float(n_muon / max(total_muons, 1)),
				"fp_contribution": int(n_nonmuon),
			}
		)
	muon_intervals = [iv for iv in intervals if iv["kind"] == "muon_pure"]
	nonmuon_intervals = [iv for iv in intervals if iv["kind"] == "nonmuon_pure"]
	best_muon = muon_intervals[0] if muon_intervals else None
	return {
		"empirical_purity_intervals": {
			"n_bins": 3,
			"purity_threshold": 1.0,
			"binning": "fixed_rule_thresholds",
			"intervals": intervals,
		},
		"empirical_purity_summary": {
			"n_muon_pure_intervals": int(len(muon_intervals)),
			"n_nonmuon_pure_intervals": int(len(nonmuon_intervals)),
			"best_muon_interval_left": None if best_muon is None else float(best_muon["left"]),
			"best_muon_interval_right": None if best_muon is None else float(best_muon["right"]),
			"best_muon_interval_precision": None if best_muon is None else float(best_muon["muon_fraction"]),
			"best_muon_interval_recall_contribution": None if best_muon is None else float(best_muon["recall_contribution"]),
			"total_muon_recall_in_muon_pure_intervals": float(sum(float(iv["recall_contribution"]) for iv in muon_intervals)),
			"total_fp_in_muon_pure_intervals": int(sum(int(iv["fp_contribution"]) for iv in muon_intervals)),
		},
	}


def _print_purity_interval_summary(results: List[Dict[str, Any]]) -> None:
	"""Print concise purity interval summaries for each feature."""
	for row in results:
		feature = str(row.get("feature", ""))
		interval_block = row.get("empirical_purity_intervals", {})
		summary = row.get("empirical_purity_summary", {})
		if not isinstance(interval_block, dict) or not isinstance(summary, dict):
			continue
		intervals = interval_block.get("intervals", [])
		if not isinstance(intervals, list):
			continue
		muon_intervals = [iv for iv in intervals if isinstance(iv, dict) and iv.get("kind") == "muon_pure"]
		nonmuon_intervals = [iv for iv in intervals if isinstance(iv, dict) and iv.get("kind") == "nonmuon_pure"]
		if not muon_intervals and not nonmuon_intervals:
			continue
		print(f"feature: {feature}")
		if muon_intervals:
			print("  muon-pure intervals:")
			for iv in muon_intervals:
				print(
					f"    [{float(iv.get('left', 0.0)):.6g}, {float(iv.get('right', 0.0)):.6g}] "
					f"n={int(iv.get('n_total', 0))} muon={int(iv.get('n_muon', 0))} non={int(iv.get('n_nonmuon', 0))} "
					f"precision={float(iv.get('muon_fraction', 0.0)):.3f} recall_part={float(iv.get('recall_contribution', 0.0)):.3f}"
				)
		if nonmuon_intervals:
			print("  nonmuon-pure intervals:")
			for iv in nonmuon_intervals:
				print(
					f"    [{float(iv.get('left', 0.0)):.6g}, {float(iv.get('right', 0.0)):.6g}] "
					f"n={int(iv.get('n_total', 0))} muon={int(iv.get('n_muon', 0))} non={int(iv.get('n_nonmuon', 0))} "
					f"precision_non={float(iv.get('nonmuon_fraction', 0.0)):.3f}"
				)


def _save_auc_barplot(results: List[Dict[str, Any]], out_path: Path, top_k: int) -> None:
	valid = [
		r for r in results
		if np.isfinite(r.get("auc_feature_oriented", np.nan))
	]
	if not valid:
		return
	valid.sort(key=lambda r: float(r["auc_feature_oriented"]), reverse=True)
	top = valid[: max(1, int(top_k))]
	names = [str(r["feature"]) for r in top][::-1]
	vals = [float(r["auc_feature_oriented"]) for r in top][::-1]

	plt.figure(figsize=(8, max(4, 0.35 * len(top) + 1)))
	plt.barh(names, vals)
	plt.axvline(0.5, linestyle="--", linewidth=1.0)
	plt.xlabel("AUC (best direction)")
	plt.title("Top features by oriented AUC")
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def _compute_midrank(x: np.ndarray) -> np.ndarray:
	J = np.argsort(x)
	Z = x[J]
	N = len(x)
	T = np.zeros(N, dtype=float)
	i = 0
	while i < N:
		j = i
		while j < N and Z[j] == Z[i]:
			j += 1
		T[i:j] = 0.5 * (i + j - 1) + 1.0
		i = j
	T2 = np.empty(N, dtype=float)
	T2[J] = T
	return T2


def _fast_delong(preds_sorted_t: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
	n = preds_sorted_t.shape[1] - m
	k = preds_sorted_t.shape[0]
	pos = preds_sorted_t[:, :m]
	neg = preds_sorted_t[:, m:]

	tx = np.empty((k, m), dtype=float)
	ty = np.empty((k, n), dtype=float)
	tz = np.empty((k, m + n), dtype=float)
	for r in range(k):
		tx[r, :] = _compute_midrank(pos[r, :])
		ty[r, :] = _compute_midrank(neg[r, :])
		tz[r, :] = _compute_midrank(preds_sorted_t[r, :])

	aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
	v01 = (tz[:, :m] - tx) / n
	v10 = 1.0 - (tz[:, m:] - ty) / m
	sx = np.cov(v01)
	sy = np.cov(v10)
	del_cov = sx / m + sy / n
	return aucs, del_cov


def _delong_2sample(y_true: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> Dict[str, float]:
	y = y_true.astype(int)
	m = int(np.sum(y == 1))
	n = int(np.sum(y == 0))
	if m < 2 or n < 2:
		return {'z': float("nan"), 'p_value': float("nan")}
	order = np.argsort(-y)
	preds = np.vstack([s1[order], s2[order]])
	aucs, cov = _fast_delong(preds, m)
	var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
	if var <= 0:
		return {'z': float("nan"), 'p_value': float("nan")}
	z = float((aucs[0] - aucs[1]) / np.sqrt(var))
	p = float(2.0 * (1.0 - norm.cdf(abs(z))))
	return {'z': z, 'p_value': p}


def _bootstrap_auc_ci(
		y_true: np.ndarray,
		score: np.ndarray,
		n_boot: int = 1000,
		seed: int = 0,
) -> Dict[str, float]:
	y = y_true.astype(int)
	n = y.size
	rng = np.random.default_rng(seed)
	vals = []
	for _ in range(max(10, int(n_boot))):
		idx = rng.integers(0, n, size=n)
		yb = y[idx]
		if np.unique(yb).size < 2:
			continue
		sb = score[idx]
		a = _roc_auc_binary(yb, sb)
		if np.isfinite(a):
			vals.append(float(a))
	if not vals:
		return {'auc_boot_mean': float("nan"), 'auc_boot_ci_lo': float("nan"), 'auc_boot_ci_hi': float("nan")}
	arr = np.array(vals, dtype=float)
	return {
		'auc_boot_mean': float(np.mean(arr)),
		'auc_boot_ci_lo': float(np.quantile(arr, 0.025)),
		'auc_boot_ci_hi': float(np.quantile(arr, 0.975)),
	}


def _save_auc_ci_plot(results: List[Dict[str, Any]], out_path: Path, top_k: int) -> None:
	valid = [r for r in results if np.isfinite(r.get('auc_feature_oriented', np.nan))]
	if not valid:
		return
	valid.sort(key=lambda r: float(r['auc_feature_oriented']), reverse=True)
	top = valid[: max(1, int(top_k))]
	names = [str(r['feature']) for r in top][::-1]
	auc = [float(r['auc_feature_oriented']) for r in top][::-1]
	lo = [float(r.get('auc_boot_ci_lo', np.nan)) for r in top][::-1]
	hi = [float(r.get('auc_boot_ci_hi', np.nan)) for r in top][::-1]
	err_left = np.array([max(0.0, a - l) if np.isfinite(l) else 0.0 for a, l in zip(auc, lo)], dtype=float)
	err_right = np.array([max(0.0, h - a) if np.isfinite(h) else 0.0 for a, h in zip(auc, hi)], dtype=float)

	plt.figure(figsize=(9, max(4, 0.35 * len(top) + 1)))
	plt.barh(names, auc, xerr=np.vstack([err_left, err_right]), capsize=2)
	plt.axvline(0.5, linestyle='--', linewidth=1.0, color="k")
	plt.xlabel("AUC (oriented) with 95 % bootstrap CI")
	plt.title("Top features: AUC with bootstrap CI")
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def _save_auc_mi_scatter(results: List[Dict[str, Any]], out_path: Path, top_k: int) -> None:
	valid = [
		r for r in results
		if np.isfinite(r.get('auc_feature_oriented', np.nan))
		and np.isfinite(r.get('mutual_info_bits', np.nan))
	]
	if not valid:
		return
	auc = np.array([float(r['auc_feature_oriented']) for r in valid], dtype=float)
	mi = np.array([float(r['mutual_info_bits']) for r in valid], dtype=float)
	names = [str(r['feature']) for r in valid]

	plt.figure(figsize=(7, 5))
	plt.scatter(auc, mi, alpha=0.75, s=28)
	plt.axvline(0.5, linestyle='--', linewidth=1.0, color="k")
	plt.xlabel("AUC (oriented)")
	plt.ylabel("Mutual info (bits)")
	plt.title("Feature map: discrimination vs. information")

	# annotate top-k by AUC
	order = np.argsort(-auc)
	for j in order[: max(1, int(top_k))]:
		plt.annotate(names[int(j)], (auc[int(j)], mi[int(j)]), fontsize=8, alpha=0.9)

	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def _show_interactive_auc_mi_scatter(results: List[Dict[str, Any]]) -> None:
	valid = [
		r for r in results
		if np.isfinite(r.get('auc_feature_oriented', np.nan))
		and np.isfinite(r.get('mutual_info_bits', np.nan))
	]
	if not valid:
		return
	auc = np.array([float(r['auc_feature_oriented']) for r in valid], dtype=float)
	mi = np.array([float(r['mutual_info_bits']) for r in valid], dtype=float)
	names = [str(r['feature']) for r in valid]

	def _color(a: float) -> str:
		if a < 0.75:
			return "#1f77b4"
		if a < 0.90:
			return "#ff7f0e"
		return "#2ca02c"

	colors = [_color(v) for v in auc]
	fig, ax = plt.subplots(figsize=(8, 6))
	sc = ax.scatter(auc, mi, c=colors, s=32, alpha=0.8)
	ax.axvline(0.5, linestyle='--', linewidth=1.0, color="k")
	ax.set_xlabel("AUC (oriented)")
	ax.set_ylabel("Mutual info (bits)")
	ax.set_title("Feature map (interactive): hover for feature name")

	annot = ax.annotate(
		"",
		xy=(0, 0),
		xytext=(12, 12),
		textcoords="offset points",
		bbox=dict(boxstyle="round", fc="w", alpha=0.85)
	)
	annot.set_visible(False)
	highlight = ax.scatter([], [], s=90, facecolors="none", edgecolors="red", linewidths=1.4, zorder=5)
	plt.xlim(left=0.5)
	plt.grid(alpha=0.25)

	def _clear_hover() -> None:
		if annot.get_visible():
			annot.set_visible(False)
			highlight.set_offsets(np.empty((0, 2)))
			fig.canvas.draw_idle()

	def _nearest_point_index(event) -> int | None:
		if event.x is None or event.y is None:
			return None
		points_px = ax.transData.transform(np.column_stack([auc, mi]))
		cursor_px = np.array([float(event.x), float(event.y)], dtype=float)
		d2 = np.sum((points_px - cursor_px) ** 2, axis=1)
		if d2.size == 0:
			return None
		i = int(np.argmin(d2))
		max_dist_px = 14.0
		if float(d2[i]) > max_dist_px * max_dist_px:
			return None
		return i

	def _on_move(event):
		if event.inaxes != ax:
			_clear_hover()
			return
		i = _nearest_point_index(event)
		if i is None:
			_clear_hover()
			return
		annot.xy = (auc[i], mi[i])
		annot.set_text(names[i])
		annot.set_visible(True)
		highlight.set_offsets(np.array([[auc[i], mi[i]]], dtype=float))
		fig.canvas.draw_idle()

	fig.canvas.mpl_connect("motion_notify_event", _on_move)
	plt.tight_layout()
	plt.show()


def _save_logreg_curves(
		paired: List[Tuple[Dict[str, Any], int]],
		features: List[str],
		out_dir: Path,
) -> None:
	for feat in features:
		x = np.array([float(r.get(feat, np.nan)) for r, _ in paired], dtype=float)
		y = np.array([int(lbl) for _, lbl in paired], dtype=float)
		mask = np.isfinite(x)
		if np.count_nonzero(mask) < 5:
			continue
		xv = x[mask]
		yv = y[mask]
		res = _fit_univariate_logreg(xv, yv, quadratic=True)
		mu = float(np.mean(xv))
		sd = float(np.std(xv))
		if sd <= 0:
			continue
		xx = np.linspace(float(np.min(xv)), float(np.max(xv)), 250)
		xz = np.clip((xx - mu) / sd, -12.0, 12.0)
		b0 = float(res.get('intercept', 0.0))
		b1 = float(res.get('coef_linear_z', 0.0))
		b2 = float(res.get('coef_quadratic_z2', 0.0))
		pp = _sigmoid(b0 + b1 * xz + b2 * (xz ** 2))

		plt.figure(figsize=(6.5,  4.2))
		plt.scatter(xv, yv, s=14, alpha=0.3, label="labels (0/1)")
		plt.plot(xx, pp, color="C3", linewidth=2.0, label="logreg fit")
		plt.ylim(-0.05, 1.05)
		plt.xlabel(feat)
		plt.ylabel("P(muon)")
		plt.title(f"Logistic curve: {feat}")
		plt.grid(alpha=0.25)
		plt.legend(loc="best")
		plt.tight_layout()
		plt.savefig(out_dir / f"logreg_curve_{feat}.png", dpi=150)
		plt.close()


def _sanitize_nonfinite(v: Any) -> Any:
	if isinstance(v, dict):
		return {k: _sanitize_nonfinite(x) for k, x in v.items()}
	if isinstance(v, list):
		return [_sanitize_nonfinite(x) for x in v]
	if isinstance(v, (float, np.floating)):
		return float(v) if np.isfinite(v) else 0.0
	return v


def _is_pipeline_feature_key(name: str) -> bool:
	"""Keep default stats focused on the current core spike/PCE pipeline."""
	key = str(name).strip()
	nm = key.lower()
	if key.startswith("spike_score_v1"):
		return True
	if key == "ss4" or key.startswith("ss4_"):
		return True
	if key.startswith("spike_score_v4"):
		return True
	if nm.startswith("rise_slope") or nm.startswith("fall_slope"):
		return True
	if nm.startswith("ss_"):
		return True
	if nm.startswith(("width_", "width_ratio_")):
		return True
	if key == "pce_negpref_t098_evidence_signed":
		return True
	if key in {
		"peak_curvature_extreme",
		"peak_curvature_extreme_z",
		"peak_curvature_extreme_negpref_t098",
		"peak_curvature_extreme_negpref_t098_z",
		"peak_curvature_extreme_negpref_t098_switched",
	}:
		return True
	if nm.startswith("pce_negpref_t098_"):
		return True
	if nm.startswith(("recdw_", "rucdw_")):
		return True
	return False


def _build_spike_score_v4_reason_breakdown(
		paired: List[Tuple[Dict[str, Any], int]],
		class_arr: np.ndarray,
) -> List[Dict[str, Any]]:
	"""Count v4 rule reasons, optionally split by ternary label_class."""
	reason_order = [
		"ss_blue",
		"ss_orange_pce_red",
		"ss_orange_pce_blue_rve_red",
		"ss_orange_pce_blue_rve_blue",
		"ss_red_pce_red",
		"ss_red_pce_blue_rve_red",
		"ss_red_pce_blue_rve_blue",
		"review_missing",
	]
	rows_by_reason: Dict[str, Dict[str, int]] = {
		reason: {
			"n_total": 0,
			"n_muon": 0,
			"n_raman": 0,
			"n_noise": 0,
			"n_nonmuon_binary": 0,
		}
		for reason in reason_order
	}
	has_classes = bool(class_arr.size == len(paired))
	for idx, (row, binary_label) in enumerate(paired):
		reason = str(row.get("ss4_reason", row.get("spike_score_v4_reason", "")) or "missing")
		if reason not in rows_by_reason:
			rows_by_reason[reason] = {
				"n_total": 0,
				"n_muon": 0,
				"n_raman": 0,
				"n_noise": 0,
				"n_nonmuon_binary": 0,
			}
		dst = rows_by_reason[reason]
		dst["n_total"] += 1
		if has_classes:
			cls = str(class_arr[idx]).strip().lower()
			if cls == "muon":
				dst["n_muon"] += 1
			elif cls == "raman":
				dst["n_raman"] += 1
			elif cls == "noise":
				dst["n_noise"] += 1
		else:
			if int(binary_label) == 1:
				dst["n_muon"] += 1
			else:
				dst["n_nonmuon_binary"] += 1
	out: List[Dict[str, Any]] = []
	ordered_reasons = reason_order + sorted(r for r in rows_by_reason.keys() if r not in reason_order)
	for reason in ordered_reasons:
		counts = rows_by_reason.get(reason, {})
		if int(counts.get("n_total", 0)) <= 0:
			continue
		out.append({"reason": reason, **counts})
	return out


def _spike_score_v4_candidate_row(row: Dict[str, Any]) -> Dict[str, Any]:
	return {
		"y": int(row.get("y", -1)),
		"x": int(row.get("x", -1)),
		"peak_index": int(row.get("peak_index", -1)),
		"peak_position_cm1": row.get("peak_position_cm1"),
		"spike_score_v1": row.get("spike_score_v1"),
		"pce_negpref_t098_evidence_signed": row.get("pce_negpref_t098_evidence_signed"),
		"ss4_rve_value": row.get("ss4_rve_value", row.get("spike_score_v4_edge_value")),
		"ss4_decision": row.get("ss4_decision", row.get("spike_score_v4_decision")),
		"ss4_reason": row.get("ss4_reason", row.get("spike_score_v4_reason")),
		"ss4_ss_zone": row.get("ss4_ss_zone", row.get("spike_score_v4_ss_zone")),
		"ss4_pce_zone": row.get("ss4_pce_zone", row.get("spike_score_v4_pce_zone")),
		"ss4_rve_zone": row.get("ss4_rve_zone", row.get("spike_score_v4_edge_zone")),
		"spike_score_v4_edge_value": row.get("spike_score_v4_edge_value"),
		"spike_score_v4_decision": row.get("spike_score_v4_decision"),
		"spike_score_v4_reason": row.get("spike_score_v4_reason"),
		"spike_score_v4_ss_zone": row.get("spike_score_v4_ss_zone"),
		"spike_score_v4_pce_zone": row.get("spike_score_v4_pce_zone"),
		"spike_score_v4_edge_zone": row.get("spike_score_v4_edge_zone"),
	}


def _build_spike_score_v4_confusion(paired: List[Tuple[Dict[str, Any], int]]) -> Dict[str, Any]:
	tp = fp = fn = tn = skipped = 0
	false_positives: List[Dict[str, Any]] = []
	false_negatives: List[Dict[str, Any]] = []
	for row, label in paired:
		try:
			score = float(row.get("ss4", row.get("spike_score_v4", row.get("spike_score_v4_three_friends", np.nan))))
		except Exception:
			score = float("nan")
		if not np.isfinite(score):
			skipped += 1
			continue
		pred_muon = bool(score >= 0.5)
		true_muon = bool(int(label) == 1)
		if pred_muon and true_muon:
			tp += 1
		elif pred_muon and not true_muon:
			fp += 1
			false_positives.append(_spike_score_v4_candidate_row(row))
		elif (not pred_muon) and true_muon:
			fn += 1
			false_negatives.append(_spike_score_v4_candidate_row(row))
		else:
			tn += 1
	n_eval = tp + fp + fn + tn
	return {
		"tp": int(tp),
		"fp": int(fp),
		"fn": int(fn),
		"tn": int(tn),
		"skipped_review_or_missing": int(skipped),
		"n_eval": int(n_eval),
		"accuracy": float((tp + tn) / max(1, n_eval)),
		"precision": float(tp / max(1, tp + fp)),
		"recall": float(tp / max(1, tp + fn)),
		"specificity": float(tn / max(1, tn + fp)),
		"false_positives": false_positives[:100],
		"false_negatives": false_negatives[:100],
	}


def _save_corr_heatmap(names: List[str], mat: np.ndarray, out_path: Path, title: str) -> None:
	if mat.size == 0:
		return
	plt.figure(figsize=(max(6, 0.32 * len(names)), max(5, 0.28 * len(names))))
	im = plt.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
	plt.colorbar(im, fraction=0.046, pad=0.04)
	labels = [_display_feature_name(name, max_len=32) for name in names]
	plt.xticks(np.arange(len(names)), labels, rotation=90, fontsize=7)
	plt.yticks(np.arange(len(names)), labels, fontsize=7)
	plt.title(title)
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Correlation helper for debug_report spikes.",
		allow_abbrev=False,
	)
	parser.add_argument("--h", action="help", help="Show this help message and exit.")
	parser.add_argument("--report", type=Path, required=True, help="Path to debug_report/debug.json")
	parser.add_argument(
		"--labels-csv",
		type=Path,
		required=True,
		help="CSV labels with columns y,x,peak_index,is_muon and optionally label_class.",
	)
	parser.add_argument(
		"--label-mode",
		type=str,
		default="binary",
		choices=["binary", "ternary"],
		help="Evaluation label mode. Default binary preserves the legacy muon/non-muon workflow.",
	)
	parser.add_argument(
		"--unlabeled-as-noise",
		action="store_true",
		help="In ternary mode only, treat unknown/unlabeled CSV rows as noise.",
	)
	parser.add_argument("--out-json", type=Path, default=None, help="Optional output JSON path")
	parser.add_argument(
		"--logreg-quadratic",
		action="store_true",
		help="Fit quadratic univariate logistic regression (z + z^2) in addition to linear term."
	)
	parser.add_argument(
		"--mi-bins",
		type=int,
		default=10,
		help="Number of quantile bins for mutual information estimate."
	)
	parser.add_argument(
		"--plots-dir", type=Path, default=None, help="Optional director for summary plots."
	)
	parser.add_argument(
		"--top-k", type=int, default=12, help="Top K features for ranking plot."
	)
	parser.add_argument(
		"--bootstrap-iters", type=int, default=1000, help="Bootstrap iterations for AUC CI."
	)
	parser.add_argument(
		"--bootstrap-seed", type=int, default=42, help="Random seed for bootstrap."
	)
	parser.add_argument(
		"--interactive-scatter",
		action="store_true",
		help="Show interactive AUC-vs-MI scatter with hover labels.",
	)
	parser.add_argument(
		"--plot-features",
		type=str,
		default="",
		help="Comma-separated feature names for logistic curve PNG plots.",
	)
	parser.add_argument(
		"--include-all-features",
		action="store_true",
		help="Include all numeric features (default hides weaker legacy features)."
	)
	parser.add_argument(
		"--high-zone-quantile",
		type=float,
		default=0.95,
		help="Quantile of oriented score used as the default high-score operating zone."
	)
	parser.add_argument(
		"--no-progress",
		action="store_true",
		help="Disable tqdm progress bars even if tqdm is installed.",
	)
	parser.add_argument(
		"--purity-bins",
		type=int,
		default=20,
		help="Number of quantile bins used for empirical purity intervals.",
	)
	parser.add_argument(
		"--purity-threshold",
		type=float,
		default=0.8,
		help="Bin purity threshold used to label bins as muon-pure or nonmuon-pure.",
	)
	parser.add_argument(
		"--print-purity-intervals",
		action="store_true",
		help="Print concise empirical purity intervals for each feature.",
	)
	args = parser.parse_args()
	show_progress = not bool(args.no_progress)

	report_path = Path(args.report)
	report = json.loads(report_path.read_text(encoding="utf-8"))
	if not isinstance(report, dict) or not isinstance(report.get("per_spectrum"), list):
		raise ValueError(
			f"--report must point to debug_report/debug.json with a 'per_spectrum' section, got {report_path}. "
			"Did you accidentally pass config.json?"
		)
	rows = _flatten_spikes(report)
	label_mode = str(args.label_mode).strip().lower()
	if label_mode == "ternary":
		all_label_classes = load_label_classes(Path(args.labels_csv), include_unknown=True, unlabeled_as_noise=False)
		label_classes = load_label_classes(
			Path(args.labels_csv),
			include_unknown=False,
			unlabeled_as_noise=bool(args.unlabeled_as_noise),
		)
		labels = {key: (1 if cls == "muon" else 0) for key, cls in label_classes.items()}
	else:
		all_label_classes = {}
		label_classes = {}
		labels = load_binary_labels(Path(args.labels_csv))

	paired = []
	paired_classes: List[str] = []
	for r in rows:
		key = (int(r['y']), int(r['x']), int(r['peak_index']))
		if key in labels:
			paired.append((r, labels[key]))
			if label_mode == "ternary":
				paired_classes.append(str(label_classes[key]))

	if not paired:
		report_keys = sorted({(int(r.get("y", -1)), int(r.get("x", -1)), int(r.get("peak_index", -1))) for r in rows})[:5]
		label_keys = sorted(labels.keys())[:5]
		raise ValueError(
			"No overlapping labeled spikes found between report and labels CSV. "
			f"report={args.report} rows={len(rows)} labels={len(labels)} "
			f"example_report_keys={report_keys} example_label_keys={label_keys}"
		)

	feature_names_all = sorted(
		k for k, v in paired[0][0].items()
		if k not in {"y", "x", "feature_source", "muon_score_components"} and isinstance(v, (int, float))
	)
	if bool(args.include_all_features):
		feature_names = list(feature_names_all)
	else:
		feature_names = [
			f for f in feature_names_all
			if _is_pipeline_feature_key(f)
		]

	y = np.array([lbl for _, lbl in paired], dtype=float)
	class_arr = np.asarray(paired_classes, dtype=str) if label_mode == "ternary" else np.asarray([], dtype=str)
	results = []
	for feat in _progress(feature_names, desc="Feature stats", enabled=show_progress, total=len(feature_names)):
		x = np.array([float(r.get(feat, np.nan)) for r, _ in paired], dtype=float)
		mask = np.isfinite(x)
		if np.count_nonzero(mask) < 3:
			continue
		xv_raw = x[mask]
		yv = y[mask]
		xv, transform = _feature_transform(feat, xv_raw)
		auc_raw = _roc_auc_binary(yv, xv)
		auc_oriented = float(max(auc_raw, 1.0 - auc_raw)) if np.isfinite(auc_raw) else float("nan")
		auc_direction = "positive" if (np.isfinite(auc_raw) and auc_raw >= 0.5) else "negative"
		ternary_stats = {}
		if label_mode == "ternary" and class_arr.size == y.size:
			ternary_stats = _ternary_feature_stats(xv, class_arr[mask])
		results.append(
			{
				"feature": feat,
				"n": int(xv.size),
				"transform": transform,
				"pearson_label": _pearson(xv, yv),
				"spearman_label": _spearman(xv, yv),
				"auc_feature_raw": auc_raw,
				"auc_feature_oriented": auc_oriented,
				"auc_direction": auc_direction,
				"operating_point_stats": compute_operating_point_stats(
					values=xv,
					y_true=yv,
					auc_direction=auc_direction,
					high_zone_quantile=float(args.high_zone_quantile),
				),
				**(
					_compute_fixed_rule_intervals(feat, xv, yv)
					or compute_empirical_purity_intervals(
						values=xv,
						labels=yv,
						n_bins=max(2, int(args.purity_bins)),
						purity_threshold=float(args.purity_threshold),
						binning="quantile",
					)
				),
				"mutual_info_bits": _mutual_info_binary_discrete(xv, yv, bins=max(2, int(args.mi_bins))),
				"logreg": _fit_univariate_logreg(xv, yv, quadratic=bool(args.logreg_quadratic)),
				**ternary_stats,
				"_x_eval": xv,
				"_y_eval": yv,
			}
		)

	# bootstrap CI per feature
	for i, r in _progress(
		list(enumerate(results)),
		desc="Bootstrap AUC CI",
		enabled=show_progress,
		total=len(results),
	):
		xv = np.array(r.get('_x_eval', []), dtype=float)
		yv = np.array(r.get('_y_eval', []), dtype=float)
		if xv.size < 4:
			continue
		orient = -1.0 if str(r.get('auc_direction')) == "negative" else 1.0
		boot = _bootstrap_auc_ci(
			y_true=yv,
			score=orient * xv,
			n_boot=int(args.bootstrap_iters),
			seed=int(args.bootstrap_seed + i),
		)
		r.update(boot)

	results.sort(
		key=lambda r: r["auc_feature_oriented"] if np.isfinite(r["auc_feature_oriented"]) else -1,
		reverse=True,
	)

	# feature redundancy matrices (pairwise complete-case)
	corr_features = [str(r['feature']) for r in results]
	nf = len(corr_features)
	pear_mat = np.full((nf, nf), np.nan, dtype=float)
	spea_mat = np.full((nf, nf), np.nan, dtype=float)
	for i, fi in _progress(
		list(enumerate(corr_features)),
		desc="Feature correlation matrix",
		enabled=show_progress,
		total=len(corr_features),
	):
		xi = np.array([float(rr.get(fi, np.nan)) for rr, _ in paired], dtype=float)
		xi, _ = _feature_transform(fi, xi)
		for j, fj in enumerate(corr_features):
			xj = np.array([float(rr.get(fj, np.nan)) for rr, _ in paired], dtype=float)
			xj, _ = _feature_transform(fj, xj)
			m = np.isfinite(xi) & np.isfinite(xj)
			if np.count_nonzero(m) < 3:
				continue
			pear_mat[i, j] = _pearson(xi[m], xj[m])
			spea_mat[i, j] = _spearman(xi[m], xj[m])

	# DeLong: compare each feature against the best-oriented feature
	delong_vs_best: List[Dict[str, Any]] = []
	if results:
		best = results[0]
		xb = np.array(best.get('_x_eval', []), dtype=float)
		yb = np.array(best.get('_y_eval', []), dtype=float)
		best_orient = -1.0 if str(best.get('auc_direction')) == "negative" else 1.0
		sb = best_orient * xb
		for r in _progress(
			results[1:],
			desc="DeLong vs best",
			enabled=show_progress,
			total=max(0, len(results) - 1),
		):
			xr = np.array(r.get('_x_eval', []), dtype=float)
			yr = np.array(r.get('_y_eval', []), dtype=float)
			if xr.size != sb.size or yr.size != yb.size:
				continue
			orient = -1.0 if str(r.get('auc_direction')) == "negative" else 1.0
			sr = orient * xr
			d = _delong_2sample(yb, sb, sr)
			delong_vs_best.append(
				{
					'best_feature': str(best['feature']),
					'other_feature': str(r['feature']),
					'auc_best': float(best.get('auc_feature_oriented', float("nan"))),
					'auc_other': float(r.get('auc_feature_oriented', float("nan"))),
					'delta_auc': float(best.get('auc_feature_oriented', float("nan")) - r.get('auc_feature_oriented', float("nan"))),
					'z': float(d.get('z', float("nan"))),
					'p_value': float(d.get('p_value', float("nan"))),
				}
			)

	for r in results:
		r.pop('_x_eval', None)
		r.pop('_y_eval', None)

	def _summary_rows_sorted(stat_key: str, reverse: bool = False, limit: int = 12) -> List[Dict[str, Any]]:
		valid = []
		for row in results:
			ops = row.get("operating_point_stats", {})
			if not isinstance(ops, dict):
				continue
			v = ops.get(stat_key, None)
			try:
				vf = float(v)
			except Exception:
				continue
			if not np.isfinite(vf):
				continue
			valid.append((vf, row))
		valid.sort(key=lambda t: t[0], reverse=bool(reverse))
		out_rows: List[Dict[str, Any]] = []
		for vf, row in valid[: max(1, int(limit))]:
			ops = row.get("operating_point_stats", {})
			out_rows.append(
				{
					"feature": str(row.get("feature", "")),
					"metric": float(vf),
					"auc_feature_oriented": float(row.get("auc_feature_oriented", float("nan"))),
					"fp_at_100_recall": ops.get("fp_at_100_recall"),
					"precision_high_zone": ops.get("precision_high_zone"),
				}
			)
		return out_rows

	out = {
		'n_labeled_spikes': len(paired),
		'label_mode': label_mode,
		'ternary_unlabeled_as_noise': bool(args.unlabeled_as_noise),
		'ternary_label_counts': (
			{
				"muon": int(sum(1 for v in all_label_classes.values() if v == "muon")),
				"raman": int(sum(1 for v in all_label_classes.values() if v == "raman")),
				"noise": int(sum(1 for v in all_label_classes.values() if v == "noise")),
				"unknown_excluded": int(sum(1 for v in all_label_classes.values() if v == "unknown" and not bool(args.unlabeled_as_noise))),
			}
			if label_mode == "ternary" else {}
		),
		'correlations': results,
		'conditional_debug_summary': _build_conditional_subset_summary(paired, results),
		'ss4_confusion': _build_spike_score_v4_confusion(paired),
		'ss4_reason_breakdown': _build_spike_score_v4_reason_breakdown(paired, class_arr),
		'delong_vs_best': delong_vs_best,
		'operating_point_summary': {
			'high_zone_quantile': float(args.high_zone_quantile),
			'n_features_with_stats': int(sum(1 for r in results if isinstance(r.get("operating_point_stats"), dict))),
			'best_features_by_fp_at_100_recall': _summary_rows_sorted("fp_at_100_recall", reverse=False, limit=12),
			'best_features_by_precision_high_zone': _summary_rows_sorted("precision_high_zone", reverse=True, limit=12),
		},
		'feature_correlation': {
			'features': corr_features,
			'pearson_matrix': pear_mat.tolist(),
			'spearman_matrix': spea_mat.tolist(),
		},
	}

	if args.plots_dir:
		p = Path(args.plots_dir)
		p.mkdir(parents=True, exist_ok=True)
		_save_auc_barplot(results, p / "feature_auc_ranking.png", top_k=int(args.top_k))
		# _save_auc_ci_plot(results, p / "feature_auc_bootstrap_ci.png", top_k=int(args.top_k))
		# _save_auc_mi_scatter(results, p / "feature_auc_mi_scatter.png", top_k=int(args.top_k))
		_save_corr_heatmap(corr_features, pear_mat, p / "feature_corr_pearson.png", "Feature correlation (Pearson)")
		_save_corr_heatmap(corr_features, spea_mat, p / "feature_corr_spearman.png", "Feature correlation (Spearman)")
		if str(args.plot_features).strip():
			feats = [t.strip() for t in str(args.plot_features).split(",") if t.strip()]
			_save_logreg_curves(paired, feats, p)

	out = _sanitize_nonfinite(out)

	if bool(args.interactive_scatter):
		_show_interactive_auc_mi_scatter(out.get('correlations', []))

	if bool(args.print_purity_intervals):
		_print_purity_interval_summary(out.get("correlations", []))

	if args.out_json:
		Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

	# print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
