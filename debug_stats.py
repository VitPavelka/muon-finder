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


def _flatten_spikes(report: Dict[str, Any]) -> List[Dict[str, Any]]:
	per = report.get("per_spectrum", [])
	rows: List[Dict[str, Any]] = []
	for spec in per:
		y = int(spec.get("y", -1))
		x = int(spec.get("x", -1))
		for sp in spec.get("spikes", []):
			row = {"y": y, "x": x}
			row.update(sp)
			rows.append(row)
	return rows


def _read_labels(path: Path) -> Dict[Tuple[int, int, int], int]:
	out: Dict[Tuple[int, int, int], int] = {}
	with Path(path).open("r", encoding="utf-8", newline="") as f:
		r = csv.DictReader(f)
		for row in r:
			y = int(row['y'])
			x = int(row['x'])
			p = int(row['peak_index'])
			lbl = int(row['is_muon'])
			out[(y, x, p)] = 1 if lbl else 0
	return out


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


def main() -> None:
	parser = argparse.ArgumentParser(description="Correlation helper for debug_report spikes.")
	parser.add_argument("--report", type=Path, required=True, help="Path ot debug_report.json")
	parser.add_argument(
		"--labels-csv",
		type=Path,
		required=True,
		help="CSV labels with columns: y,x,peak_index,is_muon",
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
	args = parser.parse_args()

	report = json.loads(Path(args.report).read_text(encoding="utf-8"))
	rows = _flatten_spikes(report)
	labels = _read_labels(Path(args.labels_csv))

	paired = []
	for r in rows:
		key = (int(r['y']), int(r['x']), int(r['peak_index']))
		if key in labels:
			paired.append((r, labels[key]))

	if not paired:
		raise ValueError("No overlapping labeled spikes found between report and labels CSV.")

	feature_names = sorted(
		k for k, v in paired[0][0].items()
		if k not in {"y", "x", "feature_source", "muon_score_components"} and isinstance(v, (int, float))
	)

	y = np.array([lbl for _, lbl in paired], dtype=float)
	results = []
	for feat in feature_names:
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
				"mutual_info_bits": _mutual_info_binary_discrete(xv, yv, bins=max(2, int(args.mi_bins))),
				"logreg": _fit_univariate_logreg(xv, yv, quadratic=bool(args.logreg_quadratic)),
				"_x_eval": xv,
				"_y_eval": yv,
			}
		)

	# bootstrap CI per feature
	for i, r in enumerate(results):
		xv = np.array(r.get('_x_eval', []), dtype=float)
		yv = np.array(r.get('y_eval', []), dtype=float)
		if xv.size < 4:
			continue
		boot = _bootstrap_auc_ci(
			y_true=yv,
			score=xv,
			n_boot=int(args.bootstrap_iters),
			seed=int(args.bootstrap_seed + i),
		)
		r.update(boot)

	results.sort(
		key=lambda r: r["auc_feature_oriented"] if np.isfinite(r["auc_feature_oriented"]) else -1,
		reverse=True,
	)

	# DeLong: compare each feature against the best-oriented feature
	delong_vs_best: List[Dict[str, Any]] = []
	if results:
		best = results[0]
		xb = np.array(best.get('_x_eval', []), dtype=float)
		yb = np.array(best.get('y_eval', []), dtype=float)
		best_orient = -1.0 if str(best.get('auc_direction')) == "negative" else 1.0
		sb = best_orient * xb
		for r in results[1:]:
			xr = np.array(r.get('_x_eval', []), dtype=float)
			yr = np.array(r.get('y_eval', []), dtype=float)
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

	out = {
		'n_labeled_spikes': len(paired),
		'correlations': results,
		'delong_vs_best': delong_vs_best,
	}

	if args.plots_dir:
		p = Path(args.plots_dir)
		p.mkdir(parents=True, exist_ok=True)
		_save_auc_barplot(results, p / "feature_auc_ranking.png", top_k=int(args.top_k))
		_save_auc_ci_plot(results, p / "feature_auc_bootstrap_ci.png", top_k=int(args.top_k))

	if args.out_json:
		Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

	print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
