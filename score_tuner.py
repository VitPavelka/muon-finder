from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, TextBox
from scipy.optimize import minimize
from scipy.stats import rankdata

from candidate_labels import load_binary_labels


def _flatten_spikes(report: Dict) -> List[Dict]:
	rows: List[Dict] = []
	for spec in report.get('per_spectrum', []):
		y = int(spec.get("y", -1))
		x = int(spec.get("x", -1))
		for sp in spec.get('spikes', []):
			row = {"y": y, "x": x}
			row.update(sp)
			rows.append(row)
	return rows


def _read_labels(path: Path) -> Dict[Tuple[int, int, int], int]:
	return load_binary_labels(Path(path))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
	xm = x - np.mean(x)
	ym = y - np.mean(y)
	den = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
	if den <= 0:
		return float("nan")
	return float(np.sum(xm * ym) / den)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
	return _pearson(rankdata(x), rankdata(y))


def _roc_auc_binary(y_true: np.ndarray, score: np.ndarray) -> float:
	y = y_true.astype(int)
	n_pos = int(np.count_nonzero(y == 1))
	n_neg = int(np.count_nonzero(y == 0))
	if n_pos == 0 or n_neg == 0:
		return float("nan")
	ranks = rankdata(score, method="average")
	sum_pos = float(np.sum(ranks[y == 1]))
	u = sum_pos - (n_pos * (n_pos + 1) / 2.0)
	return float(u / (n_pos + n_neg))


def _roc_curve(y_true: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	order = np.argsort(-score)
	y = y_true[order].astype(int)
	pos = np.sum(y == 1)
	neg = np.sum(y == 0)
	if pos == 0 or neg == 0:
		return np.array([0.0, 1.0]), np.array([0.0, 1.0])
	tp = np.cumsum(y == 1)
	fp = np.cumsum(y == 0)
	tpr = np.concatenate([[0.0], tp / pos, [1.0]])
	fpr = np.concatenate([[0.0], fp / neg, [1.0]])
	return fpr, tpr


def _fit_logreg_auc(x: np.ndarray, y: np.ndarray) -> float:
	mu = float(np.mean(x))
	sd = float(np.std(x))
	if sd <= 0:
		return float("nan")
	xz = np.clip((x - mu) / sd, -12, 12)
	X = np.vstack([np.ones_like(xz), xz, xz * xz]).T

	def _sig(z: np.ndarray) -> np.ndarray:
		return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

	def nll(beta: np.ndarray) -> float:
		p = _sig(X @ beta)
		eps = 1e-12
		return float(-np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

	res = minimize(nll, x0=np.zeros(X.shape[1], dtype=float), method="BFGS")
	if not np.all(np.isfinite(res.x)):
		return float("nan")
	p_hat = _sig(X @ res.x)
	return _roc_auc_binary(y, p_hat)


def _feature_transform(name: str, x: np.ndarray) -> np.ndarray:
	if str(name) == "fall_slope":
		return np.abs(x)
	return x


def main() -> None:
	parser = argparse.ArgumentParser(description="Interactive spike_score tuner from debug_report + labels.")
	parser.add_argument("--report", type=Path, required=True)
	parser.add_argument("--labels-csv", type=Path, required=True)
	parser.add_argument(
		"--features",
		type=str,
		default="raw_d3_max_abs_z,raw_d2_max_abs_z,fall_slope_z,peak_curvature_z",
		help="Comma-separated features to combine in weighted score."
	)
	parser.add_argument("--slider-max", type=float, default=1000.0, help="Max slider value before normalization.")
	args = parser.parse_args()

	report = json.loads(Path(args.report).read_text(encoding="utf-8"))
	rows = _flatten_spikes(report)
	labels = _read_labels(Path(args.labels_csv))
	features = [f.strip() for f in str(args.features).split(",") if f.strip()]
	if not features:
		raise ValueError("No features specified.")

	paired: List[Tuple[Dict, int]] = []
	for r in rows:
		k = (int(r['y']), int(r['x']), int(r['peak_index']))
		if k in labels:
			paired.append((r, labels[k]))
	if not paired:
		raise ValueError("No overlap between debug_report spikes and labels.csv.")

	y = np.array([lbl for _, lbl in paired], dtype=float)
	ids: List[str] = [f"{int(r['y'])},{int(r['x'])}, {int(r['peak_index'])}" for r, _ in paired]
	X: Dict[str, np.ndarray] = {}
	for f in features:
		x = np.array([float(r.get(f, np.nan)) for r, _ in paired], dtype=float)
		X[f] = _feature_transform(f, x)

	fig = plt.figure(figsize=(13, 8))
	ax_roc = fig.add_axes((0.06, 0.56, 0.38, 0.38))
	ax_hist = fig.add_axes((0.52, 0.56, 0.42, 0.38))
	ax_text = fig.add_axes((0.06, 0.46, 0.88, 0.08))
	ax_text.axis("off")

	slider_axes = []
	text_axes = []
	sliders: List[Slider] = []
	boxes: List[TextBox] = []
	for i, f in enumerate(features):
		y0 = 0.40 - i * 0.06
		sax = fig.add_axes((0.06, y0, 0.62, 0.035))
		tax = fig.add_axes((0.70, y0, 0.10, 0.04))
		slider_axes.append(sax)
		text_axes.append(tax)
		s = Slider(sax, f, valmin=0.0, valmax=float(args.slider_max), valinit=float(args.slider_max / len(features)))
		tb = TextBox(tax, "", initial=f"{s.val:.1f}")
		sliders.append(s)
		boxes.append(tb)

	ln_roc, = ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
	ln_diag, = ax_roc.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
	ax_roc.set_xlabel("FPR")
	ax_roc.set_ylabel("TPR")
	ax_roc.set_title("ROC of weighted score")

	info_txt = ax_text.text(0.01, 0.5, "", va="center", ha="left", fontsize=11, family="monospace")
	updating = {'state': False}

	def _weights() -> np.ndarray:
		w = np.array([float(s.val) for s in sliders], dtype=float)
		if np.sum(w) <= 0:
			w[:] = 1.0
		return w / np.sum(w)

	def _compute_score() -> Tuple[np.ndarray, np.ndarray, List[str]]:
		w = _weights()
		feat_stack = np.vstack([X[f] for f in features])
		mask = np.all(np.isfinite(feat_stack), axis=0)
		if not np.any(mask):
			return np.array([], dtype=float), np.array([], dtype=float), []
		s = np.zeros(np.count_nonzero(mask), dtype=float)
		for wi, f in zip(w, features):
			s += wi * X[f][mask]
		yv = y[mask]
		idsv = [ids[i] for i in np.where(mask)[0].tolist()]
		return s, yv, idsv

	def _refresh(_event=None) -> None:
		if updating['state']:
			return
		score, yv, idv = _compute_score()
		if score.size < 3:
			return
		auc = _roc_auc_binary(yv, score)
		auc_oriented = float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan")
		pear = _pearson(score, yv)
		spea = _spearman(score, yv)
		auc_logreg = _fit_logreg_auc(score, yv)
		fpr, tpr = _roc_curve(yv, score)
		ln_roc.set_data(fpr, tpr)
		ax_roc.set_title(f"ROC of weighted score | AUC={auc:.4f}")

		ax_hist.clear()
		ax_hist.hist(score[yv == 0], bins=105, alpha=0.55, label="signal (label=0)")
		ax_hist.hist(score[yv == 1], bins=35, alpha=0.55, label="muon (label=1)")
		ax_hist.set_title("Score distribution")
		ax_hist.legend(loc="best")
		ax_hist.grid(alpha=0.2)

		w = _weights()
		w_txt = " | ".join(f"{f}: {wi:.3f}" for f, wi in zip(features, w))
		info_txt.set_text(
			f"AUC{auc:.5f} AUC_oriented={auc_oriented:.5f} AUC_logreg={auc_logreg:.5f} Pearson={pear:.5f} Spearman={spea:.5f}\n{w_txt}"
		)
		fig.canvas.draw_idle()

	def _slider_changed(_val):
		if updating['state']:
			return
		updating['state'] = True
		try:
			for s, tb in zip(sliders, boxes):
				tb.set_val(f"{s.val:.1f}")
		finally:
			updating['state'] = False
		_refresh()

	def _make_box_submit(i: int):
		def _submit(txt: str):
			try:
				v = float(txt.strip())
			except Exception:
				return
			v = float(np.clip(v, 0.0, float(args.slider_max)))
			sliders[i].set_val(v)
		return _submit

	for i, s in enumerate(sliders):
		s.on_changed(_slider_changed)
		boxes[i].on_submit(_make_box_submit(i))

	_refresh()
	plt.show()


if __name__ == "__main__":
	main()
