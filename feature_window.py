from __future__ import annotations

from typing import Tuple, Literal, List, Optional

import numpy as np
from morph1d import erosion_1d


def expand_interval_to_signal_foot(
		sig: np.ndarray,
		left: int,
		right: int,
		peak: int,
		enabled: bool = True,
		k_mad: float = 2.0,
		min_run: int = 2,
		method: Literal["mad_run", "erosion_touch"] = "mad_run",
		erosion_se_size: int = 5,
) -> Tuple[int, int]:
	"""
	Expand [left, right] interval toward baseline "foot" using a robust threshold.

	The foot is detected as the nearest contiguous run of at least `min_run` samples
	under `median + k_mad * MAD`, searched outward from current anchors.
	This implementation uses vectoriyed run detection (convolution) to avoid
	per-sample Python loops.
	"""
	n = int(sig.size)
	if n <= 2:
		return int(left), int(right)

	a = int(np.clip(left, 0, n - 1))
	b = int(np.clip(right, 0, n - 1))
	p = int(np.clip(peak, 0, n - 1))
	if not bool(enabled) or not (a < p < b):
		return a, b

	m = str(method).strip().lower()
	if m == "erosion_touch":
		ero = erosion_1d(sig.astype(float), se_size=max(1, int(erosion_se_size))).astype(float)
		diff = np.abs(sig - ero)
		tol = float(max(1e-12, np.median(diff) + 1.5 * np.median(np.abs(diff - np.median(diff)))))

		left_hits = np.where(diff[: p + 1] <= tol)[0]
		right_hits = np.where(diff[p:] <= tol)[0]
		if left_hits.size and right_hits.size:
			a2 = int(left_hits.max())
			b2 = int(p + right_hits.min())
			if a2 < p < b2:
				return a2, b2
		return a, b

	bg = np.concatenate([sig[:a], sig[b + 1:]])
	if bg.size < 8:
		bg = sig
	bg_med = float(np.median(bg))
	bg_mad = float(np.median(np.abs(bg - bg_med)))
	bg_mad = max(bg_mad, 1e-12)
	thr = float(bg_med + float(k_mad) * bg_mad)

	run = max(1, int(min_run))
	below = (sig <= thr).astype(np.int8)
	if run == 1:
		run_start = np.where(below > 0)[0]
	else:
		conv = np.convolve(below, np.ones(run, dtype=np.int16), mode="valid")
		run_start = np.where(conv >= run)[0]

	if run_start.size == 0:
		return a, b

	run_end = run_start + run - 1

	left_end = run_end[run_end <= a]
	if left_end.size:
		a2 = int(left_end.max())
	else:
		a2 = a

	right_start = run_start[run_start >= b]
	if right_start.size:
		b2 = int(right_start.min())
	else:
		b2 = b

	if a2 < p < b2:
		return a2, b2
	return a, b


def enforce_shared_boundaries_by_minima(
		peaks: List[int],
		lefts: List[int],
		rights: List[int],
		signal: np.ndarray,
) -> Tuple[List[int], List[int]]:
	"""
	If neighboring windows overlap, split them at local minimum of selected signal
	between neighboring peak positions.
	"""
	n = len(peaks)
	if n <= 1:
		return lefts, rights

	p = [int(v) for v in peaks]
	l = [int(v) for v in lefts]
	r = [int(v) for v in rights]
	sig = signal.astype(float)

	order = np.argsort(np.array(p, dtype=int))
	for oi in range(len(order) - 1):
		i = int(order[oi])
		j = int(order[oi + 1])
		if r[i] < l[j]:
			continue

		pi = int(p[i])
		pj = int(p[j])
		if pj <= pi + 1:
			m = int((pi + pj) // 2)
		else:
			seg = sig[pi:pj + 1]
			m = int(pi + int(np.argmin(seg)))
			m = max(pi + 1, min(pj - 1, m))

		r[i] = min(r[j], m)
		l[j] = max(l[j], m)

		if not (l[i] < p[i] < r[i]):
			l[i] = min(l[i], p[i] - 1)
			r[i] = max(r[i], p[i] + 1)
		if not (l[j] < p[j] < r[j]):
			l[j] = min(l[j], p[j] - 1)
			r[j] = max(r[j], p[j] + 1)

	return l, r
