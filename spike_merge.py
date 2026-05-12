from __future__ import annotations

from typing import List, Optional
import numpy as np

from muon_pipeline import SpikeSegment


def merge_spike_segments(
		segs: List[SpikeSegment],
		merge_adjacent: bool = True,
		peak_distance_max: Optional[int] = None,
) -> List[SpikeSegment]:
	"""
	Merge spike segments by interval overlap (and optionally adjacency).

	:param segs: Spike candidate segments.
	:param merge_adjacent: if True -> [a,b] and [b+1,c] are merged
	:param peak_distance_max: ignore peak-index distance during merge
	"""
	if not segs:
		return []

	sorted_segs = sorted(segs, key=lambda s: (int(s.start), int(s.end), int(s.peak_index)))
	out: List[SpikeSegment] = []
	for s in sorted_segs:
		if not out:
			out.append(s)
			continue

		last = out[-1]
		allow_gap = 1 if merge_adjacent else 0
		overlap_or_adjacent = max(int(last.start), int(s.start)) <= (min(int(last.end), int(s.end)) + allow_gap)
		peak_ok = True if peak_distance_max is None else abs(int(last.peak_index) - int(s.peak_index)) <= int(peak_distance_max)

		if overlap_or_adjacent and peak_ok:
			new_start = min(int(last.start), int(s.start))
			new_end = max(int(last.end), int(s.end))
			if float(s.peak_height) >= float(last.peak_height):
				best_peak = int(s.peak_index)
				best_height = float(s.peak_height)
			else:
				best_peak = int(last.peak_index)
				best_height = float(last.peak_height)
			out[-1] = SpikeSegment(
				y=int(last.y),
				x=int(last.x),
				peak_index=best_peak,
				start=new_start,
				end=new_end,
				peak_height=best_height,
				area=float(last.area) + float(s.area),
			)
		else:
			out.append(s)

	return out


def _stable_low_run_exists_between_peaks(
		signal: np.ndarray,
		left_peak: int,
		right_peak: int,
		union_left: int,
		union_right: int,
		k_mad: float,
		min_run: int,
) -> bool:
	"""
	Check whether the connection between two peaks contains a stable low-signal run.

	This uses the same robust idea as feature-foot expansion: if the signal returns
	to local background for at least `min_run` samples, the connection should be
	treated as a separator rather than a continuous multispike structure.
	"""
	x = np.asarray(signal, dtype=float)
	n = int(x.size)
	if n <= 0:
		return False

	lp = int(np.clip(min(left_peak, right_peak), 0, n - 1))
	rp = int(np.clip(max(left_peak, right_peak), 0, n - 1))
	if rp - lp <= 1:
		return False

	bg = np.concatenate([x[: max(0, int(union_left))], x[min(n, int(union_right) + 1) :]])
	if bg.size < 8:
		bg = x
	bg_med = float(np.median(bg))
	bg_mad = float(np.median(np.abs(bg - bg_med)))
	bg_mad = max(bg_mad, 1e-12)
	thr = float(bg_med + float(k_mad) * bg_mad)

	inner = x[lp + 1 : rp]
	if inner.size == 0:
		return False
	run = max(1, int(min_run))
	below = (inner <= thr).astype(np.int8)
	if run == 1:
		return bool(np.any(below > 0))
	conv = np.convolve(below, np.ones(run, dtype=np.int16), mode="valid")
	return bool(np.any(conv >= run))


def merge_spike_segments_by_signal_foot(
		segs: List[SpikeSegment],
		signal: np.ndarray,
		k_mad: float = 2.0,
		min_run: int = 2,
		max_width_pts: Optional[int] = None,
		merge_adjacent: bool = True,
		peak_distance_max: Optional[int] = None,
) -> List[SpikeSegment]:
	"""
	Merge only those prepared segments whose expanded spike-foot regions overlap
	or touch and whose connection does not contain a stable low-signal run.

	This is stricter than plain interval overlap merge and is intended to keep
	narrow Raman structure separated when the gradient already returned to
	background between neighboring candidates.
	"""
	if not segs:
		return []

	sorted_segs = sorted(segs, key=lambda s: (int(s.start), int(s.end), int(s.peak_index)))
	out: List[SpikeSegment] = []
	for s in sorted_segs:
		if not out:
			out.append(s)
			continue

		last = out[-1]
		allow_gap = 1 if merge_adjacent else 0
		overlap_or_adjacent = max(int(last.start), int(s.start)) <= (min(int(last.end), int(s.end)) + allow_gap)
		peak_ok = True if peak_distance_max is None else abs(int(last.peak_index) - int(s.peak_index)) <= int(peak_distance_max)
		if not overlap_or_adjacent or not peak_ok:
			out.append(s)
			continue

		new_start = min(int(last.start), int(s.start))
		new_end = max(int(last.end), int(s.end))
		merged_width = int(new_end - new_start - 1)
		if max_width_pts is not None and merged_width > int(max_width_pts):
			out.append(s)
			continue

		low_run_exists = _stable_low_run_exists_between_peaks(
			signal=np.asarray(signal, dtype=float),
			left_peak=int(last.peak_index),
			right_peak=int(s.peak_index),
			union_left=new_start,
			union_right=new_end,
			k_mad=float(k_mad),
			min_run=int(min_run),
		)
		if low_run_exists:
			out.append(s)
			continue

		if float(s.peak_height) >= float(last.peak_height):
			best_peak = int(s.peak_index)
			best_height = float(s.peak_height)
		else:
			best_peak = int(last.peak_index)
			best_height = float(last.peak_height)
		out[-1] = SpikeSegment(
			y=int(last.y),
			x=int(last.x),
			peak_index=best_peak,
			start=new_start,
			end=new_end,
			peak_height=best_height,
			area=float(last.area) + float(s.area),
		)

	return out
