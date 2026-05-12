from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np

from feature_window import expand_interval_to_signal_foot, enforce_shared_boundaries_by_minima
from muon_pipeline import SpikeSegment
from spike_merge import merge_spike_segments_by_signal_foot


def prepare_primary_ss4_segments(
		*,
		y: int,
		x: int,
		segs: List[SpikeSegment],
		feature_signal: Optional[np.ndarray],
		boundary_signal: Optional[np.ndarray],
		merge_signal: Optional[np.ndarray],
		feature_expand_to_gradient_foot: bool = True,
		feature_foot_k_mad: float = 2.0,
		feature_foot_min_run: int = 2,
		feature_window_method: Literal["mad_run", "erosion_touch"] = "mad_run",
		feature_erosion_se_size: int = 5,
		merge_duplicate_segments: bool = False,
		merge_max_width_pts: Optional[int] = None,
) -> List[SpikeSegment]:
	"""Prepare original primary candidates for primary SS4 evaluation.

	This is the shared primary-candidate window preparation used by the debug
	report and the fast SS4 path: expand to signal foot, enforce shared
	boundaries, then merge duplicate/overlapping prepared candidates before
	metrics are computed.
	"""
	if not segs:
		return []
	src_sig = np.asarray(feature_signal, dtype=float) if feature_signal is not None else None
	boundary_sig = np.asarray(boundary_signal, dtype=float) if boundary_signal is not None else None
	merge_sig = np.asarray(merge_signal, dtype=float) if merge_signal is not None else None
	peaks: List[int] = []
	lefts: List[int] = []
	rights: List[int] = []
	for s in segs:
		a0 = int(s.start)
		b0 = int(s.end)
		p0 = int(s.peak_index)
		if src_sig is not None:
			a0, b0 = expand_interval_to_signal_foot(
				sig=src_sig,
				left=a0,
				right=b0,
				peak=p0,
				enabled=bool(feature_expand_to_gradient_foot),
				k_mad=float(feature_foot_k_mad),
				min_run=int(feature_foot_min_run),
				method=feature_window_method,
				erosion_se_size=int(feature_erosion_se_size),
			)
		peaks.append(p0)
		lefts.append(int(a0))
		rights.append(int(b0))

	if boundary_sig is not None:
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
	if bool(merge_duplicate_segments) and merge_sig is not None:
		prepared = merge_spike_segments_by_signal_foot(
			prepared,
			signal=merge_sig,
			k_mad=float(feature_foot_k_mad),
			min_run=int(feature_foot_min_run),
			max_width_pts=(None if merge_max_width_pts is None else int(merge_max_width_pts)),
			merge_adjacent=True,
			peak_distance_max=None,
		)
	return prepared
