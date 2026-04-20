from __future__ import annotations

from typing import List, Optional

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