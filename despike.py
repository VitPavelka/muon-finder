# despike.py
from __future__ import annotations

from typing import Iterable, Dict, List, Tuple
import numpy as np

from muon_pipeline import SpikeSegment


def apply_despike(
		x_axis: np.ndarray,
		spectra: np.ndarray,  # (H,W,N)
		accepted_spikes: Iterable[SpikeSegment],
) -> np.ndarray:
	"""
	Replace each spike interior (start+1 .. end-1) by a straight line between anchors
	at start and end. Anchors must satisfy start < peak_index < end.
	"""
	out = spectra.copy()
	H, W, N = out.shape

	# group spikes by pixel
	by_pix: Dict[Tuple[int, int], List[SpikeSegment]] = {}
	for s in accepted_spikes:
		by_pix.setdefault((s.y, s.x), []).append(s)

	for (y, x), segs in by_pix.items():
		# sort by left anchor
		segs_sorted = sorted(segs, key=lambda s: (s.start, s.end))

		# (optional) merge overlapping/adjacent anchor intervals
		merged: List[SpikeSegment] = []
		for s in segs_sorted:
			if not merged:
				merged.append(s)
				continue

			last = merged[-1]
			# merge only when overlapping AND likely representing same peak
			if s.start < last.end and abs(s.peak_index - last.peak_index) <= 1:
				# keep the wider anchors; keep the higher peak as representative
				new_start = min(last.start, s.start)
				new_end = max(last.end, s.end)
				if s.peak_height >= last.peak_height:
					peak_index = s.peak_index
					peak_height = s.peak_height
				else:
					peak_index = last.peak_index
					peak_height = last.peak_height

				merged[-1] = SpikeSegment(
					y=last.y, x=last.x,
					peak_index=peak_index,
					start=new_start, end=new_end,
					peak_height=peak_height,
					area=last.area + s.area,
				)
			else:
				merged.append(s)

		for s in merged:
			a = int(s.start)
			b = int(s.end)

			# require strict anchors so interior is non-empty
			if a < 0 or b >= N or a >= b -1:
				continue
			if not (a < s.peak_index < b):
				continue

			x0 = float(x_axis[a]); y0 = float(out[y, x, a])
			x1 = float(x_axis[b]); y1 = float(out[y, x, b])

			xs = x_axis[a:b+1].astype(float)
			if x1 == x0:
				ys = np.full_like(xs, y0, dtype=float)
			else:
				ys = y0 + (y1 - y0) * (xs - x0) / (x1 - x0)

			# replace ONLY interior
			out[y, x, a+1:b] = ys[1:-1].astype(out.dtype)

	return out
