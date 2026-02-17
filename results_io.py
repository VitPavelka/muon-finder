# results_io.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, Any, List

import json
import numpy as np

from wdf_io import WdfMap
from muon_pipeline import SpikeSegment


# --- Helpers ---
def _spikes_to_struct_array(spikes: List[SpikeSegment]) -> np.ndarray:
	dtype = np.dtype(
		[
			("y", np.int32),
			("x", np.int32),
			("peak_index", np.int32),
			("start", np.int32),
			("end", np.int32),
			("peak_height", np.float64),
			("area", np.float64),
		]
	)
	arr = np.empty((len(spikes),), dtype=dtype)
	for i, s in enumerate(spikes):
		arr[i] = (s.y, s.x, s.peak_index, s.start, s.end, s.peak_height, s.area)
	return arr


# --- Savers ---
def save_result_npz(
		out_path: Path,
		ds: WdfMap,
		score_map: np.ndarray,
		threshold: float,
		candidate_mask: np.ndarray,
		spikes: List[SpikeSegment],
		overlays: Optional[Dict[str, np.ndarray]] = None,
) -> None:
	out_path = Path(out_path)

	spikes_arr = _spikes_to_struct_array(spikes)

	meta: Dict[str, Any] = {
		"source": str(ds.path),
		"backend": ds.meta.get("backend"),
		"shape": list(ds.spectra.shape),
		"threshold": float(threshold),
		"extra": ds.meta,
	}
	meta_json = json.dumps(meta, ensure_ascii=False)

	payload = dict(
		x_axis=ds.x_axis,
		score_map=score_map,
		threshold=np.array([threshold], dtype=float),
		candidate_mask=candidate_mask.astype(np.uint8),
		spikes=spikes_arr,
		meta_json=np.array([meta_json]),
	)

	if ds.xpos is not None:
		payload["xpos"] = ds.xpos
	if ds.ypos is not None:
		payload["ypos"] = ds.ypos

	if overlays is not None:
		for k, v in overlays.items():
			payload[f"overlay_{k}"] = v

	np.savez_compressed(out_path, **payload)


def save_spikes_csv(path: Path, spikes: List[SpikeSegment]) -> None:
	path = Path(path)
	lines = ["y,x,peak_index,start,end,peak_height,area\n"]
	for s in spikes:
		lines.append(
			f"{s.y},{s.x},{s.peak_index},{s.start},{s.end},{s.peak_height:.10g},{s.area:.10g}"
		)
	path.write_text("".join(lines), encoding="utf-8")




