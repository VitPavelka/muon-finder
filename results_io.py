# results_io.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import json
import numpy as np

from wdf_io import WdfMap
from muon_pipeline import SpikeSegment


def _json_clean(value: Any) -> Any:
	if isinstance(value, np.integer):
		return int(value)
	if isinstance(value, np.floating):
		value = float(value)
	if isinstance(value, float):
		return value if np.isfinite(value) else None
	if isinstance(value, dict):
		return {str(k): _json_clean(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_json_clean(v) for v in value]
	return value


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
		corrected_spectra: Optional[np.ndarray] = None,
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

	if corrected_spectra is not None:
		payload["corrected_spectra"] = corrected_spectra.astype(np.float32)

	np.savez_compressed(out_path, **payload)


def save_spikes_csv(path: Path, spikes: List[SpikeSegment]) -> None:
	path = Path(path)
	lines = ["y,x,peak_index,start,end,peak_height,area\n"]
	for s in spikes:
		lines.append(
			f"{s.y},{s.x},{s.peak_index},{s.start},{s.end},{s.peak_height:.10g},{s.area:.10g}"
		)
	path.write_text("".join(lines), encoding="utf-8")


def _spikes_by_pixel_to_jsonable(spikes_by_pixel: Dict[Tuple[int, int], List[SpikeSegment]]) -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	for (y, x), segs in spikes_by_pixel.items():
		for s in segs:
			rows.append(
				{
					"y": int(y),
					"x": int(x),
					"peak_index": int(s.peak_index),
					"start": int(s.start),
					"end": int(s.end),
					"peak_height": float(s.peak_height),
					"area": float(s.area),
				}
			)
	return rows


def _jsonable_to_spikes_by_pixel(rows: List[Dict[str, Any]]) -> Dict[Tuple[int, int], List[SpikeSegment]]:
	out: Dict[Tuple[int, int], List[SpikeSegment]] = {}
	for row in rows:
		y = int(row["y"])
		x = int(row["x"])
		out.setdefault((y, x), []).append(
			SpikeSegment(
				y=y,
				x=x,
				peak_index=int(row["peak_index"]),
				start=int(row["start"]),
				end=int(row["end"]),
				peak_height=float(row.get("peak_height", 0.0)),
				area=float(row.get("area", 0.0)),
			)
		)
	return out


def _coord_map_to_jsonable(coord_map: Optional[Dict[Tuple[int, int], Tuple[int, int]]]) -> List[Dict[str, int]]:
	if not coord_map:
		return []
	rows: List[Dict[str, int]] = []
	for (cy, cx), (sy, sx) in coord_map.items():
		rows.append(
			{
				"compact_y": int(cy),
				"compact_x": int(cx),
				"source_y": int(sy),
				"source_x": int(sx),
			}
		)
	return rows


def _ss4_metrics_to_jsonable(metrics_by_pixel: Optional[Dict[Tuple[int, int], List[Dict[str, object]]]]) -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	for (y, x), items in (metrics_by_pixel or {}).items():
		for item in items:
			row = dict(item)
			row["y"] = int(row.get("y", y))
			row["x"] = int(row.get("x", x))
			rows.append(row)
	return rows


def _jsonable_to_ss4_metrics(rows: List[Dict[str, Any]]) -> Dict[Tuple[int, int], List[Dict[str, object]]]:
	out: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
	for row in rows:
		y = int(row["y"])
		x = int(row["x"])
		out.setdefault((y, x), []).append(dict(row))
	return out


def _jsonable_to_coord_map(rows: List[Dict[str, Any]]) -> Dict[Tuple[int, int], Tuple[int, int]]:
	out: Dict[Tuple[int, int], Tuple[int, int]] = {}
	for row in rows:
		out[(int(row["compact_y"]), int(row["compact_x"]))] = (int(row["source_y"]), int(row["source_x"]))
	return out


def save_viewer_cache(
		out_path: Path,
		*,
		x_axis: np.ndarray,
		spectra: np.ndarray,
		score_map: np.ndarray,
		candidate_mask: np.ndarray,
		spikes_by_pixel: Dict[Tuple[int, int], List[SpikeSegment]],
		overlays: Dict[str, np.ndarray],
		corrected_spectra: Optional[np.ndarray] = None,
		source_coords_map: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
		despike_contact_analysis: Optional[Dict[str, Any]] = None,
		ss4_candidate_metrics_by_pixel: Optional[Dict[Tuple[int, int], List[Dict[str, object]]]] = None,
		despike_diagnostics: Optional[List[Dict[str, object]]] = None,
		viewer_status_text: Optional[str] = None,
) -> None:
	out_path = Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	payload: Dict[str, Any] = {
		"x_axis": np.asarray(x_axis),
		"spectra": np.asarray(spectra),
		"score_map": np.asarray(score_map),
		"candidate_mask": np.asarray(candidate_mask, dtype=np.uint8),
		"spikes_by_pixel_json": np.array([json.dumps(_json_clean(_spikes_by_pixel_to_jsonable(spikes_by_pixel)), ensure_ascii=False)], dtype=object),
		"source_coords_map_json": np.array([json.dumps(_json_clean(_coord_map_to_jsonable(source_coords_map)), ensure_ascii=False)], dtype=object),
		"despike_contact_analysis_json": np.array([json.dumps(_json_clean(despike_contact_analysis or {}), ensure_ascii=False)], dtype=object),
		"ss4_candidate_metrics_json": np.array([json.dumps(_json_clean(_ss4_metrics_to_jsonable(ss4_candidate_metrics_by_pixel)), ensure_ascii=False)], dtype=object),
		"despike_diagnostics_json": np.array([json.dumps(_json_clean(despike_diagnostics or []), ensure_ascii=False)], dtype=object),
		"viewer_status_text": np.array([str(viewer_status_text or "")], dtype=object),
	}
	for name, arr in overlays.items():
		payload[f"overlay_{name}"] = np.asarray(arr)
	if corrected_spectra is not None:
		payload["corrected_spectra"] = np.asarray(corrected_spectra)
	np.savez_compressed(out_path, **payload)


def load_viewer_cache(path: Path) -> Dict[str, Any]:
	path = Path(path)
	data = np.load(path, allow_pickle=True)
	overlays: Dict[str, np.ndarray] = {}
	for key in data.files:
		if key.startswith("overlay_"):
			overlays[key[len("overlay_"):]] = np.asarray(data[key])

	def _json_item(name: str, default: Any) -> Any:
		if name not in data:
			return default
		raw = data[name]
		if isinstance(raw, np.ndarray):
			if raw.size == 0:
				return default
			value = raw.reshape(-1)[0]
		else:
			value = raw
		if value is None:
			return default
		text = str(value)
		if not text:
			return default
		try:
			return json.loads(text)
		except Exception:
			return text

	return {
		"x_axis": np.asarray(data["x_axis"]),
		"spectra": np.asarray(data["spectra"]),
		"score_map": np.asarray(data["score_map"]),
		"candidate_mask": np.asarray(data["candidate_mask"]).astype(bool),
		"spikes_by_pixel": _jsonable_to_spikes_by_pixel(_json_item("spikes_by_pixel_json", [])),
		"source_coords_map": _jsonable_to_coord_map(_json_item("source_coords_map_json", [])),
		"overlays": overlays,
		"corrected_spectra": np.asarray(data["corrected_spectra"]) if "corrected_spectra" in data else None,
		"despike_contact_analysis": _json_item("despike_contact_analysis_json", {}),
		"ss4_candidate_metrics_by_pixel": _jsonable_to_ss4_metrics(_json_item("ss4_candidate_metrics_json", [])),
		"despike_diagnostics": _json_item("despike_diagnostics_json", []),
		"viewer_status_text": str(_json_item("viewer_status_text", "")),
	}




