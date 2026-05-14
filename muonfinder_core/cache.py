from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .data_model import PipelineArtifacts
from .utils import dumps_json, loads_json


def save_viewer_cache(path: Path | str, artifacts: PipelineArtifacts) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "x_axis": np.asarray(artifacts.x_axis),
        "spectra": np.asarray(artifacts.spectra),
        "corrected_spectra": np.asarray(artifacts.corrected_spectra),
        "score_map": np.asarray(artifacts.score_map),
        "candidate_mask": np.asarray(artifacts.candidate_mask, dtype=np.uint8),
        "metadata_json": np.array([dumps_json(artifacts.metadata)], dtype=object),
        "candidate_records_json": np.array([dumps_json(_flatten_records(artifacts.candidate_records_by_pixel))], dtype=object),
        "candidates_json": np.array([dumps_json(_flatten_candidates(artifacts.candidates_by_pixel))], dtype=object),
        "coord_map_json": np.array([dumps_json(_coord_map_rows(artifacts.source_coords_map))], dtype=object),
        "small_morphology_json": np.array([dumps_json(_small_morph_rows(artifacts.small_morphology_by_pixel))], dtype=object),
        "despike_stages_json": np.array([dumps_json([stage.__dict__ for stage in artifacts.despike_stages])], dtype=object),
        "despike_chords_json": np.array([dumps_json([chord.__dict__ for chord in artifacts.despike_chords])], dtype=object),
    }
    for overlay_name, by_window in artifacts.overlays.items():
        for window, arr in by_window.items():
            payload[f"overlay_{overlay_name}_w{int(window)}"] = np.asarray(arr)
    np.savez_compressed(out_path, **payload)


def load_viewer_cache(path: Path | str) -> dict[str, Any]:
    data = np.load(Path(path), allow_pickle=True)
    overlays: dict[str, dict[int, np.ndarray]] = {}
    for key in data.files:
        if not key.startswith("overlay_"):
            continue
        rest = key[len("overlay_") :]
        name, _, window_tag = rest.rpartition("_w")
        window = int(window_tag)
        overlays.setdefault(name, {})[window] = np.asarray(data[key])
    return {
        "x_axis": np.asarray(data["x_axis"]),
        "spectra": np.asarray(data["spectra"]),
        "corrected_spectra": np.asarray(data["corrected_spectra"]),
        "score_map": np.asarray(data["score_map"]),
        "candidate_mask": np.asarray(data["candidate_mask"]).astype(bool),
        "metadata": loads_json(str(np.asarray(data["metadata_json"]).reshape(-1)[0]), {}),
        "candidate_records": loads_json(str(np.asarray(data["candidate_records_json"]).reshape(-1)[0]), []),
        "candidates": loads_json(str(np.asarray(data["candidates_json"]).reshape(-1)[0]), []),
        "coord_map": loads_json(str(np.asarray(data["coord_map_json"]).reshape(-1)[0]), []),
        "small_morphology": loads_json(str(np.asarray(data["small_morphology_json"]).reshape(-1)[0]), []),
        "despike_stages": loads_json(str(np.asarray(data["despike_stages_json"]).reshape(-1)[0]), []),
        "despike_chords": loads_json(str(np.asarray(data["despike_chords_json"]).reshape(-1)[0]), []),
        "overlays": overlays,
    }


def _flatten_candidates(by_pixel):
    rows = []
    for (_, _), segs in by_pixel.items():
        for seg in segs:
            rows.append(
                {
                    "candidate_id": seg.candidate_id,
                    "y": int(seg.y),
                    "x": int(seg.x),
                    "peak_index": int(seg.peak_index),
                    "start": int(seg.start),
                    "end": int(seg.end),
                    "peak_height": float(seg.peak_height),
                    "area": float(seg.area),
                }
            )
    return rows


def _flatten_records(by_pixel):
    rows = []
    for (_, _), items in by_pixel.items():
        rows.extend(dict(item) for item in items)
    return rows


def _coord_map_rows(coord_map):
    return [
        {
            "compact_y": int(cy),
            "compact_x": int(cx),
            "source_y": int(sy),
            "source_x": int(sx),
        }
        for (cy, cx), (sy, sx) in coord_map.items()
    ]


def _small_morph_rows(by_pixel):
    rows = []
    for (y, x), payload in by_pixel.items():
        row = {"y": int(y), "x": int(x)}
        row.update(payload)
        rows.append(row)
    return rows
