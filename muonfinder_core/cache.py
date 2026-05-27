from __future__ import annotations

import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from .data_model import PipelineArtifacts
from .utils import dumps_json, loads_json

_CHUNK_SIZE = 4 * 1024 * 1024


def _fmt_mib(num_bytes: int) -> str:
    return f"{float(num_bytes) / (1024.0 * 1024.0):.1f} MiB"


class _CacheWriteProgress:
    def __init__(self, total_bytes: int, total_chunks: int) -> None:
        self.total_bytes = int(max(0, total_bytes))
        self.total_chunks = int(max(1, total_chunks))
        self._bytes_done = 0
        self._chunks_done = 0
        self._last_print = time.perf_counter()
        self._bar = None
        if tqdm is not None:
            self._bar = tqdm(
                total=self.total_bytes if self.total_bytes > 0 else None,
                desc="viewer cache",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                mininterval=0.25,
            )

    def update(self, nbytes: int) -> None:
        self._bytes_done += int(max(0, nbytes))
        self._chunks_done += 1
        if self._bar is not None:
            self._bar.update(int(max(0, nbytes)))
            return
        now = time.perf_counter()
        if now - self._last_print >= 5.0 or self._chunks_done >= self.total_chunks:
            self._last_print = now
            print(
                f"[viewer-cache] writing viewer cache: "
                f"{self._chunks_done}/{self.total_chunks} chunks "
                f"({_fmt_mib(self._bytes_done)}/{_fmt_mib(self.total_bytes)})"
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def _build_cache_metadata(artifacts: PipelineArtifacts, config_path: Path | str | None) -> dict[str, Any]:
    metadata = dict(artifacts.metadata)
    metadata["viewer_cache_identity"] = {
        "input_data_path": str(artifacts.dataset.path),
        "spectra_shape": [int(v) for v in np.asarray(artifacts.spectra).shape],
        "corrected_shape": [int(v) for v in np.asarray(artifacts.corrected_spectra).shape],
        "x_axis_length": int(np.asarray(artifacts.x_axis).size),
        "created_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": (str(Path(config_path)) if config_path is not None else ""),
    }
    return metadata


def _build_viewer_cache_payload(artifacts: PipelineArtifacts, config_path: Path | str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "x_axis": np.asarray(artifacts.x_axis),
        "spectra": np.asarray(artifacts.spectra),
        "corrected_spectra": np.asarray(artifacts.corrected_spectra),
        "score_map": np.asarray(artifacts.score_map),
        "candidate_mask": np.asarray(artifacts.candidate_mask, dtype=np.uint8),
        "metadata_json": np.array([dumps_json(_build_cache_metadata(artifacts, config_path))], dtype=object),
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
    return payload


def _write_npz_payload(out_path: Path, payload: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="viewer_cache_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        staged: list[tuple[str, Path, int]] = []
        for key, value in payload.items():
            tmp_file = tmpdir / f"{key}.npy"
            with tmp_file.open("wb") as f:
                np.save(f, np.asarray(value), allow_pickle=True)
            staged.append((key, tmp_file, int(tmp_file.stat().st_size)))
        total_bytes = sum(size for _key, _path, size in staged)
        total_chunks = sum(max(1, (size + _CHUNK_SIZE - 1) // _CHUNK_SIZE) for _key, _path, size in staged)
        progress = _CacheWriteProgress(total_bytes=total_bytes, total_chunks=total_chunks)
        try:
            with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
                for key, tmp_file, _size in staged:
                    with tmp_file.open("rb") as src, zf.open(f"{key}.npy", mode="w", force_zip64=True) as dst:
                        while True:
                            chunk = src.read(_CHUNK_SIZE)
                            if not chunk:
                                break
                            dst.write(chunk)
                            progress.update(len(chunk))
        finally:
            progress.close()


def save_viewer_cache(path: Path | str, artifacts: PipelineArtifacts, config_path: Path | str | None = None) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_viewer_cache_payload(artifacts, config_path=config_path)
    _write_npz_payload(out_path, payload)


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
