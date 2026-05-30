from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

if __package__ in {None, ""}:
    import sys

    _THIS_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _THIS_DIR.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from muonfinder_core.cache import load_viewer_cache_light
    from muonfinder_core.config import load_config
else:
    from .cache import load_viewer_cache_light
    from .config import load_config


def _load_json_array_field(data: np.lib.npyio.NpzFile, key: str, default: Any) -> Any:
    if key not in data.files:
        return default
    return json.loads(str(np.asarray(data[key]).reshape(-1)[0]))


def _load_corrected_bundle(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    files = set(data.files)
    required = {"corrected_spectra", "x_axis"}
    missing = sorted(required - files)
    if missing:
        available = ", ".join(sorted(files))
        raise KeyError(
            f"Corrected despike bundle is missing required keys {missing}. "
            f"Available keys: {available}"
        )
    return {
        "path": path,
        "available_keys": sorted(files),
        "raw_spectra": (np.asarray(data["raw_spectra"]) if "raw_spectra" in files else None),
        "corrected_spectra": np.asarray(data["corrected_spectra"]),
        "x_axis": np.asarray(data["x_axis"], dtype=float),
        "coord_map": _load_json_array_field(data, "coord_map_json", []),
        "per_spectrum_correction_counts": _load_json_array_field(data, "per_spectrum_correction_counts_json", {}),
        "metadata": _load_json_array_field(data, "metadata_json", {}),
    }


def _build_coord_lookup(coord_rows: list[dict[str, Any]]) -> dict[tuple[int, int], tuple[int, int]]:
    lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for row in coord_rows:
        try:
            cy = int(row["compact_y"])
            cx = int(row["compact_x"])
            sy = int(row.get("source_y", cy))
            sx = int(row.get("source_x", cx))
        except Exception:
            continue
        lookup[(cy, cx)] = (sy, sx)
    return lookup


def _iter_spectrum_positions(spectra: np.ndarray) -> Iterable[tuple[int, int, np.ndarray]]:
    arr = np.asarray(spectra)
    if arr.ndim == 3:
        ny, nx, _ = arr.shape
        for y in range(ny):
            for x in range(nx):
                yield int(y), int(x), np.asarray(arr[y, x, :], dtype=float)
        return
    if arr.ndim == 2:
        n_spec, _ = arr.shape
        for y in range(n_spec):
            yield int(y), 0, np.asarray(arr[y, :], dtype=float)
        return
    raise ValueError(f"Unsupported corrected_spectra shape {tuple(arr.shape)}. Expected 2D or 3D array.")


def _resolve_coords(compact_y: int, compact_x: int, coord_lookup: dict[tuple[int, int], tuple[int, int]]) -> tuple[int, int]:
    return coord_lookup.get((int(compact_y), int(compact_x)), (int(compact_y), int(compact_x)))


def _default_text_output_path(corrected_path: Path) -> Path:
    return corrected_path.with_name(f"{corrected_path.stem}_wire_like.txt")


def _default_figures_dir(corrected_path: Path) -> Path:
    return corrected_path.with_name(f"{corrected_path.stem}_figures")


def _correction_count_for_pixel(counts: dict[str, Any], compact_y: int, compact_x: int) -> int:
    value = counts.get(f"{int(compact_y)}:{int(compact_x)}", 0)
    try:
        return int(value)
    except Exception:
        return 0


def _collect_corrected_pixels(
    corrected_spectra: np.ndarray,
    per_spectrum_counts: dict[str, Any],
    raw_spectra: np.ndarray | None,
) -> list[tuple[int, int]]:
    pixels: list[tuple[int, int]] = []
    for compact_y, compact_x, corrected_trace in _iter_spectrum_positions(corrected_spectra):
        count = _correction_count_for_pixel(per_spectrum_counts, compact_y, compact_x)
        if count > 0:
            pixels.append((compact_y, compact_x))
            continue
        if raw_spectra is None:
            continue
        raw_trace = (
            np.asarray(raw_spectra[compact_y, compact_x, :], dtype=float)
            if np.asarray(raw_spectra).ndim == 3
            else np.asarray(raw_spectra[compact_y, :], dtype=float)
        )
        if not np.allclose(raw_trace, corrected_trace, atol=1e-9, rtol=0.0):
            pixels.append((compact_y, compact_x))
    return pixels


def _accepted_parent_candidate_count(metadata: dict[str, Any]) -> int | None:
    value = metadata.get("accepted_ss6_parent_candidates")
    try:
        return int(value)
    except Exception:
        return None


def _load_raw_spectra_from_cache(cache_path: Path) -> np.ndarray:
    cache = load_viewer_cache_light(cache_path, verbose=True)
    spectra = np.asarray(cache.get("spectra"))
    if spectra.size == 0:
        raise ValueError(f"Viewer cache {cache_path} did not provide raw spectra.")
    return spectra


def export_wire_like_txt(
    corrected_spectra: np.ndarray,
    x_axis: np.ndarray,
    coord_lookup: dict[tuple[int, int], tuple[int, int]],
    out_path: Path,
) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spectra_written = 0
    rows_written = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write("#X\t#Y\t#Wave\t#Intensity\n")
        for compact_y, compact_x, spectrum in _iter_spectrum_positions(corrected_spectra):
            source_y, source_x = _resolve_coords(compact_y, compact_x, coord_lookup)
            spectra_written += 1
            for wave, intensity in zip(x_axis, spectrum, strict=False):
                f.write(f"{source_x}\t{source_y}\t{float(wave):.12g}\t{float(intensity):.12g}\n")
                rows_written += 1
    return spectra_written, rows_written


def export_figures(
    corrected_spectra: np.ndarray,
    raw_spectra: np.ndarray,
    x_axis: np.ndarray,
    coord_lookup: dict[tuple[int, int], tuple[int, int]],
    per_spectrum_counts: dict[str, Any],
    figures_dir: Path,
) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    corrected_pixels = _collect_corrected_pixels(corrected_spectra, per_spectrum_counts, raw_spectra)
    raw_arr = np.asarray(raw_spectra)
    for compact_y, compact_x in corrected_pixels:
        corrected_trace = (
            np.asarray(corrected_spectra[compact_y, compact_x, :], dtype=float)
            if np.asarray(corrected_spectra).ndim == 3
            else np.asarray(corrected_spectra[compact_y, :], dtype=float)
        )
        raw_trace = (
            np.asarray(raw_arr[compact_y, compact_x, :], dtype=float)
            if raw_arr.ndim == 3
            else np.asarray(raw_arr[compact_y, :], dtype=float)
        )
        source_y, source_x = _resolve_coords(compact_y, compact_x, coord_lookup)
        fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=110)
        ax.plot(x_axis, raw_trace, color="#d62728", linewidth=1.2, label="raw")
        ax.plot(x_axis, corrected_trace, color="#2ca02c", linewidth=1.2, label="despiked")
        ax.set_title(f"spectrum @ source(y={source_y}, x={source_x})")
        ax.set_xlabel("Wave")
        ax.set_ylabel("Intensity")
        ax.legend(loc="best")
        fig.tight_layout()
        file_name = f"spectrum_sourceY{source_y:03d}_sourceX{source_x:03d}.png"
        fig.savefig(figures_dir / file_name)
        plt.close(fig)
        saved += 1
    return saved


def _resolve_paths_from_args(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None, Path | None]:
    cfg = None
    if args.config is not None:
        cfg = load_config(args.config, path_config_path=args.path_config)
    corrected_path = Path(args.corrected) if args.corrected is not None else (
        Path(str(cfg.despike["corrected_path"])) if cfg is not None else None
    )
    if corrected_path is None:
        raise ValueError("No corrected despike bundle was provided. Use --corrected or --config.")
    out_path = Path(args.out) if args.out is not None else _default_text_output_path(corrected_path)
    cache_path = Path(args.cache) if args.cache is not None else (
        Path(str(cfg.paths["viewer_cache_path"])) if cfg is not None else None
    )
    figures_dir = Path(args.figures_dir) if args.figures_dir is not None else _default_figures_dir(corrected_path)
    return corrected_path, out_path, cache_path, figures_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export despiked spectra to a WiRE Batch File Converter-like tab-separated TXT file.")
    parser.add_argument("--config", type=Path, default=None, help="Main MuonFinder config JSON.")
    parser.add_argument("--path-config", type=Path, default=None, help="Optional machine-specific path config JSON.")
    parser.add_argument("--corrected", type=Path, default=None, help="Despike corrected NPZ bundle.")
    parser.add_argument("--out", type=Path, default=None, help="Output TXT path.")
    parser.add_argument("--cache", type=Path, default=None, help="Viewer cache path used only as a raw spectra fallback for --figures.")
    parser.add_argument("--figures", action="store_true", help="Export PNG figures for spectra where despiking occurred.")
    parser.add_argument("--figures-dir", type=Path, default=None, help="Directory for exported PNG figures.")
    args = parser.parse_args()

    corrected_path, out_path, cache_path, figures_dir = _resolve_paths_from_args(args)
    bundle = _load_corrected_bundle(corrected_path)
    corrected_spectra = np.asarray(bundle["corrected_spectra"])
    x_axis = np.asarray(bundle["x_axis"], dtype=float)
    coord_lookup = _build_coord_lookup(list(bundle.get("coord_map", [])))
    spectra_written, rows_written = export_wire_like_txt(corrected_spectra, x_axis, coord_lookup, out_path)

    figures_saved = 0
    corrected_pixel_count = len(
        _collect_corrected_pixels(
            corrected_spectra,
            dict(bundle.get("per_spectrum_correction_counts", {})),
            np.asarray(bundle["raw_spectra"]) if bundle.get("raw_spectra") is not None else None,
        )
    )
    accepted_parent_candidates = _accepted_parent_candidate_count(dict(bundle.get("metadata", {})))
    if args.figures:
        raw_spectra = bundle.get("raw_spectra")
        if raw_spectra is None:
            if cache_path is None:
                raise FileNotFoundError(
                    "Figure export requires raw spectra, but raw_spectra is missing in the corrected NPZ. "
                    "Pass --cache or regenerate despike output with raw_spectra included."
                )
            raw_spectra = _load_raw_spectra_from_cache(cache_path)
        figures_saved = export_figures(
            corrected_spectra=corrected_spectra,
            raw_spectra=np.asarray(raw_spectra),
            x_axis=x_axis,
            coord_lookup=coord_lookup,
            per_spectrum_counts=dict(bundle.get("per_spectrum_correction_counts", {})),
            figures_dir=figures_dir,
        )

    print(f"input corrected npz: {corrected_path}")
    print(f"output txt path: {out_path}")
    print(f"exported spectra: {spectra_written}")
    print(f"exported rows: {rows_written}")
    print(f"spectra with despike corrections: {corrected_pixel_count}")
    if accepted_parent_candidates is not None:
        print(f"accepted parent ss6 candidates: {accepted_parent_candidates}")
    print(f"figures exported: {'yes' if args.figures else 'no'}")
    print(f"figure files saved: {figures_saved}")
    if args.figures:
        print(f"figures dir: {figures_dir}")


if __name__ == "__main__":
    main()
