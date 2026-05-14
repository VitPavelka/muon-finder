from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .data_model import SpectrumDataset


def _try_get_xy_from_origins(data, WdfDataType, h: int, w: int):
    if not hasattr(data, "origins"):
        return None, None
    origins = data.origins
    x = None
    y = None
    if WdfDataType is not None:
        for name in ("X", "Y"):
            if hasattr(WdfDataType, name):
                key = getattr(WdfDataType, name)
                try:
                    arr = np.asarray(origins[key], dtype=float)
                except Exception:
                    continue
                if name == "X":
                    x = arr
                else:
                    y = arr
    if x is None or y is None:
        for key in origins:
            key_name = getattr(key, "name", str(key)).lower()
            try:
                arr = np.asarray(origins[key], dtype=float)
            except Exception:
                continue
            if x is None and key_name in {"x", "xpos", "stagex", "x_position"}:
                x = arr
            if y is None and key_name in {"y", "ypos", "stagey", "y_position"}:
                y = arr
    try:
        if x is not None:
            x = x.reshape(h, w)
        if y is not None:
            y = y.reshape(h, w)
    except Exception:
        x, y = None, None
    return x, y


def load_wdf_map(path: Path) -> SpectrumDataset:
    path = Path(path)
    try:
        from wdf import Wdf  # type: ignore
        try:
            from wdf import WdfDataType  # type: ignore
        except Exception:
            WdfDataType = None
        with Wdf(path) as data:
            x_axis = np.asarray(list(data.xlist()), dtype=float)
            w = int(data.map_area.count.x)
            h = int(data.map_area.count.y)
            spectra = np.empty((h, w, x_axis.size), dtype=np.float32)
            for idx in range(h * w):
                yy = idx // w
                xx = idx % w
                spectra[yy, xx, :] = np.asarray(data[idx], dtype=np.float32)
            xpos, ypos = _try_get_xy_from_origins(data, WdfDataType, h, w)
            return SpectrumDataset(
                path=path,
                x_axis=x_axis,
                spectra=spectra,
                xpos=xpos,
                ypos=ypos,
                meta={"backend": "renishaw-wdf", "h": h, "w": w},
            )
    except ImportError:
        pass
    except Exception as exc:
        print(f"[io] renishaw-wdf failed: {exc!r}; trying renishawWiRE fallback")

    try:
        from renishawWiRE import WDFReader  # type: ignore
        reader = WDFReader(str(path))
        x_axis = np.asarray(reader.xdata, dtype=float)
        spectra = np.asarray(reader.spectra, dtype=np.float32)
        if spectra.ndim != 3:
            raise RuntimeError(f"Expected grid map (H,W,N), got ndim={spectra.ndim}")
        h, w, _ = spectra.shape
        xpos = None
        ypos = None
        try:
            xpos = np.asarray(reader.xpos, dtype=float).reshape(h, w)
            ypos = np.asarray(reader.ypos, dtype=float).reshape(h, w)
        except Exception:
            pass
        try:
            reader.close()
        except Exception:
            pass
        return SpectrumDataset(
            path=path,
            x_axis=x_axis,
            spectra=spectra,
            xpos=xpos,
            ypos=ypos,
            meta={"backend": "renishawWiRE", "h": h, "w": w},
        )
    except ImportError as exc:
        raise RuntimeError(
            "No WDF reader is installed. Use renishaw-wdf or renishawWiRE."
        ) from exc


def load_npz_map(path: Path) -> SpectrumDataset:
    data = np.load(path, allow_pickle=True)
    x_axis = np.asarray(data["x_axis"], dtype=float)
    if "corrected_spectra" in data:
        spectra = np.asarray(data["corrected_spectra"], dtype=np.float32)
    elif "spectra" in data:
        spectra = np.asarray(data["spectra"], dtype=np.float32)
    else:
        raise KeyError("NPZ must contain either 'spectra' or 'corrected_spectra'.")
    return SpectrumDataset(
        path=Path(path),
        x_axis=x_axis,
        spectra=spectra,
        xpos=np.asarray(data["xpos"], dtype=float) if "xpos" in data else None,
        ypos=np.asarray(data["ypos"], dtype=float) if "ypos" in data else None,
        meta={"backend": "npz"},
    )


def load_dataset(path: Path | str, input_format: str = "auto") -> SpectrumDataset:
    dataset_path = Path(path)
    fmt = str(input_format).strip().lower()
    suffix = dataset_path.suffix.lower()
    if fmt == "auto":
        if suffix == ".wdf":
            return load_wdf_map(dataset_path)
        if suffix == ".npz":
            return load_npz_map(dataset_path)
        raise ValueError(f"Unsupported input type: {suffix}")
    if fmt == "wdf":
        return load_wdf_map(dataset_path)
    if fmt == "npz":
        return load_npz_map(dataset_path)
    raise ValueError(f"Unsupported input_format: {input_format!r}")


def load_target_coords_csv(path: Path | str, shape_hw: tuple[int, int]) -> list[tuple[int, int]]:
    csv_path = Path(path)
    first_line = csv_path.read_text(encoding="utf-8").splitlines()[0]
    delimiter = ";" if ";" in first_line else ","
    raw = np.genfromtxt(csv_path, delimiter=delimiter, names=True, dtype=None, encoding="utf-8")
    if getattr(raw, "size", 0) == 0:
        return []
    h, w = shape_hw
    cols = {c.lower(): c for c in (raw.dtype.names or [])}
    y_col = cols.get("y")
    x_col = cols.get("x")
    if y_col is None or x_col is None:
        raise ValueError("coords CSV must contain 'y' and 'x' headers.")
    seen: set[tuple[int, int]] = set()
    coords: list[tuple[int, int]] = []
    for yv, xv in zip(np.atleast_1d(raw[y_col]), np.atleast_1d(raw[x_col])):
        y = int(yv)
        x = int(xv)
        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        key = (y, x)
        if key in seen:
            continue
        seen.add(key)
        coords.append(key)
    return coords
