# wdf_io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np


@dataclass(frozen=True)
class WdfMap:
	path: Path
	x_axis: np.ndarray          # (N,)
	spectra: np.ndarray         # (H, W, N)
	xpos: Optional[np.ndarray]  # (H, W) or None
	ypos: Optional[np.ndarray]  # (H, W) or None
	meta: Dict[str, Any]


# --- Helpers ---
def _try_get_xy_from_origins(data, WdfDataType, h: int, w: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
	"""
	renishaw-wdf: X/Y is usually in data.origins[...] (per-spectrum).

	`enum`'s name can differ based on the WiRE version. It tries:
		- WdfDataType.X / .Y if exists
		- fallback to `name` heuristics
	"""
	if not hasattr(data, "origins"):
		return None, None

	origins = data.origins
	x, y = None, None

	# 1) direct enums
	if WdfDataType is not None:
		for cand_name in ("X", "Y"):
			if hasattr(WdfDataType, cand_name):
				key = getattr(WdfDataType, cand_name)
				try:
					arr = np.asarray(origins[key], dtype=float)
					if cand_name == "X":
						x = arr
					else:
						y = arr
				except Exception:
					pass

	# 2 heuristics over key names
	if x is None or y is None:
		for key in origins:
			nm = getattr(key, "name", str(key)).lower()
			try:
				arr = np.asarray(origins[key], dtype=float)
			except Exception:
				continue

			if x is None and nm in {"x", "xpos", "xposition", "stagex", "x_position"}:
				x = arr
			if y is None and nm in {"y", "ypos", "yposition", "stagey", "y_position"}:
				y = arr

	# reshape to a map, if possible
	try:
		if x is not None:
			x = x.reshape(h, w)
		if y is not None:
			y = y.reshape(h, w)
	except Exception:
		x, y = None, None

	return x, y


# --- Loaders ---
def load_wdf_map(path: Path) -> WdfMap:
	"""
	Loads a Renishaw .wdf map into numpy (H, W, N).

	Primarily using `renishaw-wdf` (module `wdf`, class `Wdf`).
	Fallback to `renishawWiRE` (py-wdf-reader), class `WDFReader`.

	If not a map, it throws a `RunTimeError` (for now).
	"""
	path = Path(path)

	# 1) renishaw-wdf (official)
	try:
		from wdf import Wdf  # type: ignore

		# some enums may not exist in older versions
		try:
			from wdf import WdfDataType  # type: ignore
		except Exception:
			WdfDataType = None  # noqa: N806

		with Wdf(path) as data:
			x_axis = np.asarray(list(data.xlist()), dtype=float)

			if not hasattr(data, "map_area") or not hasattr(data.map_area, "count"):
				raise RuntimeError("File does not look like a map (data.map_area.count is missing).")

			w = int(data.map_area.count.x)
			h = int(data.map_area.count.y)
			n_expected = w * h
			n_points = x_axis.size

			spectra = np.empty((h, w, n_points), dtype=np.float32)

			# order: index raises in X (columns), then in Y (rows)
			for idx in range(n_expected):
				yy = idx // w
				xx = idx % w
				spectra[yy, xx, :] = np.asarray(data[idx], dtype=np.float32)

			xpos, ypos = _try_get_xy_from_origins(data, WdfDataType, h=h, w=w)

			meta = {
				"backend": "renishaw-wdf",
				"w": w,
				"h": h
			}
			return WdfMap(path=path, x_axis=x_axis, spectra=spectra, xpos=xpos, ypos=ypos, meta=meta)

	except ImportError:
		print("Consider installing `renishaw-wdf`:\n"
		      "  pip install renishaw-wdf")
		pass
	except Exception as e:
		# if renishaw-wdf is installed, but fails reading specific file, fallback
		print(f"[wdf_io] renishaw-wdf failed: {e!r} → trying renishawWiRE fallback.")

	# 2) renishawWiRE (py-wdf-reader)
	try:
		from renishawWiRE import WDFReader  # type: ignore

		reader = WDFReader(str(path))
		x_axis = np.asarray(reader.xdata, dtype=float)
		spectra = np.asarray(reader.spectra, dtype=np.float32)

		if spectra.ndim != 3:
			raise RuntimeError(f"As of now, only grid map (H, W, N) are supported. Here spectra.ndim={spectra.ndim}")

		h, w, _n = spectra.shape
		xpos = None
		ypos = None
		try:
			xpos = np.asarray(reader.xpos, dtype=float).reshape(h, w)
			ypos = np.asarray(reader.ypos, dtype=float).reshape(h, w)
		except Exception:
			pass

		meta = {
			"backend": "renishawWiRE",
			"w": w,
			"h": h,
			"measurement_type": getattr(reader, "measurement_type", None),
		}
		try:
			reader.close()
		except Exception:
			pass

		return WdfMap(path=path, x_axis=x_axis, spectra=spectra, xpos=xpos, ypos=ypos, meta=meta)

	except ImportError as e:
		raise RuntimeError(
			"There is no WDF reader. Install either `renishaw-wdf`, or `renishawWiRE`.\n"
			"  pip install renishaw-wdf\n"
			"  pip install renishawWiRE"
		) from e
