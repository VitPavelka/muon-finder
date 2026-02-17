# morph1d.py
from __future__ import annotations

import numpy as np


# --- Helpers ---
def _grey_filter_1d(arr: np.ndarray, half_window: int, mode: str) -> np.ndarray:
	"""
	Grayscale min/max filter over the last axes.
	Prefers SciPy (fast, memory OK). Fallback: numpy sliding window (slow, memory demanding).
	"""
	if half_window < 0:
		raise ValueError("Half window cannot be negative")
	if half_window == 0:
		return arr.copy()

	size_last = 2 * half_window + 1

	# SciPy
	try:
		from scipy.ndimage import grey_erosion, grey_dilation  # type: ignore

		size = [1] * arr.ndim
		size[-1] = size_last
		if mode == "min":
			return grey_erosion(arr, size=tuple(size), mode="reflect")
		if mode == "max":
			return grey_dilation(arr, size=tuple(size), mode="reflect")
		raise ValueError("`mode` must be either 'min' or 'max'")
	except ImportError:
		print("Consider installing `scipy`:\n"
		      "  pip install scipy")
		pass

	# Numpy fallback (sliding_window_view can be RAM heavy for large maps)
	from numpy.lib.stride_tricks import sliding_window_view

	pad = [(0, 0)] * arr.ndim
	pad[-1] = (half_window, half_window)
	padded = np.pad(arr, pad_width=pad, mode="reflect")
	windows = sliding_window_view(padded, window_shape=size_last, axis=-1)

	if mode == "min":
		return windows.min(axis=-1)
	if mode == "max":
		return windows.max(axis=-1)
	raise ValueError("`mode` must be either 'min' or 'max'")


# --- Morphology ---
def erosion_1d(arr: np.ndarray, half_window: int) -> np.ndarray:
	return _grey_filter_1d(arr, half_window=half_window, mode="min")


def dilation_1d(arr: np.ndarray, half_window: int) -> np.ndarray:
	return _grey_filter_1d(arr, half_window=half_window, mode="max")


def opening_1d(arr: np.ndarray, half_window: int) -> np.ndarray:
	return dilation_1d(erosion_1d(arr, half_window), half_window)


def top_hat_1d(arr: np.ndarray, half_window: int) -> np.ndarray:
	op = opening_1d(arr, half_window)
	out = arr - op

	out[out < 0] = 0
	return out
