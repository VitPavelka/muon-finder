# morph1d.py
from __future__ import annotations

import numpy as np


# --- Helpers ---
def _minmax_filter_1d(arr: np.ndarray, se_size: int, mode: str) -> np.ndarray:
	"""
	Grayscale min/max filter over the last axis.
	Prefers SciPy (fast, memory OK). Fallback: numpy sliding window (slow, memory demanding).
	Note: even se_size is allowed; SciPy origin=0 makes it slightly left-biased for even sizes.
	"""
	if se_size < 1:
		raise ValueError(f"se_size must be >= 1: {se_size}")
	if se_size == 1:
		return arr.copy()

	# SciPy
	try:
		from scipy.ndimage import minimum_filter1d, maximum_filter1d  # type: ignore

		if mode == "min":
			return minimum_filter1d(arr, size=se_size, axis=-1, mode="reflect", origin=0)
		if mode == "max":
			return maximum_filter1d(arr, size=se_size, axis=-1, mode="reflect", origin=0)
		raise ValueError("`mode` must be either 'min' or 'max'")
	except ImportError:
		print("Consider installing `scipy`:\n"
		      "  pip install scipy")
		pass

	# Numpy fallback (sliding_window_view can be RAM heavy for large maps)
	from numpy.lib.stride_tricks import sliding_window_view

	left = se_size // 2
	right = se_size - 1 - left  # matches SciPy origin=0 behavior for even sizes
	pad = [(0, 0)] * arr.ndim
	pad[-1] = (left, right)
	padded = np.pad(arr, pad_width=pad, mode="reflect")
	windows = sliding_window_view(padded, window_shape=se_size, axis=-1)

	if mode == "min":
		return windows.min(axis=-1)
	if mode == "max":
		return windows.max(axis=-1)
	raise ValueError("`mode` must be either 'min' or 'max'")


# --- Morphology ---
def erosion_1d(arr: np.ndarray, se_size: int) -> np.ndarray:
	return _minmax_filter_1d(arr, se_size=se_size, mode="min")


def dilation_1d(arr: np.ndarray, se_size: int) -> np.ndarray:
	return _minmax_filter_1d(arr, se_size=se_size, mode="max")


def opening_1d(arr: np.ndarray, se_size: int) -> np.ndarray:
	return dilation_1d(erosion_1d(arr, se_size), se_size)


def top_hat_1d(arr: np.ndarray, se_size: int) -> np.ndarray:
	op = opening_1d(arr, se_size)
	out = arr - op

	out[out < 0] = 0
	return out
