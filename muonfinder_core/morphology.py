from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _minmax_filter_1d(arr: np.ndarray, se_size: int, mode: str) -> np.ndarray:
    if se_size < 1:
        raise ValueError("se_size must be >= 1")
    if se_size == 1:
        return np.asarray(arr).copy()
    try:
        from scipy.ndimage import maximum_filter1d, minimum_filter1d  # type: ignore

        if mode == "min":
            return minimum_filter1d(arr, size=se_size, axis=-1, mode="reflect", origin=0)
        if mode == "max":
            return maximum_filter1d(arr, size=se_size, axis=-1, mode="reflect", origin=0)
    except ImportError:
        pass
    from numpy.lib.stride_tricks import sliding_window_view

    left = se_size // 2
    right = se_size - 1 - left
    pad = [(0, 0)] * arr.ndim
    pad[-1] = (left, right)
    padded = np.pad(arr, pad_width=pad, mode="reflect")
    windows = sliding_window_view(padded, window_shape=se_size, axis=-1)
    if mode == "min":
        return windows.min(axis=-1)
    if mode == "max":
        return windows.max(axis=-1)
    raise ValueError("mode must be 'min' or 'max'")


def erosion_1d(arr: np.ndarray, se_size: int) -> np.ndarray:
    return _minmax_filter_1d(arr, se_size=se_size, mode="min")


def dilation_1d(arr: np.ndarray, se_size: int) -> np.ndarray:
    return _minmax_filter_1d(arr, se_size=se_size, mode="max")


def opening_1d(arr: np.ndarray, se_size: int) -> np.ndarray:
    return dilation_1d(erosion_1d(arr, se_size=se_size), se_size=se_size)


def top_hat_1d(arr: np.ndarray, se_size: int) -> np.ndarray:
    opening = opening_1d(arr, se_size=se_size)
    out = np.asarray(arr, dtype=float) - np.asarray(opening, dtype=float)
    out[out < 0] = 0
    return out


@dataclass(frozen=True)
class MorphologyWindowSet:
    window: int
    erosion: np.ndarray
    dilation: np.ndarray
    opening: np.ndarray
    top_hat: np.ndarray
    gradient: np.ndarray


def compute_morphology_windows(spectra: np.ndarray, windows: list[int]) -> dict[int, MorphologyWindowSet]:
    out: dict[int, MorphologyWindowSet] = {}
    for window in sorted({int(w) for w in windows if int(w) >= 1}):
        erosion = erosion_1d(spectra, window)
        dilation = dilation_1d(spectra, window)
        opening = opening_1d(spectra, window)
        top_hat = np.asarray(spectra, dtype=float) - np.asarray(opening, dtype=float)
        top_hat[top_hat < 0] = 0
        gradient = np.asarray(dilation, dtype=float) - np.asarray(erosion, dtype=float)
        out[window] = MorphologyWindowSet(
            window=window,
            erosion=np.asarray(erosion),
            dilation=np.asarray(dilation),
            opening=np.asarray(opening),
            top_hat=np.asarray(top_hat),
            gradient=np.asarray(gradient),
        )
    return out
