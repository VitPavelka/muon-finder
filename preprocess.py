from __future__ import annotations

from typing import Tuple
import numpy as np


def resample_axis_and_spectra(
		x_axis: np.ndarray,
		spectra: np.ndarray,  # (..., N)
		factor: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Linear resampling along spectral axis.
	factor=1 -> identity.
	New length  = (N-1)*factor + 1
	"""
	if factor <= 1:
		return np.asarray(x_axis, dtype=float), np.asarray(spectra, dtype=float).copy()

	x = np.asarray(x_axis, dtype=float)
	arr = np.asarray(spectra, dtype=float)
	n = x.size
	if n < 2:
		return x, arr.copy()
	if arr.shape[-1] != n:
		raise ValueError("spectra last dimension must match x_axis length")

	dx = np.diff(x)
	if np.all(dx > 0):
		x_work = x
		flip = False
	elif np.all(dx < 0):
		x_work = x[::-1]
		flip = True
	else:
		raise ValueError("x_axis must be strictly monotonic (increasing or decreasing)")

	n_new = (n - 1) * int(factor) + 1
	x_new = np.linspace(float(x[0]), float(x[-1]), n_new, dtype=float)
	x_new_work = x_new[::-1] if flip else x_new

	flat = arr.reshape(-1, n)
	if flip:
		flat = flat[:, ::-1]

	out = np.empty((flat.shape[0], n_new), dtype=float)
	for i in range(flat.shape[0]):
		out[i] = np.interp(x_new_work, x_work, flat[i])
	if flip:
		out = out[:, ::-1]

	new_shape = arr.shape[:-1] + (n_new,)
	return x_new, out.reshape(new_shape)
