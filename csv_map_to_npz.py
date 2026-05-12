from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


_SEP_RE = re.compile(r"[\t,; ]+")


def _human_text_key(text: str) -> List[object]:
	parts = re.split(r"(\d+)", text)
	key: List[object] = []
	for part in parts:
		if not part:
			continue
		if part.isdigit():
			key.append(int(part))
		else:
			key.append(part.replace(" ", "").lower())
	return key


def _path_sort_key(path: Path) -> List[object]:
	return _human_text_key(path.name)


def _read_text_lines(path: Path) -> List[str]:
	raw = path.read_bytes()
	if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
		return raw.decode("utf-16").splitlines()
	if raw.startswith(b"\xef\xbb\xbf"):
		return raw.decode("utf-8-sig").splitlines()

	for encoding in ("utf-8", "cp1250", "cp1252", "latin-1"):
		try:
			return raw.decode(encoding).splitlines()
		except UnicodeDecodeError:
			continue

	raise UnicodeDecodeError(
		"csv_map_to_npz",
		raw,
		0,
		min(len(raw), 4),
		f"Could not decode {path} with supported encodings",
	)


def _expand_inputs(inputs: Sequence[Path], pattern: str) -> List[Path]:
	files: List[Path] = []
	for item in inputs:
		if item.is_dir():
			files.extend(sorted((p for p in item.glob(pattern) if p.is_file()), key=_path_sort_key))
		elif item.is_file():
			files.append(item)
		else:
			raise FileNotFoundError(f"Input path does not exist: {item}")

	seen = set()
	unique: List[Path] = []
	for path in files:
		key = str(path.resolve())
		if key in seen:
			continue
		seen.add(key)
		unique.append(path)
	unique.sort(key=_path_sort_key)
	return unique


def _parse_two_column_text(path: Path, skip_rows: int) -> Tuple[np.ndarray, np.ndarray]:
	x_vals: List[float] = []
	y_vals: List[float] = []

	for line_no, raw_line in enumerate(_read_text_lines(path), start=1):
		if line_no <= skip_rows:
			continue

		line = raw_line.strip()
		if not line or line.startswith("#"):
			continue

		parts = [part for part in _SEP_RE.split(line) if part]
		if len(parts) != 2:
			raise ValueError(
				f"{path}: line {line_no} does not have exactly 2 columns: {raw_line.rstrip()!r}"
			)

		try:
			x_vals.append(float(parts[0]))
			y_vals.append(float(parts[1]))
		except ValueError as exc:
			raise ValueError(
				f"{path}: line {line_no} contains non-numeric values: {raw_line.rstrip()!r}"
			) from exc

	if not x_vals:
		raise ValueError(f"{path}: no numeric rows were found")

	x_axis = np.asarray(x_vals, dtype=float)
	intensity = np.asarray(y_vals, dtype=np.float32)
	return x_axis, intensity


def _common_x_axis_intersection(
		x_axes: Sequence[np.ndarray],
		x_tol: float,
) -> np.ndarray:
	if not x_axes:
		raise ValueError("No x-axes were provided.")

	starts = np.asarray([float(x[0]) for x in x_axes], dtype=float)
	ends = np.asarray([float(x[-1]) for x in x_axes], dtype=float)
	steps = np.asarray(
		[
			float(np.median(np.diff(x)))
			for x in x_axes
			if x.size >= 2
		],
		dtype=float,
	)
	if steps.size == 0:
		raise ValueError("Could not determine a common x-axis step.")

	step = float(np.median(steps))
	if not np.allclose(steps, step, rtol=0.0, atol=max(x_tol, 1e-12)):
		raise ValueError("Input files do not share a compatible x-axis step.")

	start = float(np.max(starts))
	end = float(np.min(ends))
	if end <= start + x_tol:
		raise ValueError("Input files do not have a shared x-axis overlap.")

	count = int(np.floor((end - start) / step + x_tol)) + 1
	if count < 2:
		raise ValueError("Shared x-axis overlap is too short.")

	common = start + step * np.arange(count, dtype=float)
	common = common[common <= end + x_tol]
	if common.size < 2:
		raise ValueError("Shared x-axis overlap is too short.")
	return common


def _load_csv_spectra(
		paths: Sequence[Path],
		skip_rows: int,
		x_tol: float,
		x_axis_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
	x_axes: List[np.ndarray] = []
	intensities: List[np.ndarray] = []

	for path in paths:
		x_axis, intensity = _parse_two_column_text(path, skip_rows=skip_rows)
		x_axes.append(x_axis)
		intensities.append(intensity)

	ref_x = x_axes[0]
	identical = all(
		x.shape == ref_x.shape and np.allclose(x, ref_x, rtol=0.0, atol=x_tol)
		for x in x_axes[1:]
	)
	if identical:
		return ref_x, np.stack(intensities, axis=0)

	mode = str(x_axis_mode).strip().lower()
	if mode == "strict":
		bad = next(
			path for path, x_axis in zip(paths[1:], x_axes[1:])
			if x_axis.shape != ref_x.shape or not np.allclose(x_axis, ref_x, rtol=0.0, atol=x_tol)
		)
		raise ValueError(
			f"{bad}: x-axis does not match the first file. "
			"Use --x-axis-mode auto or --x-axis-mode intersection to align spectra."
		)
	if mode not in {"auto", "intersection"}:
		raise ValueError(f"Unsupported --x-axis-mode: {x_axis_mode}")

	common_x = _common_x_axis_intersection(x_axes, x_tol=x_tol)
	stacked = np.stack(
		[
			np.interp(common_x, x_axis, intensity).astype(np.float32)
			for x_axis, intensity in zip(x_axes, intensities)
		],
		axis=0,
	)
	return common_x, stacked


def _report_x_axis_mode(
		paths: Sequence[Path],
		x_axis: np.ndarray,
		spectra_2d: np.ndarray,
) -> None:
	lengths = [int(spectra_2d.shape[1])] * len(paths)
	if x_axis.size:
		print(
			f"x-axis range: {float(x_axis[0]):.10g} .. {float(x_axis[-1]):.10g} "
			f"({int(x_axis.size)} points)"
		)
	print(f"loaded spectra: {len(lengths)}")


def _load_csv_spectra_and_report(
		paths: Sequence[Path],
		skip_rows: int,
		x_tol: float,
		x_axis_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
	x_axis, stacked = _load_csv_spectra(paths, skip_rows=skip_rows, x_tol=x_tol, x_axis_mode=x_axis_mode)
	_report_x_axis_mode(paths, x_axis, stacked)
	return x_axis, stacked


def _reshape_map(spectra_2d: np.ndarray, rows: int | None, cols: int | None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	n_files, n_points = spectra_2d.shape

	if rows is None and cols is None:
		rows = n_files
		cols = 1
	elif rows is None:
		if cols is None or cols <= 0:
			raise ValueError("--cols must be a positive integer")
		if n_files % cols != 0:
			raise ValueError(f"{n_files} spectra cannot be reshaped into (?, {cols})")
		rows = n_files // cols
	elif cols is None:
		if rows <= 0:
			raise ValueError("--rows must be a positive integer")
		if n_files % rows != 0:
			raise ValueError(f"{n_files} spectra cannot be reshaped into ({rows}, ?)")
		cols = n_files // rows

	if rows <= 0 or cols <= 0:
		raise ValueError("--rows/--cols must be positive integers")
	if rows * cols != n_files:
		raise ValueError(f"{n_files} spectra cannot be reshaped into ({rows}, {cols})")

	spectra_3d = spectra_2d.reshape(rows, cols, n_points)
	ypos, xpos = np.indices((rows, cols), dtype=np.float32)
	return spectra_3d, xpos, ypos


def _save_npz(
	out_path: Path,
	x_axis: np.ndarray,
	spectra: np.ndarray,
	xpos: np.ndarray,
	ypos: np.ndarray,
	source_files: Iterable[Path],
) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(
		out_path,
		x_axis=x_axis.astype(float),
		spectra=spectra.astype(np.float32),
		xpos=xpos,
		ypos=ypos,
		source_files=np.asarray([str(path) for path in source_files]),
	)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Convert individual 2-column CSV spectra into NPZ map format readable by the MuonFinder pipeline."
	)
	parser.add_argument(
		"inputs",
		nargs="+",
		type=Path,
		help="Input CSV files and/or directories. Directories are expanded by --pattern.",
	)
	parser.add_argument("--out-npz", type=Path, required=True, help="Output NPZ path.")
	parser.add_argument(
		"--pattern",
		default="*.csv",
		help="Glob used when an input is a directory. Default: *.csv",
	)
	parser.add_argument(
		"--rows",
		type=int,
		default=None,
		help="Optional map row count. Default layout is N x 1.",
	)
	parser.add_argument(
		"--cols",
		type=int,
		default=None,
		help="Optional map column count. Default layout is N x 1.",
	)
	parser.add_argument(
		"--skip-rows",
		type=int,
		default=0,
		help="Number of initial rows to skip in each file. Default: 0.",
	)
	parser.add_argument(
		"--x-tol",
		type=float,
		default=1e-9,
		help="Absolute tolerance for x-axis equality across files. Default: 1e-9.",
	)
	parser.add_argument(
		"--x-axis-mode",
		type=str,
		choices=["auto", "strict", "intersection"],
		default="auto",
		help="How to reconcile differing x-axes. 'strict' requires exact match; 'auto' and 'intersection' align to the shared overlap.",
	)
	args = parser.parse_args()

	paths = _expand_inputs(args.inputs, pattern=args.pattern)
	if not paths:
		raise ValueError("No input CSV files were found.")

	x_axis, spectra_2d = _load_csv_spectra_and_report(
		paths,
		skip_rows=args.skip_rows,
		x_tol=args.x_tol,
		x_axis_mode=args.x_axis_mode,
	)
	spectra_3d, xpos, ypos = _reshape_map(spectra_2d, rows=args.rows, cols=args.cols)
	_save_npz(args.out_npz, x_axis, spectra_3d, xpos, ypos, paths)

	print(f"Loaded {len(paths)} CSV files")
	print(f"Saved NPZ map to {args.out_npz}")
	print(f"spectra shape: {tuple(spectra_3d.shape)}")


if __name__ == "__main__":
	main()
