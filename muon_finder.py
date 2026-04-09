# muon_finder.py
from __future__ import annotations

import argparse
import json
import math
import pathlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

from wdf_io import load_dataset
from muon_pipeline import (
	compute_morph_overlays,
	score_map_from_top_hat,
	threshold_score_map,
	extract_spikes_for_candidates,
	neighbour_filter_ratio,
	SpikeSegment
)
from despike import apply_despike
from viewer import show_hover_map
from results_io import save_result_npz, save_spikes_csv
from debug_report import build_debug_report, save_debug_report_json

DEFAULT_CONFIG: Dict[str, Any] = {
	"input_path": "",
	"se_size": 5,
	"score_mode": "max",
	"threshold_method": "quantile",
	"threshold_quantile": 0.1,
	"threshold_k_mad": 20.0,
	"threshold_min_abs": None,
	"k_mad_pixel": 8.0,
	"min_peak": 0.0,
	"baseline_se_size": 11,
	"edge_k_mad": 2.0,
	"max_width_pts": 20,
	"pad_pts": 0,
	"neighbor_filter_enabled": False,
	"neighbor_radius": 1,
	"neighbor_ratio_min": 3.0,
	"coords_csv": None,
	"use_compact_coords_view": True,
	"despike_enabled": True,
	"save_npz_path": None,
	"save_spikes_csv_path": None,
	"save_corrected_in_npz": True,
	"save_overlays_in_npz": True,
	"debug_report_path": None,
	"debug_include_per_spectrum": True,
	"debug_top_pixels": 25,
}


def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
	cfg = dict(DEFAULT_CONFIG)
	if config_path is None:
		return cfg
	path = Path(config_path)
	user_cfg = json.loads(path.read_text(encoding="utf-8"))
	cfg.update(user_cfg)
	return cfg


def load_target_coords_csv(path: Path, shape_hw: Tuple[int, int]) -> List[Tuple[int, int]]:
	h, w = shape_hw
	raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
	if raw.size == 0:
		return []

	cols = {c.lower(): c for c in raw.dtype.names or []}
	y_col = cols.get("y")
	x_col = cols.get("x")
	if y_col is None or x_col is None:
		raise ValueError("coords CSV must contain headers 'y' and 'x'.")

	coords: List[Tuple[int, int]] = []
	seen = set()
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


def build_compact_subset(
		x_axis: np.ndarray,
		raw_spectra: np.ndarray,
		score_map: np.ndarray,
		candidate_mask: np.ndarray,
		overlays: Dict[str, np.ndarray],
		spikes_by_pixel: Dict[Tuple[int, int], List[SpikeSegment]],
		coords: List[Tuple[int, int]],
		corrected_spectra: Optional[np.ndarray] = None,
) -> Tuple[
	np.ndarray, np.ndarray, np.ndarray,
	Dict[str, np.ndarray], Dict[Tuple[int, int], List[SpikeSegment]],
	Dict[Tuple[int, int], Tuple[int, int]], Optional[np.ndarray]
]:
	n = len(coords)
	if n == 0:
		raise ValueError("No coordinates for compact subset.")

	grid_w = int(math.ceil(math.sqrt(n)))
	grid_h = int(math.ceil(n / grid_w))
	spec_n = x_axis.size

	raw_compact = np.zeros((grid_h, grid_w, spec_n), dtype=raw_spectra.dtype)
	corr_compact = None
	if corrected_spectra is not None:
		corr_compact = np.zeros((grid_h, grid_w, spec_n), dtype=corrected_spectra.dtype)
	score_compact = np.zeros((grid_h, grid_w), dtype=score_map.dtype)
	cand_compact = np.zeros((grid_h, grid_w), dtype=bool)
	overlay_compact: Dict[str, np.ndarray] = {
		k: np.zeros((grid_h, grid_w, spec_n), dtype=v.dtype) for k, v in overlays.items()
	}
	spikes_compact: Dict[Tuple[int, int], List[SpikeSegment]] = {}
	coord_map: Dict[Tuple[int, int], Tuple[int, int]] = {}

	for i, (src_y, src_x) in enumerate(coords):
		yy = i // grid_w
		xx = i % grid_w
		coord_map[(yy, xx)] = (src_y, src_x)

		raw_compact[yy, xx, :] = raw_spectra[src_y, src_x, :]
		if corr_compact is not None:
			corr_compact[yy, xx, :] = corrected_spectra[src_y, src_x, :]
		score_compact[yy, xx] = score_map[src_y, src_x]
		cand_compact[yy, xx] = bool(candidate_mask[src_y, src_x])
		for k in overlay_compact:
			overlay_compact[k][yy, xx, :] = overlays[k][src_y, src_x, :]

		src_spikes = spikes_by_pixel.get((src_y, src_x), [])
		if src_spikes:
			spikes_compact[(yy, xx)] = [
				SpikeSegment(
					y=yy,
					x=xx,
					peak_index=s.peak_index,
					start=s.start,
					end=s.end,
					peak_height=s.peak_height,
					area=s.area,
				)
				for s in src_spikes
			]

	return raw_compact, score_compact, cand_compact, overlay_compact, spikes_compact, coord_map, corr_compact


def run(cfg: Dict[str, Any]) -> None:

	# 0) config file
	in_path = Path(cfg['input_path'])

	# 1) load data
	ds = load_dataset(in_path)
	x_axis = ds.x_axis
	raw = ds.spectra

	# 2) morphology
	overlays = compute_morph_overlays(raw, se_size=int(cfg['se_size']))

	# 3a) score map + candidates
	score_map = score_map_from_top_hat(overlays['top_hat'], mode=str(cfg['score_mode']))
	thr = threshold_score_map(
		score_map=score_map,
		method=str(cfg['threshold_method']),
		quantile=float(cfg['threshold_quantile']),
		k_mad=float(cfg['threshold_k_mad']),
		min_abs=cfg['threshold_min_abs'],
	)
	candidate_mask = score_map >= thr

	target_coords: Optional[List[Tuple[int, int]]] = None
	if cfg.get("coords_csv"):
		target_coords = load_target_coords_csv(Path(cfg['coords_csv']), shape_hw=score_map.shape)
		coord_mask = np.zeros_like(candidate_mask, dtype=bool)
		for yy, xx in target_coords:
			coord_mask[yy, xx] = True
		# For coordinate-based optimization we run extraction only on requested spectra
		candidate_mask = coord_mask

	# 3b) spike segments (narrow bands) for candidate pixels
	spikes, spikes_by_pixel = extract_spikes_for_candidates(
		x_axis=x_axis,
		top_hat=overlays['top_hat'],
		candidate_mask=candidate_mask,
		raw_spectra=raw,
		max_width_pts=int(cfg['max_width_pts']),
		k_mad_pixel=float(cfg['k_mad_pixel']),
		min_peak=float(cfg['min_peak']),
		baseline_se_size=cfg['baseline_se_size'],
		edge_k_mad=float(cfg['edge_k_mad']),
		pad_pts=int(cfg['pad_pts']),
	)

	# 3) comparison with neighbors
	if bool(cfg['neighbor_filter_enabled']):
		accepted = neighbour_filter_ratio(
			top_hat=overlays['top_hat'],
			spikes=spikes,
			radius=int(cfg['neighbor_radius']),
			ratio_min=float(cfg['neighbor_ratio_min']),
		)
		accepted_set = set(accepted)

		# rebuild spikes_by_pixel + rebuild cand from kept spikes
		spikes_by_pixel = {
			pix: [s for s in segs if s in accepted_set]
			for pix, segs in spikes_by_pixel.items()
		}
		spikes_by_pixel = {pix: segs for pix, segs in spikes_by_pixel.items() if segs}
		spikes = accepted

	corrected = raw

	# 5) Despike
	if bool(cfg['despike_enabled']):
		corrected = apply_despike(
			x_axis=x_axis, spectra=raw, accepted_spikes=spikes
		)

	# 6) Viewer (hover)
	view_x = x_axis
	view_spectra = raw
	view_score = score_map
	view_mask = candidate_mask
	view_overlays = overlays
	view_spikes_by_pixel = spikes_by_pixel
	source_coords_map = None
	view_corrected = corrected if bool(cfg['despike_enabled']) else None

	if target_coords and bool(cfg['use_compact_coords_view']):
		(
			view_spectra,
			view_score,
			view_mask,
			view_overlays,
			view_spikes_by_pixel,
			source_coords_map,
			view_corrected,
		) = build_compact_subset(
			x_axis=view_x,
			raw_spectra=view_spectra,
			score_map=view_score,
			candidate_mask=view_mask,
			overlays=view_overlays,
			spikes_by_pixel=view_spikes_by_pixel,
			coords=target_coords,
			corrected_spectra=view_corrected,
		)
	show_hover_map(
		x_axis=view_x,
		spectra=view_spectra,
		score_map=view_score,
		candidate_mask=view_mask,
		spikes_by_pixel=view_spikes_by_pixel,
		overlays=view_overlays,
		source_coords_map=source_coords_map,
		corrected_spectra=view_corrected,
		initial_checked={"raw": True, "top_hat": True, "corrected:": True},
	)

	# 7) Save data and report
	if cfg.get("save_npz_path"):
		save_result_npz(
			out_path=Path(cfg['save_npz_path']),
			ds=ds,
			score_map=score_map,
			threshold=float(thr),
			candidate_mask=candidate_mask,
			spikes=spikes,
			corrected_spectra=corrected if bool(cfg['save_corrected_in_npz']) else None,
			overlays=overlays if bool(cfg['save_overlays_in_npz']) else None,
		)

	if cfg.get("save_spikes_csv_path"):
		save_spikes_csv(path=Path(cfg["save_spikes_csv_path"]), spikes=spikes)

	if cfg.get("debug_report_path"):
		report = build_debug_report(
			score_map=score_map,
			candidate_mask=candidate_mask,
			spikes_by_pixel=spikes_by_pixel,
			threshold=float(thr),
			target_coords=target_coords,
			include_per_spectrum=bool(cfg['debug_include_per_spectrum']),
			max_top_pixels=int(cfg['debug_top_pixels']),
		)
		save_debug_report_json(Path(cfg['debug_report_path']), report)


def main() -> None:
	parser = argparse.ArgumentParser(description="Muon finder")
	parser.add_argument("--config", type=Path, default=None, help="Path to JSON config file.")
	parser.add_argument("--input", type=Path, default=None, help="Input .wdf or .npz path (overrides config).")
	parser.add_argument("--coords-csv", type=Path, default=None, help="CSV with columns y,x for targeted spectra.")
	args = parser.parse_args()

	cfg = load_config(args.config)
	if args.input is not None:
		cfg['input_path'] = str(args.input)
	if args.coords_csv is not None:
		cfg['coords_csv'] = str(args.coords_csv)
	if not cfg.get("input_path"):
		raise ValueError("Set input path via config 'input_path' or CLI --input.")

	run(cfg)


if __name__ == "__main__":
	main()
