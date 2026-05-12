from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from feature_discrimination import (
	compute_peak_curvature_features,
	compute_spike_score_v2_features,
	estimate_background_mad,
)
from muon_pipeline import SpikeSegment
from muon_decision import compute_ss4


def _clean_value(value: Any) -> Any:
	if isinstance(value, (np.integer,)):
		return int(value)
	if isinstance(value, (np.floating,)):
		value = float(value)
	if isinstance(value, float):
		return value if math.isfinite(value) else None
	if isinstance(value, dict):
		return {str(k): _clean_value(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_clean_value(v) for v in value]
	return value


def _mad(values: np.ndarray) -> float:
	x = np.asarray(values, dtype=float)
	x = x[np.isfinite(x)]
	if x.size == 0:
		return 1.0
	med = float(np.median(x))
	mad = float(np.median(np.abs(x - med)))
	if not np.isfinite(mad) or mad <= 1e-12:
		std = float(np.nanstd(x))
		return std if np.isfinite(std) and std > 1e-12 else 1.0
	return 1.4826 * mad


def _cell_kind(
		*,
		contains_parent_apex: bool,
		overlaps_parent_segment: bool,
		n_dilation_contacts: int,
		salience: float,
		high_salience: float,
) -> str:
	if contains_parent_apex:
		return "parent_apex_cell"
	if n_dilation_contacts >= 2 and salience >= high_salience:
		return "candidate_multispike_cell"
	if n_dilation_contacts >= 1 and salience >= high_salience:
		return "candidate_spike_cell"
	if salience < 1.0:
		return "noise_like_cell"
	if overlaps_parent_segment:
		return "mixed_or_uncertain_cell"
	return "context_cell"


def _finite_minmax_norm(values: List[float]) -> List[float]:
	arr = np.asarray(values, dtype=float)
	finite = arr[np.isfinite(arr)]
	if finite.size == 0:
		return [float("nan") for _ in values]
	lo = float(np.min(finite))
	hi = float(np.max(finite))
	if hi <= lo + 1e-12:
		fill = 1.0 if hi > 0.0 else 0.0
		return [float(fill) if np.isfinite(v) else float("nan") for v in arr]
	return [float((v - lo) / (hi - lo)) if np.isfinite(v) else float("nan") for v in arr]


def _secondary_edge_rve_proxy(raw: np.ndarray, left: int, right: int, local_noise: float) -> Tuple[float, Dict[str, Any]]:
	"""Small diagnostic Raman-veto proxy for one contact cell.

	The production RVE metric is dataset-normalized and cannot be reproduced for
	an isolated contact cell. This proxy keeps the same orientation/range for
	secondary diagnostics only: narrow cells trend negative/spike-compatible,
	broad upper-level cells trend positive/Raman-like.
	"""
	n = int(raw.size)
	a = int(np.clip(left, 0, n - 1))
	b = int(np.clip(right, 0, n - 1))
	if b < a:
		a, b = b, a
	seg = np.asarray(raw[a:b + 1], dtype=float)
	debug: Dict[str, Any] = {
		"secondary_edge_kind": "local_cell_upper_width_proxy",
		"left": int(a),
		"right": int(b),
		"valid_n": 0,
	}
	if seg.size < 2 or not np.any(np.isfinite(seg)):
		debug["reason"] = "invalid_segment"
		return float("nan"), debug
	baseline = float(np.nanmin(seg))
	ymax = float(np.nanmax(seg))
	amp = float(ymax - baseline)
	noise = float(local_noise) if np.isfinite(local_noise) and local_noise > 1e-12 else 0.0
	debug.update({"baseline": baseline, "ymax": ymax, "amp": amp, "noise": noise})
	if not np.isfinite(amp) or amp <= max(noise, 1e-12):
		debug["reason"] = "below_noise_or_flat"
		return float("nan"), debug
	widths: List[float] = []
	levels: List[int] = []
	for pct in range(5, 95, 5):
		level = baseline + (float(pct) / 100.0) * amp
		if level <= baseline + noise:
			continue
		idx = np.flatnonzero(seg >= level)
		if idx.size == 0:
			continue
		widths.append(float(int(idx[-1]) - int(idx[0]) + 1))
		levels.append(int(pct))
	max_sum = float(max(seg.size, 1) * max(len(widths), 1))
	width_sum = float(np.nansum(widths)) if widths else float("nan")
	structure01 = float(width_sum / max_sum) if np.isfinite(width_sum) and max_sum > 0 else float("nan")
	rve = float(np.clip(2.0 * structure01 - 1.0, -1.0, 1.0)) if np.isfinite(structure01) else float("nan")
	debug.update({
		"reason": "ok" if np.isfinite(rve) else "no_valid_levels",
		"levels": levels,
		"widths": widths,
		"width_sum": width_sum,
		"structure01": structure01,
		"rve_proxy": rve,
		"valid_n": int(len(widths)),
	})
	return rve, debug


def _secondary_ss4_for_cell(
		*,
		raw: np.ndarray,
		gradient: Optional[np.ndarray],
		left: int,
		right: int,
		preferred_apex: Optional[int],
		dilation_contacts: List[int],
		local_noise: float,
		ss_blue_max: float,
		ss_red_min: float,
		pce_red_min: float,
		rve_red_max: float,
		missing_policy: str,
		edge_rescue_ss_min: float,
) -> Dict[str, Any]:
	n = int(raw.size)
	a = int(np.clip(left, 0, n - 1))
	b = int(np.clip(right, 0, n - 1))
	if b < a:
		a, b = b, a
	raw_seg = np.asarray(raw[a:b + 1], dtype=float)
	if raw_seg.size < 3:
		return {
			"secondary_ss4_ran": True,
			"secondary_ss4_invalid_reason": "cell_too_short",
			"secondary_ss1": float("nan"),
			"secondary_pce": float("nan"),
			"secondary_edge": float("nan"),
			"secondary_ss4": float("nan"),
			"secondary_ss4_decision": "review",
			"secondary_ss4_reason": "review_missing",
		}
	if preferred_apex is not None and a <= int(preferred_apex) <= b:
		apex = int(preferred_apex)
		anchor_source = "parent_apex"
	else:
		anchor_candidates = [int(v) for v in dilation_contacts if a <= int(v) <= b]
		if anchor_candidates:
			apex = max(anchor_candidates, key=lambda idx: float(raw[int(idx)]))
			anchor_source = "dilation_contact_max_raw"
		else:
			apex = int(a + int(np.nanargmax(raw_seg)))
			anchor_source = "cell_max_raw"
	apex_rel = int(np.clip(apex - a, 0, raw_seg.size - 1))
	sig = np.asarray(gradient, dtype=float) if gradient is not None and np.asarray(gradient).size == n else raw
	sig_seg = np.asarray(sig[a:b + 1], dtype=float)
	if sig_seg.size < 3 or not (0 < apex_rel < sig_seg.size - 1):
		ss1 = float("nan")
	else:
		rise = np.diff(sig_seg)[: max(1, apex_rel)]
		fall = np.diff(sig_seg)[max(1, apex_rel):]
		rise_slope = float(np.nanmax(rise)) if rise.size else 0.0
		fall_slope = float(np.nanmin(fall)) if fall.size else 0.0
		bg = estimate_background_mad(sig, a, b)
		bg = bg if np.isfinite(bg) and bg > 1e-12 else max(float(local_noise), 1e-12)
		ss1 = float(0.5 * np.tanh((rise_slope / bg) / 6.0) + 0.5 * np.tanh((abs(fall_slope) / bg) / 6.0))
	pce = float("nan")
	try:
		bg = max(float(local_noise), 1e-12)
		pce_features = compute_peak_curvature_features(sig_seg, bg, peak_rel=apex_rel)
		tmp = dict(pce_features)
		tmp["spike_score_v1"] = ss1
		tmp.update(compute_spike_score_v2_features(tmp))
		pce = float(tmp.get("pce_negpref_t098_evidence_signed", np.nan))
	except Exception:
		pce = float("nan")
	rve, edge_debug = _secondary_edge_rve_proxy(raw, a, b, local_noise)
	ss4 = compute_ss4(
		ss1,
		pce,
		rve,
		ss_blue_max=ss_blue_max,
		ss_red_min=ss_red_min,
		pce_red_min=pce_red_min,
		rve_red_max=rve_red_max,
		missing_policy=missing_policy,
	)
	if (
			str(ss4.get("ss4_decision", "")) != "spike"
			and np.isfinite(ss1)
			and np.isfinite(rve)
			and float(ss1) >= float(edge_rescue_ss_min)
			and float(rve) <= float(rve_red_max)
	):
		ss4 = dict(ss4)
		ss4.update({
			"ss4": 1.0,
			"ss4_decision": "spike",
			"ss4_reason": "secondary_ss_low_but_edge_red",
			"ss4_rve_zone": "red",
			"ss4_is_rve_red": 1.0,
			"ss4_is_raman_veto": 0.0,
		})
	return {
		"secondary_ss4_ran": True,
		"secondary_anchor_index": int(apex),
		"secondary_anchor_source": anchor_source,
		"secondary_ss1": float(ss1),
		"secondary_pce": float(pce),
		"secondary_edge": float(rve),
		"secondary_edge_debug": edge_debug,
		"secondary_ss4": ss4.get("ss4"),
		"secondary_ss4_decision": ss4.get("ss4_decision"),
		"secondary_ss4_reason": ss4.get("ss4_reason"),
		"secondary_ss4_ss_zone": ss4.get("ss4_ss_zone"),
		"secondary_ss4_pce_zone": ss4.get("ss4_pce_zone"),
		"secondary_ss4_rve_zone": ss4.get("ss4_rve_zone"),
	}


def _line_between(raw: np.ndarray, left: int, right: int) -> np.ndarray:
	if right <= left:
		return np.asarray([float(raw[left])], dtype=float)
	return np.linspace(float(raw[left]), float(raw[right]), int(right - left + 1))


def _chord_stats(raw: np.ndarray, left: int, right: int, chord: np.ndarray, eps: float) -> Dict[str, Any]:
	seg = np.asarray(raw[left:right + 1], dtype=float)
	overshoot = np.asarray(chord, dtype=float) - seg
	pos = overshoot > float(eps)
	return {
		"max_overshoot": float(np.nanmax(overshoot)) if overshoot.size else 0.0,
		"crossing_count": int(np.count_nonzero(pos)),
	}


def _required_points_supported(raw: np.ndarray, left: int, right: int, chord: np.ndarray, required_indices: Iterable[int], eps: float) -> bool:
	for idx in required_indices:
		try:
			i = int(idx)
		except Exception:
			continue
		if not (int(left) <= i <= int(right)):
			return False
		rel = int(i - int(left))
		if rel < 0 or rel >= int(chord.size):
			return False
		if float(chord[rel]) > float(raw[i]) + float(eps):
			return False
	return True


def _best_supporting_tangent(raw: np.ndarray, left: int, right: int, required_indices: Iterable[int], eps: float) -> Dict[str, Any]:
	best: Optional[Dict[str, Any]] = None
	required = [int(v) for v in required_indices]

	def _consider(fixed_side: str, a: int, b: int, touch_index: int) -> None:
		nonlocal best
		if b <= a:
			return
		chord = _line_between(raw, a, b)
		if not _required_points_supported(raw, a, b, chord, required, eps):
			return
		stats = _chord_stats(raw, a, b, chord, eps)
		if int(stats["crossing_count"]) > 0 or float(stats["max_overshoot"]) > float(eps):
			return
		score = (int(b - a + 1), float(np.nansum(chord)))
		cand = {
			"score": score,
			"chord_method": "supporting_tangent",
			"fixed_side": fixed_side,
			"final_left_edge": int(a),
			"final_right_edge": int(b),
			"touch_index": int(touch_index),
			"chord_x_index": [int(v) for v in range(a, b + 1)],
			"chord_y": [float(v) for v in chord],
			"max_overshoot_after": float(stats["max_overshoot"]),
			"crossing_count_after": int(stats["crossing_count"]),
		}
		if best is None or score > best["score"]:
			best = cand

	for k in range(right, left, -1):
		_consider("left", left, int(k), int(k))
	for k in range(left, right):
		_consider("right", int(k), right, int(k))
	if best is not None:
		best.pop("score", None)
		return best
	chord = _line_between(raw, left, right)
	raw_span = np.asarray(raw[left:right + 1], dtype=float)
	overshoot = chord - raw_span
	max_overshoot = float(np.nanmax(overshoot)) if overshoot.size else 0.0
	if max_overshoot > float(eps):
		chord = chord - max_overshoot - float(eps)
	stats = _chord_stats(raw, left, right, chord, eps)
	return {
		"chord_method": "supporting_tangent_fallback_shifted_chord",
		"fixed_side": "both_shifted",
		"final_left_edge": int(left),
		"final_right_edge": int(right),
		"touch_index": None,
		"vertical_shift": float(max(max_overshoot, 0.0) + float(eps)),
		"chord_x_index": [int(v) for v in range(left, right + 1)],
		"chord_y": [float(v) for v in chord],
		"max_overshoot_after": float(stats["max_overshoot"]),
		"crossing_count_after": int(stats["crossing_count"]),
	}


def _build_final_chord(raw: np.ndarray, left: int, right: int, required_indices: Optional[Iterable[int]] = None, eps: float = 1e-12) -> Dict[str, Any]:
	left = int(left)
	right = int(right)
	if right < left:
		left, right = right, left
	required = [int(v) for v in (required_indices or []) if left <= int(v) <= right]
	ordinary = _line_between(raw, left, right)
	before = _chord_stats(raw, left, right, ordinary, eps)
	base = {
		"original_left_edge": int(left),
		"original_right_edge": int(right),
		"max_overshoot_before": float(before["max_overshoot"]),
		"crossing_count_before": int(before["crossing_count"]),
	}
	if (
			int(before["crossing_count"]) == 0
			and float(before["max_overshoot"]) <= float(eps)
			and _required_points_supported(raw, left, right, ordinary, required, eps)
	):
		after = _chord_stats(raw, left, right, ordinary, eps)
		return {
			**base,
			"chord_method": "ordinary",
			"fixed_side": "both",
			"final_left_edge": int(left),
			"final_right_edge": int(right),
			"touch_index": None,
			"chord_x_index": [int(v) for v in range(left, right + 1)],
			"chord_y": [float(v) for v in ordinary],
			"max_overshoot_after": float(after["max_overshoot"]),
			"crossing_count_after": int(after["crossing_count"]),
			"required_indices": required,
		}
	out = {**base, **_best_supporting_tangent(raw, left, right, required, eps)}
	out["required_indices"] = required
	return out


def apply_contact_cell_despike_chords(raw_spectra: np.ndarray, analysis: Mapping[str, Any]) -> np.ndarray:
	"""Apply final contact-cell despike chords to a corrected copy."""
	out = np.asarray(raw_spectra, dtype=float).copy()
	for parent in analysis.get("parents", []) or []:
		if not isinstance(parent, Mapping):
			continue
		try:
			y = int(parent.get("y"))
			x = int(parent.get("x"))
		except Exception:
			continue
		if not (0 <= y < out.shape[0] and 0 <= x < out.shape[1]):
			continue
		cells_by_index = {
			int(c.get("cell_index")): c
			for c in (parent.get("cells", []) or [])
			if isinstance(c, Mapping) and c.get("cell_index") is not None
		}
		for chord in (parent.get("summary", {}) or {}).get("final_despike_chords", []) or []:
			try:
				cell_indices = [int(v) for v in chord.get("cell_indices", [])]
				if not cell_indices:
					continue
				if any(not bool(cells_by_index.get(int(ci), {}).get("secondary_final_is_spike", False)) for ci in cell_indices):
					continue
				left = int(chord.get("final_left_edge"))
				right = int(chord.get("final_right_edge"))
				vals = np.asarray(chord.get("chord_y", []), dtype=float)
			except Exception:
				continue
			if right < left:
				left, right = right, left
			left = int(np.clip(left, 0, out.shape[2] - 1))
			right = int(np.clip(right, 0, out.shape[2] - 1))
			if vals.size != right - left + 1 or vals.size == 0:
				continue
			out[y, x, left:right + 1] = vals
	return out


def analyze_erosion_dilation_contact_cells(
		*,
		x_axis: np.ndarray,
		raw_spectra: np.ndarray,
		erosion: np.ndarray,
		dilation: np.ndarray,
		gradient: Optional[np.ndarray] = None,
		parents: Iterable[SpikeSegment],
		parent_metadata: Optional[Mapping[Tuple[int, int, int, int, int], Mapping[str, Any]]] = None,
		context_pad_pts: int = 4,
		strict_equal: bool = True,
		despike_sensitivity: float = 1.0,
		secondary_noise_thr: float = 0.05,
		secondary_uncertain_thr: float = 0.5,
		ss4_ss_blue_max: float = 0.95,
		ss4_ss_red_min: float = 0.9999,
		ss4_pce_red_min: float = 0.4,
		ss4_rve_red_max: float = -0.1,
		ss4_missing_policy: str = "review",
		secondary_edge_rescue_ss_min: float = 0.85,
) -> Dict[str, Any]:
	"""Build diagnostic erosion-contact cells for already accepted parent spikes."""
	parent_metadata = parent_metadata or {}
	h, w, n = raw_spectra.shape
	pad = max(0, int(context_pad_pts))
	sensitivity = max(float(despike_sensitivity), 1e-6)
	parent_rows: List[Dict[str, Any]] = []
	for parent in parents:
		y = int(parent.y)
		x = int(parent.x)
		if not (0 <= y < h and 0 <= x < w):
			continue
		raw = raw_spectra[y, x, :].astype(float)
		ero = erosion[y, x, :].astype(float)
		dil = dilation[y, x, :].astype(float)
		grad = gradient[y, x, :].astype(float) if gradient is not None and np.asarray(gradient).shape == raw_spectra.shape else None
		start = int(np.clip(parent.start, 0, n - 1))
		end = int(np.clip(parent.end, 0, n - 1))
		apex = int(np.clip(parent.peak_index, 0, n - 1))
		if start > end:
			start, end = end, start
		cl = max(0, start - pad)
		cr = min(n - 1, end + pad)
		raw_ctx = raw[cl:cr + 1]
		local_noise = _mad(np.diff(raw_ctx)) / math.sqrt(2.0) if raw_ctx.size >= 3 else _mad(raw_ctx)
		local_noise = local_noise if np.isfinite(local_noise) and local_noise > 1e-12 else 1.0
		if bool(strict_equal):
			erosion_contacts = (np.flatnonzero(raw[cl:cr + 1] == ero[cl:cr + 1]) + cl).astype(int)
			dilation_contacts = (np.flatnonzero(raw[cl:cr + 1] == dil[cl:cr + 1]) + cl).astype(int)
		else:
			erosion_contacts = (np.flatnonzero(np.isclose(raw[cl:cr + 1], ero[cl:cr + 1])) + cl).astype(int)
			dilation_contacts = (np.flatnonzero(np.isclose(raw[cl:cr + 1], dil[cl:cr + 1])) + cl).astype(int)
		dilation_set = set(int(v) for v in dilation_contacts.tolist())
		cells: List[Dict[str, Any]] = []
		parent_apex_cell_index: Optional[int] = None
		if erosion_contacts.size >= 2:
			for idx, (left, right) in enumerate(zip(erosion_contacts[:-1], erosion_contacts[1:])):
				left = int(left)
				right = int(right)
				if right <= left:
					continue
				xs = np.asarray(x_axis[left:right + 1], dtype=float)
				ys = raw[left:right + 1].astype(float)
				chord = np.linspace(float(raw[left]), float(raw[right]), right - left + 1)
				excess = ys - chord
				above = np.maximum(excess, 0.0)
				below = np.maximum(-excess, 0.0)
				chord_area_above = float(np.nansum(above))
				chord_height_above = float(np.nanmax(above)) if above.size else 0.0
				chord_area_below = float(np.nansum(below))
				chord_area_total = float(chord_area_above + chord_area_below)
				height_abs = float(np.nanmax(np.abs(excess))) if excess.size else 0.0
				width = int(right - left + 1)
				height_z = chord_height_above / local_noise
				area_z = chord_area_above / (local_noise * max(width, 1))
				salience = float(max(height_z, math.sqrt(max(area_z, 0.0))))
				height_abs_z = float(height_abs / local_noise)
				area_total_z = float(chord_area_total / (local_noise * max(width, 1)))
				salience_total = float(max(height_abs_z, math.sqrt(max(area_total_z, 0.0))))
				salience_density = float(salience_total / math.sqrt(max(width, 1)))
				dil_inside = [int(v) for v in dilation_contacts.tolist() if left <= int(v) <= right]
				contains_apex = bool(left <= apex <= right)
				overlaps_parent = bool(max(left, start) <= min(right, end))
				if contains_apex:
					parent_apex_cell_index = int(idx)
				kind = _cell_kind(
					contains_parent_apex=contains_apex,
					overlaps_parent_segment=overlaps_parent,
					n_dilation_contacts=len(dil_inside),
					salience=salience,
					high_salience=3.0 / sensitivity,
				)
				cells.append({
					"cell_index": int(idx),
					"cell_left": int(left),
					"cell_right": int(right),
					"cell_width": int(width),
					"contains_parent_apex": contains_apex,
					"overlaps_parent_segment": overlaps_parent,
					"is_left_of_parent": bool(right < start),
					"is_right_of_parent": bool(left > end),
					"dilation_contact_indices_inside_cell": dil_inside,
					"n_dilation_contacts_inside_cell": int(len(dil_inside)),
					"contains_dilation_contact": bool(len(dil_inside) > 0),
					"parent_apex_is_dilation_contact": bool(apex in dilation_set),
					"chord_area_above": chord_area_above,
					"chord_height_above": chord_height_above,
					"chord_area_below": chord_area_below,
					"chord_area_total": chord_area_total,
					"height_abs": height_abs,
					"chord_crosses_raw": bool(np.any(below > 1e-12)),
					"chord_compactness": float(chord_area_above / max(width * chord_height_above, 1e-12)) if chord_height_above > 0 else 0.0,
					"height_z": float(height_z),
					"area_z": float(area_z),
					"salience": salience,
					"height_abs_z": height_abs_z,
					"area_total_z": area_total_z,
					"salience_total": salience_total,
					"salience_density": salience_density,
					"cell_label": kind,
					"chord_x": [float(v) for v in xs],
					"chord_y": [float(v) for v in chord],
				})

		if cells:
			s_norm = _finite_minmax_norm([float(c.get("salience", np.nan)) for c in cells])
			t_norm = _finite_minmax_norm([float(c.get("salience_total", np.nan)) for c in cells])
			d_norm = _finite_minmax_norm([float(c.get("salience_density", np.nan)) for c in cells])
			for c, sv, tv, dv in zip(cells, s_norm, t_norm, d_norm):
				c["salience_norm"] = sv
				c["salience_total_norm"] = tv
				c["salience_density_norm"] = dv
				dil_inside = [int(v) for v in c.get("dilation_contact_indices_inside_cell", []) or []]
				if bool(c.get("contains_parent_apex", False)):
					c["secondary_anchor_index"] = int(apex)
					c["secondary_anchor_source"] = "parent_apex"
				elif dil_inside:
					c["secondary_anchor_index"] = int(max(dil_inside, key=lambda idx: float(raw[int(idx)])))
					c["secondary_anchor_source"] = "dilation_contact_max_raw"
				else:
					c["secondary_anchor_index"] = int((int(c["cell_left"]) + int(c["cell_right"])) // 2)
					c["secondary_anchor_source"] = "cell_center"
				if bool(c.get("contains_parent_apex", False)):
					pre = "definite_parent_spike"
				elif not np.isfinite(tv):
					pre = "uncertain"
				elif tv < float(secondary_noise_thr):
					pre = "definite_noise"
				elif tv >= float(secondary_uncertain_thr):
					pre = "definite_spike"
				else:
					pre = "uncertain"
				c["secondary_preclassification"] = pre
				if pre == "definite_parent_spike":
					c["secondary_final_class"] = "spike"
					c["secondary_final_is_spike"] = True
					c["secondary_final_source"] = "primary_ss4_parent_apex"
					c["secondary_ss4_ran"] = False
				elif pre == "definite_noise":
					c["secondary_final_class"] = "non_spike"
					c["secondary_final_is_spike"] = False
					c["secondary_final_source"] = "t_norm_noise"
					c["secondary_ss4_ran"] = False
				elif pre == "definite_spike":
					c["secondary_final_class"] = "spike"
					c["secondary_final_is_spike"] = True
					c["secondary_final_source"] = "t_norm_spike"
					c["secondary_ss4_ran"] = False
				else:
					sec = _secondary_ss4_for_cell(
						raw=raw,
						gradient=grad,
						left=int(c["cell_left"]),
						right=int(c["cell_right"]),
						preferred_apex=(int(apex) if bool(c.get("contains_parent_apex", False)) else None),
						dilation_contacts=[int(v) for v in c.get("dilation_contact_indices_inside_cell", []) or []],
						local_noise=local_noise,
						ss_blue_max=float(ss4_ss_blue_max),
						ss_red_min=float(ss4_ss_red_min),
						pce_red_min=float(ss4_pce_red_min),
						rve_red_max=float(ss4_rve_red_max),
						missing_policy=str(ss4_missing_policy),
						edge_rescue_ss_min=float(secondary_edge_rescue_ss_min),
					)
					c.update(sec)
					is_spike = bool(str(sec.get("secondary_ss4_decision", "")) == "spike" and float(sec.get("secondary_ss4", np.nan)) == 1.0)
					c["secondary_final_class"] = "spike" if is_spike else "non_spike"
					c["secondary_final_is_spike"] = is_spike
					c["secondary_final_source"] = "secondary_ss4"

		cells_overlapping = [int(c["cell_index"]) for c in cells if bool(c["overlaps_parent_segment"])]
		cells_inside = [int(c["cell_index"]) for c in cells if int(c["cell_left"]) >= start and int(c["cell_right"]) <= end]
		cells_left = [int(c["cell_index"]) for c in cells if bool(c["is_left_of_parent"])]
		cells_right = [int(c["cell_index"]) for c in cells if bool(c["is_right_of_parent"])]
		candidate_spike = [int(c["cell_index"]) for c in cells if c["cell_label"] == "candidate_spike_cell"]
		candidate_multi = [int(c["cell_index"]) for c in cells if c["cell_label"] == "candidate_multispike_cell"]
		uncertain = [int(c["cell_index"]) for c in cells if c["cell_label"] == "mixed_or_uncertain_cell"]
		secondary_spike_cells = [int(c["cell_index"]) for c in cells if bool(c.get("secondary_final_is_spike", False))]
		secondary_uncertain_cells = [int(c["cell_index"]) for c in cells if str(c.get("secondary_preclassification", "")).startswith("uncertain")]
		secondary_groups: List[Dict[str, Any]] = []
		if secondary_spike_cells:
			by_idx = {int(c["cell_index"]): c for c in cells}
			cur: List[int] = []
			for idx in sorted(secondary_spike_cells):
				if not cur or idx == cur[-1] + 1:
					cur.append(idx)
				else:
					group_cells = [by_idx[i] for i in cur if i in by_idx]
					if group_cells:
						secondary_groups.append({
							"cell_indices": [int(i) for i in cur],
							"left": int(min(int(c["cell_left"]) for c in group_cells)),
							"right": int(max(int(c["cell_right"]) for c in group_cells)),
							"merge_rule": "neighboring_spike_cells",
						})
					cur = [idx]
			group_cells = [by_idx[i] for i in cur if i in by_idx]
			if group_cells:
				secondary_groups.append({
					"cell_indices": [int(i) for i in cur],
					"left": int(min(int(c["cell_left"]) for c in group_cells)),
					"right": int(max(int(c["cell_right"]) for c in group_cells)),
					"merge_rule": "neighboring_spike_cells",
				})
		final_chords: List[Dict[str, Any]] = []
		for group in secondary_groups:
			try:
				gl = int(group["left"])
				gr = int(group["right"])
			except Exception:
				continue
			required = [
				int(c.get("secondary_anchor_index"))
				for c in cells
				if int(c.get("cell_index", -1)) in set(int(v) for v in group.get("cell_indices", []))
				and c.get("secondary_anchor_index") is not None
			]
			chord = _build_final_chord(raw, gl, gr, required_indices=required)
			chord["cell_indices"] = [int(v) for v in group.get("cell_indices", [])]
			chord["y"] = int(y)
			chord["x"] = int(x)
			chord["chord_x"] = [float(x_axis[int(v)]) for v in chord.get("chord_x_index", [])]
			final_chords.append(chord)
		saliences = [float(c["salience"]) for c in cells]
		parent_salience = None
		if parent_apex_cell_index is not None:
			for c in cells:
				if int(c["cell_index"]) == int(parent_apex_cell_index):
					parent_salience = float(c["salience"])
					break
		island = []
		if parent_apex_cell_index is not None:
			by_idx = {int(c["cell_index"]): c for c in cells}
			island = [int(parent_apex_cell_index)]
			base_salience = max(float(parent_salience or 0.0), 1e-12)
			for direction in (-1, 1):
				cur = int(parent_apex_cell_index)
				while True:
					cur += direction
					c = by_idx.get(cur)
					if c is None:
						break
					compatible = float(c["salience"]) >= max(1.0, 0.35 * base_salience / sensitivity)
					has_dilation = bool(c["contains_dilation_contact"])
					if compatible or has_dilation:
						island.append(int(cur))
					else:
						break
			island = sorted(set(island))
		if island:
			island_cells = [c for c in cells if int(c["cell_index"]) in set(island)]
			island_left = min(int(c["cell_left"]) for c in island_cells)
			island_right = max(int(c["cell_right"]) for c in island_cells)
			island_reason = "grown_from_parent_apex_cell"
		else:
			island_left = None
			island_right = None
			island_reason = "no_parent_apex_cell"
		key = (y, x, apex, start, end)
		meta = dict(parent_metadata.get(key, {}))
		summary = {
			"n_erosion_contacts": int(erosion_contacts.size),
			"n_dilation_contacts": int(dilation_contacts.size),
			"n_cells": int(len(cells)),
			"parent_apex_cell_index": parent_apex_cell_index,
			"cells_overlapping_parent_segment": cells_overlapping,
			"cells_inside_parent_segment": cells_inside,
			"cells_left_of_parent": cells_left,
			"cells_right_of_parent": cells_right,
			"n_dilation_contacts_inside_parent_segment": int(sum(1 for v in dilation_contacts.tolist() if start <= int(v) <= end)),
			"dilation_contacts_inside_parent_segment": [int(v) for v in dilation_contacts.tolist() if start <= int(v) <= end],
			"max_cell_salience": max(saliences) if saliences else None,
			"salience_of_parent_apex_cell": parent_salience,
			"candidate_spike_cell_indices": candidate_spike,
			"candidate_multispike_cell_indices": candidate_multi,
			"uncertain_cell_indices": uncertain,
			"secondary_noise_thr": float(secondary_noise_thr),
			"secondary_uncertain_thr": float(secondary_uncertain_thr),
			"secondary_spike_cell_indices": secondary_spike_cells,
			"secondary_uncertain_cell_indices": secondary_uncertain_cells,
			"secondary_final_spike_intervals": secondary_groups,
			"final_despike_chords": final_chords,
			"preliminary_island_cell_indices": island,
			"preliminary_island_left": island_left,
			"preliminary_island_right": island_right,
			"preliminary_island_reason": island_reason,
			"incomplete_contact_segmentation": bool(erosion_contacts.size < 2),
		}
		parent_rows.append({
			"y": y,
			"x": x,
			"parent_start": int(start),
			"parent_end": int(end),
			"parent_apex": int(apex),
			"parent_peak_height": float(parent.peak_height),
			"parent_ss4_value": _clean_value(meta.get("ss4", 1.0)),
			"parent_ss4_reason": meta.get("ss4_reason"),
			"parent_ss1": _clean_value(meta.get("spike_score_v1")),
			"parent_pce": _clean_value(meta.get("pce_negpref_t098_evidence_signed")),
			"parent_edge": _clean_value(meta.get("recdw_sum_0_90_raman_veto_evidence_signed")),
			"parent_edge_feature": meta.get("ss4_rve_feature"),
			"context_left": int(cl),
			"context_right": int(cr),
			"local_noise": float(local_noise),
			"erosion_contacts": [int(v) for v in erosion_contacts.tolist()],
			"erosion_contact_x": [float(x_axis[int(v)]) for v in erosion_contacts.tolist()],
			"erosion_contact_y": [float(raw[int(v)]) for v in erosion_contacts.tolist()],
			"dilation_contacts": [int(v) for v in dilation_contacts.tolist()],
			"dilation_contact_x": [float(x_axis[int(v)]) for v in dilation_contacts.tolist()],
			"dilation_contact_y": [float(raw[int(v)]) for v in dilation_contacts.tolist()],
			"cells": cells,
			"summary": summary,
		})
	result = {
		"method": "erosion_dilation_contact_cell_analysis",
		"n_parent_segments": int(len(parent_rows)),
		"config": {
			"despike_contact_context_pad_pts": int(context_pad_pts),
			"despike_contact_strict_equal": bool(strict_equal),
			"despike_sensitivity": float(despike_sensitivity),
			"secondary_noise_thr": float(secondary_noise_thr),
			"secondary_uncertain_thr": float(secondary_uncertain_thr),
			"ss4_ss_blue_max": float(ss4_ss_blue_max),
			"ss4_ss_red_min": float(ss4_ss_red_min),
			"ss4_pce_red_min": float(ss4_pce_red_min),
			"ss4_rve_red_max": float(ss4_rve_red_max),
			"secondary_edge_rescue_ss_min": float(secondary_edge_rescue_ss_min),
		},
		"parents": parent_rows,
	}
	return _clean_value(result)


def save_despike_contact_debug_json(path: Path, analysis: Mapping[str, Any]) -> None:
	Path(path).write_text(json.dumps(_clean_value(dict(analysis)), indent=2), encoding="utf-8")
