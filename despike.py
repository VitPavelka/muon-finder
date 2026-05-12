# despike.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage

from muon_pipeline import SpikeSegment


def _finite_float(value: Any, default: float = float("nan")) -> float:
	try:
		v = float(value)
	except Exception:
		return float(default)
	return v if np.isfinite(v) else float(default)


def _json_float(value: Any) -> float:
	v = _finite_float(value)
	return float(v) if np.isfinite(v) else float("nan")


def _odd_size(width: int, n: int) -> int:
	w = max(3, int(width))
	if w % 2 == 0:
		w += 1
	if n > 0:
		w = min(w, n if n % 2 == 1 else max(1, n - 1))
	return max(1, int(w))


def _mad(values: np.ndarray) -> float:
	x = np.asarray(values, dtype=float)
	x = x[np.isfinite(x)]
	if x.size == 0:
		return float("nan")
	med = float(np.median(x))
	return float(np.median(np.abs(x - med)))


def _connected_runs(mask: np.ndarray, offset: int = 0) -> List[Tuple[int, int]]:
	m = np.asarray(mask, dtype=bool)
	if m.size == 0:
		return []
	runs: List[Tuple[int, int]] = []
	i = 0
	while i < m.size:
		if not m[i]:
			i += 1
			continue
		j = i
		while j + 1 < m.size and bool(m[j + 1]):
			j += 1
		runs.append((int(i + offset), int(j + offset)))
		i = j + 1
	return runs


def _merge_runs(runs: Sequence[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
	if not runs:
		return []
	out: List[Tuple[int, int]] = []
	for left, right in sorted((int(a), int(b)) for a, b in runs):
		if not out:
			out.append((left, right))
			continue
		prev_l, prev_r = out[-1]
		if left - prev_r - 1 <= int(max_gap):
			out[-1] = (prev_l, max(prev_r, right))
		else:
			out.append((left, right))
	return out


def _merge_spike_segments(
		segs: Sequence[SpikeSegment],
		*,
		group_merge_gap_pts: int = 2,
		group_merge_peak_distance_pts: int = 6,
		legacy_same_peak_only: bool = False,
) -> List[SpikeSegment]:
	segs_sorted = sorted(segs, key=lambda s: (int(s.start), int(s.end), int(s.peak_index)))
	merged: List[SpikeSegment] = []
	for s in segs_sorted:
		if not merged:
			merged.append(s)
			continue
		last = merged[-1]
		overlap = int(s.start) <= int(last.end)
		close_gap = int(s.start) - int(last.end) - 1 <= int(group_merge_gap_pts)
		close_peak = abs(int(s.peak_index) - int(last.peak_index)) <= int(group_merge_peak_distance_pts)
		if legacy_same_peak_only:
			do_merge = int(s.start) < int(last.end) and abs(int(s.peak_index) - int(last.peak_index)) <= 1
		else:
			do_merge = bool(overlap or close_gap or (close_peak and close_gap))
		if not do_merge:
			merged.append(s)
			continue
		new_start = min(int(last.start), int(s.start))
		new_end = max(int(last.end), int(s.end))
		if float(s.peak_height) >= float(last.peak_height):
			peak_index = int(s.peak_index)
			peak_height = float(s.peak_height)
		else:
			peak_index = int(last.peak_index)
			peak_height = float(last.peak_height)
		merged[-1] = SpikeSegment(
			y=int(last.y),
			x=int(last.x),
			peak_index=int(peak_index),
			start=int(new_start),
			end=int(new_end),
			peak_height=float(peak_height),
			area=float(last.area) + float(s.area),
		)
	return merged


def _linear_patch_values(x_axis: np.ndarray, signal: np.ndarray, left: int, right: int, left_value: Optional[float] = None, right_value: Optional[float] = None) -> np.ndarray:
	xs = np.asarray(x_axis[left:right + 1], dtype=float)
	y0 = float(signal[left] if left_value is None else left_value)
	y1 = float(signal[right] if right_value is None else right_value)
	x0 = float(xs[0])
	x1 = float(xs[-1])
	if xs.size == 0:
		return np.asarray([], dtype=float)
	if x1 == x0:
		return np.full(xs.shape, y0, dtype=float)
	return y0 + (y1 - y0) * (xs - x0) / (x1 - x0)


def _apply_simple_linear_existing(
		out: np.ndarray,
		x_axis: np.ndarray,
		s: SpikeSegment,
) -> Dict[str, Any]:
	_, _, n = out.shape
	y = int(s.y)
	x = int(s.x)
	a = int(s.start)
	b = int(s.end)
	diag: Dict[str, Any] = {
		"method": "simple_linear_existing",
		"y": y,
		"x": x,
		"original_start": a,
		"original_end": b,
		"peak_index": int(s.peak_index),
		"fallback_used": False,
	}
	if a < 0 or b >= n or a >= b - 1:
		diag["skipped_reason"] = "invalid_anchor_interval"
		return diag
	if not (a < int(s.peak_index) < b):
		diag["skipped_reason"] = "peak_not_between_anchors"
		return diag
	ys = _linear_patch_values(x_axis, out[y, x, :], a, b)
	if ys.size != b - a + 1 or not np.all(np.isfinite(ys)):
		diag["skipped_reason"] = "nonfinite_simple_patch"
		return diag
	patch = np.minimum(ys[1:-1].astype(float), out[y, x, a + 1:b].astype(float))
	out[y, x, a + 1:b] = patch.astype(out.dtype, copy=False)
	diag.update({
		"support_left": int(a + 1),
		"support_right": int(b - 1),
		"support_width": int(max(0, b - a - 1)),
		"patch_method": "simple_linear_existing",
		"despike_enforce_not_above_raw": True,
		"skipped_reason": "",
	})
	return diag


def _anchor_zone(left: int, right: int, n: int) -> Optional[Tuple[int, int]]:
	l = max(0, int(left))
	r = min(n - 1, int(right))
	if l > r:
		return None
	return l, r


def _estimate_noise(residual_ctx: np.ndarray, raw_ctx: np.ndarray, seg_left_ctx: int, seg_right_ctx: int) -> float:
	outside = np.concatenate([
		residual_ctx[:max(0, int(seg_left_ctx))],
		residual_ctx[min(residual_ctx.size, int(seg_right_ctx) + 1):],
	])
	if np.count_nonzero(np.isfinite(outside)) >= 5:
		v = 1.4826 * _mad(outside)
		if np.isfinite(v) and v > 1e-12:
			return float(v)
	if raw_ctx.size >= 3:
		d = np.diff(raw_ctx.astype(float))
		v = 1.4826 * _mad(d) / np.sqrt(2.0)
		if np.isfinite(v) and v > 1e-12:
			return float(v)
	v = 1.4826 * _mad(residual_ctx)
	return float(v) if np.isfinite(v) and v > 1e-12 else 1e-12


def _segment_residual_diagnostics(
		signal: np.ndarray,
		s: SpikeSegment,
		*,
		context_pad_pts: int,
		baseline_method: str,
		baseline_width: int,
		low_rel: float,
		low_k_mad: float,
) -> Dict[str, Any]:
	n = int(signal.size)
	l0 = int(np.clip(s.start, 0, n - 1))
	r0 = int(np.clip(s.end, 0, n - 1))
	pad = max(0, int(context_pad_pts))
	cl = max(0, l0 - pad)
	cr = min(n - 1, r0 + pad)
	raw_ctx = signal[cl:cr + 1].astype(float, copy=True)
	if raw_ctx.size < 3:
		return {"context_left": cl, "context_right": cr, "residual_valid": False}
	method = str(baseline_method).strip().lower()
	w = _odd_size(int(baseline_width), int(raw_ctx.size))
	if method == "median":
		baseline = ndimage.median_filter(raw_ctx, size=w, mode="nearest").astype(float)
	else:
		baseline = ndimage.grey_opening(raw_ctx, size=w).astype(float)
	residual = np.maximum(raw_ctx - baseline, 0.0)
	seg_lc = int(l0 - cl)
	seg_rc = int(r0 - cl)
	seg_res = residual[seg_lc:seg_rc + 1]
	res_max = float(np.nanmax(seg_res)) if seg_res.size and np.any(np.isfinite(seg_res)) else float("nan")
	noise = _estimate_noise(residual, raw_ctx, seg_lc, seg_rc)
	low_thr = max(float(low_rel) * max(res_max, 0.0), float(low_k_mad) * float(noise)) if np.isfinite(res_max) else float("nan")
	mask = residual >= low_thr if np.isfinite(low_thr) else np.zeros_like(residual, dtype=bool)
	runs = _connected_runs(mask, offset=cl)
	apex = int(np.clip(s.peak_index, 0, n - 1))
	selected = [(a, b) for a, b in runs if a <= apex <= b or max(a, l0) <= min(b, r0)]
	selected = _merge_runs(selected, max_gap=0)
	if selected:
		rl = min(a for a, _ in selected)
		rr = max(b for _, b in selected)
	else:
		rl = rr = None
	return {
		"context_left": int(cl),
		"context_right": int(cr),
		"baseline_method": method,
		"baseline_width": int(w),
		"noise_mad": _json_float(noise),
		"res_max": _json_float(res_max),
		"low_thr": _json_float(low_thr),
		"residual_support_left": int(rl) if rl is not None else None,
		"residual_support_right": int(rr) if rr is not None else None,
		"residual_support_width": int(rr - rl + 1) if rl is not None and rr is not None else 0,
		"residual_valid": True,
	}


def _find_apex_centered_despike_support(
		signal: np.ndarray,
		s: SpikeSegment,
		*,
		context_pad_pts: int,
		baseline_method: str,
		baseline_width: int,
		low_rel: float,
		low_k_mad: float,
		min_width: int,
		max_width: int,
		segment_as_max_bounds: bool,
		edge_expand_pts: int = 0,
) -> Dict[str, Any]:
	n = int(signal.size)
	l0 = int(np.clip(s.start, 0, n - 1))
	r0 = int(np.clip(s.end, 0, n - 1))
	apex = int(np.clip(s.peak_index, 0, n - 1))
	res_diag = _segment_residual_diagnostics(
		signal,
		s,
		context_pad_pts=int(context_pad_pts),
		baseline_method=str(baseline_method),
		baseline_width=int(baseline_width),
		low_rel=float(low_rel),
		low_k_mad=float(low_k_mad),
	)
	out: Dict[str, Any] = dict(res_diag)
	out["support_mode"] = "apex_component"
	out["segment_as_max_bounds"] = bool(segment_as_max_bounds)
	if not bool(res_diag.get("residual_valid", False)):
		out.update({"support_left": apex, "support_right": apex, "support_width": 1, "support_fallback": "apex_only_no_residual"})
		return out
	cl = int(res_diag.get("context_left", max(0, l0 - int(context_pad_pts))))
	cr = int(res_diag.get("context_right", min(n - 1, r0 + int(context_pad_pts))))
	raw_ctx = signal[cl:cr + 1].astype(float)
	w = _odd_size(int(baseline_width), int(raw_ctx.size))
	if str(baseline_method).strip().lower() == "median":
		baseline = ndimage.median_filter(raw_ctx, size=w, mode="nearest").astype(float)
	else:
		baseline = ndimage.grey_opening(raw_ctx, size=w).astype(float)
	residual = np.maximum(raw_ctx - baseline, 0.0)
	thr = _finite_float(res_diag.get("low_thr"))
	if not np.isfinite(thr):
		thr = 0.0
	left_bound = l0 if bool(segment_as_max_bounds) else cl
	right_bound = r0 if bool(segment_as_max_bounds) else cr
	apex_c = apex - cl
	mask = residual >= thr
	runs = [
		(int(max(a, left_bound)), int(min(b, right_bound)))
		for a, b in _connected_runs(mask, offset=cl)
		if max(a, left_bound) <= min(b, right_bound)
	]
	if runs:
		best_i = min(
			range(len(runs)),
			key=lambda i: 0 if runs[i][0] <= apex <= runs[i][1] else min(abs(apex - runs[i][0]), abs(apex - runs[i][1])),
		)
		support_left, support_right = runs[best_i]
		fallback = "" if support_left <= apex <= support_right else "nearest_component"
		# A narrow spike can split into two residual lobes around a small notch.
		# Merge only immediately adjacent/nearby lobes and only while the support
		# remains spike-sized; this prevents jumping into broad Raman structure.
		changed = True
		while changed:
			changed = False
			for a, b in runs:
				if b < support_left:
					gap = support_left - b - 1
					new_left, new_right = a, support_right
				elif a > support_right:
					gap = a - support_right - 1
					new_left, new_right = support_left, b
				else:
					gap = 0
					new_left, new_right = min(support_left, a), max(support_right, b)
				if gap <= 2 and (int(max_width) <= 0 or new_right - new_left + 1 <= int(max_width)):
					if new_left != support_left or new_right != support_right:
						support_left, support_right = int(new_left), int(new_right)
						changed = True
	else:
		support_left = support_right = apex
		fallback = "apex_only_no_component"

	width = int(support_right - support_left + 1)
	truncated_by_width = False
	max_w = int(max_width)
	if max_w > 0 and width > max_w:
		truncated_by_width = True
		half_l = max_w // 2
		half_r = max_w - half_l - 1
		support_left = max(left_bound, apex - half_l)
		support_right = min(right_bound, apex + half_r)
		if support_right - support_left + 1 < max_w:
			if support_left == left_bound:
				support_right = min(right_bound, support_left + max_w - 1)
			elif support_right == right_bound:
				support_left = max(left_bound, support_right - max_w + 1)
	width = int(support_right - support_left + 1)
	if width < int(min_width):
		support_left = max(left_bound, apex - max(0, int(min_width) // 2))
		support_right = min(right_bound, support_left + int(min_width) - 1)
		if support_right - support_left + 1 < int(min_width):
			support_left = max(left_bound, support_right - int(min_width) + 1)
	edge_expanded_left = 0
	edge_expanded_right = 0
	relaxed_thr = max(0.0, 0.5 * float(thr))
	for _ in range(max(0, int(edge_expand_pts))):
		if int(max_width) > 0 and support_right - support_left + 1 >= int(max_width):
			break
		idx = support_left - 1
		if idx >= left_bound and 0 <= idx - cl < residual.size and residual[idx - cl] >= relaxed_thr:
			support_left = int(idx)
			edge_expanded_left += 1
		idx = support_right + 1
		if int(max_width) > 0 and support_right - support_left + 1 >= int(max_width):
			break
		if idx <= right_bound and 0 <= idx - cl < residual.size and residual[idx - cl] >= relaxed_thr:
			support_right = int(idx)
			edge_expanded_right += 1
	out.update({
		"support_left": int(support_left),
		"support_right": int(support_right),
		"support_width": int(support_right - support_left + 1),
		"support_fallback": fallback,
		"support_truncated_by_width_guard": bool(truncated_by_width),
		"neighbor_structure_guard_triggered": bool(truncated_by_width),
		"support_candidate_components": [{"left": int(a), "right": int(b)} for a, b in runs],
		"support_edge_expanded_left_pts": int(edge_expanded_left),
		"support_edge_expanded_right_pts": int(edge_expanded_right),
	})
	return out


def _apply_segment_linear_expanded(
		out: np.ndarray,
		x_axis: np.ndarray,
		s: SpikeSegment,
		*,
		expand_left_pts: int,
		expand_right_pts: int,
		preserve_anchor_points: bool,
		patch_method: str,
		auto_expand_edges: bool,
		auto_expand_max_pts: int,
		auto_expand_rel: float,
		auto_expand_k_mad: float,
		context_pad_pts: int,
		baseline_method: str,
		baseline_width: int,
		low_rel: float,
		low_k_mad: float,
		feather_pts: int,
		support_mode: str,
		segment_as_max_bounds: bool,
		support_min_width: int,
		support_max_width: int,
		use_external_anchor_zones: bool,
		anchor_pad_pts: int,
		anchor_width_pts: int,
		enforce_not_above_raw: bool,
		max_overshoot_eps: float,
) -> Dict[str, Any]:
	_, _, n = out.shape
	y = int(s.y)
	x = int(s.x)
	l0 = int(np.clip(s.start, 0, n - 1))
	r0 = int(np.clip(s.end, 0, n - 1))
	apex = int(np.clip(s.peak_index, 0, n - 1))
	diag: Dict[str, Any] = {
		"method": "segment_linear_expanded",
		"y": y,
		"x": x,
		"original_start": int(s.start),
		"original_end": int(s.end),
		"peak_index": int(s.peak_index),
		"fallback_used": False,
		"preserve_anchor_points": bool(preserve_anchor_points),
	}
	if not (0 <= l0 <= apex <= r0 < n):
		diag["skipped_reason"] = "invalid_segment_geometry"
		return diag
	raw_signal = out[y, x, :].astype(float)
	if str(support_mode).strip().lower() in {"apex_component", "component"}:
		support_diag = _find_apex_centered_despike_support(
			raw_signal,
			s,
			context_pad_pts=int(context_pad_pts),
			baseline_method=str(baseline_method),
			baseline_width=int(baseline_width),
			low_rel=float(low_rel),
			low_k_mad=float(low_k_mad),
			min_width=int(support_min_width),
			max_width=int(support_max_width),
			segment_as_max_bounds=bool(segment_as_max_bounds),
			edge_expand_pts=1,
		)
	else:
		support_diag = _segment_residual_diagnostics(
			raw_signal,
			s,
			context_pad_pts=int(context_pad_pts),
			baseline_method=str(baseline_method),
			baseline_width=int(baseline_width),
			low_rel=float(low_rel),
			low_k_mad=float(low_k_mad),
		)
		support_diag.update({
			"support_left": max(0, l0 - max(0, int(expand_left_pts))),
			"support_right": min(n - 1, r0 + max(0, int(expand_right_pts))),
			"support_mode": "segment_bounds",
		})
		support_diag["support_width"] = int(support_diag["support_right"] - support_diag["support_left"] + 1)
	diag.update(support_diag)
	support_left = int(np.clip(int(support_diag.get("support_left", apex)), 0, n - 1))
	support_right = int(np.clip(int(support_diag.get("support_right", apex)), 0, n - 1))
	if support_left > support_right:
		diag["skipped_reason"] = "invalid_support_interval"
		return diag
	if bool(auto_expand_edges) and not bool(segment_as_max_bounds) and bool(support_diag.get("residual_valid", False)):
		res_max = _finite_float(support_diag.get("res_max"))
		noise = _finite_float(support_diag.get("noise_mad"), 0.0)
		thr = max(float(auto_expand_rel) * max(res_max, 0.0), float(auto_expand_k_mad) * max(noise, 0.0))
		cl = int(support_diag.get("context_left", max(0, l0 - int(context_pad_pts))))
		cr = int(support_diag.get("context_right", min(n - 1, r0 + int(context_pad_pts))))
		raw_ctx = out[y, x, cl:cr + 1].astype(float)
		w = _odd_size(int(baseline_width), int(raw_ctx.size))
		if str(baseline_method).strip().lower() == "median":
			baseline = ndimage.median_filter(raw_ctx, size=w, mode="nearest").astype(float)
		else:
			baseline = ndimage.grey_opening(raw_ctx, size=w).astype(float)
		residual = np.maximum(raw_ctx - baseline, 0.0)
		left_expanded = 0
		while support_left > 0 and support_left - 1 >= cl and left_expanded < int(auto_expand_max_pts):
			cur = residual[support_left - cl]
			prev = residual[support_left - 1 - cl]
			if (np.isfinite(cur) and cur > thr) or (np.isfinite(prev) and prev > thr):
				support_left -= 1
				left_expanded += 1
			else:
				break
		right_expanded = 0
		while support_right < n - 1 and support_right + 1 <= cr and right_expanded < int(auto_expand_max_pts):
			cur = residual[support_right - cl]
			nxt = residual[support_right + 1 - cl]
			if (np.isfinite(cur) and cur > thr) or (np.isfinite(nxt) and nxt > thr):
				support_right += 1
				right_expanded += 1
			else:
				break
		diag["auto_expand_thr"] = _json_float(thr)
		diag["auto_expanded_left_pts"] = int(left_expanded)
		diag["auto_expanded_right_pts"] = int(right_expanded)

	if support_left > support_right:
		diag["skipped_reason"] = "invalid_support_interval"
		return diag
	if bool(use_external_anchor_zones):
		apad = max(0, int(anchor_pad_pts))
		aw = max(1, int(anchor_width_pts))
		lz = _anchor_zone(support_left - apad - aw, support_left - apad - 1, n)
		rz = _anchor_zone(support_right + apad + 1, support_right + apad + aw, n)
		if lz is None and support_left > 0:
			lz = (support_left - 1, support_left - 1)
		if rz is None and support_right < n - 1:
			rz = (support_right + 1, support_right + 1)
		if lz is None or rz is None:
			diag["skipped_reason"] = "missing_external_anchor_zone"
			return diag
		lvals = out[y, x, lz[0]:lz[1] + 1].astype(float)
		rvals = out[y, x, rz[0]:rz[1] + 1].astype(float)
		if lvals.size == 0 or rvals.size == 0 or not np.any(np.isfinite(lvals)) or not np.any(np.isfinite(rvals)):
			diag["skipped_reason"] = "invalid_external_anchor_values"
			return diag
		left_anchor_idx = int(round((lz[0] + lz[1]) / 2.0))
		right_anchor_idx = int(round((rz[0] + rz[1]) / 2.0))
		left_anchor_value = float(np.nanmedian(lvals))
		right_anchor_value = float(np.nanmedian(rvals))
	else:
		left_anchor_idx = int(support_left)
		right_anchor_idx = int(support_right)
		left_anchor_value = float(out[y, x, left_anchor_idx])
		right_anchor_value = float(out[y, x, right_anchor_idx])
	line = _linear_patch_values(
		x_axis,
		out[y, x, :].astype(float),
		left_anchor_idx,
		right_anchor_idx,
		left_value=left_anchor_value,
		right_value=right_anchor_value,
	)
	if line.size != right_anchor_idx - left_anchor_idx + 1 or not np.all(np.isfinite(line)):
		diag["skipped_reason"] = "nonfinite_segment_line"
		return diag
	replace_left = support_left
	replace_right = support_right
	if preserve_anchor_points and not bool(use_external_anchor_zones):
		replace_left = support_left + 1
		replace_right = support_right - 1
	if replace_left > replace_right:
		diag["skipped_reason"] = "empty_replace_interval"
		return diag
	patch = line[replace_left - left_anchor_idx:replace_right - left_anchor_idx + 1].astype(float)
	if str(patch_method).strip().lower() == "line_with_guard" and bool(support_diag.get("residual_valid", False)):
		cl = int(support_diag.get("context_left", support_left))
		cr = int(support_diag.get("context_right", support_right))
		raw_ctx = out[y, x, cl:cr + 1].astype(float)
		w = _odd_size(int(baseline_width), int(raw_ctx.size))
		if str(baseline_method).strip().lower() == "median":
			baseline = ndimage.median_filter(raw_ctx, size=w, mode="nearest").astype(float)
		else:
			baseline = ndimage.grey_opening(raw_ctx, size=w).astype(float)
		noise = _finite_float(support_diag.get("noise_mad"), 0.0)
		base_patch = baseline[replace_left - cl:replace_right - cl + 1].astype(float)
		if base_patch.size == patch.size:
			patch = np.maximum(patch, base_patch - float(low_k_mad) * max(noise, 0.0))
	if not np.all(np.isfinite(patch)):
		diag["skipped_reason"] = "nonfinite_patch"
		return diag
	f = max(0, int(feather_pts))
	if f > 0 and patch.size >= 2 * f + 5:
		orig = out[y, x, replace_left:replace_right + 1].astype(float)
		for i in range(f):
			t = float(i + 1) / float(f + 1)
			patch[i] = (1.0 - t) * orig[i] + t * patch[i]
			j = patch.size - 1 - i
			patch[j] = (1.0 - t) * orig[j] + t * patch[j]
	orig_replace = out[y, x, replace_left:replace_right + 1].astype(float)
	patch, cap_info = _cap_patch_to_raw(
		patch,
		orig_replace,
		enforce=bool(enforce_not_above_raw),
		eps=float(max_overshoot_eps),
		strategy=str(cap_strategy),
	)
	out[y, x, replace_left:replace_right + 1] = patch.astype(out.dtype, copy=False)
	diag.update({
		"expanded_left": int(support_left),
		"expanded_right": int(support_right),
		"support_left": int(support_left),
		"support_right": int(support_right),
		"support_width": int(support_right - support_left + 1),
		"replaced_left": int(replace_left),
		"replaced_right": int(replace_right),
		"anchor_left_zone": [int(lz[0]), int(lz[1])] if bool(use_external_anchor_zones) else [int(left_anchor_idx), int(left_anchor_idx)],
		"anchor_right_zone": [int(rz[0]), int(rz[1])] if bool(use_external_anchor_zones) else [int(right_anchor_idx), int(right_anchor_idx)],
		"left_anchor_index": int(left_anchor_idx),
		"right_anchor_index": int(right_anchor_idx),
		"left_anchor_value": _json_float(left_anchor_value),
		"right_anchor_value": _json_float(right_anchor_value),
		"patch_method": str(patch_method),
		"feather_pts": int(f),
		"despike_enforce_not_above_raw": bool(enforce_not_above_raw),
		"replacement_x": [float(v) for v in np.asarray(x_axis[left_anchor_idx:right_anchor_idx + 1], dtype=float)],
		"replacement_y": [float(v) for v in line],
		"final_patch_x": [float(v) for v in np.asarray(x_axis[replace_left:replace_right + 1], dtype=float)],
		"final_patch_y": [float(v) for v in patch],
		"skipped_reason": "",
	})
	diag.update(cap_info)
	return diag


def _cap_patch_to_raw(
		patch: np.ndarray,
		raw_values: np.ndarray,
		*,
		enforce: bool,
		eps: float,
		strategy: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
	out = np.asarray(patch, dtype=float).copy()
	raw = np.asarray(raw_values, dtype=float)
	info: Dict[str, Any] = {
		"cap_strategy": str(strategy),
		"vertical_shift_applied": False,
		"vertical_shift_value": 0.0,
		"pointwise_clamp_final_n": 0,
		"pointwise_clamp_final_max": 0.0,
		"despike_overshoot_clamped_n": 0,
		"despike_overshoot_clamped_max": 0.0,
	}
	if not bool(enforce) or out.size == 0:
		return out, info
	limit = raw + float(eps)
	mode = str(strategy).strip().lower()
	if mode == "none":
		return out, info
	overshoot = out - limit
	finite_over = overshoot[np.isfinite(overshoot)]
	max_over = float(np.max(finite_over)) if finite_over.size else 0.0
	if mode == "vertical_shift" and max_over > 0.0:
		shift = max_over + float(eps)
		out = out - shift
		info["vertical_shift_applied"] = True
		info["vertical_shift_value"] = _json_float(shift)
	elif mode == "pointwise_min":
		mask0 = np.isfinite(overshoot) & (overshoot > 0.0)
		info["despike_overshoot_clamped_n"] = int(np.count_nonzero(mask0))
		info["despike_overshoot_clamped_max"] = _json_float(float(np.max(overshoot[mask0])) if np.any(mask0) else 0.0)
		out = np.minimum(out, limit)
	final_over = out - limit
	mask = np.isfinite(final_over) & (final_over > 0.0)
	if np.any(mask):
		info["pointwise_clamp_final_n"] = int(np.count_nonzero(mask))
		info["pointwise_clamp_final_max"] = _json_float(float(np.max(final_over[mask])))
		out = np.minimum(out, limit)
	info["despike_overshoot_clamped_n"] = max(int(info["despike_overshoot_clamped_n"]), int(info["pointwise_clamp_final_n"]))
	info["despike_overshoot_clamped_max"] = _json_float(max(float(info["despike_overshoot_clamped_max"]), float(info["pointwise_clamp_final_max"])))
	return out, info


def _shift_anchor_zone(
		zone: Optional[Tuple[int, int]],
		*,
		direction: int,
		n: int,
		avoid: Sequence[Tuple[int, int]],
		max_shift: int,
) -> Tuple[Optional[Tuple[int, int]], int, bool]:
	if zone is None:
		return None, 0, False
	z = (int(zone[0]), int(zone[1]))
	for shift in range(0, int(max_shift) + 1):
		if z[0] < 0 or z[1] >= n or z[0] > z[1]:
			return None, shift, False
		overlaps = any(max(z[0], int(a)) <= min(z[1], int(b)) for a, b in avoid)
		if not overlaps:
			return z, shift, True
		z = (z[0] + int(direction), z[1] + int(direction))
	return z, int(max_shift), False


def _merge_support_groups(
		items: Sequence[Dict[str, Any]],
		*,
		gap_pts: int,
		max_width: int,
) -> List[Dict[str, Any]]:
	groups: List[Dict[str, Any]] = []
	for item in sorted(items, key=lambda d: (int(d["support_left"]), int(d["support_right"]))):
		if not groups:
			groups.append({
				"support_left": int(item["support_left"]),
				"support_right": int(item["support_right"]),
				"items": [item],
				"group_merge_reason": "seed",
			})
			continue
		g = groups[-1]
		gap = int(item["support_left"]) - int(g["support_right"]) - 1
		new_l = min(int(g["support_left"]), int(item["support_left"]))
		new_r = max(int(g["support_right"]), int(item["support_right"]))
		can_merge = gap <= int(gap_pts) and (int(max_width) <= 0 or new_r - new_l + 1 <= int(max_width))
		if can_merge:
			g["support_left"] = new_l
			g["support_right"] = new_r
			g["items"].append(item)
			g["group_merge_reason"] = "overlap_or_gap"
		else:
			groups.append({
				"support_left": int(item["support_left"]),
				"support_right": int(item["support_right"]),
				"items": [item],
				"group_merge_reason": "seed",
			})
	return groups


def _apply_support_group_patch(
		out: np.ndarray,
		original_signal: np.ndarray,
		x_axis: np.ndarray,
		*,
		y: int,
		x: int,
		group: Dict[str, Any],
		avoid_supports: Sequence[Tuple[int, int]],
		anchor_pad_pts: int,
		anchor_width_pts: int,
		anchor_avoid_all_supports: bool,
		anchor_max_shift_pts: int,
		patch_method: str,
		cap_strategy: str,
		enforce_not_above_raw: bool,
		max_overshoot_eps: float,
		feather_pts: int,
) -> Dict[str, Any]:
	_, _, n = out.shape
	support_left = int(np.clip(int(group["support_left"]), 0, n - 1))
	support_right = int(np.clip(int(group["support_right"]), 0, n - 1))
	items = list(group.get("items", []))
	diag: Dict[str, Any] = {
		"method": "segment_linear_expanded",
		"y": int(y),
		"x": int(x),
		"support_mode": "apex_component_group",
		"support_left": support_left,
		"support_right": support_right,
		"support_width": int(support_right - support_left + 1),
		"group_support_left": support_left,
		"group_support_right": support_right,
		"group_support_width": int(support_right - support_left + 1),
		"grouped_peak_indices": [int(it.get("peak_index", -1)) for it in items],
		"group_merge_reason": str(group.get("group_merge_reason", "")),
		"n_initial_supports": int(len(items)),
		"fallback_used": False,
		"preserve_anchor_points": False,
	}
	if support_left > support_right:
		diag["skipped_reason"] = "invalid_group_support"
		return diag
	apad = max(0, int(anchor_pad_pts))
	aw = max(1, int(anchor_width_pts))
	avoid = list(avoid_supports) if bool(anchor_avoid_all_supports) else [(support_left, support_right)]
	lz0 = _anchor_zone(support_left - apad - aw, support_left - apad - 1, n)
	rz0 = _anchor_zone(support_right + apad + 1, support_right + apad + aw, n)
	lz, lshift, lclean = _shift_anchor_zone(lz0, direction=-1, n=n, avoid=avoid, max_shift=int(anchor_max_shift_pts))
	rz, rshift, rclean = _shift_anchor_zone(rz0, direction=1, n=n, avoid=avoid, max_shift=int(anchor_max_shift_pts))
	diag["left_anchor_shift_pts"] = int(lshift)
	diag["right_anchor_shift_pts"] = int(rshift)
	diag["anchor_clean_left"] = bool(lclean)
	diag["anchor_clean_right"] = bool(rclean)
	if lz is None or rz is None or not lclean or not rclean:
		diag["skipped_reason"] = "no_clean_anchor"
		return diag
	lvals = original_signal[lz[0]:lz[1] + 1].astype(float)
	rvals = original_signal[rz[0]:rz[1] + 1].astype(float)
	if lvals.size == 0 or rvals.size == 0 or not np.any(np.isfinite(lvals)) or not np.any(np.isfinite(rvals)):
		diag["skipped_reason"] = "invalid_anchor_values"
		return diag
	left_anchor_idx = int(round((lz[0] + lz[1]) / 2.0))
	right_anchor_idx = int(round((rz[0] + rz[1]) / 2.0))
	left_anchor_value = float(np.nanmedian(lvals))
	right_anchor_value = float(np.nanmedian(rvals))
	line = _linear_patch_values(
		x_axis,
		original_signal,
		left_anchor_idx,
		right_anchor_idx,
		left_value=left_anchor_value,
		right_value=right_anchor_value,
	)
	if line.size != right_anchor_idx - left_anchor_idx + 1 or not np.all(np.isfinite(line)):
		diag["skipped_reason"] = "nonfinite_group_line"
		return diag
	patch = line[support_left - left_anchor_idx:support_right - left_anchor_idx + 1].astype(float)
	if patch.size != support_right - support_left + 1:
		diag["skipped_reason"] = "group_patch_size_mismatch"
		return diag
	if max(0, int(feather_pts)) > 0 and patch.size >= 2 * int(feather_pts) + 5:
		orig = original_signal[support_left:support_right + 1].astype(float)
		for i in range(int(feather_pts)):
			t = float(i + 1) / float(int(feather_pts) + 1)
			patch[i] = (1.0 - t) * orig[i] + t * patch[i]
			j = patch.size - 1 - i
			patch[j] = (1.0 - t) * orig[j] + t * patch[j]
	patch, cap_info = _cap_patch_to_raw(
		patch,
		original_signal[support_left:support_right + 1],
		enforce=bool(enforce_not_above_raw),
		eps=float(max_overshoot_eps),
		strategy=str(cap_strategy),
	)
	out[y, x, support_left:support_right + 1] = patch.astype(out.dtype, copy=False)
	diag.update(cap_info)
	diag.update({
		"original_start": int(min(int(it.get("original_start", support_left)) for it in items)) if items else support_left,
		"original_end": int(max(int(it.get("original_end", support_right)) for it in items)) if items else support_right,
		"peak_index": int(items[0].get("peak_index", support_left)) if items else support_left,
		"replaced_left": int(support_left),
		"replaced_right": int(support_right),
		"anchor_left_zone": [int(lz[0]), int(lz[1])],
		"anchor_right_zone": [int(rz[0]), int(rz[1])],
		"left_anchor_index": int(left_anchor_idx),
		"right_anchor_index": int(right_anchor_idx),
		"left_anchor_value": _json_float(left_anchor_value),
		"right_anchor_value": _json_float(right_anchor_value),
		"patch_method": str(patch_method),
		"feather_pts": int(max(0, int(feather_pts))),
		"despike_enforce_not_above_raw": bool(enforce_not_above_raw),
		"replacement_x": [float(v) for v in np.asarray(x_axis[left_anchor_idx:right_anchor_idx + 1], dtype=float)],
		"replacement_y": [float(v) for v in line],
		"final_patch_x": [float(v) for v in np.asarray(x_axis[support_left:support_right + 1], dtype=float)],
		"final_patch_y": [float(v) for v in patch],
		"individual_supports": [
			{"left": int(it.get("support_left", -1)), "right": int(it.get("support_right", -1)), "peak_index": int(it.get("peak_index", -1))}
			for it in items
		],
		"skipped_reason": "",
	})
	return diag


def _apply_residual_hysteresis_guarded_line(
		out: np.ndarray,
		x_axis: np.ndarray,
		s: SpikeSegment,
		*,
		context_pad_pts: int,
		baseline_method: str,
		baseline_width: int,
		high_rel: float,
		low_rel: float,
		high_k_mad: float,
		low_k_mad: float,
		merge_gap_pts: int,
		min_support_width: int,
		max_support_width: int,
		anchor_pad_pts: int,
		anchor_width_pts: int,
		guard_tolerance_k_mad: float,
		feather_pts: int,
) -> Dict[str, Any]:
	_, _, n = out.shape
	y = int(s.y)
	x = int(s.x)
	l0 = int(np.clip(s.start, 0, n - 1))
	r0 = int(np.clip(s.end, 0, n - 1))
	apex = int(np.clip(s.peak_index, 0, n - 1))
	diag: Dict[str, Any] = {
		"method": "residual_hysteresis_guarded_line",
		"y": y,
		"x": x,
		"original_start": int(s.start),
		"original_end": int(s.end),
		"peak_index": int(s.peak_index),
		"merged_start": l0,
		"merged_end": r0,
		"fallback_used": False,
	}
	if not (0 <= l0 <= apex <= r0 < n):
		diag["skipped_reason"] = "invalid_segment_geometry"
		return diag

	pad = max(0, int(context_pad_pts))
	cl = max(0, l0 - pad)
	cr = min(n - 1, r0 + pad)
	raw_ctx = out[y, x, cl:cr + 1].astype(float, copy=True)
	diag["context_left"] = int(cl)
	diag["context_right"] = int(cr)
	diag["baseline_method"] = str(baseline_method)
	if raw_ctx.size < 3:
		diag["skipped_reason"] = "context_too_short"
		return diag

	method = str(baseline_method).strip().lower()
	if method == "median":
		w = _odd_size(int(baseline_width), int(raw_ctx.size))
		baseline = ndimage.median_filter(raw_ctx, size=w, mode="nearest").astype(float)
	else:
		w = _odd_size(int(baseline_width), int(raw_ctx.size))
		baseline = ndimage.grey_opening(raw_ctx, size=w).astype(float)
	residual = np.maximum(raw_ctx - baseline, 0.0)
	seg_lc = int(l0 - cl)
	seg_rc = int(r0 - cl)
	apex_c = int(apex - cl)
	seg_res = residual[seg_lc:seg_rc + 1]
	res_max = float(np.nanmax(seg_res)) if seg_res.size else float("nan")
	noise = _estimate_noise(residual, raw_ctx, seg_lc, seg_rc)
	high_thr = max(float(high_rel) * max(res_max, 0.0), float(high_k_mad) * float(noise))
	low_thr = max(float(low_rel) * max(res_max, 0.0), float(low_k_mad) * float(noise))
	diag.update({
		"baseline_width": int(w),
		"noise_mad": _json_float(noise),
		"res_max": _json_float(res_max),
		"high_thr": _json_float(high_thr),
		"low_thr": _json_float(low_thr),
	})
	if not np.isfinite(res_max) or res_max <= 0.0:
		diag["skipped_reason"] = "nonpositive_residual"
		return diag

	seed_mask = residual >= high_thr
	grow_mask = residual >= low_thr
	seed_runs = _connected_runs(seed_mask, offset=cl)
	grow_runs = _connected_runs(grow_mask, offset=cl)
	diag["seed_components"] = [{"left": int(a), "right": int(b)} for a, b in seed_runs]
	diag["grow_components"] = [{"left": int(a), "right": int(b)} for a, b in grow_runs]
	selected: List[Tuple[int, int]] = []
	for gl, gr in grow_runs:
		contains_apex = int(gl) <= apex <= int(gr)
		has_seed_in_original = any(
			max(int(sl), l0) <= min(int(sr), r0) and max(int(sl), int(gl)) <= min(int(sr), int(gr))
			for sl, sr in seed_runs
		)
		if contains_apex or has_seed_in_original:
			selected.append((int(gl), int(gr)))
	selected = _merge_runs(selected, max_gap=int(merge_gap_pts))
	if not selected:
		diag["skipped_reason"] = "no_hysteresis_support"
		return diag
	support_left = min(a for a, _ in selected)
	support_right = max(b for _, b in selected)
	width = int(support_right - support_left + 1)
	diag.update({
		"support_left": int(support_left),
		"support_right": int(support_right),
		"support_width": int(width),
	})
	if width < int(min_support_width):
		diag["skipped_reason"] = "support_too_narrow"
		return diag
	if int(max_support_width) > 0 and width > int(max_support_width):
		diag["skipped_reason"] = "support_too_wide"
		return diag

	apad = max(0, int(anchor_pad_pts))
	aw = max(1, int(anchor_width_pts))
	lz = _anchor_zone(support_left - apad - aw, support_left - apad - 1, n)
	rz = _anchor_zone(support_right + apad + 1, support_right + apad + aw, n)
	if lz is None or rz is None:
		diag["skipped_reason"] = "missing_anchor_zone"
		return diag
	lvals = out[y, x, lz[0]:lz[1] + 1].astype(float)
	rvals = out[y, x, rz[0]:rz[1] + 1].astype(float)
	if lvals.size == 0 or rvals.size == 0 or not np.any(np.isfinite(lvals)) or not np.any(np.isfinite(rvals)):
		diag["skipped_reason"] = "invalid_anchor_values"
		return diag
	left_anchor_value = float(np.nanmedian(lvals))
	right_anchor_value = float(np.nanmedian(rvals))
	left_anchor_idx = int(round((lz[0] + lz[1]) / 2.0))
	right_anchor_idx = int(round((rz[0] + rz[1]) / 2.0))
	diag.update({
		"left_anchor_zone": [int(lz[0]), int(lz[1])],
		"right_anchor_zone": [int(rz[0]), int(rz[1])],
		"left_anchor_index": int(left_anchor_idx),
		"right_anchor_index": int(right_anchor_idx),
		"left_anchor_value": _json_float(left_anchor_value),
		"right_anchor_value": _json_float(right_anchor_value),
	})

	line_full = _linear_patch_values(
		x_axis,
		out[y, x, :].astype(float),
		left_anchor_idx,
		right_anchor_idx,
		left_value=left_anchor_value,
		right_value=right_anchor_value,
	)
	if line_full.size != right_anchor_idx - left_anchor_idx + 1 or not np.all(np.isfinite(line_full)):
		diag["skipped_reason"] = "nonfinite_guarded_line"
		return diag
	p0 = int(support_left - left_anchor_idx)
	p1 = int(support_right - left_anchor_idx + 1)
	line_patch = line_full[p0:p1].astype(float)
	base_patch = baseline[support_left - cl:support_right - cl + 1].astype(float)
	if line_patch.size != width or base_patch.size != width:
		diag["skipped_reason"] = "patch_size_mismatch"
		return diag
	patch = np.maximum(line_patch, base_patch - float(guard_tolerance_k_mad) * float(noise))
	if not np.all(np.isfinite(patch)):
		diag["skipped_reason"] = "nonfinite_patch"
		return diag

	orig = out[y, x, support_left:support_right + 1].astype(float)
	f = max(0, int(feather_pts))
	if f > 0 and patch.size > 1:
		for i in range(min(f, patch.size)):
			t = float(i + 1) / float(f + 1)
			patch[i] = (1.0 - t) * orig[i] + t * patch[i]
			j = patch.size - 1 - i
			patch[j] = (1.0 - t) * orig[j] + t * patch[j]
	out[y, x, support_left:support_right + 1] = patch.astype(out.dtype, copy=False)
	diag.update({
		"patch_method": "line_with_guard",
		"guard_tolerance_k_mad": float(guard_tolerance_k_mad),
		"feather_pts": int(f),
		"skipped_reason": "",
	})
	return diag


def apply_despike(
		x_axis: np.ndarray,
		spectra: np.ndarray,  # (H,W,N)
		accepted_spikes: Iterable[SpikeSegment],
		*,
		method: str = "segment_linear_expanded",
		fallback_method: str = "simple_linear_existing",
		allow_fallback: bool = True,
		return_diagnostics: bool = False,
		expand_left_pts: int = 1,
		expand_right_pts: int = 1,
		preserve_anchor_points: bool = True,
		patch_method: str = "plain_line",
		auto_expand_edges: bool = True,
		auto_expand_max_pts: int = 3,
		auto_expand_rel: float = 0.10,
		auto_expand_k_mad: float = 2.0,
		cap_strategy: str = "vertical_shift",
		support_mode: str = "apex_component",
		segment_as_max_bounds: bool = True,
		support_min_width: int = 1,
		support_max_width: int = 12,
		group_supports_before_patch: bool = True,
		support_group_gap_pts: int = 2,
		support_group_max_width: int = 28,
		support_edge_expand_pts: int = 1,
		anchor_avoid_all_supports: bool = True,
		anchor_max_shift_pts: int = 4,
		use_external_anchor_zones: bool = True,
		enforce_not_above_raw: bool = True,
		max_overshoot_eps: float = 0.0,
		context_pad_pts: int = 8,
		baseline_method: str = "opening",
		baseline_width: int = 9,
		high_rel: float = 0.35,
		low_rel: float = 0.08,
		high_k_mad: float = 5.0,
		low_k_mad: float = 2.0,
		merge_gap_pts: int = 2,
		min_support_width: int = 1,
		max_support_width: int = 20,
		anchor_pad_pts: int = 1,
		anchor_width_pts: int = 3,
		guard_tolerance_k_mad: float = 1.0,
		feather_pts: int = 0,
		group_merge_gap_pts: int = 2,
		group_merge_peak_distance_pts: int = 6,
) -> Union[np.ndarray, Tuple[np.ndarray, List[Dict[str, Any]]]]:
	"""
	Despike only already accepted spike candidates.

	Default method trusts the accepted SpikeSegment interval, expands it
	conservatively, and replaces it with a plain line. Residual logic may expand
	the interval but never shrink it. The old and hysteresis methods remain
	available for fallback/experiments.
	"""
	out = spectra.copy()
	_, _, n = out.shape
	diagnostics: List[Dict[str, Any]] = []

	by_pix: Dict[Tuple[int, int], List[SpikeSegment]] = {}
	for s in accepted_spikes:
		by_pix.setdefault((int(s.y), int(s.x)), []).append(s)

	method_key = str(method).strip().lower()
	fallback_key = str(fallback_method).strip().lower()
	for (y, x), segs in by_pix.items():
		if method_key == "segment_linear_expanded" and bool(group_supports_before_patch):
			original_signal = spectra[int(y), int(x), :].astype(float)
			support_items: List[Dict[str, Any]] = []
			for s in sorted(segs, key=lambda sp: (int(sp.peak_index), int(sp.start), int(sp.end))):
				if not (0 <= int(s.y) < out.shape[0] and 0 <= int(s.x) < out.shape[1]):
					diagnostics.append({
						"method": method_key,
						"y": int(s.y),
						"x": int(s.x),
						"peak_index": int(s.peak_index),
						"skipped_reason": "pixel_out_of_bounds",
					})
					continue
				sd = _find_apex_centered_despike_support(
					original_signal,
					s,
					context_pad_pts=int(context_pad_pts),
					baseline_method=str(baseline_method),
					baseline_width=int(baseline_width),
					low_rel=float(low_rel),
					low_k_mad=float(low_k_mad),
					min_width=int(support_min_width),
					max_width=int(support_max_width),
					segment_as_max_bounds=bool(segment_as_max_bounds),
					edge_expand_pts=int(support_edge_expand_pts),
				)
				sd.update({
					"method": method_key,
					"y": int(s.y),
					"x": int(s.x),
					"original_start": int(s.start),
					"original_end": int(s.end),
					"peak_index": int(s.peak_index),
				})
				support_items.append(sd)
			groups = _merge_support_groups(
				support_items,
				gap_pts=int(support_group_gap_pts),
				max_width=int(support_group_max_width),
			)
			avoid_supports = [(int(g["support_left"]), int(g["support_right"])) for g in groups]
			for g in groups:
				diag = _apply_support_group_patch(
					out,
					original_signal,
					x_axis,
					y=int(y),
					x=int(x),
					group=g,
					avoid_supports=avoid_supports,
					anchor_pad_pts=int(anchor_pad_pts),
					anchor_width_pts=int(anchor_width_pts),
					anchor_avoid_all_supports=bool(anchor_avoid_all_supports),
					anchor_max_shift_pts=int(anchor_max_shift_pts),
					patch_method=str(patch_method),
					cap_strategy=str(cap_strategy),
					enforce_not_above_raw=bool(enforce_not_above_raw),
					max_overshoot_eps=float(max_overshoot_eps),
					feather_pts=int(feather_pts),
				)
				diag["n_merged_support_groups"] = int(len(groups))
				if diag.get("skipped_reason") and bool(allow_fallback) and fallback_key == "simple_linear_existing":
					# Fallback only to the grouped interval, not to the broad original band.
					fb_seg = SpikeSegment(
						y=int(y),
						x=int(x),
						peak_index=int(diag.get("peak_index", int(g["support_left"]))),
						start=int(g["support_left"]),
						end=int(g["support_right"]),
						peak_height=0.0,
						area=0.0,
					)
					fb = _apply_simple_linear_existing(out, x_axis, fb_seg)
					fb["fallback_from_method"] = method_key
					fb["fallback_original_skipped_reason"] = str(diag.get("skipped_reason", ""))
					if fb.get("skipped_reason"):
						diag["fallback_attempted"] = True
						diag["fallback_skipped_reason"] = str(fb.get("skipped_reason", ""))
						diagnostics.append(diag)
					else:
						fb["fallback_used"] = True
						diagnostics.append(fb)
				else:
					diagnostics.append(diag)
			continue
		merged = _merge_spike_segments(
			segs,
			group_merge_gap_pts=int(group_merge_gap_pts),
			group_merge_peak_distance_pts=int(group_merge_peak_distance_pts),
			legacy_same_peak_only=(method_key == "simple_linear_existing"),
		)
		for s in merged:
			if not (0 <= int(s.y) < out.shape[0] and 0 <= int(s.x) < out.shape[1]):
				diagnostics.append({
					"method": method_key,
					"y": int(s.y),
					"x": int(s.x),
					"peak_index": int(s.peak_index),
					"skipped_reason": "pixel_out_of_bounds",
				})
				continue
			if method_key == "simple_linear_existing":
				diag = _apply_simple_linear_existing(out, x_axis, s)
				diagnostics.append(diag)
				continue
			if method_key == "segment_linear_expanded":
				diag = _apply_segment_linear_expanded(
					out,
					x_axis,
					s,
					expand_left_pts=int(expand_left_pts),
					expand_right_pts=int(expand_right_pts),
					preserve_anchor_points=bool(preserve_anchor_points),
					patch_method=str(patch_method),
					auto_expand_edges=bool(auto_expand_edges),
					auto_expand_max_pts=int(auto_expand_max_pts),
					auto_expand_rel=float(auto_expand_rel),
					auto_expand_k_mad=float(auto_expand_k_mad),
					cap_strategy=str(cap_strategy),
					support_mode=str(support_mode),
					segment_as_max_bounds=bool(segment_as_max_bounds),
					support_min_width=int(support_min_width),
					support_max_width=int(support_max_width),
					support_edge_expand_pts=int(support_edge_expand_pts),
					use_external_anchor_zones=bool(use_external_anchor_zones),
					anchor_pad_pts=int(anchor_pad_pts),
					anchor_width_pts=int(anchor_width_pts),
					enforce_not_above_raw=bool(enforce_not_above_raw),
					max_overshoot_eps=float(max_overshoot_eps),
					context_pad_pts=int(context_pad_pts),
					baseline_method=str(baseline_method),
					baseline_width=int(baseline_width),
					low_rel=float(low_rel),
					low_k_mad=float(low_k_mad),
					feather_pts=int(feather_pts),
				)
			elif method_key == "residual_hysteresis_guarded_line":
				diag = _apply_residual_hysteresis_guarded_line(
					out,
					x_axis,
					s,
					context_pad_pts=int(context_pad_pts),
					baseline_method=str(baseline_method),
					baseline_width=int(baseline_width),
					high_rel=float(high_rel),
					low_rel=float(low_rel),
					high_k_mad=float(high_k_mad),
					low_k_mad=float(low_k_mad),
					merge_gap_pts=int(merge_gap_pts),
					min_support_width=int(min_support_width),
					max_support_width=int(max_support_width),
					anchor_pad_pts=int(anchor_pad_pts),
					anchor_width_pts=int(anchor_width_pts),
					guard_tolerance_k_mad=float(guard_tolerance_k_mad),
					feather_pts=int(feather_pts),
				)
			else:
				diag = {
					"method": method_key,
					"y": int(s.y),
					"x": int(s.x),
					"peak_index": int(s.peak_index),
					"original_start": int(s.start),
					"original_end": int(s.end),
					"skipped_reason": f"unsupported_method:{method_key}",
				}
			if diag.get("skipped_reason") and bool(allow_fallback) and fallback_key == "simple_linear_existing":
				fb = _apply_simple_linear_existing(out, x_axis, s)
				fb["fallback_from_method"] = method_key
				fb["fallback_original_skipped_reason"] = str(diag.get("skipped_reason", ""))
				if fb.get("skipped_reason"):
					diag["fallback_attempted"] = True
					diag["fallback_skipped_reason"] = str(fb.get("skipped_reason", ""))
					diagnostics.append(diag)
				else:
					fb["fallback_used"] = True
					diagnostics.append(fb)
			else:
				diagnostics.append(diag)

	if return_diagnostics:
		return out, diagnostics
	return out
