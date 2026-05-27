from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np

from .data_model import CandidateSegment
from .morphology import dilation_1d, erosion_1d, opening_1d


def _split_group_into_peak_regions(indices: np.ndarray, signal: np.ndarray) -> list[tuple[int, int, int]]:
    if indices.size == 0:
        return []
    g = np.asarray(indices, dtype=int)
    vals = np.asarray(signal[g], dtype=float)
    local_pos: list[int] = []
    for i in range(vals.size):
        lv = vals[i - 1] if i > 0 else -np.inf
        rv = vals[i + 1] if i < vals.size - 1 else -np.inf
        if vals[i] >= lv and vals[i] >= rv:
            local_pos.append(i)
    if not local_pos:
        peak = int(g[int(np.argmax(vals))])
        return [(int(g[0]), int(g[-1]), peak)]
    peak_pos: list[int] = []
    local_sorted = sorted(set(local_pos))
    run = [local_sorted[0]]
    for pos in local_sorted[1:]:
        if pos == run[-1] + 1:
            run.append(pos)
            continue
        best = max(run, key=lambda idx: vals[idx])
        peak_pos.append(int(best))
        run = [pos]
    best = max(run, key=lambda idx: vals[idx])
    peak_pos.append(int(best))
    peaks = [int(g[pos]) for pos in peak_pos]
    if len(peaks) == 1:
        return [(int(g[0]), int(g[-1]), peaks[0])]
    splits = [int(g[0])]
    for a, b in zip(peak_pos[:-1], peak_pos[1:]):
        if b <= a + 1:
            splits.append(int(g[b]))
            continue
        valley = int(np.argmin(vals[a : b + 1]) + a)
        splits.append(int(g[valley]))
    splits.append(int(g[-1]))
    regions: list[tuple[int, int, int]] = []
    for i, peak in enumerate(peaks):
        left = int(splits[i])
        right = int(splits[i + 1])
        if right < left:
            left, right = right, left
        regions.append((left, right, peak))
    return regions


def score_map_from_top_hat(top_hat: np.ndarray, mode: str = "max") -> np.ndarray:
    if mode == "max":
        return np.max(top_hat, axis=-1)
    if mode == "sum":
        return np.sum(top_hat, axis=-1)
    if mode == "l2":
        return np.sqrt(np.sum(top_hat * top_hat, axis=-1))
    raise ValueError("mode must be one of: max, sum, l2")


def _local_maxima_indices(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float).reshape(-1)
    if x.size == 0:
        return np.asarray([], dtype=int)
    maxima: list[int] = []
    for i in range(x.size):
        lv = x[i - 1] if i > 0 else -np.inf
        rv = x[i + 1] if i < x.size - 1 else -np.inf
        if x[i] >= lv and x[i] >= rv:
            maxima.append(int(i))
    return np.asarray(maxima, dtype=int)


def _positive_local_maxima_values(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float).reshape(-1)
    maxima = _local_maxima_indices(x)
    if maxima.size == 0:
        return np.asarray([], dtype=float)
    values = np.asarray(x[maxima], dtype=float)
    return values[np.isfinite(values) & (values > 0.0)]


def _estimate_noise_from_morph_range_local(morph_range: np.ndarray) -> tuple[float, np.ndarray, str]:
    x = np.asarray(morph_range, dtype=float)
    valid = np.flatnonzero(np.isfinite(x)).astype(int)
    if valid.size < 9:
        return float("nan"), np.asarray([], dtype=int), "insufficient"
    vals = x[valid]
    q1, q3 = np.percentile(vals, [25.0, 75.0])
    iqr = float(q3 - q1)
    upper = float(q3 + 1.5 * iqr) if iqr > 1e-12 else float(np.percentile(vals, 80.0))
    lower = float(max(0.0, q1 - 1.5 * iqr))
    keep = valid[(vals >= lower) & (vals <= upper)]
    if keep.size < 9:
        p80 = float(np.percentile(vals, 80.0))
        keep = valid[vals <= p80]
    if keep.size < 9:
        return float("nan"), np.asarray([], dtype=int), "insufficient"
    noise_height = float(np.median(x[keep]))
    if not np.isfinite(noise_height) or noise_height <= 0.0:
        return float("nan"), np.asarray([], dtype=int), "insufficient"
    return noise_height, keep, "ok"


def _small_morphology_for_signal(raw_signal: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw_signal, dtype=float).reshape(1, 1, -1)
    erosion = np.asarray(erosion_1d(raw, int(window_size)).reshape(-1), dtype=float)
    dilation = np.asarray(dilation_1d(raw, int(window_size)).reshape(-1), dtype=float)
    morph_range = np.asarray(dilation - erosion, dtype=float)
    return erosion, dilation, morph_range


def threshold_score_map(
    score_map: np.ndarray,
    method: str = "quantile",
    quantile: float = 0.999,
    k_mad: float = 20.0,
    min_abs: float | None = None,
) -> float:
    if min_abs is not None:
        return float(min_abs)
    values = np.asarray(score_map, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("inf")
    if method == "quantile":
        return float(np.percentile(values, float(np.clip(quantile, 0.0, 1.0)) * 100.0))
    if method == "mad":
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        return float(med + float(k_mad) * mad)
    raise ValueError("method must be 'quantile' or 'mad'")


def _segments_from_thresholded_spectrum(
    *,
    y: int,
    x: int,
    x_axis: np.ndarray,
    top_hat_signal: np.ndarray,
    raw_signal: np.ndarray,
    threshold: float,
    baseline_window: int,
    max_width_pts: int,
    edge_k_mad: float,
    pad_pts: int,
) -> list[CandidateSegment]:
    th = np.asarray(top_hat_signal, dtype=float)
    idx_hi = np.where(th > float(threshold))[0]
    if idx_hi.size == 0:
        return []
    groups = np.split(idx_hi, np.where(np.diff(idx_hi) > 1)[0] + 1)
    raw = np.asarray(raw_signal, dtype=float)
    baseline = np.asarray(opening_1d(raw, se_size=int(baseline_window)), dtype=float)
    resid = raw - baseline
    res_med = float(np.median(resid))
    res_mad = float(np.median(np.abs(resid - res_med)))
    tol = float(res_med + float(edge_k_mad) * res_mad)
    pixel_segments: list[CandidateSegment] = []
    for group in groups:
        for g0, g1, peak_index in _split_group_into_peak_regions(group, th):
            res_peak = float(resid[peak_index])
            tol_eff = min(tol, 0.5 * res_peak)
            left = int(g0)
            while left > 0 and resid[left] > tol_eff:
                left -= 1
            right = int(g1)
            while right < raw.size - 1 and resid[right] > tol_eff:
                right += 1
            if left >= peak_index:
                left = peak_index - 1
            if right <= peak_index:
                right = peak_index + 1
            if left < 0 or right >= raw.size or left >= right - 1:
                continue
            left2 = max(0, left - int(pad_pts))
            right2 = min(raw.size - 1, right + int(pad_pts))
            if (right2 - left2 - 1) > int(max_width_pts):
                continue
            inner_start = int(left2 + 1)
            inner_end = int(right2)
            if inner_end <= inner_start:
                continue
            raw_window = raw[inner_start:inner_end]
            peak_index = int(inner_start + int(np.argmax(raw_window)))
            peak_height = float(raw[peak_index])
            seg = np.maximum(resid[left2 + 1 : right2], 0.0)
            area = float(abs(np.trapezoid(seg, x_axis[left2 + 1 : right2])))
            pixel_segments.append(
                CandidateSegment(
                    y=int(y),
                    x=int(x),
                    peak_index=int(peak_index),
                    start=int(left2),
                    end=int(right2),
                    peak_height=peak_height,
                    area=area,
                )
            )
    return pixel_segments


def extract_top_hat_candidates(
    *,
    x_axis: np.ndarray,
    top_hat: np.ndarray,
    candidate_mask: np.ndarray,
    raw_spectra: np.ndarray,
    max_width_pts: int,
    k_mad_pixel: float,
    min_peak: float,
    baseline_window: int,
    edge_k_mad: float,
    pad_pts: int,
) -> tuple[list[CandidateSegment], dict[tuple[int, int], list[CandidateSegment]]]:
    h, w, n = top_hat.shape
    spikes: list[CandidateSegment] = []
    by_pixel: dict[tuple[int, int], list[CandidateSegment]] = {}
    audit_by_pixel: dict[tuple[int, int], dict[str, Any]] = {}
    ys, xs = np.where(candidate_mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        th = np.asarray(top_hat[y, x, :], dtype=float)
        med = float(np.median(th))
        mad = float(np.median(np.abs(th - med)))
        thr_high = max(med + float(k_mad_pixel) * mad, float(min_peak))
        raw_peak_count = int(np.count_nonzero(th > thr_high))
        raw_local_peak_count = int(np.count_nonzero(th[_local_maxima_indices(th)] > thr_high))
        pixel_segments = _segments_from_thresholded_spectrum(
            y=int(y),
            x=int(x),
            x_axis=np.asarray(x_axis, dtype=float),
            top_hat_signal=th,
            raw_signal=np.asarray(raw_spectra[y, x, :], dtype=float),
            threshold=float(thr_high),
            baseline_window=int(baseline_window),
            max_width_pts=int(max_width_pts),
            edge_k_mad=float(edge_k_mad),
            pad_pts=int(pad_pts),
        )
        audit_by_pixel[(int(y), int(x))] = {
            "raw_candidate_score_peaks": int(raw_local_peak_count),
            "raw_candidate_score_points": int(raw_peak_count),
            "post_filter_candidates": int(len(pixel_segments)),
            "candidate_threshold_used": float(thr_high),
            "max_candidate_score": float(np.nanmax(th)) if th.size else float("nan"),
        }
        if not pixel_segments:
            continue
        if pixel_segments:
            spikes.extend(pixel_segments)
            by_pixel[(int(y), int(x))] = pixel_segments
    return spikes, by_pixel, audit_by_pixel


def extract_top_hat_candidates_local_1d(
    *,
    x_axis: np.ndarray,
    top_hat: np.ndarray,
    processed_mask: np.ndarray,
    raw_spectra: np.ndarray,
    max_width_pts: int,
    min_peak: float,
    baseline_window: int,
    edge_k_mad: float,
    pad_pts: int,
    threshold_method: str,
    threshold_quantile: float,
    threshold_k_mad: float,
    threshold_min_abs: float | None,
    noise_height_factor: float,
    noise_window: int,
) -> tuple[list[CandidateSegment], dict[tuple[int, int], list[CandidateSegment]], dict[tuple[int, int], dict[str, Any]]]:
    h, w, _n = top_hat.shape
    spikes: list[CandidateSegment] = []
    by_pixel: dict[tuple[int, int], list[CandidateSegment]] = {}
    audit_by_pixel: dict[tuple[int, int], dict[str, Any]] = {}
    ys, xs = np.where(processed_mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        th = np.asarray(top_hat[y, x, :], dtype=float)
        raw = np.asarray(raw_spectra[y, x, :], dtype=float)
        threshold_values = _positive_local_maxima_values(th)
        if threshold_values.size == 0:
            threshold_values = np.asarray(th[np.isfinite(th) & (th > 0.0)], dtype=float)
        score_threshold = threshold_score_map(
            threshold_values,
            method=str(threshold_method),
            quantile=float(threshold_quantile),
            k_mad=float(threshold_k_mad),
            min_abs=threshold_min_abs,
        )
        mad_threshold = threshold_score_map(
            threshold_values,
            method="mad",
            quantile=float(threshold_quantile),
            k_mad=float(threshold_k_mad),
            min_abs=None,
        )
        _erosion, _dilation, morph_range = _small_morphology_for_signal(raw, int(noise_window))
        noise_value, keep_idx, noise_status = _estimate_noise_from_morph_range_local(morph_range)
        noise_threshold = float(noise_height_factor) * float(noise_value) if np.isfinite(noise_value) and noise_value > 0.0 else float("nan")
        candidate_threshold = max(
            float(min_peak),
            float(noise_threshold) if np.isfinite(noise_threshold) else float(min_peak),
        )
        maxima = _local_maxima_indices(th)
        raw_local_peak_count = int(np.count_nonzero(th[maxima] > candidate_threshold)) if maxima.size else 0
        pixel_segments = _segments_from_thresholded_spectrum(
            y=int(y),
            x=int(x),
            x_axis=np.asarray(x_axis, dtype=float),
            top_hat_signal=th,
            raw_signal=raw,
            threshold=float(candidate_threshold),
            baseline_window=int(baseline_window),
            max_width_pts=int(max_width_pts),
            edge_k_mad=float(edge_k_mad),
            pad_pts=int(pad_pts),
        )
        audit_by_pixel[(int(y), int(x))] = {
            "raw_candidate_score_peaks": int(raw_local_peak_count),
            "raw_candidate_score_points": int(np.count_nonzero(th > candidate_threshold)),
            "post_filter_candidates": int(len(pixel_segments)),
            "candidate_threshold_used": float(candidate_threshold),
            "effective_threshold": float(candidate_threshold),
            "min_peak_used": float(min_peak),
            "noise_height_factor_used": float(noise_height_factor),
            "score_threshold_used": float(score_threshold) if np.isfinite(score_threshold) else float("nan"),
            "mad_threshold_used": float(mad_threshold) if np.isfinite(mad_threshold) else float("nan"),
            "noise_threshold_used": float(noise_threshold) if np.isfinite(noise_threshold) else float("nan"),
            "mad_threshold_ignored_in_local_1d": 1,
            "max_candidate_score": float(np.nanmax(th)) if th.size else float("nan"),
            "per_spectrum_noise_value": float(noise_value) if np.isfinite(noise_value) else float("nan"),
            "per_spectrum_noise_source": "morph_range",
            "noise_reference_status": str(noise_status),
            "noise_reference_n_points": int(np.asarray(keep_idx).size),
        }
        if not pixel_segments:
            continue
        spikes.extend(pixel_segments)
        by_pixel[(int(y), int(x))] = pixel_segments
    return spikes, by_pixel, audit_by_pixel


def expand_interval_to_signal_foot(
    sig: np.ndarray,
    left: int,
    right: int,
    peak: int,
    *,
    enabled: bool,
    k_mad: float,
    min_run: int,
    method: Literal["mad_run", "erosion_touch"],
    erosion_window: int,
) -> tuple[int, int]:
    n = int(sig.size)
    a = int(np.clip(left, 0, n - 1))
    b = int(np.clip(right, 0, n - 1))
    p = int(np.clip(peak, 0, n - 1))
    if not enabled or not (a < p < b):
        return a, b
    if method == "erosion_touch":
        ero = np.asarray(erosion_1d(sig.astype(float), se_size=max(1, int(erosion_window))), dtype=float)
        diff = np.abs(sig - ero)
        tol = float(max(1e-12, np.median(diff) + 1.5 * np.median(np.abs(diff - np.median(diff)))))
        left_hits = np.where(diff[: p + 1] <= tol)[0]
        right_hits = np.where(diff[p:] <= tol)[0]
        if left_hits.size and right_hits.size:
            a2 = int(left_hits.max())
            b2 = int(p + right_hits.min())
            if a2 < p < b2:
                return a2, b2
        return a, b
    bg = np.concatenate([sig[:a], sig[b + 1 :]])
    if bg.size < 8:
        bg = sig
    bg_med = float(np.median(bg))
    bg_mad = max(float(np.median(np.abs(bg - bg_med))), 1e-12)
    thr = float(bg_med + float(k_mad) * bg_mad)
    run = max(1, int(min_run))
    below = (sig <= thr).astype(np.int8)
    if run == 1:
        run_start = np.where(below > 0)[0]
    else:
        conv = np.convolve(below, np.ones(run, dtype=np.int16), mode="valid")
        run_start = np.where(conv >= run)[0]
    if run_start.size == 0:
        return a, b
    run_end = run_start + run - 1
    left_end = run_end[run_end <= a]
    right_start = run_start[run_start >= b]
    a2 = int(left_end.max()) if left_end.size else a
    b2 = int(right_start.min()) if right_start.size else b
    return (a2, b2) if a2 < p < b2 else (a, b)


def enforce_shared_boundaries_by_minima(
    peaks: list[int],
    lefts: list[int],
    rights: list[int],
    signal: np.ndarray,
) -> tuple[list[int], list[int]]:
    if len(peaks) <= 1:
        return lefts, rights
    p = [int(v) for v in peaks]
    l = [int(v) for v in lefts]
    r = [int(v) for v in rights]
    sig = np.asarray(signal, dtype=float)
    order = np.argsort(np.asarray(p, dtype=int))
    for oi in range(len(order) - 1):
        i = int(order[oi])
        j = int(order[oi + 1])
        if r[i] < l[j]:
            continue
        pi = int(p[i])
        pj = int(p[j])
        if pj <= pi + 1:
            minimum = int((pi + pj) // 2)
        else:
            minimum = int(pi + int(np.argmin(sig[pi : pj + 1])))
            minimum = max(pi + 1, min(pj - 1, minimum))
        r[i] = min(r[j], minimum)
        l[j] = max(l[j], minimum)
    return l, r


def merge_spike_segments_by_signal_foot(
    segs: list[CandidateSegment],
    signal: np.ndarray,
    *,
    k_mad: float,
    min_run: int,
    max_width_pts: Optional[int],
    merge_adjacent: bool = True,
) -> list[CandidateSegment]:
    if not segs:
        return []
    sorted_segs = sorted(segs, key=lambda seg: (seg.start, seg.end, seg.peak_index))
    out: list[CandidateSegment] = []
    x = np.asarray(signal, dtype=float)
    for seg in sorted_segs:
        if not out:
            out.append(seg)
            continue
        last = out[-1]
        allow_gap = 1 if merge_adjacent else 0
        overlap = max(int(last.start), int(seg.start)) <= (min(int(last.end), int(seg.end)) + allow_gap)
        if not overlap:
            out.append(seg)
            continue
        new_start = min(int(last.start), int(seg.start))
        new_end = max(int(last.end), int(seg.end))
        if max_width_pts is not None and (new_end - new_start - 1) > int(max_width_pts):
            out.append(seg)
            continue
        lp = min(int(last.peak_index), int(seg.peak_index))
        rp = max(int(last.peak_index), int(seg.peak_index))
        bg = np.concatenate([x[:new_start], x[new_end + 1 :]])
        if bg.size < 8:
            bg = x
        bg_med = float(np.median(bg))
        bg_mad = max(float(np.median(np.abs(bg - bg_med))), 1e-12)
        thr = float(bg_med + float(k_mad) * bg_mad)
        inner = x[lp + 1 : rp]
        below = (inner <= thr).astype(np.int8)
        low_run_exists = False
        if inner.size:
            run = max(1, int(min_run))
            if run == 1:
                low_run_exists = bool(np.any(below > 0))
            else:
                conv = np.convolve(below, np.ones(run, dtype=np.int16), mode="valid")
                low_run_exists = bool(np.any(conv >= run))
        if low_run_exists:
            out.append(seg)
            continue
        best = seg if float(seg.peak_height) >= float(last.peak_height) else last
        out[-1] = CandidateSegment(
            y=int(best.y),
            x=int(best.x),
            peak_index=int(best.peak_index),
            start=int(new_start),
            end=int(new_end),
            peak_height=float(best.peak_height),
            area=float(last.area) + float(seg.area),
        )
    return out


def prepare_primary_candidates(
    *,
    y: int,
    x: int,
    segs: list[CandidateSegment],
    feature_signal: Optional[np.ndarray],
    boundary_signal: Optional[np.ndarray],
    merge_signal: Optional[np.ndarray],
    feature_expand_to_gradient_foot: bool,
    feature_foot_k_mad: float,
    feature_foot_min_run: int,
    feature_window_method: Literal["mad_run", "erosion_touch"],
    feature_erosion_window: int,
    merge_duplicate_segments: bool,
    merge_max_width_pts: Optional[int],
) -> list[CandidateSegment]:
    if not segs:
        return []
    src = np.asarray(feature_signal, dtype=float) if feature_signal is not None else None
    boundary = np.asarray(boundary_signal, dtype=float) if boundary_signal is not None else None
    merge_sig = np.asarray(merge_signal, dtype=float) if merge_signal is not None else None
    peaks: list[int] = []
    lefts: list[int] = []
    rights: list[int] = []
    for seg in segs:
        left = int(seg.start)
        right = int(seg.end)
        peak = int(seg.peak_index)
        if src is not None:
            left, right = expand_interval_to_signal_foot(
                src,
                left,
                right,
                peak,
                enabled=feature_expand_to_gradient_foot,
                k_mad=float(feature_foot_k_mad),
                min_run=int(feature_foot_min_run),
                method=feature_window_method,
                erosion_window=int(feature_erosion_window),
            )
        peaks.append(peak)
        lefts.append(left)
        rights.append(right)
    if boundary is not None:
        lefts, rights = enforce_shared_boundaries_by_minima(peaks, lefts, rights, boundary)
    prepared = [
        CandidateSegment(
            y=int(y),
            x=int(x),
            peak_index=int(peaks[i]),
            start=int(lefts[i]),
            end=int(rights[i]),
            peak_height=float(segs[i].peak_height),
            area=float(segs[i].area),
        )
        for i in range(len(segs))
    ]
    if merge_duplicate_segments and merge_sig is not None:
        prepared = merge_spike_segments_by_signal_foot(
            prepared,
            merge_sig,
            k_mad=float(feature_foot_k_mad),
            min_run=int(feature_foot_min_run),
            max_width_pts=merge_max_width_pts,
        )
    return prepared
