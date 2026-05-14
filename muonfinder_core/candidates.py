from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .data_model import CandidateSegment
from .morphology import erosion_1d, opening_1d


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
    ys, xs = np.where(candidate_mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        th = np.asarray(top_hat[y, x, :], dtype=float)
        med = float(np.median(th))
        mad = float(np.median(np.abs(th - med)))
        thr_high = max(med + float(k_mad_pixel) * mad, float(min_peak))
        idx_hi = np.where(th > thr_high)[0]
        if idx_hi.size == 0:
            continue
        groups = np.split(idx_hi, np.where(np.diff(idx_hi) > 1)[0] + 1)
        raw = np.asarray(raw_spectra[y, x, :], dtype=float)
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
        if pixel_segments:
            spikes.extend(pixel_segments)
            by_pixel[(int(y), int(x))] = pixel_segments
    return spikes, by_pixel


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
