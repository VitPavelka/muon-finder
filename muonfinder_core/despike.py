from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .cap_metrics import join_extra_feature_rows, load_extra_feature_rows
from .cache import load_viewer_cache
from .data_model import CorrectionResult, DespikeChord, DespikeStage
from .morphology import dilation_1d, erosion_1d


@dataclass(frozen=True)
class DespikeArtifacts:
    final_corrected: np.ndarray
    chords: list[DespikeChord]
    debug_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class ContactCell:
    left: int
    right: int
    morph_window: int
    contains_peak: bool
    nearest_dilation_contact: int | None
    contains_dilation_contact: bool
    left_anchor: int
    right_anchor: int
    chord_width_pts: int
    cell_width_pts: int
    height_above_chord: float
    height_above_chord_noise_z: float
    chord_overshoot_max: float
    original_peak_value: float
    corrected_peak_value: float
    correction_height: float


def build_placeholder_correction(raw_spectra: np.ndarray) -> CorrectionResult:
    return CorrectionResult(
        corrected_spectra=np.asarray(raw_spectra, dtype=np.float32).copy(),
        stages=[DespikeStage(stage_id="stage:0", name="preview", description="Raw passthrough placeholder")],
        chords=[],
    )


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _candidate_key(row: dict[str, Any]) -> str:
    candidate_id = str(row.get("candidate_id", "")).strip()
    if candidate_id:
        return candidate_id
    return f"{int(row.get('source_y', row.get('y', -1)))}:{int(row.get('source_x', row.get('x', -1)))}:{int(row.get('peak_index', -1))}"


def _to_float_array(signal: np.ndarray) -> np.ndarray:
    return np.asarray(signal, dtype=float).reshape(-1)


def _compute_contacts(signal: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = _to_float_array(signal)
    arr = x.reshape(1, 1, -1)
    erosion = np.asarray(erosion_1d(arr, int(window)).reshape(-1), dtype=float)
    dilation = np.asarray(dilation_1d(arr, int(window)).reshape(-1), dtype=float)
    erosion_mask = np.isclose(x, erosion) if not np.any(x == erosion) else (x == erosion)
    dilation_mask = np.isclose(x, dilation) if not np.any(x == dilation) else (x == dilation)
    return erosion, dilation, np.flatnonzero(erosion_mask).astype(int), np.flatnonzero(dilation_mask).astype(int)


def _local_noise_fallback(signal: np.ndarray, peak_idx: int, pad: int = 6) -> float:
    x = _to_float_array(signal)
    left = max(0, int(peak_idx) - int(pad))
    right = min(len(x) - 1, int(peak_idx) + int(pad))
    seg = x[left : right + 1]
    med = float(np.median(seg)) if seg.size else 0.0
    mad = float(np.median(np.abs(seg - med))) if seg.size else float("nan")
    return float(max(mad, 1e-12)) if np.isfinite(mad) else float("nan")


def _select_noise(row: dict[str, Any], signal: np.ndarray, peak_idx: int) -> tuple[float, str, bool]:
    for key, source in (
        ("candidate_noise_estimate_used", "candidate_noise_estimate_used"),
        ("noise_height_morph_range", "noise_height_morph_range"),
    ):
        value = _safe_float(row.get(key))
        if np.isfinite(value) and value > 0.0:
            return float(value), str(source), False
    fallback = _local_noise_fallback(signal, peak_idx)
    return float(fallback), "local_mad_fallback", True


def _chord_values(left_idx: int, right_idx: int, left_val: float, right_val: float) -> np.ndarray:
    if right_idx <= left_idx:
        return np.asarray([float(left_val)], dtype=float)
    xs = np.arange(int(left_idx), int(right_idx) + 1, dtype=float)
    return np.interp(xs, [float(left_idx), float(right_idx)], [float(left_val), float(right_val)]).astype(float)


def _cell_height_metrics(signal: np.ndarray, peak_idx: int, left_anchor: int, right_anchor: int, noise_value: float) -> tuple[float, float, float, float]:
    x = _to_float_array(signal)
    chord = _chord_values(left_anchor, right_anchor, float(x[left_anchor]), float(x[right_anchor]))
    peak_pos = int(np.clip(int(peak_idx) - int(left_anchor), 0, max(0, chord.size - 1)))
    original_peak = float(x[int(peak_idx)])
    corrected_peak = float(chord[peak_pos])
    height = float(original_peak - corrected_peak)
    noise_z = float(height / max(float(noise_value), 1e-12)) if np.isfinite(noise_value) and noise_value > 0.0 else float("nan")
    overshoot = np.asarray(chord[1:-1] - x[int(left_anchor) + 1 : int(right_anchor)], dtype=float)
    overshoot_max = float(np.max(overshoot)) if overshoot.size else 0.0
    return height, noise_z, corrected_peak, overshoot_max


def _local_maximum(signal: np.ndarray, idx: int, left: int, right: int) -> bool:
    x = _to_float_array(signal)
    left = max(0, int(left))
    right = min(len(x) - 1, int(right))
    if not (left <= int(idx) <= right):
        return False
    local = x[left : right + 1]
    return bool(local.size and int(idx) == left + int(np.argmax(local)))


def _find_parent_contact_cell(
    signal: np.ndarray,
    peak_idx: int,
    *,
    morph_windows: list[int],
    max_cell_width_pts: int,
    max_half_width_pts: int,
    min_height_above_chord_noise_z: float,
    anchor_overshoot_noise_factor: float,
    allow_spectrum_edge_anchors: bool,
    noise_value: float,
    local_bounds: tuple[int, int] | None = None,
) -> tuple[ContactCell | None, dict[str, Any]]:
    x = _to_float_array(signal)
    n = len(x)
    search_left = 0 if local_bounds is None else max(0, int(local_bounds[0]))
    search_right = n - 1 if local_bounds is None else min(n - 1, int(local_bounds[1]))
    debug: dict[str, Any] = {"skipped_reason": "skipped_no_parent_contact_cell"}
    for window in morph_windows:
        _erosion, _dilation, erosion_contacts, dilation_contacts = _compute_contacts(x, int(window))
        contacts = [idx for idx in erosion_contacts.tolist() if search_left <= int(idx) <= search_right]
        if len(contacts) < 2:
            continue
        for left_anchor, right_anchor in zip(contacts[:-1], contacts[1:]):
            left_anchor = int(left_anchor)
            right_anchor = int(right_anchor)
            if right_anchor - left_anchor < 2:
                continue
            contains_peak = bool(left_anchor < int(peak_idx) < right_anchor)
            if not contains_peak:
                continue
            cell_width = int(right_anchor - left_anchor)
            if cell_width > int(max_cell_width_pts):
                debug = {
                    "morph_window": int(window),
                    "cell_left": left_anchor,
                    "cell_right": right_anchor,
                    "cell_width_pts": cell_width,
                    "skipped_reason": "skipped_cell_too_wide",
                }
                continue
            half_width = int(min(int(peak_idx) - left_anchor, right_anchor - int(peak_idx)))
            if half_width > int(max_half_width_pts):
                debug = {
                    "morph_window": int(window),
                    "cell_left": left_anchor,
                    "cell_right": right_anchor,
                    "cell_width_pts": cell_width,
                    "skipped_reason": "skipped_cell_too_wide",
                }
                continue
            if not allow_spectrum_edge_anchors and (left_anchor <= 0 or right_anchor >= n - 1):
                debug = {
                    "morph_window": int(window),
                    "cell_left": left_anchor,
                    "cell_right": right_anchor,
                    "cell_width_pts": cell_width,
                    "skipped_reason": "skipped_parent_uses_spectrum_edge_anchor",
                }
                continue
            dilation_inside = [int(idx) for idx in dilation_contacts.tolist() if left_anchor <= int(idx) <= right_anchor]
            nearest_dilation = min(dilation_inside, key=lambda idx: abs(idx - int(peak_idx))) if dilation_inside else None
            has_dilation = bool(dilation_inside)
            is_local_max = _local_maximum(x, int(peak_idx), left_anchor, right_anchor)
            if not has_dilation and not is_local_max:
                debug = {
                    "morph_window": int(window),
                    "cell_left": left_anchor,
                    "cell_right": right_anchor,
                    "cell_width_pts": cell_width,
                    "skipped_reason": "skipped_no_dilation_contact_or_local_max",
                }
                continue
            height, noise_z, corrected_peak, overshoot_max = _cell_height_metrics(x, int(peak_idx), left_anchor, right_anchor, noise_value)
            if not np.isfinite(noise_z):
                debug = {
                    "morph_window": int(window),
                    "cell_left": left_anchor,
                    "cell_right": right_anchor,
                    "cell_width_pts": cell_width,
                    "skipped_reason": "skipped_missing_noise",
                }
                continue
            if noise_z < float(min_height_above_chord_noise_z):
                debug = {
                    "morph_window": int(window),
                    "cell_left": left_anchor,
                    "cell_right": right_anchor,
                    "cell_width_pts": cell_width,
                    "height_above_chord": float(height),
                    "height_above_chord_noise_z": float(noise_z),
                    "skipped_reason": "skipped_height_below_noise_threshold",
                }
                continue
            if np.isfinite(overshoot_max) and overshoot_max > float(anchor_overshoot_noise_factor) * max(float(noise_value), 1e-12):
                debug = {
                    "morph_window": int(window),
                    "cell_left": left_anchor,
                    "cell_right": right_anchor,
                    "cell_width_pts": cell_width,
                    "chord_overshoot_max": float(overshoot_max),
                    "skipped_reason": "skipped_chord_overshoots_signal",
                }
                continue
            return (
                ContactCell(
                    left=left_anchor,
                    right=right_anchor,
                    morph_window=int(window),
                    contains_peak=True,
                    nearest_dilation_contact=(None if nearest_dilation is None else int(nearest_dilation)),
                    contains_dilation_contact=bool(has_dilation),
                    left_anchor=left_anchor,
                    right_anchor=right_anchor,
                    chord_width_pts=int(max(0, cell_width - 1)),
                    cell_width_pts=cell_width,
                    height_above_chord=float(height),
                    height_above_chord_noise_z=float(noise_z),
                    chord_overshoot_max=float(overshoot_max),
                    original_peak_value=float(x[int(peak_idx)]),
                    corrected_peak_value=float(corrected_peak),
                    correction_height=float(height),
                ),
                {
                    "morph_window": int(window),
                    "cell_left": left_anchor,
                    "cell_right": right_anchor,
                    "cell_width_pts": cell_width,
                },
            )
    return None, debug


def _find_recheck_cell(
    signal: np.ndarray,
    *,
    local_left: int,
    local_right: int,
    morph_windows: list[int],
    max_cell_width_pts: int,
    max_half_width_pts: int,
    min_height_above_chord_noise_z: float,
    anchor_overshoot_noise_factor: float,
    allow_spectrum_edge_anchors: bool,
    noise_value: float,
) -> tuple[ContactCell | None, dict[str, Any]]:
    x = _to_float_array(signal)
    best_cell: ContactCell | None = None
    best_debug: dict[str, Any] = {"skipped_reason": "skipped_no_parent_contact_cell"}
    for window in morph_windows:
        _erosion, _dilation, erosion_contacts, dilation_contacts = _compute_contacts(x, int(window))
        contacts = [idx for idx in erosion_contacts.tolist() if int(local_left) <= int(idx) <= int(local_right)]
        for left_anchor, right_anchor in zip(contacts[:-1], contacts[1:]):
            left_anchor = int(left_anchor)
            right_anchor = int(right_anchor)
            if right_anchor - left_anchor < 2:
                continue
            local_peak = left_anchor + int(np.argmax(x[left_anchor : right_anchor + 1]))
            cell, dbg = _find_parent_contact_cell(
                x,
                local_peak,
                morph_windows=[int(window)],
                max_cell_width_pts=int(max_cell_width_pts),
                max_half_width_pts=int(max_half_width_pts),
                min_height_above_chord_noise_z=float(min_height_above_chord_noise_z),
                anchor_overshoot_noise_factor=float(anchor_overshoot_noise_factor),
                allow_spectrum_edge_anchors=bool(allow_spectrum_edge_anchors),
                noise_value=float(noise_value),
                local_bounds=(int(local_left), int(local_right)),
            )
            if cell is None:
                best_debug = dbg
                continue
            if best_cell is None or float(cell.height_above_chord_noise_z) > float(best_cell.height_above_chord_noise_z):
                best_cell = cell
                best_debug = dbg
    return best_cell, best_debug


def compute_despike_from_cache_and_ss6(
    *,
    cache_path: Path,
    ss6_path: Path,
    corrected_path: Path | None,
    debug_path: Path | None,
    summary_path: Path | None,
    max_iterations: int,
    morph_windows: list[int],
    max_cell_width_pts: int,
    max_half_width_pts: int,
    min_height_above_chord_noise_z: float,
    anchor_overshoot_noise_factor: float,
    allow_spectrum_edge_anchors: bool,
    recheck_enabled: bool,
    recheck_context_pad_pts: int,
    skip_overlapping_corrections: bool,
    progress_iter: Callable[[list[tuple[tuple[int, int], list[dict[str, Any]]]]], Any] | None = None,
    timings_out: dict[str, float] | None = None,
) -> DespikeArtifacts:
    t0 = time.perf_counter()
    cache = load_viewer_cache(cache_path)
    if timings_out is not None:
        timings_out["load cache"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rows = [dict(row) for row in cache.get("candidate_records", [])]
    ss6_rows, _ = load_extra_feature_rows(ss6_path)
    if timings_out is not None:
        timings_out["load ss6 decisions"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    join_info = join_extra_feature_rows(rows, ss6_rows)
    accepted_rows = [row for row in rows if int(_safe_float(row.get("ss6_accept", 0))) == 1]
    accepted_by_pixel: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in accepted_rows:
        accepted_by_pixel.setdefault((int(row["y"]), int(row["x"])), []).append(row)
    for pixel_rows in accepted_by_pixel.values():
        pixel_rows.sort(key=lambda row: int(row.get("peak_index", -1)))
    if timings_out is not None:
        timings_out["group accepted candidates"] = time.perf_counter() - t0

    raw_spectra = np.asarray(cache["spectra"], dtype=np.float32)
    corrected = np.asarray(raw_spectra, dtype=np.float32).copy()
    chords: list[DespikeChord] = []
    debug_rows: list[dict[str, Any]] = []
    per_spectrum_correction_counts: dict[str, int] = {}

    summary_counts = {
        "corrected_spikes": 0,
        "skipped_candidates": 0,
        "repeated_corrections": 0,
        "max_iteration_reached_count": 0,
        "skipped_no_parent_contact_cell": 0,
        "skipped_cell_too_wide": 0,
        "skipped_chord_overshoots_signal": 0,
        "skipped_missing_noise": 0,
        "skipped_overlaps_previous_correction": 0,
    }

    spectrum_items = list(accepted_by_pixel.items())
    iterator = progress_iter(spectrum_items) if progress_iter is not None else spectrum_items
    t0 = time.perf_counter()
    for (y, x), pixel_rows in iterator:
        corrected_intervals: list[tuple[int, int]] = []
        spectrum_corrections = 0
        for row in pixel_rows:
            peak_idx = int(row.get("peak_index", -1))
            signal = np.asarray(corrected[y, x, :], dtype=float)
            noise_value, noise_source, noise_fallback_used = _select_noise(row, signal, peak_idx)
            if not np.isfinite(noise_value) or noise_value <= 0.0:
                summary_counts["skipped_candidates"] += 1
                summary_counts["skipped_missing_noise"] += 1
                debug_rows.append(
                    {
                        "source_y": int(row.get("source_y", y)),
                        "source_x": int(row.get("source_x", x)),
                        "compact_y": int(y),
                        "compact_x": int(x),
                        "peak_index": int(peak_idx),
                        "candidate_id": str(row.get("candidate_id", "")),
                        "ss6_branch": str(row.get("ss6_branch", "")),
                        "iteration": 0,
                        "morph_window": "",
                        "status": "skipped",
                        "skipped_reason": "skipped_missing_noise",
                        "cell_left": "",
                        "cell_right": "",
                        "cell_width_pts": "",
                        "left_anchor": "",
                        "right_anchor": "",
                        "chord_width_pts": "",
                        "contains_peak": "",
                        "contains_dilation_contact": "",
                        "nearest_dilation_contact": "",
                        "height_above_chord": "",
                        "height_above_chord_noise_z": "",
                        "chord_overshoot_max": "",
                        "noise_value": "",
                        "noise_source": str(noise_source),
                        "noise_fallback_used": int(noise_fallback_used),
                        "original_peak_value": "",
                        "corrected_peak_value": "",
                        "correction_height": "",
                    }
                )
                continue

            if skip_overlapping_corrections and any(not (peak_idx < left or peak_idx > right) for left, right in corrected_intervals):
                summary_counts["skipped_candidates"] += 1
                summary_counts["skipped_overlaps_previous_correction"] += 1
                debug_rows.append(
                    {
                        "source_y": int(row.get("source_y", y)),
                        "source_x": int(row.get("source_x", x)),
                        "compact_y": int(y),
                        "compact_x": int(x),
                        "peak_index": int(peak_idx),
                        "candidate_id": str(row.get("candidate_id", "")),
                        "ss6_branch": str(row.get("ss6_branch", "")),
                        "iteration": 0,
                        "morph_window": "",
                        "status": "skipped",
                        "skipped_reason": "skipped_overlaps_previous_correction",
                        "cell_left": "",
                        "cell_right": "",
                        "cell_width_pts": "",
                        "left_anchor": "",
                        "right_anchor": "",
                        "chord_width_pts": "",
                        "contains_peak": "",
                        "contains_dilation_contact": "",
                        "nearest_dilation_contact": "",
                        "height_above_chord": "",
                        "height_above_chord_noise_z": "",
                        "chord_overshoot_max": "",
                        "noise_value": float(noise_value),
                        "noise_source": str(noise_source),
                        "noise_fallback_used": int(noise_fallback_used),
                        "original_peak_value": "",
                        "corrected_peak_value": "",
                        "correction_height": "",
                    }
                )
                continue

            local_bounds: tuple[int, int] | None = None
            candidate_corrected = False
            for iteration in range(int(max_iterations)):
                signal = np.asarray(corrected[y, x, :], dtype=float)
                if iteration == 0:
                    cell, dbg = _find_parent_contact_cell(
                        signal,
                        peak_idx,
                        morph_windows=morph_windows,
                        max_cell_width_pts=int(max_cell_width_pts),
                        max_half_width_pts=int(max_half_width_pts),
                        min_height_above_chord_noise_z=float(min_height_above_chord_noise_z),
                        anchor_overshoot_noise_factor=float(anchor_overshoot_noise_factor),
                        allow_spectrum_edge_anchors=bool(allow_spectrum_edge_anchors),
                        noise_value=float(noise_value),
                    )
                else:
                    if local_bounds is None:
                        break
                    cell, dbg = _find_recheck_cell(
                        signal,
                        local_left=local_bounds[0],
                        local_right=local_bounds[1],
                        morph_windows=morph_windows,
                        max_cell_width_pts=int(max_cell_width_pts),
                        max_half_width_pts=int(max_half_width_pts),
                        min_height_above_chord_noise_z=float(min_height_above_chord_noise_z),
                        anchor_overshoot_noise_factor=float(anchor_overshoot_noise_factor),
                        allow_spectrum_edge_anchors=bool(allow_spectrum_edge_anchors),
                        noise_value=float(noise_value),
                    )
                if cell is None:
                    if not candidate_corrected:
                        summary_counts["skipped_candidates"] += 1
                        reason = str(dbg.get("skipped_reason", "skipped_no_parent_contact_cell"))
                        summary_counts[reason] = summary_counts.get(reason, 0) + 1
                        debug_rows.append(
                            {
                                "source_y": int(row.get("source_y", y)),
                                "source_x": int(row.get("source_x", x)),
                                "compact_y": int(y),
                                "compact_x": int(x),
                                "peak_index": int(peak_idx),
                                "candidate_id": str(row.get("candidate_id", "")),
                                "ss6_branch": str(row.get("ss6_branch", "")),
                                "iteration": int(iteration),
                                "morph_window": dbg.get("morph_window", ""),
                                "status": "skipped",
                                "skipped_reason": reason,
                                "cell_left": dbg.get("cell_left", ""),
                                "cell_right": dbg.get("cell_right", ""),
                                "cell_width_pts": dbg.get("cell_width_pts", ""),
                                "left_anchor": "",
                                "right_anchor": "",
                                "chord_width_pts": "",
                                "contains_peak": "",
                                "contains_dilation_contact": "",
                                "nearest_dilation_contact": "",
                                "height_above_chord": dbg.get("height_above_chord", ""),
                                "height_above_chord_noise_z": dbg.get("height_above_chord_noise_z", ""),
                                "chord_overshoot_max": dbg.get("chord_overshoot_max", ""),
                                "noise_value": float(noise_value),
                                "noise_source": str(noise_source),
                                "noise_fallback_used": int(noise_fallback_used),
                                "original_peak_value": "",
                                "corrected_peak_value": "",
                                "correction_height": "",
                            }
                        )
                    break

                left_anchor = int(cell.left_anchor)
                right_anchor = int(cell.right_anchor)
                if right_anchor - left_anchor < 2:
                    break
                chord_vals = _chord_values(left_anchor, right_anchor, float(signal[left_anchor]), float(signal[right_anchor]))
                corrected[y, x, left_anchor + 1 : right_anchor] = chord_vals[1:-1].astype(np.float32)
                corrected_intervals.append((left_anchor + 1, right_anchor - 1))
                candidate_corrected = True
                spectrum_corrections += 1
                summary_counts["corrected_spikes"] += 1
                if iteration > 0:
                    summary_counts["repeated_corrections"] += 1
                if iteration + 1 >= int(max_iterations):
                    summary_counts["max_iteration_reached_count"] += 1
                chord_id = f"despike:{y}:{x}:{peak_idx}:{iteration}:{left_anchor}:{right_anchor}"
                chords.append(
                    DespikeChord(
                        chord_id=chord_id,
                        stage_id=f"corr:{y}:{x}:{peak_idx}:{iteration}",
                        y=int(y),
                        x=int(x),
                        left=int(left_anchor),
                        right=int(right_anchor),
                        method="morph_contact_cells",
                        y_left=float(signal[left_anchor]),
                        y_right=float(signal[right_anchor]),
                    )
                )
                debug_rows.append(
                    {
                        "source_y": int(row.get("source_y", y)),
                        "source_x": int(row.get("source_x", x)),
                        "compact_y": int(y),
                        "compact_x": int(x),
                        "peak_index": int(peak_idx),
                        "candidate_id": str(row.get("candidate_id", "")),
                        "ss6_branch": str(row.get("ss6_branch", "")),
                        "iteration": int(iteration),
                        "morph_window": int(cell.morph_window),
                        "status": "corrected",
                        "skipped_reason": "",
                        "cell_left": int(cell.left),
                        "cell_right": int(cell.right),
                        "cell_width_pts": int(cell.cell_width_pts),
                        "left_anchor": int(cell.left_anchor),
                        "right_anchor": int(cell.right_anchor),
                        "chord_width_pts": int(cell.chord_width_pts),
                        "contains_peak": int(cell.contains_peak),
                        "contains_dilation_contact": int(cell.contains_dilation_contact),
                        "nearest_dilation_contact": "" if cell.nearest_dilation_contact is None else int(cell.nearest_dilation_contact),
                        "height_above_chord": float(cell.height_above_chord),
                        "height_above_chord_noise_z": float(cell.height_above_chord_noise_z),
                        "chord_overshoot_max": float(cell.chord_overshoot_max),
                        "noise_value": float(noise_value),
                        "noise_source": str(noise_source),
                        "noise_fallback_used": int(noise_fallback_used),
                        "original_peak_value": float(cell.original_peak_value),
                        "corrected_peak_value": float(cell.corrected_peak_value),
                        "correction_height": float(cell.correction_height),
                    }
                )
                if not bool(recheck_enabled):
                    break
                local_bounds = (
                    max(0, int(left_anchor) - int(recheck_context_pad_pts)),
                    min(len(signal) - 1, int(right_anchor) + int(recheck_context_pad_pts)),
                )
            per_spectrum_correction_counts[f"{int(y)}:{int(x)}"] = int(spectrum_corrections)
    if timings_out is not None:
        timings_out["despike correction"] = time.perf_counter() - t0

    corrected_attempt_rows = [row for row in debug_rows if str(row.get("status", "")) == "corrected"]
    max_correction_width = max((int(row.get("chord_width_pts", 0) or 0) for row in corrected_attempt_rows), default=0)
    max_iteration_used = max((int(row.get("iteration", 0) or 0) for row in corrected_attempt_rows), default=-1) + (1 if corrected_attempt_rows else 0)

    summary = {
        "accepted_ss6_spikes": int(len(accepted_rows)),
        "spectra_with_accepted_spikes": int(len(accepted_by_pixel)),
        "corrected_spikes": int(summary_counts["corrected_spikes"]),
        "skipped_candidates": int(summary_counts["skipped_candidates"]),
        "repeated_corrections": int(summary_counts["repeated_corrections"]),
        "max_iteration_reached_count": int(summary_counts["max_iteration_reached_count"]),
        "skipped_no_parent_contact_cell": int(summary_counts.get("skipped_no_parent_contact_cell", 0)),
        "skipped_cell_too_wide": int(summary_counts.get("skipped_cell_too_wide", 0)),
        "skipped_chord_overshoots_signal": int(summary_counts.get("skipped_chord_overshoots_signal", 0)),
        "skipped_missing_noise": int(summary_counts.get("skipped_missing_noise", 0)),
        "skipped_overlaps_previous_correction": int(summary_counts.get("skipped_overlaps_previous_correction", 0)),
        "max_correction_width_pts": int(max_correction_width),
        "max_iterations_used": int(max_iteration_used),
        "join_info": join_info,
        "output_paths": {
            "corrected_path": str(corrected_path) if corrected_path is not None else "",
            "debug_path": str(debug_path) if debug_path is not None else "",
            "summary_path": str(summary_path) if summary_path is not None else "",
        },
    }

    t0 = time.perf_counter()
    if corrected_path is not None:
        corrected_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "corrected_spectra": np.asarray(corrected, dtype=np.float32),
            "x_axis": np.asarray(cache["x_axis"], dtype=float),
            "coord_map_json": np.array([json.dumps(cache.get("coord_map", []), ensure_ascii=False)], dtype=object),
            "despike_chords_json": np.array([json.dumps([chord.__dict__ for chord in chords], ensure_ascii=False)], dtype=object),
            "despike_debug_rows_json": np.array([json.dumps(debug_rows, ensure_ascii=False)], dtype=object),
            "per_spectrum_correction_counts_json": np.array([json.dumps(per_spectrum_correction_counts, ensure_ascii=False)], dtype=object),
            "metadata_json": np.array([json.dumps(summary, ensure_ascii=False)], dtype=object),
        }
        np.savez_compressed(corrected_path, **payload)
    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "source_y", "source_x", "compact_y", "compact_x", "peak_index", "candidate_id", "ss6_branch",
            "iteration", "morph_window", "status", "skipped_reason",
            "cell_left", "cell_right", "cell_width_pts", "left_anchor", "right_anchor", "chord_width_pts",
            "contains_peak", "contains_dilation_contact", "nearest_dilation_contact",
            "height_above_chord", "height_above_chord_noise_z", "chord_overshoot_max",
            "noise_value", "noise_source", "noise_fallback_used",
            "original_peak_value", "corrected_peak_value", "correction_height",
        ]
        with debug_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in debug_rows:
                writer.writerow(row)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if timings_out is not None:
        timings_out["write outputs"] = time.perf_counter() - t0
    return DespikeArtifacts(
        final_corrected=np.asarray(corrected, dtype=np.float32),
        chords=chords,
        debug_rows=debug_rows,
        summary=summary,
    )


def load_despike_bundle(path: Path | str) -> dict[str, Any]:
    data = np.load(Path(path), allow_pickle=True)
    return {
        "corrected_spectra": np.asarray(data["corrected_spectra"]),
        "x_axis": np.asarray(data["x_axis"]) if "x_axis" in data.files else None,
        "coord_map": json.loads(str(np.asarray(data["coord_map_json"]).reshape(-1)[0])) if "coord_map_json" in data.files else [],
        "despike_chords": json.loads(str(np.asarray(data["despike_chords_json"]).reshape(-1)[0])) if "despike_chords_json" in data.files else [],
        "despike_debug_rows": json.loads(str(np.asarray(data["despike_debug_rows_json"]).reshape(-1)[0])) if "despike_debug_rows_json" in data.files else [],
        "per_spectrum_correction_counts": json.loads(str(np.asarray(data["per_spectrum_correction_counts_json"]).reshape(-1)[0])) if "per_spectrum_correction_counts_json" in data.files else {},
        "metadata": json.loads(str(np.asarray(data["metadata_json"]).reshape(-1)[0])) if "metadata_json" in data.files else {},
    }
