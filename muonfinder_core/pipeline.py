from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
from tqdm import tqdm

if __package__ in {None, ""}:
    import sys

    _THIS_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _THIS_DIR.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from muonfinder_core.cache import save_viewer_cache
    from muonfinder_core.candidates import (
        CandidateSegment,
        extract_top_hat_candidates,
        prepare_primary_candidates,
        score_map_from_top_hat,
        threshold_score_map,
    )
    from muonfinder_core.config import CoreConfig, load_config
    from muonfinder_core.data_model import PipelineArtifacts, SpectrumDataset
    from muonfinder_core.decision import compute_ss4, compute_ss5
    from muonfinder_core.despike import build_placeholder_correction
    from muonfinder_core.io import load_dataset, load_target_coords_csv
    from muonfinder_core.metrics import (
        MetricComputationContext,
        compute_raw_edge_metric,
        compute_ss1_pce_features,
        finalize_edge_evidence,
    )
    from muonfinder_core.morphology import compute_morphology_windows
    from muonfinder_core.noise import apply_global_metric_ranks, evaluate_candidate_noise_prefilter, get_or_compute_small_morphology
    from muonfinder_core.utils import dumps_json
else:
    from .cache import save_viewer_cache
    from .candidates import (
        CandidateSegment,
        extract_top_hat_candidates,
        prepare_primary_candidates,
        score_map_from_top_hat,
        threshold_score_map,
    )
    from .config import CoreConfig, load_config
    from .data_model import PipelineArtifacts, SpectrumDataset
    from .decision import compute_ss4, compute_ss5
    from .despike import build_placeholder_correction
    from .io import load_dataset, load_target_coords_csv
    from .metrics import (
        MetricComputationContext,
        compute_raw_edge_metric,
        compute_ss1_pce_features,
        finalize_edge_evidence,
    )
    from .morphology import compute_morphology_windows
    from .noise import apply_global_metric_ranks, evaluate_candidate_noise_prefilter, get_or_compute_small_morphology
    from .utils import dumps_json


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _fmt_s(seconds: float) -> str:
    return f"{float(seconds):.2f}s"


def _phase_print(label: str, started_at: float, extra: str | None = None) -> None:
    msg = f"[{_ts()}] {label} done in {_fmt_s(time.perf_counter() - started_at)}"
    if extra:
        msg += f" | {extra}"
    print(msg)


def _build_compact_subset(dataset: SpectrumDataset, coords: list[tuple[int, int]]) -> tuple[SpectrumDataset, dict[tuple[int, int], tuple[int, int]]]:
    n = len(coords)
    if n == 0:
        raise ValueError("No coordinates for compact subset.")
    grid_w = int(np.ceil(np.sqrt(n)))
    grid_h = int(np.ceil(n / grid_w))
    spec_n = int(dataset.x_axis.size)
    compact = np.zeros((grid_h, grid_w, spec_n), dtype=dataset.spectra.dtype)
    xpos = np.zeros((grid_h, grid_w), dtype=float) if dataset.xpos is not None else None
    ypos = np.zeros((grid_h, grid_w), dtype=float) if dataset.ypos is not None else None
    coord_map: dict[tuple[int, int], tuple[int, int]] = {}
    for idx, (sy, sx) in enumerate(coords):
        cy = idx // grid_w
        cx = idx % grid_w
        coord_map[(cy, cx)] = (int(sy), int(sx))
        compact[cy, cx, :] = dataset.spectra[sy, sx, :]
        if xpos is not None and dataset.xpos is not None:
            xpos[cy, cx] = float(dataset.xpos[sy, sx])
        if ypos is not None and dataset.ypos is not None:
            ypos[cy, cx] = float(dataset.ypos[sy, sx])
    compact_ds = SpectrumDataset(
        path=dataset.path,
        x_axis=np.asarray(dataset.x_axis, dtype=float),
        spectra=np.asarray(compact),
        xpos=xpos,
        ypos=ypos,
        meta=dict(dataset.meta, compact_subset=True, compact_shape=[grid_h, grid_w]),
    )
    return compact_ds, coord_map


def _prepare_candidate_rows(
    *,
    raw: np.ndarray,
    overlays: dict[str, dict[int, np.ndarray]],
    candidates_by_pixel: dict[tuple[int, int], list[CandidateSegment]],
    cfg: CoreConfig,
    source_coords_map: dict[tuple[int, int], tuple[int, int]],
) -> tuple[
    dict[tuple[int, int], list[dict[str, Any]]],
    dict[tuple[int, int], dict[str, Any]],
    dict[tuple[int, int], dict[str, Any]],
    dict[str, int],
]:
    rows_by_pixel: dict[tuple[int, int], list[dict[str, Any]]] = {}
    small_morphology_payloads: dict[tuple[int, int], dict[str, Any]] = {}
    summary_by_pixel: dict[tuple[int, int], dict[str, Any]] = {}
    small_morph_cache = {}
    tophat_window = int(cfg.morphology["tophat_window"])
    noise_window = int(cfg.morphology["noise_window"])
    gradient = overlays["gradient"][tophat_window]
    boundary_signal = gradient
    merge_signal = gradient
    metric_ctx = MetricComputationContext(
        feature_signal_source="gradient",
        edge_context_pad_pts=int(cfg.noise.get("edge_dense_context_pad_pts", 20)),
        edge_context_min_pad_pts=int(cfg.noise.get("edge_dense_context_min_pad_pts", 10)),
        edge_context_max_pad_pts=int(cfg.noise.get("edge_dense_context_max_pad_pts", 80)),
        edge_dense_min_snr=float(cfg.noise.get("edge_dense_min_snr", 3.0)),
        edge_robust_reference_enabled=bool(cfg.noise.get("edge_robust_reference_enabled", True)),
        edge_noise_guard_enabled=bool(cfg.noise.get("edge_noise_guard_enabled", True)),
        edge_noise_guard_factor=float(cfg.noise.get("candidate_noise_height_factor", 5.0)),
        edge_use_enhanced_spike_mapping=bool(cfg.noise.get("edge_use_enhanced_spike_mapping", True)),
        edge_mapping_enable_merge_guard=bool(cfg.noise.get("edge_mapping_enable_merge_guard", True)),
        edge_mapping_noise_guard_enabled=bool(cfg.noise.get("edge_mapping_noise_guard_enabled", False)),
        edge_mapping_min_level_percent=int(cfg.noise.get("edge_mapping_min_level_percent", 1)),
    )
    all_metric_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    total_raw_candidates = 0
    total_merged_candidates = 0
    total_after_noise = 0
    n_pixels = len(candidates_by_pixel)
    print(f"[{_ts()}] [primary] preparing features for {n_pixels} candidate pixels")
    pixel_iter = tqdm(
        list(candidates_by_pixel.items()),
        desc="Primary features",
        unit="px",
        dynamic_ncols=True,
        mininterval=0.25,
        miniters=1,
    )
    for (y, x), segs in pixel_iter:
        total_raw_candidates += int(len(segs))
        raw_sig = np.asarray(raw[y, x, :], dtype=float)
        grad_sig = np.asarray(gradient[y, x, :], dtype=float)
        prepared = prepare_primary_candidates(
            y=int(y),
            x=int(x),
            segs=list(segs),
            feature_signal=grad_sig if cfg.candidates.get("feature_expand_to_gradient_foot", True) else raw_sig,
            boundary_signal=np.asarray(boundary_signal[y, x, :], dtype=float),
            merge_signal=np.asarray(merge_signal[y, x, :], dtype=float),
            feature_expand_to_gradient_foot=bool(cfg.candidates.get("feature_expand_to_gradient_foot", True)),
            feature_foot_k_mad=float(cfg.candidates.get("feature_foot_k_mad", 2.0)),
            feature_foot_min_run=int(cfg.candidates.get("feature_foot_min_run", 2)),
            feature_window_method=str(cfg.candidates.get("feature_window_method", "mad_run")),
            feature_erosion_window=int(cfg.morphology.get("feature_erosion_window", 20)),
            merge_duplicate_segments=bool(cfg.candidates.get("merge_duplicate_segments", True)),
            merge_max_width_pts=int(cfg.candidates.get("max_width_pts", 24)),
        )
        total_merged_candidates += int(len(prepared))
        small = get_or_compute_small_morphology(small_morph_cache, raw, int(y), int(x), window_size=noise_window)
        prefilter_rows, prefilter_summary = evaluate_candidate_noise_prefilter(
            segs=prepared,
            raw_signal=raw_sig,
            small_morphology=small,
            enabled=bool(cfg.noise.get("candidate_noise_prefilter_enabled", True)),
            mode=str(cfg.noise.get("candidate_noise_prefilter_mode", "morph_range_chord")),
            height_factor=float(cfg.noise.get("candidate_noise_height_factor", 5.0)),
        )
        total_after_noise += int(prefilter_summary.get("n_candidates_after_noise_prefilter", 0))
        noise_by_candidate = {str(row["candidate_id"]): row for row in prefilter_rows}
        out_rows: list[dict[str, Any]] = []
        for seg in prepared:
            source_y, source_x = source_coords_map.get((int(seg.y), int(seg.x)), (int(seg.y), int(seg.x)))
            base_row = {
                "candidate_id": seg.candidate_id,
                "parent_id": seg.candidate_id,
                "y": int(seg.y),
                "x": int(seg.x),
                "source_y": int(source_y),
                "source_x": int(source_x),
                "peak_index": int(seg.peak_index),
                "start": int(seg.start),
                "end": int(seg.end),
                "peak_height": float(seg.peak_height),
                "area": float(seg.area),
                "primary_active_decision_profile": str(cfg.decision_profile),
                "source_record_origin": "pipeline_primary",
            }
            base_row.update(noise_by_candidate.get(seg.candidate_id, {}))
            compute_metrics = str(base_row.get("candidate_noise_prefilter_status", "")) != "rejected_noise"
            if compute_metrics:
                base_row.update(
                    compute_ss1_pce_features(
                        raw_signal=raw_sig,
                        gradient_signal=grad_sig,
                        seg=seg,
                        feature_signal_source="gradient",
                    )
                )
                base_row.update(
                    compute_raw_edge_metric(
                        raw_signal=raw_sig,
                        seg=seg,
                        candidate_noise_estimate=base_row.get("candidate_noise_estimate_used"),
                        ctx=metric_ctx,
                    )
                )
                all_edge_rows.append(base_row)
            out_rows.append(base_row)
        rows_by_pixel[(int(y), int(x))] = out_rows
        all_metric_rows.extend(out_rows)
        small_morphology_payloads[(int(y), int(x))] = {
            "erosion_contacts": [int(v) for v in small.erosion_contacts.tolist()],
            "dilation_contacts": [int(v) for v in small.dilation_contacts.tolist()],
            "noise_reference_spans": prefilter_summary.get("noise_reference_spans", []),
            "noise_height_morph_range": prefilter_summary.get("noise_height_morph_range"),
            "noise_reference_status": prefilter_summary.get("noise_reference_status"),
            "noise_reference_method": prefilter_summary.get("noise_reference_method"),
        }
        summary_by_pixel[(int(y), int(x))] = prefilter_summary
        pixel_iter.set_postfix(
            candidates=len(out_rows),
            kept=sum(1 for row in out_rows if str(row.get("candidate_noise_prefilter_status", "")) != "rejected_noise"),
            refresh=False,
        )
    finalize_edge_evidence(all_edge_rows, metric_ctx)
    for rows in rows_by_pixel.values():
        for row in rows:
            ss4 = compute_ss4(
                float(row.get("spike_score_v1", np.nan)),
                float(row.get("pce_negpref_t098_evidence_signed", np.nan)),
                float(row.get("recdw_sum_0_90_raman_veto_evidence_signed", np.nan)),
                ss_blue_max=float(cfg.ss4["ss_blue_max"]),
                ss_red_min=float(cfg.ss4["ss_red_min"]),
                pce_red_min=float(cfg.ss4["pce_red_min"]),
                edge_red_max=float(cfg.ss4["edge_red_max"]),
                pce_dead_zone_enabled=bool(cfg.ss4.get("pce_dead_zone_enabled", False)),
                pce_dead_zone_low=float(cfg.ss4.get("pce_dead_zone_low", -0.8)),
                pce_dead_zone_high=float(cfg.ss4.get("pce_dead_zone_high", -0.2)),
                missing_policy=str(cfg.ss4.get("missing_policy", "review")),
            )
            ss5 = compute_ss5(
                float(row.get("spike_score_v1", np.nan)),
                float(row.get("pce_negpref_t098_evidence_signed", np.nan)),
                float(row.get("recdw_sum_0_90_raman_veto_evidence_signed", np.nan)),
                ss1_threshold=float(cfg.ss5["ss1_threshold"]),
                pce_spike_min=float(cfg.ss5["pce_spike_min"]),
                edge_spike_max=float(cfg.ss5["edge_spike_max"]),
            )
            row.update(ss4)
            row.update(ss5)
            row["primary_spike_score_v1"] = row.get("spike_score_v1")
            row["primary_pce_negpref_t098_evidence_signed"] = row.get("pce_negpref_t098_evidence_signed")
            row["primary_recdw_sum_0_90_raman_veto_evidence_signed"] = row.get("recdw_sum_0_90_raman_veto_evidence_signed")
            row["primary_ss4"] = row.get("ss4")
            row["primary_ss4_decision"] = row.get("ss4_decision")
            row["primary_ss4_reason"] = row.get("ss4_reason")
            row["primary_ss5"] = row.get("ss5")
            row["primary_ss5_decision"] = row.get("ss5_decision")
            row["primary_ss5_reason"] = row.get("ss5_reason")
            if str(cfg.decision_profile) == "ss5":
                row["primary_active_score"] = row.get("ss5")
                row["primary_active_decision"] = row.get("ss5_decision")
                row["primary_active_reason"] = row.get("ss5_reason")
            else:
                row["primary_active_score"] = row.get("ss4")
                row["primary_active_decision"] = row.get("ss4_decision")
                row["primary_active_reason"] = row.get("ss4_reason")
            row["primary_is_spike"] = bool(
                str(row.get("candidate_noise_prefilter_status", "")) != "rejected_noise"
                and str(row.get("primary_active_decision", "non_spike")) == "spike"
            )
    apply_global_metric_ranks(all_metric_rows)
    return rows_by_pixel, small_morphology_payloads, summary_by_pixel, {
        "raw_candidates": int(total_raw_candidates),
        "merged_candidates": int(total_merged_candidates),
        "noise_kept_candidates": int(total_after_noise),
    }


def run_pipeline(cfg: CoreConfig) -> PipelineArtifacts:
    run_started = time.perf_counter()
    print(f"[{_ts()}] [pipeline] start")

    t0 = time.perf_counter()
    dataset = load_dataset(cfg.paths["input_path"], input_format=str(cfg.data.get("input_format", "auto")))
    _phase_print(
        "data load",
        t0,
        extra=f"shape={tuple(int(v) for v in dataset.spectra.shape)} x_axis={int(dataset.x_axis.size)}",
    )

    source_coords_map: dict[tuple[int, int], tuple[int, int]]
    if bool(cfg.data.get("use_compact_coords_view", True)) and cfg.paths.get("coords_csv"):
        t0 = time.perf_counter()
        coords = load_target_coords_csv(cfg.paths["coords_csv"], dataset.spectra.shape[:2])
        dataset, source_coords_map = _build_compact_subset(dataset, coords)
        _phase_print(
            "compact subset",
            t0,
            extra=f"n_coords={len(coords)} compact_shape={tuple(int(v) for v in dataset.spectra.shape[:2])}",
        )
    else:
        h, w = dataset.spectra.shape[:2]
        source_coords_map = {(y, x): (y, x) for y in range(h) for x in range(w)}
    raw = np.asarray(dataset.spectra, dtype=np.float32)
    windows = sorted({int(v) for v in cfg.morphology.get("morphology_windows", [3, 5, 7])})
    if int(cfg.morphology["tophat_window"]) not in windows:
        windows.append(int(cfg.morphology["tophat_window"]))

    t0 = time.perf_counter()
    morph = compute_morphology_windows(raw, windows)
    _phase_print("morphology", t0, extra=f"windows={windows}")

    overlays = {
        "dilation": {window: payload.dilation for window, payload in morph.items()},
        "erosion": {window: payload.erosion for window, payload in morph.items()},
        "opening": {window: payload.opening for window, payload in morph.items()},
        "top_hat": {window: payload.top_hat for window, payload in morph.items()},
        "gradient": {window: payload.gradient for window, payload in morph.items()},
    }
    th_window = int(cfg.morphology["tophat_window"])

    t0 = time.perf_counter()
    score_map = score_map_from_top_hat(overlays["top_hat"][th_window], mode=str(cfg.candidates.get("score_mode", "max")))
    threshold = threshold_score_map(
        score_map,
        method=str(cfg.candidates.get("threshold_method", "quantile")),
        quantile=float(cfg.candidates.get("threshold_quantile", 0.1)),
        k_mad=float(cfg.candidates.get("threshold_k_mad", 20.0)),
        min_abs=cfg.candidates.get("threshold_min_abs"),
    )
    _phase_print(
        "score map + threshold",
        t0,
        extra=f"window={th_window} threshold={float(threshold):.4f}",
    )

    candidate_mask = np.asarray(score_map >= threshold, dtype=bool)
    processed_mask = np.asarray(candidate_mask, dtype=bool)
    processed_spectra = int(np.count_nonzero(processed_mask))
    if bool(cfg.data.get("use_compact_coords_view", True)) and cfg.paths.get("coords_csv"):
        processed_mask = np.zeros_like(candidate_mask, dtype=bool)
        for cy, cx in source_coords_map.keys():
            processed_mask[int(cy), int(cx)] = True
        processed_spectra = int(np.count_nonzero(processed_mask))

    t0 = time.perf_counter()
    _, candidates_by_pixel = extract_top_hat_candidates(
        x_axis=np.asarray(dataset.x_axis, dtype=float),
        top_hat=np.asarray(overlays["top_hat"][th_window], dtype=float),
        candidate_mask=processed_mask,
        raw_spectra=np.asarray(raw, dtype=float),
        max_width_pts=int(cfg.candidates.get("max_width_pts", 24)),
        k_mad_pixel=float(cfg.candidates.get("k_mad_pixel", 8.0)),
        min_peak=float(cfg.candidates.get("min_peak", 80.0)),
        baseline_window=int(cfg.morphology.get("baseline_window", 5)),
        edge_k_mad=float(cfg.candidates.get("edge_k_mad", 2.0)),
        pad_pts=int(cfg.candidates.get("pad_pts", 0)),
    )
    raw_candidate_count = sum(len(v) for v in candidates_by_pixel.values())
    _phase_print(
        "candidate detection",
        t0,
        extra=f"processed_spectra={processed_spectra} candidate_pixels={len(candidates_by_pixel)} raw_candidates={raw_candidate_count}",
    )

    t0 = time.perf_counter()
    candidate_records_by_pixel, small_morphology_by_pixel, prefilter_summary_by_pixel, candidate_stats = _prepare_candidate_rows(
        raw=np.asarray(raw, dtype=float),
        overlays=overlays,
        candidates_by_pixel=candidates_by_pixel,
        cfg=cfg,
        source_coords_map=source_coords_map,
    )
    _phase_print("primary metrics + decisions", t0)

    t0 = time.perf_counter()
    correction = build_placeholder_correction(raw)
    _phase_print("correction placeholder", t0, extra=f"stages={len(correction.stages)} chords={len(correction.chords)}")

    all_rows = [row for rows in candidate_records_by_pixel.values() for row in rows]
    ss4_spikes = sum(1 for row in all_rows if str(row.get("ss4_decision")) == "spike")
    ss5_spikes = sum(1 for row in all_rows if str(row.get("ss5_decision")) == "spike")
    accepted_both = sum(1 for row in all_rows if str(row.get("ss4_decision")) == "spike" and str(row.get("ss5_decision")) == "spike")
    accepted_ss4_only = sum(1 for row in all_rows if str(row.get("ss4_decision")) == "spike" and str(row.get("ss5_decision")) != "spike")
    accepted_ss5_only = sum(1 for row in all_rows if str(row.get("ss5_decision")) == "spike" and str(row.get("ss4_decision")) != "spike")
    rejected_both = sum(1 for row in all_rows if str(row.get("ss4_decision")) != "spike" and str(row.get("ss5_decision")) != "spike")
    active_primary_spikes = sum(1 for row in all_rows if bool(row.get("primary_is_spike")))
    prefilter_before = sum(int(v.get("n_candidates_before_noise_prefilter", 0)) for v in prefilter_summary_by_pixel.values())
    prefilter_kept = sum(int(v.get("n_candidates_after_noise_prefilter", 0)) for v in prefilter_summary_by_pixel.values())
    prefilter_rejected = sum(int(v.get("n_candidates_rejected_by_noise_prefilter", 0)) for v in prefilter_summary_by_pixel.values())
    sufficient_noise_refs = sum(1 for v in prefilter_summary_by_pixel.values() if str(v.get("noise_reference_status")) == "ok")
    insufficient_noise_refs = sum(1 for v in prefilter_summary_by_pixel.values() if str(v.get("noise_reference_status")) != "ok")
    metadata = {
        "input_path": str(dataset.path),
        "decision_profile": str(cfg.decision_profile),
        "threshold": float(threshold),
        "morphology_windows": windows,
        "tophat_window": int(th_window),
        "noise_window": int(cfg.morphology["noise_window"]),
        "n_candidate_pixels": int(np.sum(candidate_mask)),
        "n_processed_spectra": int(processed_spectra),
        "n_candidates": int(len(all_rows)),
        "n_raw_candidates": int(candidate_stats["raw_candidates"]),
        "n_merged_candidates": int(candidate_stats["merged_candidates"]),
        "n_candidates_after_noise_prefilter": int(candidate_stats["noise_kept_candidates"]),
        "n_primary_accepted_candidates": int(active_primary_spikes),
        "ss4_spikes": int(ss4_spikes),
        "ss5_spikes": int(ss5_spikes),
        "accepted_by_both": int(accepted_both),
        "accepted_by_ss4_only": int(accepted_ss4_only),
        "accepted_by_ss5_only": int(accepted_ss5_only),
        "rejected_by_both": int(rejected_both),
        "prefilter_summaries": {
            f"{y},{x}": value for (y, x), value in prefilter_summary_by_pixel.items()
        },
    }
    print(
        f"[{_ts()}] [summary] processed_spectra={processed_spectra} candidate_pixels={len(candidates_by_pixel)} "
        f"raw_candidates={candidate_stats['raw_candidates']} "
        f"merged_candidates={candidate_stats['merged_candidates']}"
    )
    print(
        f"[{_ts()}] [summary] noise_prefilter before={prefilter_before} kept={prefilter_kept} "
        f"rejected_noise={prefilter_rejected}"
    )
    print(
        f"[{_ts()}] [summary] noise_reference ok={sufficient_noise_refs} insufficient={insufficient_noise_refs}"
    )
    print(
        f"[{_ts()}] [summary] {str(cfg.decision_profile).upper()} active_spikes="
        f"{active_primary_spikes} "
        f"| ss4={ss4_spikes} ss5={ss5_spikes} both={accepted_both} "
        f"ss4_only={accepted_ss4_only} ss5_only={accepted_ss5_only} neither={rejected_both}"
    )
    print(f"[{_ts()}] [pipeline] finished in {_fmt_s(time.perf_counter() - run_started)}")
    return PipelineArtifacts(
        dataset=dataset,
        x_axis=np.asarray(dataset.x_axis, dtype=float),
        spectra=np.asarray(raw),
        corrected_spectra=np.asarray(correction.corrected_spectra),
        score_map=np.asarray(score_map),
        candidate_mask=np.asarray(candidate_mask),
        overlays=overlays,
        candidates_by_pixel=candidates_by_pixel,
        candidate_records_by_pixel=candidate_records_by_pixel,
        source_coords_map=source_coords_map,
        small_morphology_by_pixel=small_morphology_by_pixel,
        despike_stages=correction.stages,
        despike_chords=correction.chords,
        metadata=metadata,
    )


def _write_light_debug(cfg: CoreConfig, artifacts: PipelineArtifacts) -> None:
    if not bool(cfg.outputs.get("despike_debug_lite_enabled", cfg.outputs.get("save_light_debug", True))):
        return
    out_path = Path(cfg.paths["light_debug_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dumps_json(artifacts.metadata), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clean MuonFinder core pipeline.")
    parser.add_argument("--config", required=True, help="Path to config_core.json")
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    artifacts = run_pipeline(cfg)
    t0 = time.perf_counter()
    if bool(cfg.outputs.get("save_viewer_cache", True)):
        save_viewer_cache(Path(cfg.paths["viewer_cache_path"]), artifacts)
        _phase_print("viewer cache write", t0, extra=str(cfg.paths["viewer_cache_path"]))
    t0 = time.perf_counter()
    _write_light_debug(cfg, artifacts)
    if bool(cfg.outputs.get("despike_debug_lite_enabled", cfg.outputs.get("save_light_debug", True))):
        _phase_print("light debug write", t0, extra=str(cfg.paths["light_debug_path"]))
    print(f"viewer cache: {cfg.paths['viewer_cache_path']}")
    if bool(cfg.outputs.get("despike_debug_lite_enabled", cfg.outputs.get("save_light_debug", True))):
        print(f"light debug: {cfg.paths['light_debug_path']}")
    if bool(cfg.viewer.get("open_after_pipeline", False)):
        import sys

        viewer_script = Path(__file__).resolve().with_name("viewer.py")
        viewer_cache = Path(cfg.paths["viewer_cache_path"]).resolve()
        subprocess.Popen([sys.executable, str(viewer_script), "--cache", str(viewer_cache)])


if __name__ == "__main__":
    main()
