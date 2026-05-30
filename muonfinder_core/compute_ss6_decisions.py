from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
import gc
from pathlib import Path
import time
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

if __package__ in {None, ""}:
    import sys

    _THIS_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _THIS_DIR.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from muonfinder_core.cache import load_viewer_cache_light
    from muonfinder_core.cap_metrics import join_extra_feature_rows, load_extra_feature_rows
    from muonfinder_core.compute_all_experimental_features import compute_all_experimental_features_from_config
    from muonfinder_core.config import load_config
    from muonfinder_core.experimental_common import _safe_float, write_feature_csv
    from muonfinder_core.ss6_diagnostics import generate_ss6_histograms
    from muonfinder_core.ss6_decision import compute_ss6_row, ss6_defaults
else:
    from .cache import load_viewer_cache_light
    from .cap_metrics import join_extra_feature_rows, load_extra_feature_rows
    from .compute_all_experimental_features import compute_all_experimental_features_from_config
    from .config import load_config
    from .experimental_common import _safe_float, write_feature_csv
    from .ss6_diagnostics import generate_ss6_histograms
    from .ss6_decision import compute_ss6_row, ss6_defaults


def _progress_iter(items, *, desc: str, total: int | None = None):
    if tqdm is not None:
        return tqdm(items, desc=desc, total=total, dynamic_ncols=True, mininterval=0.25)

    def _fallback():
        count = 0
        next_print = 250
        for item in items:
            count += 1
            if count == 1 or count >= next_print:
                if total is not None and total > 0:
                    print(f"{desc}: {count}/{total}")
                else:
                    print(f"{desc}: {count}")
                next_print += 250
            yield item

    return _fallback()


def _mtime(path: Path) -> float:
    return float(path.stat().st_mtime) if path.exists() else float("-inf")


def _experimental_refresh_decision(
    *,
    cache_path: Path,
    exp_path: Path,
    auto_recompute: bool,
    force_experimental: bool,
    skip_experimental_recompute: bool,
) -> tuple[bool, str]:
    if force_experimental:
        return True, "force_flag"
    if not exp_path.exists():
        if skip_experimental_recompute:
            raise FileNotFoundError(
                f"Experimental features CSV not found: {exp_path}. "
                "Re-run without --no-experimental-recompute or generate experimental features first."
            )
        return True, "missing"
    if skip_experimental_recompute:
        return False, "existing_file_no_recompute"
    if not auto_recompute:
        return False, "existing_file_auto_recompute_false"
    if _mtime(cache_path) > _mtime(exp_path):
        return True, "cache_newer_than_features"
    return False, "fresh"


def _missing_required_experimental_columns(path: Path, required_columns: list[str]) -> list[str]:
    if not path.exists():
        return list(required_columns)
    rows, columns = load_extra_feature_rows(path)
    _ = rows
    present = set(columns)
    return [col for col in required_columns if col not in present]


def _pce_debug_chosen_value(row: dict[str, Any]) -> float:
    direct = _safe_float(row.get("pce_t098_chosen_value", np.nan))
    if np.isfinite(direct):
        return direct
    debug = row.get("pce_t98_debug", {})
    if isinstance(debug, dict):
        return _safe_float(debug.get("chosen_value", np.nan))
    return float("nan")


def _write_pce_audit(
    *,
    rows: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    ss6_cfg: dict[str, Any],
    out_path: Path,
) -> tuple[int, int]:
    pce_field = str(ss6_cfg.get("metric_names", {}).get("pce", "pce_negpref_t098_evidence_signed")).strip() or "pce_negpref_t098_evidence_signed"
    fieldnames = [
        "source_y",
        "source_x",
        "compact_y",
        "compact_x",
        "peak_index",
        "candidate_id",
        "pce_debug_chosen_value",
        "configured_pce_field",
        "configured_pce_value",
        "ss6_pce",
        "pce_annotation_value",
        "mismatch_type",
    ]
    audit_rows: list[dict[str, Any]] = []
    audited = 0
    for row, out in zip(rows, outputs):
        configured_value = _safe_float(row.get(pce_field, np.nan))
        ss6_pce = _safe_float(out.get("ss6_pce", np.nan))
        if not np.isfinite(ss6_pce):
            continue
        audited += 1
        debug_chosen = _pce_debug_chosen_value(row)
        mismatch_types: list[str] = []
        if not np.isfinite(configured_value):
            mismatch_types.append("configured_field_missing")
        elif abs(configured_value - ss6_pce) > 1e-12:
            mismatch_types.append("configured_vs_ss6")
        if np.isfinite(debug_chosen) and np.isfinite(configured_value):
            if debug_chosen < 0.0 < configured_value:
                mismatch_types.append("raw_negative_vs_configured_positive")
            elif debug_chosen > 0.0 > configured_value:
                mismatch_types.append("raw_positive_vs_configured_negative")
        if mismatch_types:
            audit_rows.append(
                {
                    "source_y": int(row.get("source_y", row.get("y", -1))),
                    "source_x": int(row.get("source_x", row.get("x", -1))),
                    "compact_y": int(row.get("y", -1)),
                    "compact_x": int(row.get("x", -1)),
                    "peak_index": int(row.get("peak_index", -1)),
                    "candidate_id": row.get("candidate_id", ""),
                    "pce_debug_chosen_value": debug_chosen,
                    "configured_pce_field": pce_field,
                    "configured_pce_value": configured_value,
                    "ss6_pce": ss6_pce,
                    "pce_annotation_value": configured_value,
                    "mismatch_type": "|".join(mismatch_types),
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for audit_row in audit_rows:
            writer.writerow(audit_row)
    return audited, len(audit_rows)


def _write_ss6_decision_trace(
    *,
    rows: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    ss6_cfg: dict[str, Any],
    out_path: Path,
) -> None:
    metric_names = dict(ss6_cfg.get("metric_names", {}))
    trace_rows: list[dict[str, Any]] = []
    for row, out in zip(rows, outputs):
        trace_row: dict[str, Any] = {
            "source_y": int(row.get("source_y", row.get("y", -1))),
            "source_x": int(row.get("source_x", row.get("x", -1))),
            "compact_y": int(row.get("y", -1)),
            "compact_x": int(row.get("x", -1)),
            "peak_index": int(row.get("peak_index", -1)),
            "candidate_id": row.get("candidate_id", ""),
            "peak_x": _safe_float(row.get("peak_x", np.nan)),
            "noise_kept": int(str(row.get("candidate_noise_prefilter_status", "")) != "rejected_noise"),
            "ss6_accept": int(out.get("ss6_accept", 0)),
            "ss6_branch": str(out.get("ss6_branch", "")),
            "ss6_reason": str(out.get("ss6_reason", "")),
            "branch_fired": str(out.get("ss6_branch", "")),
        }
        for metric_key in ("ss1", "pce", "edge", "eel", "resid"):
            source_col = str(metric_names.get(metric_key, "")).strip()
            value = _safe_float(row.get(source_col, np.nan))
            trace_row[f"{metric_key}_source_column"] = source_col
            trace_row[f"{metric_key}_value"] = value
            trace_row[f"{metric_key}_finite"] = int(np.isfinite(value))
        for key, value in out.items():
            if str(key).startswith("ss6_") and str(key).endswith("_flag"):
                trace_row[str(key)] = value
        trace_rows.append(trace_row)
    write_feature_csv(out_path, trace_rows)


def run_ss6_decisions(
    *,
    config_path: Path | str,
    cache_path: Path | None = None,
    experimental_features_path: Path | None = None,
    out_path: Path | None = None,
    summary_out: Path | None = None,
    force_experimental_recompute: bool = False,
    no_experimental_recompute: bool = False,
    no_histograms: bool = False,
) -> tuple[dict[str, Any], dict[str, float]]:
    timings: dict[str, float] = {}
    t_start = time.perf_counter()
    stage_t0 = time.perf_counter()
    cfg_path = Path(config_path)
    print(f"config path: {cfg_path}")
    cfg = load_config(cfg_path)
    timings["load config"] = time.perf_counter() - stage_t0
    ss6_cfg = ss6_defaults(getattr(cfg, "ss6", {}))
    exp_cfg = dict(getattr(cfg, "experimental_features", {}))
    cache_path = Path(cache_path) if cache_path is not None else Path(str(cfg.paths["viewer_cache_path"]))
    exp_path = Path(experimental_features_path) if experimental_features_path is not None else Path(str(exp_cfg.get("features_path", "")))
    out_path = Path(out_path) if out_path is not None else Path(str(ss6_cfg["decisions_path"]))
    summary_path = Path(summary_out) if summary_out is not None else Path(str(ss6_cfg["summary_path"]))
    auto_recompute = bool(exp_cfg.get("auto_recompute", True))
    print(f"viewer cache: {cache_path}")
    print(f"experimental features path: {exp_path}")
    print(f"ss6 decisions path: {out_path}")
    recompute_exp, recompute_reason = _experimental_refresh_decision(
        cache_path=cache_path,
        exp_path=exp_path,
        auto_recompute=auto_recompute,
        force_experimental=bool(force_experimental_recompute),
        skip_experimental_recompute=bool(no_experimental_recompute),
    )
    required_exp_columns = ["exp_edge_legacy_evidence_signed_modernnorm", "exp_resid3_height_noise_z"]
    if not recompute_exp:
        missing_exp_cols = _missing_required_experimental_columns(exp_path, required_exp_columns)
        if missing_exp_cols:
            if bool(no_experimental_recompute):
                raise ValueError(f"Experimental features CSV missing required columns: {missing_exp_cols}")
            recompute_exp = True
            recompute_reason = "missing_required_d3_columns"
    if recompute_exp:
        if recompute_reason == "missing":
            print("experimental features: recomputing because file missing")
        elif recompute_reason == "force_flag":
            print("experimental features: recomputing because force flag set")
        else:
            print(f"experimental features: recomputing because {recompute_reason}")
        stage_t0 = time.perf_counter()
        compute_all_experimental_features_from_config(
            cfg,
            cache_path=cache_path,
            out_path=exp_path,
            summary_path=Path(str(exp_cfg.get("summary_path", ""))) if str(exp_cfg.get("summary_path", "")).strip() else None,
        )
        timings["experimental features"] = time.perf_counter() - stage_t0
        gc.collect()
    else:
        print("experimental features: loaded existing file")
        timings["experimental features"] = 0.0

    if not exp_path.exists():
        raise FileNotFoundError(f"Experimental features CSV not found after refresh step: {exp_path}")

    stage_t0 = time.perf_counter()
    print("load viewer cache...")
    cache = load_viewer_cache_light(cache_path, verbose=True)
    timings["load cache"] = time.perf_counter() - stage_t0
    rows = [dict(row) for row in cache.get("candidate_records", [])]
    print(f"candidate records source: {cache.get('candidate_records_source', 'unknown')}")
    print(f"candidate records loaded: {len(rows)}")

    stage_t0 = time.perf_counter()
    print("load existing experimental features...")
    extra_rows, extra_columns = load_extra_feature_rows(exp_path)
    timings["load experimental rows"] = time.perf_counter() - stage_t0
    print(f"experimental rows loaded: {len(extra_rows)}")

    stage_t0 = time.perf_counter()
    print("join candidate records with experimental features...")
    join_info = join_extra_feature_rows(rows, extra_rows)
    timings["join rows"] = time.perf_counter() - stage_t0
    print(f"rows matched: {join_info['matched_rows']}")
    print(f"rows unmatched: {join_info['unmatched_rows']}")
    required_metric_names = [
        str(ss6_cfg.get("metric_names", {}).get("ss1", "")),
        str(ss6_cfg.get("metric_names", {}).get("pce", "")),
        str(ss6_cfg.get("metric_names", {}).get("edge", "")),
        str(ss6_cfg.get("metric_names", {}).get("eel", "")),
        str(ss6_cfg.get("metric_names", {}).get("resid", "")),
    ]
    missing_required_rows = 0
    for row in rows:
        if not all(np.isfinite(_safe_float(row.get(name, np.nan))) for name in required_metric_names):
            missing_required_rows += 1
    print(f"rows with missing required SS6 metrics: {missing_required_rows}")

    stage_t0 = time.perf_counter()
    print("compute SS6 decisions...")
    outputs = [compute_ss6_row(row, ss6_cfg) for row in _progress_iter(rows, desc="SS6 decisions", total=len(rows))]
    timings["SS6 decisions"] = time.perf_counter() - stage_t0

    stage_t0 = time.perf_counter()
    print("write SS6 decisions CSV...")
    write_feature_csv(out_path, outputs)
    trace_path = out_path.with_name(f"{out_path.stem}_trace.csv")
    print("write SS6 trace CSV...")
    _write_ss6_decision_trace(rows=rows, outputs=outputs, ss6_cfg=ss6_cfg, out_path=trace_path)
    print("write SS6 summary JSON...")
    generated_hist_files: list[str] = []
    branch_counts = Counter(str(row.get("ss6_branch", "")) for row in outputs)
    finite_metric_rows = sum(
        1
        for row in outputs
        if all(np.isfinite(_safe_float(row.get(key, np.nan))) for key in ("ss6_ss1", "ss6_pce", "ss6_edge", "ss6_eel", "ss6_resid"))
    )
    accepted = sum(int(row.get("ss6_accept", 0)) == 1 for row in outputs)
    timings["write outputs"] = time.perf_counter() - stage_t0
    audited_count = 0
    mismatch_count = 0
    audit_path = out_path.with_name("pce_metric_audit.csv")
    if bool(ss6_cfg.get("write_pce_metric_audit", False)):
        audited_count, mismatch_count = _write_pce_audit(rows=rows, outputs=outputs, ss6_cfg=ss6_cfg, out_path=audit_path)
    elif audit_path.exists():
        audit_path.unlink()

    save_histograms = bool(ss6_cfg.get("save_histograms", True)) and not bool(no_histograms)
    stage_t0 = time.perf_counter()
    if save_histograms:
        print("generating SS6 histograms...")
        generated_hist_files = generate_ss6_histograms(outputs, ss6_cfg)
    else:
        print("skipping SS6 histograms")
    timings["histograms"] = time.perf_counter() - stage_t0

    summary = {
        "total_candidate_rows": int(len(outputs)),
        "rows_with_all_required_metrics_finite": int(finite_metric_rows),
        "rows_accepted_by_ss6": int(accepted),
        "rows_rejected_by_ss6": int(len(outputs) - accepted),
        "counts_per_ss6_branch": dict(sorted(branch_counts.items())),
        "thresholds_used": {
            key: ss6_cfg[key]
            for key in (
                "ss1_gate",
                "pce_strong_min",
                "pce_dead_max",
                "pce_gray_min",
                "pce_gray_max",
                "edge_spike_max",
                "eel_spike_max",
                "edge_eel_delta_min",
                "pce_dead_edge_spike_max",
                "pce_gray_delta_eel_max",
                "pce_gray_delta_min",
                "pce_gray_delta_resid_min",
                "pce_gray_high_pce_min",
                "pce_gray_high_pce_eel_max",
                "pce_gray_high_pce_resid_min",
                "pce_gray_high_resid_min",
                "pce_gray_soft_edge_max",
                "pce_gray_soft_eel_max",
                "low_pce_double_edge_edge_max",
                "low_pce_double_edge_eel_max",
                "low_pce_double_edge_resid_min",
                "resid_rescue_min",
                "resid_strong_min",
                "require_noise_kept",
            )
        },
        "metric_names_used": dict(ss6_cfg.get("metric_names", {})),
        "experimental_features_path": str(exp_path),
        "extra_feature_columns_joined": extra_columns,
        "extra_feature_join_info": join_info,
        "histograms_dir": str(ss6_cfg.get("histograms_dir", "")),
        "generated_histogram_files": generated_hist_files,
        "pce_metric_audit_path": (str(audit_path) if bool(ss6_cfg.get("write_pce_metric_audit", False)) else ""),
        "ss6_decision_trace_path": str(trace_path),
        "pce_metric_audited_rows": int(audited_count),
        "pce_metric_mismatches": int(mismatch_count),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    stage_t0 = time.perf_counter()
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    timings["write outputs"] += time.perf_counter() - stage_t0

    print(f"experimental features: {exp_path}")
    print(f"experimental rows loaded: {join_info['loaded_rows']}")
    print(f"experimental rows matched: {join_info['matched_rows']}")
    print(f"ss6 decisions rows: {len(outputs)}")
    print(f"ss6 accepted: {accepted}")
    print(f"ss6 rejected: {len(outputs) - accepted}")
    if bool(ss6_cfg.get("write_pce_metric_audit", False)):
        print(f"pce metric audit: {audited_count} audited, {mismatch_count} mismatches")
        print(f"pce metric audit csv: {audit_path}")
    else:
        print("pce metric audit: disabled")
    if generated_hist_files:
        print(f"ss6 histogram files: {len(generated_hist_files)}")
    print(f"ss6 output: {out_path}")
    print(f"ss6 trace: {trace_path}")
    print(f"ss6 summary: {summary_path}")
    timings["total"] = time.perf_counter() - t_start
    return summary, timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute experimental SS6 rule-based decisions from viewer cache and experimental features.")
    parser.add_argument("--config", type=Path, default=Path("config_core.json"), help="Core config; used to resolve viewer_cache_path, experimental features, and SS6 output paths.")
    parser.add_argument("--cache", type=Path, default=None, help="Optional explicit viewer cache path.")
    parser.add_argument("--experimental-features", type=Path, default=None, help="Optional explicit experimental_features.csv path.")
    parser.add_argument("--out", type=Path, default=None, help="Optional SS6 CSV output path.")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional SS6 summary JSON output path.")
    parser.add_argument("--force-experimental-recompute", action="store_true", help="Always recompute experimental features before SS6.")
    parser.add_argument("--no-experimental-recompute", action="store_true", help="Reuse existing experimental features and fail clearly if missing.")
    parser.add_argument("--no-histograms", action="store_true", help="Skip SS6 histogram generation.")
    args = parser.parse_args()
    summary, timings = run_ss6_decisions(
        config_path=args.config,
        cache_path=args.cache,
        experimental_features_path=args.experimental_features,
        out_path=args.out,
        summary_out=args.summary_out,
        force_experimental_recompute=bool(args.force_experimental_recompute),
        no_experimental_recompute=bool(args.no_experimental_recompute),
        no_histograms=bool(args.no_histograms),
    )
    _ = summary
    print("Timing summary:")
    for key in ("load config", "load cache", "experimental features", "load experimental rows", "join rows", "SS6 decisions", "write outputs", "histograms"):
        print(f"  {key}: {timings.get(key, 0.0):.1f} s")
    print(f"  total: {timings.get('total', 0.0):.1f} s")


if __name__ == "__main__":
    main()
