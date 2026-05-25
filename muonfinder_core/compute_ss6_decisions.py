from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import time

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

    from muonfinder_core.cache import load_viewer_cache
    from muonfinder_core.cap_metrics import join_extra_feature_rows, load_extra_feature_rows
    from muonfinder_core.compute_all_experimental_features import compute_all_experimental_features_from_config
    from muonfinder_core.config import load_config
    from muonfinder_core.experimental_common import _safe_float, write_feature_csv
    from muonfinder_core.ss6_diagnostics import generate_ss6_histograms
    from muonfinder_core.ss6_decision import compute_ss6_row, ss6_defaults
else:
    from .cache import load_viewer_cache
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

    timings: dict[str, float] = {}
    t_start = time.perf_counter()
    stage_t0 = time.perf_counter()
    cfg = load_config(args.config)
    timings["load config"] = time.perf_counter() - stage_t0
    ss6_cfg = ss6_defaults(getattr(cfg, "ss6", {}))
    exp_cfg = dict(getattr(cfg, "experimental_features", {}))
    cache_path = Path(args.cache) if args.cache is not None else Path(str(cfg.paths["viewer_cache_path"]))
    exp_path = Path(args.experimental_features) if args.experimental_features is not None else Path(str(exp_cfg.get("features_path", "")))
    out_path = Path(args.out) if args.out is not None else Path(str(ss6_cfg["decisions_path"]))
    summary_path = Path(args.summary_out) if args.summary_out is not None else Path(str(ss6_cfg["summary_path"]))
    auto_recompute = bool(exp_cfg.get("auto_recompute", True))
    print(f"viewer cache: {cache_path}")
    print(f"experimental features path: {exp_path}")
    print(f"ss6 decisions path: {out_path}")
    recompute_exp, recompute_reason = _experimental_refresh_decision(
        cache_path=cache_path,
        exp_path=exp_path,
        auto_recompute=auto_recompute,
        force_experimental=bool(args.force_experimental_recompute),
        skip_experimental_recompute=bool(args.no_experimental_recompute),
    )
    stage_t0 = time.perf_counter()
    print("load viewer cache...")
    cache = load_viewer_cache(cache_path)
    timings["load cache"] = time.perf_counter() - stage_t0
    rows = [dict(row) for row in cache.get("candidate_records", [])]
    print(f"candidate records loaded: {len(rows)}")

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
    else:
        print("experimental features: loaded existing file")
        timings["experimental features"] = 0.0

    if not exp_path.exists():
        raise FileNotFoundError(f"Experimental features CSV not found after refresh step: {exp_path}")

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

    save_histograms = bool(ss6_cfg.get("save_histograms", True)) and not bool(args.no_histograms)
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
    if generated_hist_files:
        print(f"ss6 histogram files: {len(generated_hist_files)}")
    print(f"ss6 output: {out_path}")
    print(f"ss6 summary: {summary_path}")
    print("Timing summary:")
    for key in ("load config", "load cache", "experimental features", "load experimental rows", "join rows", "SS6 decisions", "write outputs", "histograms"):
        print(f"  {key}: {timings.get(key, 0.0):.1f} s")
    print(f"  total: {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
