from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

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
        return True, "forced"
    if not exp_path.exists():
        if skip_experimental_recompute:
            raise FileNotFoundError(
                f"Experimental features CSV not found: {exp_path}. "
                "Re-run without --skip-experimental-recompute or generate experimental features first."
            )
        return True, "missing"
    if _mtime(cache_path) > _mtime(exp_path):
        if skip_experimental_recompute:
            return False, "stale_reused_by_request"
        return True, "stale"
    return False, "fresh"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute experimental SS6 rule-based decisions from viewer cache and experimental features.")
    parser.add_argument("--config", type=Path, default=Path("config_core.json"), help="Core config; used to resolve viewer_cache_path, experimental features, and SS6 output paths.")
    parser.add_argument("--cache", type=Path, default=None, help="Optional explicit viewer cache path.")
    parser.add_argument("--experimental-features", type=Path, default=None, help="Optional explicit experimental_features.csv path.")
    parser.add_argument("--out", type=Path, default=None, help="Optional SS6 CSV output path.")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional SS6 summary JSON output path.")
    parser.add_argument("--force-experimental", action="store_true", help="Always recompute experimental features before SS6.")
    parser.add_argument("--skip-experimental-recompute", action="store_true", help="Reuse existing experimental features and fail clearly if missing.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ss6_cfg = ss6_defaults(getattr(cfg, "ss6", {}))
    exp_cfg = dict(getattr(cfg, "experimental_features", {}))
    cache_path = Path(args.cache) if args.cache is not None else Path(str(cfg.paths["viewer_cache_path"]))
    exp_path = Path(args.experimental_features) if args.experimental_features is not None else Path(str(exp_cfg.get("features_path", "")))
    out_path = Path(args.out) if args.out is not None else Path(str(ss6_cfg["decisions_path"]))
    summary_path = Path(args.summary_out) if args.summary_out is not None else Path(str(ss6_cfg["summary_path"]))
    auto_recompute = bool(exp_cfg.get("auto_recompute", True))
    recompute_exp, recompute_reason = _experimental_refresh_decision(
        cache_path=cache_path,
        exp_path=exp_path,
        auto_recompute=auto_recompute,
        force_experimental=bool(args.force_experimental),
        skip_experimental_recompute=bool(args.skip_experimental_recompute),
    )

    print(f"viewer cache: {cache_path}")
    print(f"experimental features path: {exp_path}")
    if recompute_exp:
        compute_all_experimental_features_from_config(
            cfg,
            cache_path=cache_path,
            out_path=exp_path,
            summary_path=Path(str(exp_cfg.get("summary_path", ""))) if str(exp_cfg.get("summary_path", "")).strip() else None,
        )
        print(f"experimental features status: recomputed ({recompute_reason})")
    else:
        print(f"experimental features status: reused ({recompute_reason})")
    print(f"ss6 decisions path: {out_path}")

    if not exp_path.exists():
        raise FileNotFoundError(f"Experimental features CSV not found after refresh step: {exp_path}")

    cache = load_viewer_cache(cache_path)
    rows = [dict(row) for row in cache.get("candidate_records", [])]
    extra_rows, extra_columns = load_extra_feature_rows(exp_path)
    join_info = join_extra_feature_rows(rows, extra_rows)

    outputs = [compute_ss6_row(row, ss6_cfg) for row in rows]
    write_feature_csv(out_path, outputs)
    generated_hist_files: list[str] = []
    if bool(ss6_cfg.get("save_histograms", True)):
        generated_hist_files = generate_ss6_histograms(outputs, ss6_cfg)

    branch_counts = Counter(str(row.get("ss6_branch", "")) for row in outputs)
    finite_metric_rows = sum(
        1
        for row in outputs
        if all(np.isfinite(_safe_float(row.get(key, np.nan))) for key in ("ss6_ss1", "ss6_pce", "ss6_edge", "ss6_eel", "ss6_resid"))
    )
    accepted = sum(int(row.get("ss6_accept", 0)) == 1 for row in outputs)
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
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

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


if __name__ == "__main__":
    main()
