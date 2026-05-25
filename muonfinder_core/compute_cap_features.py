from __future__ import annotations

"""Compute standalone experimental CAP v3 features from `viewer_cache.npz`.

Example:
`python -m muonfinder_core.compute_cap_features --config muonfinder_core/config_core.json`
"""

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    _THIS_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _THIS_DIR.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from muonfinder_core.cache import load_viewer_cache
    from muonfinder_core.cap_metrics import (
        cap_defaults,
        compute_cap_feature_rows,
        filter_cap_candidate_rows,
        summarize_cap_rows,
        write_cap_csv,
    )
    from muonfinder_core.config import load_config
else:
    from .cache import load_viewer_cache
    from .cap_metrics import (
        cap_defaults,
        compute_cap_feature_rows,
        filter_cap_candidate_rows,
        summarize_cap_rows,
        write_cap_csv,
    )
    from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute standalone experimental CAP v3 features from viewer cache.")
    parser.add_argument("--config", type=Path, default=Path("config_core.json"), help="Core config; used to resolve viewer_cache_path and CAP output paths.")
    parser.add_argument("--cache", type=Path, default=None, help="Optional explicit viewer cache path.")
    parser.add_argument("--out", type=Path, default=None, help="Optional CAP CSV output path.")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional CAP summary JSON output path.")
    parser.add_argument("--signal-source", choices=["raw", "corrected"], default=None, help="Override CAP signal source.")
    parser.add_argument("--candidate-scope", choices=["all", "noise-kept"], default=None, help="Choose whether CAP is computed for all rows or only noise-kept rows.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cap_cfg = cap_defaults(getattr(cfg, "cap", {}))
    if args.signal_source is not None:
        cap_cfg["signal_source"] = str(args.signal_source)
    if args.candidate_scope is not None:
        cap_cfg["candidate_scope"] = str(args.candidate_scope)
    cache_path = Path(args.cache) if args.cache is not None else Path(str(cfg.paths["viewer_cache_path"]))
    out_path = Path(args.out) if args.out is not None else Path(str(cap_cfg["features_path"]))
    summary_path = Path(args.summary_out) if args.summary_out is not None else Path(str(cap_cfg["summary_path"]))

    cache = load_viewer_cache(cache_path)
    all_rows = [dict(row) for row in cache.get("candidate_records", [])]
    cap_rows_input, scope_stats = filter_cap_candidate_rows(all_rows, str(cap_cfg["candidate_scope"]))
    cap_rows = compute_cap_feature_rows(cache, cap_rows_input, cap_cfg)
    summary = summarize_cap_rows(cap_rows, scope_stats, cap_cfg)
    summary.update(
        {
            "cache_path": str(cache_path),
            "output_path": str(out_path),
            "summary_path": str(summary_path),
            "signal_source": str(cap_cfg["signal_source"]),
        }
    )

    write_cap_csv(out_path, cap_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"viewer cache: {cache_path}")
    print(f"candidate scope: {scope_stats['candidate_scope']}")
    print(f"total candidate rows loaded: {scope_stats['total_loaded_candidates']}")
    print(f"rejected by noise prefilter: {scope_stats['candidates_rejected_by_noise_filter']}")
    print(f"used for CAP: {scope_stats['candidates_used']}")
    print(f"valid CAP rows: {summary['valid_cap_rows']}")
    if int(scope_stats.get("candidates_missing_noise_status", 0)) > 0:
        print(f"warning: candidates missing noise status field: {scope_stats['candidates_missing_noise_status']}")
    print(f"cap features: {out_path}")
    print(f"cap summary: {summary_path}")


if __name__ == "__main__":
    main()
