from __future__ import annotations

import argparse
import time
from pathlib import Path

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

    from muonfinder_core.config import load_config
    from muonfinder_core.despike import compute_despike_from_cache_and_ss6
    from muonfinder_core.metrics import MetricComputationContext
else:
    from .config import load_config
    from .despike import compute_despike_from_cache_and_ss6
    from .metrics import MetricComputationContext


def _progress_iter(items):
    if tqdm is not None:
        return tqdm(items, desc="Despike spectra", total=len(items), dynamic_ncols=True, mininterval=0.25)

    def _fallback():
        next_print = 25
        for idx, item in enumerate(items, start=1):
            if idx == 1 or idx >= next_print:
                print(f"despike spectra: {idx}/{len(items)}")
                next_print += 25
            yield item

    return _fallback()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SS6-based despike correction from viewer cache and ss6_decisions.csv.")
    parser.add_argument("--config", type=Path, default=Path("config_core.json"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--ss6", type=Path, default=None)
    parser.add_argument("--corrected-out", type=Path, default=None)
    parser.add_argument("--debug-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    timings["load config"] = time.perf_counter() - t0
    despike_cfg = dict(getattr(cfg, "despike", {}))
    cache_path = Path(args.cache) if args.cache is not None else Path(str(cfg.paths["viewer_cache_path"]))
    ss6_path = Path(args.ss6) if args.ss6 is not None else Path(str(cfg.ss6["decisions_path"]))
    corrected_path = Path(args.corrected_out) if args.corrected_out is not None else Path(str(despike_cfg["corrected_path"]))
    debug_path = Path(args.debug_out) if args.debug_out is not None else Path(str(despike_cfg["debug_path"]))
    attempts_path = Path(str(despike_cfg["attempts_path"]))
    summary_path = Path(args.summary_out) if args.summary_out is not None else Path(str(despike_cfg["summary_path"]))

    noise_cfg = dict(getattr(cfg, "noise", {}))
    metric_ctx = MetricComputationContext(
        noise_source=str(noise_cfg.get("noise_source", "morph_range")),
        noise_height_factor=float(noise_cfg.get("noise_height_factor", 3.0)),
        edge_foot_method=str(noise_cfg.get("edge_foot_method", "noise_quantized_component")),
        edge_pre_level_step_noise=float(noise_cfg.get("edge_pre_level_step_noise", 1.0)),
        edge_pre_transient_tolerance_levels=int(noise_cfg.get("edge_pre_transient_tolerance_levels", 2)),
        edge_pre_min_stable_levels=int(noise_cfg.get("edge_pre_min_stable_levels", 2)),
        edge_neighbor_structure_factor=float(noise_cfg.get("edge_neighbor_structure_factor", 3.0)),
        edge_dense_context_min_pad_pts=int(noise_cfg.get("edge_dense_context_min_pad_pts", 10)),
        edge_dense_context_pad_pts=int(noise_cfg.get("edge_dense_context_pad_pts", 20)),
        edge_dense_context_max_pad_pts=int(noise_cfg.get("edge_dense_context_max_pad_pts", 120)),
        edge_context_expand_step_pts=int(noise_cfg.get("edge_context_expand_step_pts", 10)),
    )

    print(f"viewer cache: {cache_path}")
    print(f"ss6 decisions: {ss6_path}")
    print(f"despike source: {despike_cfg.get('source', 'ss6')}")

    inner_timings: dict[str, float] = {}
    print("load cache...")
    print("load ss6 decisions...")
    print("group accepted candidates...")
    print("despike correction...")
    artifacts = compute_despike_from_cache_and_ss6(
        cache_path=cache_path,
        ss6_path=ss6_path,
        corrected_path=corrected_path,
        debug_path=debug_path,
        attempts_path=attempts_path,
        summary_path=summary_path,
        morph_window=int(despike_cfg.get("morph_window", 3)),
        despike_context_window_pad=int(despike_cfg.get("despike_context_window_pad", 0)),
        noise_height_factor=float(despike_cfg.get("noise_height_factor", 3.0)),
        max_iterations=int(despike_cfg.get("max_iterations", 1000)),
        ss6_config=dict(getattr(cfg, "ss6", {})),
        metric_context=metric_ctx,
        progress_iter=_progress_iter,
        timings_out=inner_timings,
    )
    timings.update(inner_timings)

    summary = dict(artifacts.summary)
    print(f"despike corrected: {corrected_path}")
    print(f"despike debug: {debug_path}")
    print(f"despike attempts: {attempts_path}")
    print(f"despike summary: {summary_path}")
    print(f"accepted ss6 parent candidates: {summary['accepted_ss6_parent_candidates']}")
    print(f"spectra with accepted spikes: {summary['spectra_with_accepted_spikes']}")
    print(f"parent corrected: {summary['parent_corrected']}")
    print(f"parent skipped below noise height: {summary['parent_skipped_below_noise_height']}")
    print(f"parent skipped no erosion neighbors: {summary['parent_skipped_no_erosion_neighbors']}")
    print(f"local candidates from dilation contacts: {summary['local_candidates_from_dilation_contacts']}")
    print(f"local candidates rejected by noise height: {summary['local_candidates_rejected_by_noise_height']}")
    print(f"local candidates sent to ss6: {summary['local_candidates_sent_to_ss6']}")
    print(f"local candidates accepted by ss6: {summary['local_candidates_accepted_by_ss6']}")
    print(f"local candidates corrected: {summary['local_candidates_corrected']}")
    print(f"total corrections applied: {summary['total_corrections_applied']}")
    print(f"technical failures: {summary['technical_failures']}")
    print(f"max pass index used: {summary['max_pass_index_used']}")
    print("Timing summary:")
    print(f"  load config: {timings.get('load config', 0.0):.1f} s")
    print(f"  load cache: {timings.get('load cache', 0.0):.1f} s")
    print(f"  load ss6 decisions: {timings.get('load ss6 decisions', 0.0):.1f} s")
    print(f"  group accepted candidates: {timings.get('group accepted candidates', 0.0):.1f} s")
    print(f"  despike correction: {timings.get('despike correction', 0.0):.1f} s")
    print(f"  write outputs: {timings.get('write outputs', 0.0):.1f} s")
    print(f"  total: {time.perf_counter() - t_total:.1f} s")


if __name__ == "__main__":
    main()
