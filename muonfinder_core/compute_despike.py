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
else:
    from .config import load_config
    from .despike import compute_despike_from_cache_and_ss6


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
    summary_path = Path(args.summary_out) if args.summary_out is not None else Path(str(despike_cfg["summary_path"]))

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
        summary_path=summary_path,
        max_iterations=int(despike_cfg.get("max_iterations", 4)),
        morph_windows=[int(v) for v in despike_cfg.get("morph_windows", [3, 5])],
        max_cell_width_pts=int(despike_cfg.get("max_cell_width_pts", 10)),
        max_half_width_pts=int(despike_cfg.get("max_half_width_pts", 5)),
        min_height_above_chord_noise_z=float(despike_cfg.get("min_height_above_chord_noise_z", 3.0)),
        anchor_overshoot_noise_factor=float(despike_cfg.get("anchor_overshoot_noise_factor", 0.5)),
        allow_spectrum_edge_anchors=bool(despike_cfg.get("allow_spectrum_edge_anchors", False)),
        recheck_enabled=bool(despike_cfg.get("recheck_enabled", True)),
        recheck_context_pad_pts=int(despike_cfg.get("recheck_context_pad_pts", 3)),
        skip_overlapping_corrections=bool(despike_cfg.get("skip_overlapping_corrections", True)),
        progress_iter=_progress_iter,
        timings_out=inner_timings,
    )
    timings.update(inner_timings)

    summary = dict(artifacts.summary)
    print(f"despike corrected: {corrected_path}")
    print(f"despike debug: {debug_path}")
    print(f"despike summary: {summary_path}")
    print(f"accepted ss6 spikes: {summary['accepted_ss6_spikes']}")
    print(f"spectra with accepted spikes: {summary['spectra_with_accepted_spikes']}")
    print(f"corrected spikes: {summary['corrected_spikes']}")
    print(f"skipped candidates: {summary['skipped_candidates']}")
    print(f"repeated corrections: {summary['repeated_corrections']}")
    print(f"max_correction_width_pts: {summary['max_correction_width_pts']}")
    print(f"max_iterations_used: {summary['max_iterations_used']}")
    print(f"max_iteration_reached_count: {summary['max_iteration_reached_count']}")
    print(f"skipped_no_parent_contact_cell: {summary['skipped_no_parent_contact_cell']}")
    print(f"skipped_cell_too_wide: {summary['skipped_cell_too_wide']}")
    print(f"skipped_chord_overshoots_signal: {summary['skipped_chord_overshoots_signal']}")
    print(f"skipped_missing_noise: {summary['skipped_missing_noise']}")
    print(f"skipped_overlaps_previous_correction: {summary['skipped_overlaps_previous_correction']}")
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
