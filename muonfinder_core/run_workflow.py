from __future__ import annotations

import argparse
import time
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    _THIS_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _THIS_DIR.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from muonfinder_core.compute_despike import run_compute_despike
    from muonfinder_core.compute_ss6_decisions import run_ss6_decisions
    from muonfinder_core.pipeline import execute_pipeline
else:
    from .compute_despike import run_compute_despike
    from .compute_ss6_decisions import run_ss6_decisions
    from .pipeline import execute_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full MuonFinder core workflow.")
    parser.add_argument("--config", type=Path, default=Path("config_core.json"))
    parser.add_argument("--force", action="store_true", help="Force experimental feature recompute during SS6 stage.")
    parser.add_argument("--skip-ss6", action="store_true", help="Skip SS6 decision stage.")
    parser.add_argument("--skip-despike", action="store_true", help="Skip despike stage.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    stage_timings: dict[str, float] = {}
    t_total = time.perf_counter()

    print(f"[workflow] config: {cfg_path}")

    print("[workflow] stage 1/3: pipeline")
    _artifacts, pipeline_timings = execute_pipeline(cfg_path, open_viewer=False)
    stage_timings["pipeline"] = float(pipeline_timings.get("pipeline", 0.0))
    stage_timings["viewer cache"] = float(pipeline_timings.get("viewer cache", 0.0))

    if not bool(args.skip_ss6):
        print("[workflow] stage 2/3: ss6")
        _ss6_summary, ss6_timings = run_ss6_decisions(
            config_path=cfg_path,
            force_experimental_recompute=bool(args.force),
        )
        stage_timings["ss6 decisions"] = float(ss6_timings.get("total", 0.0))
    else:
        print("[workflow] stage 2/3: ss6 skipped")
        stage_timings["ss6 decisions"] = 0.0

    if not bool(args.skip_despike):
        print("[workflow] stage 3/3: despike")
        _despike_artifacts, _despike_summary, despike_timings = run_compute_despike(config_path=cfg_path)
        stage_timings["despike"] = float(despike_timings.get("total", 0.0))
    else:
        print("[workflow] stage 3/3: despike skipped")
        stage_timings["despike"] = 0.0

    stage_timings["total"] = time.perf_counter() - t_total
    print("[workflow] done")
    print("Timing summary:")
    print(f"  pipeline: {stage_timings.get('pipeline', 0.0):.1f} s")
    print(f"  viewer cache: {stage_timings.get('viewer cache', 0.0):.1f} s")
    print(f"  ss6 decisions: {stage_timings.get('ss6 decisions', 0.0):.1f} s")
    print(f"  despike: {stage_timings.get('despike', 0.0):.1f} s")
    print(f"  total: {stage_timings.get('total', 0.0):.1f} s")


if __name__ == "__main__":
    main()
