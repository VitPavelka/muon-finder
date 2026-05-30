from __future__ import annotations

"""Compute standalone experimental feature tables from `viewer_cache.npz`.

Example:
`python -m muonfinder_core.compute_all_experimental_features --config muonfinder_core/config_core.json`
"""

import argparse
import json
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

    from muonfinder_core.cache import load_viewer_cache_light
    from muonfinder_core.config import load_config
    from muonfinder_core.experimental_common import (
        _safe_float,
        build_join_row,
        experimental_defaults,
        filter_candidate_rows,
        select_experimental_noise,
        write_feature_csv,
    )
    from muonfinder_core.experimental_edge_variants import compute_experimental_edge_variants
    from muonfinder_core.experimental_residual_pce import compute_experimental_residual_features
    from muonfinder_core.metrics import MetricComputationContext, robust_center_scale, sigmoid_support
else:
    from .cache import load_viewer_cache_light
    from .config import load_config
    from .experimental_common import (
        _safe_float,
        build_join_row,
        experimental_defaults,
        filter_candidate_rows,
        select_experimental_noise,
        write_feature_csv,
    )
    from .experimental_edge_variants import compute_experimental_edge_variants
    from .experimental_residual_pce import compute_experimental_residual_features
    from .metrics import MetricComputationContext, robust_center_scale, sigmoid_support


def _group_valid_count(rows: list[dict[str, object]], columns: list[str]) -> int:
    valid = 0
    for row in rows:
        if any(np.isfinite(_safe_float(row.get(col, np.nan))) for col in columns if row.get(col) not in ("", None)):
            valid += 1
    return int(valid)


def _resolve_modern_edge_width_field(rows: list[dict[str, object]]) -> tuple[str | None, np.ndarray]:
    # `core_recdw_sum_0_90` is only used as a fallback when candidate-like rows were
    # exported from a comparison table and the same raw width metric was namespaced.
    candidate_fields = ("recdw_sum_0_90", "core_recdw_sum_0_90")
    for field in candidate_fields:
        values = np.asarray([_safe_float(row.get(field, np.nan)) for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            return field, finite
    return None, np.asarray([], dtype=float)


def signed_edge_evidence_from_width_sum(value: float, center: float, scale: float, ctx: MetricComputationContext) -> float:
    if not (np.isfinite(value) and np.isfinite(center) and np.isfinite(scale) and float(scale) > 1e-12):
        return float("nan")
    z = float((float(value) - float(center)) / float(scale))
    support = sigmoid_support(z, float(ctx.recdw_support_z_scale), float(ctx.recdw_z_clip))
    return float(2.0 * support - 1.0)


def _assign_signed_edge_column(
    rows: list[dict[str, object]],
    *,
    raw_col: str,
    signed_col: str,
    center: float,
    scale: float,
    ctx: MetricComputationContext,
) -> None:
    for row in rows:
        raw_value = _safe_float(row.get(raw_col, np.nan))
        row[signed_col] = signed_edge_evidence_from_width_sum(raw_value, center, scale, ctx)


def _column_center_scale(rows: list[dict[str, object]], column: str) -> tuple[float, float]:
    values = np.asarray([_safe_float(row.get(column, np.nan)) for row in rows], dtype=float)
    return robust_center_scale(values)


def _finalize_edge_variant_evidence(
    rows: list[dict[str, object]],
    modern_center: float,
    modern_scale: float,
) -> dict[str, float]:
    ctx = MetricComputationContext()
    legacy_center, legacy_scale = _column_center_scale(rows, "exp_edge_legacy_width_sum_0_90")
    percent0_center, percent0_scale = _column_center_scale(rows, "exp_edge_percent_0_90_width_sum")
    percent5_center, percent5_scale = _column_center_scale(rows, "exp_edge_percent_5_90_width_sum")
    noise0_center, noise0_scale = _column_center_scale(rows, "exp_edge_noise_from_0_width_sum")
    noise1_center, noise1_scale = _column_center_scale(rows, "exp_edge_noise_from_1_width_sum")

    _assign_signed_edge_column(
        rows,
        raw_col="exp_edge_percent_0_90_width_sum",
        signed_col="exp_edge_percent_0_90_evidence_signed_selfnorm",
        center=percent0_center,
        scale=percent0_scale,
        ctx=ctx,
    )
    _assign_signed_edge_column(
        rows,
        raw_col="exp_edge_percent_0_90_width_sum",
        signed_col="exp_edge_percent_0_90_evidence_signed_modernnorm",
        center=modern_center,
        scale=modern_scale,
        ctx=ctx,
    )
    _assign_signed_edge_column(
        rows,
        raw_col="exp_edge_percent_5_90_width_sum",
        signed_col="exp_edge_percent_5_90_evidence_signed_selfnorm",
        center=percent5_center,
        scale=percent5_scale,
        ctx=ctx,
    )
    _assign_signed_edge_column(
        rows,
        raw_col="exp_edge_percent_5_90_width_sum",
        signed_col="exp_edge_percent_5_90_evidence_signed_modernnorm",
        center=modern_center,
        scale=modern_scale,
        ctx=ctx,
    )
    _assign_signed_edge_column(
        rows,
        raw_col="exp_edge_noise_from_0_width_sum",
        signed_col="exp_edge_noise_from_0_evidence_signed_selfnorm",
        center=noise0_center,
        scale=noise0_scale,
        ctx=ctx,
    )
    _assign_signed_edge_column(
        rows,
        raw_col="exp_edge_noise_from_0_width_sum",
        signed_col="exp_edge_noise_from_0_evidence_signed_modernnorm",
        center=modern_center,
        scale=modern_scale,
        ctx=ctx,
    )
    _assign_signed_edge_column(
        rows,
        raw_col="exp_edge_noise_from_1_width_sum",
        signed_col="exp_edge_noise_from_1_evidence_signed_selfnorm",
        center=noise1_center,
        scale=noise1_scale,
        ctx=ctx,
    )
    _assign_signed_edge_column(
        rows,
        raw_col="exp_edge_noise_from_1_width_sum",
        signed_col="exp_edge_noise_from_1_evidence_signed_modernnorm",
        center=modern_center,
        scale=modern_scale,
        ctx=ctx,
    )
    _assign_signed_edge_column(
        rows,
        raw_col="exp_edge_legacy_width_sum_0_90",
        signed_col="exp_edge_legacy_evidence_signed_selfnorm",
        center=legacy_center,
        scale=legacy_scale,
        ctx=ctx,
    )
    _assign_signed_edge_column(
        rows,
        raw_col="exp_edge_legacy_width_sum_0_90",
        signed_col="exp_edge_legacy_evidence_signed_modernnorm",
        center=modern_center,
        scale=modern_scale,
        ctx=ctx,
    )
    for row in rows:
        row["exp_edge_percent_0_90"] = row.get("exp_edge_percent_0_90_evidence_signed_modernnorm", np.nan)
        row["exp_edge_percent_5_90"] = row.get("exp_edge_percent_5_90_evidence_signed_modernnorm", np.nan)
        row["exp_edge_noise_from_0"] = row.get("exp_edge_noise_from_0_evidence_signed_selfnorm", np.nan)
        row["exp_edge_noise_from_1"] = row.get("exp_edge_noise_from_1_evidence_signed_selfnorm", np.nan)
        row["exp_edge_legacy_like"] = row.get("exp_edge_legacy_evidence_signed_modernnorm", np.nan)
        row["exp_edge_legacy_like_evidence_signed"] = row.get("exp_edge_legacy_evidence_signed_modernnorm", np.nan)
    return {
        "modern_edge_width_center": float(modern_center) if np.isfinite(modern_center) else float("nan"),
        "modern_edge_width_scale": float(modern_scale) if np.isfinite(modern_scale) else float("nan"),
        "legacy_edge_width_center": float(legacy_center) if np.isfinite(legacy_center) else float("nan"),
        "legacy_edge_width_scale": float(legacy_scale) if np.isfinite(legacy_scale) else float("nan"),
    }


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


def compute_all_experimental_features_from_config(
    cfg: object,
    *,
    cache_path: Path | None = None,
    out_path: Path | None = None,
    summary_path: Path | None = None,
    candidate_scope: str | None = None,
) -> dict[str, object]:
    exp_cfg = experimental_defaults(getattr(cfg, "experimental_features", {}))
    if candidate_scope is not None:
        exp_cfg["candidate_scope"] = str(candidate_scope)
    cache_path = Path(cache_path) if cache_path is not None else Path(str(cfg.paths["viewer_cache_path"]))
    out_path = Path(out_path) if out_path is not None else Path(str(exp_cfg["features_path"]))
    summary_path = Path(summary_path) if summary_path is not None else Path(str(exp_cfg["summary_path"]))

    print(f"[cache] experimental features cache path: {cache_path}")
    cache = load_viewer_cache_light(cache_path, verbose=True)
    all_rows = [dict(row) for row in cache.get("candidate_records", [])]
    used_rows, scope_stats = filter_candidate_rows(all_rows, str(exp_cfg["candidate_scope"]))
    spectra = np.asarray(cache["spectra"], dtype=float)
    modern_width_field, modern_width_values = _resolve_modern_edge_width_field(used_rows)
    modern_width_missing = int(len(used_rows) - int(modern_width_values.size))
    print(f"experimental features rows used: {len(used_rows)}")
    print(f"finite raw EDGE width rows: {int(modern_width_values.size)}")
    print(f"missing/nonfinite raw EDGE width rows: {modern_width_missing}")
    if modern_width_field is not None:
        print(f"raw EDGE width field used: {modern_width_field}")
    if modern_width_values.size == 0:
        raise ValueError(
            "No finite recdw_sum_0_90 values found in viewer_cache candidate records. "
            "Re-run the main pipeline or check whether EDGE raw width metric is stored under a different field name."
        )
    modern_center, modern_scale = robust_center_scale(modern_width_values)

    rows_out: list[dict[str, object]] = []
    missing_raw_signal = 0
    missing_edge_foot = 0
    invalid_prominence = 0
    missing_noise = 0
    resid_time = 0.0
    edge_time = 0.0
    total_rows = len(used_rows)
    for row in _progress_iter(used_rows, desc="Experimental features", total=total_rows):
        out_row: dict[str, object] = build_join_row(row)
        y = int(row.get("y", -1))
        x = int(row.get("x", -1))
        if not (0 <= y < spectra.shape[0] and 0 <= x < spectra.shape[1]):
            missing_raw_signal += 1
            out_row["exp_global_status"] = "missing_raw_signal"
            rows_out.append(out_row)
            continue
        raw = np.asarray(spectra[y, x, :], dtype=float)
        noise_info = select_experimental_noise(raw, row, str(exp_cfg.get("noise_source", "morph_range")))
        out_row.update(noise_info)
        noise_value = float(noise_info["exp_noise_value"])
        if not np.isfinite(noise_value) or noise_value <= 0.0:
            missing_noise += 1

        resid_status = ""
        if bool(dict(exp_cfg.get("residual_pce", {})).get("enabled", False)) or bool(dict(exp_cfg.get("residual_threshold", {})).get("enabled", True)):
            t_part = time.perf_counter()
            resid_features = compute_experimental_residual_features(raw, row, noise_value, exp_cfg)
            out_row.update(resid_features)
            resid_status = str(resid_features.get("exp_resid3_status", ""))
            resid_time += time.perf_counter() - t_part

        edge_status = ""
        if bool(dict(exp_cfg.get("edge_variants", {})).get("enabled", True)):
            t_part = time.perf_counter()
            edge_features = compute_experimental_edge_variants(raw, row, noise_value, exp_cfg)
            out_row.update(edge_features)
            edge_status = str(edge_features.get("exp_edge_variants_status", ""))
            edge_time += time.perf_counter() - t_part

        if "missing_raw_signal" in {resid_status, edge_status}:
            missing_raw_signal += 1
        if edge_status == "missing_edge_foot":
            missing_edge_foot += 1
        if edge_status == "invalid_prominence":
            invalid_prominence += 1
        out_row["exp_global_status"] = "ok"
        if not np.isfinite(noise_value) or noise_value <= 0.0:
            out_row["exp_global_status"] = "missing_noise"
        elif edge_status == "missing_edge_foot":
            out_row["exp_global_status"] = "missing_edge_foot"
        elif edge_status == "invalid_prominence":
            out_row["exp_global_status"] = "invalid_prominence"
        elif resid_status == "missing_raw_signal":
            out_row["exp_global_status"] = "missing_raw_signal"
        rows_out.append(out_row)

    edge_norm_stats = _finalize_edge_variant_evidence(rows_out, modern_center, modern_scale)
    write_feature_csv(out_path, rows_out)
    feature_names = sorted({key for row in rows_out for key in row.keys() if key.startswith("exp_")})
    valid_rows_per_group = {
        "residual_threshold": _group_valid_count(rows_out, ["exp_resid3_height_noise_z", "exp_resid3_above_3noise"]),
        "edge_variants": _group_valid_count(
            rows_out,
            [
                "exp_edge_percent_0_90_width_sum",
                "exp_edge_percent_0_90_evidence_signed_modernnorm",
                "exp_edge_noise_from_0_width_sum",
                "exp_edge_noise_from_0_evidence_signed_selfnorm",
                "exp_edge_noise_from_1_width_sum",
                "exp_edge_noise_from_1_evidence_signed_selfnorm",
                "exp_edge_legacy_width_sum_0_90",
                "exp_edge_legacy_evidence_signed_modernnorm",
            ],
        ),
    }
    summary = {
        "total_candidates_loaded": int(scope_stats["total_loaded_candidates"]),
        "rows_skipped_by_noise_filter": int(scope_stats["total_loaded_candidates"]) - int(scope_stats["candidates_used"]),
        "rows_used": int(scope_stats["candidates_used"]),
        "candidate_scope": str(scope_stats["candidate_scope"]),
        "valid_rows_per_feature_group": valid_rows_per_group,
        "feature_names": feature_names,
        "noise_source": str(exp_cfg.get("noise_source", "morph_range")),
        "output_path": str(out_path),
        "summary_path": str(summary_path),
        "missing_raw_signal_count": int(missing_raw_signal),
        "missing_edge_foot_count": int(missing_edge_foot),
        "invalid_prominence_count": int(invalid_prominence),
        "missing_noise_count": int(missing_noise),
        "candidates_missing_noise_status": int(scope_stats.get("candidates_missing_noise_status", 0)),
        "modern_edge_width_field_used": str(modern_width_field),
        "modern_edge_width_rows_finite": int(modern_width_values.size),
        "modern_edge_width_rows_missing_or_nonfinite": int(modern_width_missing),
        "modern_edge_width_center": edge_norm_stats["modern_edge_width_center"],
        "modern_edge_width_scale": edge_norm_stats["modern_edge_width_scale"],
        "legacy_edge_width_center": edge_norm_stats["legacy_edge_width_center"],
        "legacy_edge_width_scale": edge_norm_stats["legacy_edge_width_scale"],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"experimental residual time: {resid_time:.1f} s")
    print(f"experimental edge_variants time: {edge_time:.1f} s")

    return {
        "cache_path": cache_path,
        "out_path": out_path,
        "summary_path": summary_path,
        "summary": summary,
        "scope_stats": scope_stats,
        "valid_rows_per_group": valid_rows_per_group,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute standalone experimental feature tables from viewer cache.")
    parser.add_argument("--config", type=Path, default=Path("config_core.json"), help="Core config; used to resolve viewer_cache_path and experimental output paths.")
    parser.add_argument("--cache", type=Path, default=None, help="Optional explicit viewer cache path.")
    parser.add_argument("--out", type=Path, default=None, help="Optional experimental CSV output path.")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional experimental summary JSON output path.")
    parser.add_argument("--candidate-scope", choices=["all", "noise-kept"], default=None, help="Choose whether experimental features are computed for all rows or only noise-kept rows.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = compute_all_experimental_features_from_config(
        cfg,
        cache_path=(Path(args.cache) if args.cache is not None else None),
        out_path=(Path(args.out) if args.out is not None else None),
        summary_path=(Path(args.summary_out) if args.summary_out is not None else None),
        candidate_scope=args.candidate_scope,
    )
    scope_stats = dict(result["scope_stats"])
    valid_rows_per_group = dict(result["valid_rows_per_group"])
    cache_path = Path(result["cache_path"])
    out_path = Path(result["out_path"])
    summary_path = Path(result["summary_path"])

    print(f"viewer cache: {cache_path}")
    print(f"candidate scope: {scope_stats['candidate_scope']}")
    print(f"total candidate rows loaded: {scope_stats['total_loaded_candidates']}")
    print(f"rejected by noise prefilter: {scope_stats['candidates_rejected_by_noise_filter']}")
    print(f"used for experimental features: {scope_stats['candidates_used']}")
    print(f"valid residual_threshold rows: {valid_rows_per_group['residual_threshold']}")
    print(f"valid edge_variants rows: {valid_rows_per_group['edge_variants']}")
    if int(scope_stats.get("candidates_missing_noise_status", 0)) > 0:
        print(f"warning: candidates missing noise status field: {scope_stats['candidates_missing_noise_status']}")
    print(f"experimental features: {out_path}")
    print(f"experimental summary: {summary_path}")


if __name__ == "__main__":
    main()
