from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

if __package__ in {None, ""}:
    import sys

    _THIS_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _THIS_DIR.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from muonfinder_core.cache import load_viewer_cache
    from muonfinder_core.config import load_config
    from muonfinder_core.utils import flatten_dict_keys, human_text_key
else:
    from .cache import load_viewer_cache
    from .config import load_config
    from .utils import flatten_dict_keys, human_text_key


AUC_SORT_KEYS = (
    "auc",
    "auc_feature_oriented",
    "auc_feat_oriented",
    "auc_muon_vs_rest",
    "auc_raman_vs_rest",
    "auc_noise_vs_rest",
    "auc_muon_vs_raman",
    "auc_muon_vs_noise",
    "auc_raman_vs_noise",
    "macro_ovr_auc",
    "macro_pairwise_auc",
    "best_pairwise_auc",
    "worst_pairwise_auc",
    "raman_veto_auc",
    "noise_filter_auc",
    "muon_detector_auc",
)

KEYWORD_MAP = {
    "pce": ("pce_", "peak_curvature"),
    "ss1": ("spike_score_v1", "rise_slope_z", "fall_slope_z"),
    "ss4": ("ss4", "primary_ss4"),
    "ss5": ("ss5", "primary_ss5"),
    "edge": ("recdw_", "edge_", "raw_edge_"),
    "noise": ("candidate_noise_", "noise_"),
}


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _fmt(value: object, digits: int = 4) -> str:
    v = _safe_float(value)
    if np.isfinite(v):
        return f"{v:.{digits}f}"
    if value is None:
        return "nan"
    return str(value)


def _load_labels(path: Path) -> tuple[dict[tuple[int, int, int], int], dict[tuple[int, int, int], str]]:
    binary: dict[tuple[int, int, int], int] = {}
    ternary: dict[tuple[int, int, int], str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y = int(row.get("y", -1))
            x = int(row.get("x", -1))
            peak = int(row.get("peak_index", -1))
            key = (y, x, peak)
            cls = str(row.get("label_class", "")).strip().lower()
            if cls:
                ternary[key] = cls
            is_muon = str(row.get("is_muon", "")).strip()
            if is_muon in {"0", "1"}:
                binary[key] = int(is_muon)
    return binary, ternary


def _row_label_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(row.get("source_y", row.get("y", -1))), int(row.get("source_x", row.get("x", -1))), int(row.get("peak_index", -1)))


def _auc_binary(values: list[float], labels: list[int]) -> float:
    vals = np.asarray(values, dtype=float)
    lab = np.asarray(labels, dtype=int)
    mask = np.isfinite(vals)
    vals = vals[mask]
    lab = lab[mask]
    pos = vals[lab == 1]
    neg = vals[lab == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = 0.0
    ties = 0.0
    for pv in pos:
        wins += float(np.sum(pv > neg))
        ties += float(np.sum(pv == neg))
    return float((wins + 0.5 * ties) / (pos.size * neg.size))


def _auc_pairwise(values: list[float], labels: list[str], pos_label: str, neg_label: str) -> float:
    use_vals = []
    use_lab = []
    for value, label in zip(values, labels):
        if label == pos_label:
            use_vals.append(value)
            use_lab.append(1)
        elif label == neg_label:
            use_vals.append(value)
            use_lab.append(0)
    return _auc_binary(use_vals, use_lab)


def _resolve_feature_keyword(names: list[str], token: str) -> list[str]:
    token_norm = str(token).strip().lower()
    if token_norm in KEYWORD_MAP:
        prefixes = KEYWORD_MAP[token_norm]
        return [name for name in names if any(name.lower().startswith(prefix) or prefix in name.lower() for prefix in prefixes)]
    return [token]


def _selected_features(names: list[str], explicit: list[str], features_file: Path | None) -> list[str]:
    wanted = list(explicit)
    if features_file is not None:
        wanted.extend(
            line.strip()
            for line in features_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if not wanted:
        raise ValueError("Provide --features and/or --features-file.")
    selected: list[str] = []
    seen = set()
    for token in wanted:
        for name in _resolve_feature_keyword(names, token):
            if name in names and name not in seen:
                seen.add(name)
                selected.append(name)
    if not selected:
        raise ValueError("None of the requested features were found in the current core cache.")
    return selected


def _feature_row(name: str, rows: list[dict[str, Any]], binary_labels: dict[tuple[int, int, int], int], ternary_labels: dict[tuple[int, int, int], str]) -> dict[str, Any]:
    values = []
    labels_bin = []
    labels_ter = []
    for row in rows:
        value = _safe_float(row.get(name))
        key = _row_label_key(row)
        if key in binary_labels:
            values.append(value)
            labels_bin.append(int(binary_labels[key]))
            labels_ter.append(str(ternary_labels.get(key, "unknown")))
    auc_bin = _auc_binary(values, labels_bin)
    auc_mr = _auc_pairwise(values, labels_ter, "muon", "raman")
    auc_mn = _auc_pairwise(values, labels_ter, "muon", "noise")
    auc_rn = _auc_pairwise(values, labels_ter, "raman", "noise")
    macro_pair = float(np.nanmean([auc_mr, auc_mn, auc_rn]))
    return {
        "feature": name,
        "auc": auc_bin,
        "auc_feature_oriented": auc_bin,
        "auc_feat_oriented": auc_bin,
        "auc_muon_vs_rest": auc_bin,
        "auc_raman_vs_rest": _auc_pairwise(values, labels_ter, "raman", "muon"),
        "auc_noise_vs_rest": _auc_pairwise(values, labels_ter, "noise", "muon"),
        "auc_muon_vs_raman": auc_mr,
        "auc_muon_vs_noise": auc_mn,
        "auc_raman_vs_noise": auc_rn,
        "macro_pairwise_auc": macro_pair,
        "macro_ovr_auc": float(np.nanmean([auc_bin, _auc_pairwise(values, labels_ter, "raman", "muon"), _auc_pairwise(values, labels_ter, "noise", "muon")])),
        "best_pairwise_auc": float(np.nanmax([auc_mr, auc_mn, auc_rn])) if np.any(np.isfinite([auc_mr, auc_mn, auc_rn])) else float("nan"),
        "worst_pairwise_auc": float(np.nanmin([auc_mr, auc_mn, auc_rn])) if np.any(np.isfinite([auc_mr, auc_mn, auc_rn])) else float("nan"),
        "raman_veto_auc": auc_mr,
        "noise_filter_auc": auc_mn,
        "muon_detector_auc": auc_bin,
        "n_labeled": int(len(labels_bin)),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def _short_label(name: str, limit: int = 20) -> str:
    if len(name) <= limit:
        return name
    return name[: limit - 3] + "..."


def _print_table(rows: list[dict[str, Any]], columns: list[str], title: str) -> None:
    print()
    print(title)
    if not rows:
        print("  <empty>")
        return
    widths = {col: max(len(col), max(len(_fmt(row.get(col))) for row in rows)) for col in columns}
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(_fmt(row.get(col)).ljust(widths[col]) for col in columns))


def _build_corr_matrices(selected: list[str], rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    mat_rows = []
    for name in selected:
        vals = np.asarray([_safe_float(row.get(name)) for row in rows], dtype=float)
        if not np.any(np.isfinite(vals)):
            vals = np.zeros_like(vals)
        fill = float(np.nanmedian(vals[np.isfinite(vals)])) if np.any(np.isfinite(vals)) else 0.0
        mat_rows.append(np.nan_to_num(vals, nan=fill))
    matrix = np.vstack(mat_rows) if mat_rows else np.zeros((0, 0))
    pearson = np.corrcoef(matrix) if matrix.size else np.zeros((0, 0))
    ranked = np.vstack([_rankdata(row) for row in matrix]) if matrix.size else np.zeros((0, 0))
    spearman = np.corrcoef(ranked) if ranked.size else np.zeros((0, 0))
    return pearson, spearman


def _show_matrix(selected: list[str], rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]], auc_key: str, corr_type: str) -> None:
    pearson, spearman = _build_corr_matrices(selected, rows)
    auc_by_feature = {str(row["feature"]): _safe_float(row.get(auc_key)) for row in feature_rows}
    short_labels = [_short_label(name) for name in selected]
    if corr_type == "both":
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
        mats = [("Pearson", pearson), ("Spearman", spearman)]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7.4, 6.5), constrained_layout=True)
        axes = [ax]
        mats = [(corr_type.capitalize(), pearson if corr_type == "pearson" else spearman)]
    annot = axes[0].annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points", bbox={"boxstyle": "round", "fc": "w", "alpha": 0.96})
    annot.set_visible(False)
    artists = []
    norm = Normalize(vmin=-1.0, vmax=1.0)
    for ax, (title, corr) in zip(axes, mats):
        im = ax.imshow(corr, cmap="coolwarm", norm=norm, interpolation="nearest", aspect="auto")
        ax.set_xticks(np.arange(len(short_labels)))
        ax.set_yticks(np.arange(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=90, fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_title(f"{title} correlation")
        artists.append((ax, im, title, corr))
    fig.colorbar(artists[0][1], ax=axes, fraction=0.046, pad=0.04)

    def on_move(event) -> None:
        shown = False
        for ax, _im, title, corr in artists:
            if event.inaxes is not ax or event.xdata is None or event.ydata is None:
                continue
            j = int(np.clip(round(event.xdata), 0, len(selected) - 1))
            i = int(np.clip(round(event.ydata), 0, len(selected) - 1))
            annot.xy = (j, i)
            annot.set_text(
                "\n".join(
                    [
                        f"A={selected[i]}",
                        f"B={selected[j]}",
                        f"Pearson={pearson[i, j]:.4f}",
                        f"Spearman={spearman[i, j]:.4f}",
                        f"A {auc_key}={auc_by_feature.get(selected[i], float('nan')):.4f}",
                        f"B {auc_key}={auc_by_feature.get(selected[j], float('nan')):.4f}",
                    ]
                )
            )
            annot.set_visible(True)
            shown = True
            break
        if not shown and annot.get_visible():
            annot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a manually selected feature subset from core metric rows.")
    parser.add_argument("--config", type=Path, default=Path("config_core.json"), help="Core config; used to resolve viewer_cache_path and labels_csv.")
    parser.add_argument("--corr-json", type=Path, default=None, help="Accepted for compatibility; core mode reads current metrics from viewer cache.")
    parser.add_argument("--features", nargs="*", default=[], help="Explicit feature names or supported keywords (currently: pce, ss1, ss4, ss5, edge, noise).")
    parser.add_argument("--features-file", type=Path, default=None, help="Text file with one feature name or supported keyword per line.")
    parser.add_argument("--corr-type", choices=["pearson", "spearman", "both"], default="both", help="Which correlation heatmap to display.")
    parser.add_argument("--sort-by", choices=["input", *AUC_SORT_KEYS], default="input", help="How to order selected features.")
    parser.add_argument("--save-csv", type=Path, default=None, help="Optional flat CSV export.")
    parser.add_argument("--save-json", type=Path, default=None, help="Optional JSON export.")
    parser.add_argument("--auc-min", type=float, default=None, help="Keep only selected features whose selected AUC key is >= this value.")
    parser.add_argument("--auc-min-key", choices=AUC_SORT_KEYS, default=None, help="AUC column used by --auc-min.")
    parser.add_argument("--hide-pairwise-summary", action="store_true", help="Do not print the Pairwise Summary table in the console.")
    parser.add_argument("--label-mode", choices=["binary", "ternary"], default="binary", help="Read binary or ternary labels from labels.csv.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache = load_viewer_cache(Path(cfg.paths["viewer_cache_path"]))
    rows = [dict(row) for row in cache.get("candidate_records", [])]
    names = [name for name in flatten_dict_keys(rows) if not name.endswith("_debug") and all(ch not in name for ch in ("source_record_origin", "candidate_id", "parent_id"))]
    selected = _selected_features(names, list(args.features), args.features_file)
    binary_labels, ternary_labels = _load_labels(Path(cfg.paths["labels_csv"]))

    feature_rows = [_feature_row(name, rows, binary_labels, ternary_labels) for name in selected]
    if args.auc_min is not None:
        auc_key = str(args.auc_min_key or ("auc_feature_oriented" if args.label_mode == "binary" else "macro_pairwise_auc"))
        feature_rows = [row for row in feature_rows if np.isfinite(_safe_float(row.get(auc_key))) and _safe_float(row.get(auc_key)) >= float(args.auc_min)]
        selected = [row["feature"] for row in feature_rows]
    if args.sort_by != "input":
        feature_rows.sort(key=lambda row: _safe_float(row.get(args.sort_by), float("-inf")), reverse=True)
        selected = [row["feature"] for row in feature_rows]
    summary_cols = ["feature", "auc_feature_oriented", "macro_pairwise_auc", "auc_muon_vs_raman", "auc_muon_vs_noise", "auc_raman_vs_noise", "n_labeled"]
    _print_table(feature_rows, summary_cols, "Feature Summary")

    pair_rows: list[dict[str, Any]] = []
    for i, fa in enumerate(selected):
        va = np.asarray([_safe_float(row.get(fa)) for row in rows], dtype=float)
        for fb in selected[i + 1 :]:
            vb = np.asarray([_safe_float(row.get(fb)) for row in rows], dtype=float)
            mask = np.isfinite(va) & np.isfinite(vb)
            pear = float(np.corrcoef(va[mask], vb[mask])[0, 1]) if np.sum(mask) >= 3 else float("nan")
            spear = float(np.corrcoef(_rankdata(va[mask]), _rankdata(vb[mask]))[0, 1]) if np.sum(mask) >= 3 else float("nan")
            pair_rows.append({"feature_a": fa, "feature_b": fb, "pearson_pair": pear, "spearman_pair": spear})
    if not args.hide_pairwise_summary:
        _print_table(pair_rows[: min(32, len(pair_rows))], ["feature_a", "feature_b", "pearson_pair", "spearman_pair"], "Pairwise Summary")

    if args.save_json is not None:
        args.save_json.write_text(json.dumps({"feature_summary": feature_rows, "pairwise_summary": pair_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.save_csv is not None:
        with args.save_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_cols)
            writer.writeheader()
            for row in feature_rows:
                writer.writerow({key: row.get(key) for key in summary_cols})

    if selected:
        heat_auc_key = str(args.auc_min_key or ("auc_feature_oriented" if args.label_mode == "binary" else "macro_pairwise_auc"))
        _show_matrix(selected, rows, feature_rows, heat_auc_key, args.corr_type)


if __name__ == "__main__":
    main()
