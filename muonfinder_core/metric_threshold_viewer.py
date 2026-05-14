from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import numpy as np

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


KEYWORD_MAP = {
    "pce": ("pce_negpref_t098_evidence_signed", "pce_", "peak_curvature"),
    "ss1": ("spike_score_v1",),
    "ss4": ("ss4",),
    "ss5": ("ss5",),
    "edge": ("recdw_sum_0_90_raman_veto_evidence_signed", "recdw_sum_0_90_support01", "recdw_sum_0_90_z", "recdw_", "edge_", "raw_edge_"),
    "noise": ("candidate_noise_", "noise_"),
}

TERNARY_COLORS = {
    "muon": "#d62728",
    "raman": "#1f77b4",
    "noise": "#7f7f7f",
    "unknown": "#9467bd",
}


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    return v if np.isfinite(v) else default


def _load_labels(path: Path) -> tuple[dict[tuple[int, int, int], int], dict[tuple[int, int, int], str]]:
    binary: dict[tuple[int, int, int], int] = {}
    ternary: dict[tuple[int, int, int], str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["y"]), int(row["x"]), int(row["peak_index"]))
            is_muon = str(row.get("is_muon", "")).strip()
            if is_muon in {"0", "1"}:
                binary[key] = int(is_muon)
            cls = str(row.get("label_class", "")).strip().lower()
            if cls:
                ternary[key] = cls
    return binary, ternary


def _resolve_metric(names: list[str], metric: str) -> str:
    token = str(metric).strip()
    if token in names:
        return token
    key = token.lower()
    if key in KEYWORD_MAP:
        prefixes = KEYWORD_MAP[key]
        exact = [name for name in names if any(name.lower() == prefix for prefix in prefixes)]
        if exact:
            exact.sort(key=human_text_key)
            best = exact[0]
            print(f"[metric keyword] {metric} -> {best}")
            return best
        matches = [
            name
            for name in names
            if not name.endswith("_debug")
            and any(name.lower() == prefix or name.lower().startswith(prefix) or prefix in name.lower() for prefix in prefixes)
        ]
        if not matches:
            raise ValueError(f"No current core features matched keyword {metric!r}.")
        matches.sort(key=human_text_key)
        best = matches[0]
        print(f"[metric keyword] {metric} -> {best}")
        return best
    raise ValueError(f"Metric not found: {metric}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive metric-threshold viewer for core candidate peaks")
    parser.add_argument("--config", type=Path, default=Path("config_core.json"), help="Core config; used to resolve viewer_cache_path and labels_csv.")
    parser.add_argument("--metric", type=str, required=True, help="Metric/feature in candidate rows.")
    parser.add_argument("--threshold-low", type=float, default=None, help="Lower threshold guide.")
    parser.add_argument("--threshold-high", type=float, default=None, help="Upper threshold guide.")
    parser.add_argument("--label-mode", choices=["binary", "ternary"], default="binary", help="Use binary or ternary labels.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache = load_viewer_cache(Path(cfg.paths["viewer_cache_path"]))
    binary_labels, ternary_labels = _load_labels(Path(cfg.paths["labels_csv"]))
    rows = [dict(row) for row in cache.get("candidate_records", [])]
    x_axis = np.asarray(cache["x_axis"], dtype=float)
    spectra = np.asarray(cache["spectra"], dtype=float)
    names = flatten_dict_keys(rows)
    metric = _resolve_metric(names, args.metric)

    data_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        key = (int(row.get("source_y", row.get("y", -1))), int(row.get("source_x", row.get("x", -1))), int(row.get("peak_index", -1)))
        label_name = ternary_labels.get(key, "unknown")
        if args.label_mode == "binary" and key not in binary_labels:
            continue
        if args.label_mode == "ternary" and key not in ternary_labels:
            continue
        value = _safe_float(row.get(metric))
        if not np.isfinite(value):
            continue
        data_rows.append(
            {
                "index": idx,
                "metric_value": value,
                "binary_label": int(binary_labels.get(key, 0)),
                "ternary_label": label_name,
                "row": row,
            }
        )
    if not data_rows:
        raise ValueError(f"No labeled candidate rows with finite {metric}.")

    values = np.asarray([item["metric_value"] for item in data_rows], dtype=float)
    thr_low = float(args.threshold_low if args.threshold_low is not None else np.nanmedian(values))
    thr_high = float(args.threshold_high if args.threshold_high is not None else np.nanpercentile(values, 75.0))

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 0.16], width_ratios=[1.0, 1.0], hspace=0.25, wspace=0.20)
    ax_spec = fig.add_subplot(gs[0, :])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_scatter = fig.add_subplot(gs[1, 1])
    ctl = gs[2, :].subgridspec(1, 4, wspace=0.28)
    ax_low_slider = fig.add_subplot(ctl[0, 0])
    ax_high_slider = fig.add_subplot(ctl[0, 1])
    ax_low_box = fig.add_subplot(ctl[0, 2])
    ax_high_box = fig.add_subplot(ctl[0, 3])

    if args.label_mode == "binary":
        groups = {
            "non-muon": values[[item["binary_label"] == 0 for item in data_rows]],
            "muon": values[[item["binary_label"] == 1 for item in data_rows]],
        }
        colors = {"non-muon": "#1f77b4", "muon": "#d62728"}
    else:
        groups = {}
        colors = TERNARY_COLORS
        for label_name in ("muon", "raman", "noise", "unknown"):
            group_vals = values[[item["ternary_label"] == label_name for item in data_rows]]
            if group_vals.size:
                groups[label_name] = group_vals

    bins = np.linspace(float(np.min(values)), float(np.max(values)), 50)
    for label_name, group_vals in groups.items():
        ax_hist.hist(group_vals, bins=bins, color=colors[label_name], alpha=0.55, label=label_name)
    low_line = ax_hist.axvline(thr_low, color="#ff7f0e", linestyle="--", linewidth=1.8, label="thr-low")
    high_line = ax_hist.axvline(thr_high, color="#2ca02c", linestyle="--", linewidth=1.8, label="thr-high")
    ax_hist.set_title(f"{metric} histogram")
    ax_hist.set_xlabel(metric)
    ax_hist.set_ylabel("count")
    ax_hist.legend(loc="upper right", fontsize=9)

    xs = np.arange(len(data_rows), dtype=float)
    ys = values
    cs = [
        ("#d62728" if item["binary_label"] == 1 else "#1f77b4")
        if args.label_mode == "binary"
        else colors.get(item["ternary_label"], "#9467bd")
        for item in data_rows
    ]
    scat = ax_scatter.scatter(xs, ys, c=cs, s=26, alpha=0.85, picker=True)
    low_h = ax_scatter.axhline(thr_low, color="#ff7f0e", linestyle="--", linewidth=1.4)
    high_h = ax_scatter.axhline(thr_high, color="#2ca02c", linestyle="--", linewidth=1.4)
    sel_marker, = ax_scatter.plot([], [], marker="o", markersize=10, markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.5, linestyle="None")
    ax_scatter.set_title("labeled candidates")
    ax_scatter.set_xlabel("candidate index")
    ax_scatter.set_ylabel(metric)
    annot = ax_scatter.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points", bbox={"boxstyle": "round", "fc": "w", "alpha": 0.96})
    annot.set_visible(False)
    selected = {"idx": 0}

    def _tooltip(index: int) -> str:
        item = data_rows[index]
        row = item["row"]
        label_text = (
            ("muon" if item["binary_label"] == 1 else "non-muon")
            if args.label_mode == "binary"
            else item["ternary_label"]
        )
        return "\n".join(
            [
                f"value={item['metric_value']:.4g}",
                f"label={label_text}",
                f"source=({row.get('source_y', row.get('y'))},{row.get('source_x', row.get('x'))})",
                f"compact=({row.get('y')},{row.get('x')})",
                f"peak={row.get('peak_index')}",
                f"candidate_id={row.get('candidate_id')}",
                f"ss1={_safe_float(row.get('spike_score_v1')):.4g}",
                f"pce={_safe_float(row.get('pce_negpref_t098_evidence_signed')):.4g}",
                f"edge={_safe_float(row.get('recdw_sum_0_90_raman_veto_evidence_signed')):.4g}",
                f"ss4={row.get('ss4_decision', 'n/a')}",
                f"ss5={row.get('ss5_decision', 'n/a')}",
            ]
        )

    def _draw_spectrum(index: int) -> None:
        ax_spec.clear()
        item = data_rows[index]
        row = item["row"]
        y = int(row.get("y", 0))
        x = int(row.get("x", 0))
        sig = np.asarray(spectra[y, x, :], dtype=float)
        ax_spec.plot(x_axis, sig, color="#1f77b4", linewidth=1.5, label="raw")
        peak = int(row.get("peak_index", 0))
        start = int(row.get("start", peak))
        end = int(row.get("end", peak))
        if 0 <= peak < len(x_axis):
            ax_spec.axvline(x_axis[peak], color="#d62728", linestyle="--", linewidth=1.4)
        if 0 <= start < len(x_axis) and 0 <= end < len(x_axis):
            ax_spec.axvspan(x_axis[start], x_axis[end], color="#2ca02c", alpha=0.08)
        ax_spec.set_title(
            f"{metric}={item['metric_value']:.4g} | source(y={row.get('source_y', row.get('y'))}, x={row.get('source_x', row.get('x'))}) | "
            f"peak={peak}"
        )
        ax_spec.set_xlabel("wavenumber")
        ax_spec.set_ylabel("intensity")
        ax_spec.legend(loc="upper right", fontsize=9)
        sel_marker.set_data([xs[index]], [ys[index]])

    def _sync_threshold_artists() -> None:
        low_line.set_xdata([s_low.val, s_low.val])
        high_line.set_xdata([s_high.val, s_high.val])
        low_h.set_ydata([s_low.val, s_low.val])
        high_h.set_ydata([s_high.val, s_high.val])
        low_box.set_val(f"{s_low.val:.6g}")
        high_box.set_val(f"{s_high.val:.6g}")
        fig.canvas.draw_idle()

    def on_move(event) -> None:
        if event.inaxes is not ax_scatter:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return
        contains, info = scat.contains(event)
        if not contains or not info.get("ind"):
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return
        idx = int(info["ind"][0])
        annot.xy = (xs[idx], ys[idx])
        annot.set_text(_tooltip(idx))
        annot.set_visible(True)
        fig.canvas.draw_idle()

    def on_pick(event) -> None:
        if event.artist is not scat:
            return
        if not event.ind:
            return
        idx = int(event.ind[0])
        selected["idx"] = idx
        _draw_spectrum(idx)
        fig.canvas.draw_idle()

    s_low = Slider(ax_low_slider, "thr-low", float(np.min(values)), float(np.max(values)), valinit=thr_low, valstep=max((float(np.max(values)) - float(np.min(values))) / 400.0, 1e-6))
    s_high = Slider(ax_high_slider, "thr-high", float(np.min(values)), float(np.max(values)), valinit=thr_high, valstep=max((float(np.max(values)) - float(np.min(values))) / 400.0, 1e-6))
    low_box = TextBox(ax_low_box, "thr-low", initial=f"{thr_low:.6g}")
    high_box = TextBox(ax_high_box, "thr-high", initial=f"{thr_high:.6g}")

    def on_slider(_val) -> None:
        _sync_threshold_artists()

    def _submit_box(which: str, text: str) -> None:
        try:
            value = float(text)
        except Exception:
            return
        if which == "low":
            s_low.set_val(value)
        else:
            s_high.set_val(value)

    s_low.on_changed(on_slider)
    s_high.on_changed(on_slider)
    low_box.on_submit(lambda text: _submit_box("low", text))
    high_box.on_submit(lambda text: _submit_box("high", text))
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("pick_event", on_pick)
    _draw_spectrum(0)
    _sync_threshold_artists()
    plt.show()


if __name__ == "__main__":
    main()
