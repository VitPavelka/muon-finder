from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
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

THRESH_COLORS = {
    "low": "#1f77b4",
    "mid": "#ff7f0e",
    "high": "#d62728",
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


def _row_label(row: dict[str, Any], binary_labels: dict[tuple[int, int, int], int], ternary_labels: dict[tuple[int, int, int], str], mode: str) -> tuple[str, str]:
    key = (int(row.get("source_y", row.get("y", -1))), int(row.get("source_x", row.get("x", -1))), int(row.get("peak_index", -1)))
    if mode == "binary":
        return ("muon" if binary_labels.get(key, 0) == 1 else "non-muon"), ("#d62728" if binary_labels.get(key, 0) == 1 else "#1f77b4")
    label = str(ternary_labels.get(key, "unknown"))
    return label, TERNARY_COLORS.get(label, "#9467bd")


def _threshold_color(value: float, thr_l: float, thr_h: float, reverse: bool) -> str:
    if not np.isfinite(value):
        return "#7f7f7f"
    if reverse:
        if value <= thr_l:
            return THRESH_COLORS["high"]
        if value >= thr_h:
            return THRESH_COLORS["low"]
        return THRESH_COLORS["mid"]
    if value >= thr_h:
        return THRESH_COLORS["high"]
    if value <= thr_l:
        return THRESH_COLORS["low"]
    return THRESH_COLORS["mid"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive metric-threshold viewer for core candidate peaks")
    parser.add_argument("--config", type=Path, default=Path("config_core.json"), help="Core config; used to resolve viewer_cache_path and labels_csv.")
    parser.add_argument("--metric", type=str, required=True, help="Metric/feature in candidate rows.")
    parser.add_argument("--thr-l", type=float, default=None, help="Lower threshold guide.")
    parser.add_argument("--thr-h", type=float, default=None, help="Upper threshold guide.")
    parser.add_argument("--reverse", action="store_true", help="Interpret lower / more negative values as more spike-like.")
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
    all_rows_by_spectrum: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        spec_key = (int(row.get("y", -1)), int(row.get("x", -1)))
        all_rows_by_spectrum.setdefault(spec_key, []).append(row)
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
        label_text, color = _row_label(row, binary_labels, ternary_labels, args.label_mode)
        data_rows.append(
            {
                "index": idx,
                "metric_value": value,
                "label_text": label_text,
                "color": color,
                "row": row,
            }
        )
    if not data_rows:
        raise ValueError(f"No labeled candidate rows with finite {metric}.")

    spectrum_keys: list[tuple[int, int]] = []
    spectrum_to_candidate_indices: dict[tuple[int, int], list[int]] = {}
    for data_idx, item in enumerate(data_rows):
        row = item["row"]
        key = (int(row.get("y", -1)), int(row.get("x", -1)))
        if key not in spectrum_to_candidate_indices:
            spectrum_keys.append(key)
            spectrum_to_candidate_indices[key] = []
        spectrum_to_candidate_indices[key].append(int(data_idx))

    values = np.asarray([item["metric_value"] for item in data_rows], dtype=float)
    thr_state = {
        "low": float(args.thr_l if args.thr_l is not None else (np.nanpercentile(values, 25.0) if args.reverse else np.nanmedian(values))),
        "high": float(args.thr_h if args.thr_h is not None else (np.nanmedian(values) if args.reverse else np.nanpercentile(values, 75.0))),
    }

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 0.12], width_ratios=[1.0, 1.0], hspace=0.25, wspace=0.20)
    ax_spec = fig.add_subplot(gs[0, :])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_scatter = fig.add_subplot(gs[1, 1])
    ctl = gs[2, :].subgridspec(1, 4, width_ratios=[0.18, 0.20, 0.18, 0.20], wspace=0.25)
    ax_low_label = fig.add_subplot(ctl[0, 0])
    ax_low_box = fig.add_subplot(ctl[0, 1])
    ax_high_label = fig.add_subplot(ctl[0, 2])
    ax_high_box = fig.add_subplot(ctl[0, 3])
    for ax, text in ((ax_low_label, "thr-low"), (ax_high_label, "thr-high")):
        ax.axis("off")
        ax.text(0.98, 0.5, text, ha="right", va="center", fontsize=11)

    bins = np.linspace(float(np.min(values)), float(np.max(values)), 50)
    if args.label_mode == "binary":
        group_map = {
            "non-muon": values[np.asarray([item["label_text"] == "non-muon" for item in data_rows], dtype=bool)],
            "muon": values[np.asarray([item["label_text"] == "muon" for item in data_rows], dtype=bool)],
        }
        color_map = {"non-muon": "#1f77b4", "muon": "#d62728"}
    else:
        group_map = {}
        color_map = TERNARY_COLORS
        for label_name in ("muon", "raman", "noise", "unknown"):
            vals = values[np.asarray([item["label_text"] == label_name for item in data_rows], dtype=bool)]
            if vals.size:
                group_map[label_name] = vals
    for label_name, group_vals in group_map.items():
        ax_hist.hist(group_vals, bins=bins, color=color_map[label_name], alpha=0.55)
    low_line = ax_hist.axvline(thr_state["low"], color="#ff7f0e", linestyle="--", linewidth=2.2)
    high_line = ax_hist.axvline(thr_state["high"], color="#d62728", linestyle="--", linewidth=2.2)
    ax_hist.set_title(f"{metric} histogram")
    ax_hist.set_xlabel(metric)
    ax_hist.set_ylabel("count")

    xs = np.arange(len(data_rows), dtype=float)
    ys = values
    scat = ax_scatter.scatter(
        xs,
        ys,
        c=[item["color"] for item in data_rows],
        s=30,
        alpha=0.90,
        edgecolors="black",
        linewidths=0.35,
        picker=True,
    )
    low_h = ax_scatter.axhline(thr_state["low"], color="#ff7f0e", linestyle="--", linewidth=2.0)
    high_h = ax_scatter.axhline(thr_state["high"], color="#d62728", linestyle="--", linewidth=2.0)
    sel_marker, = ax_scatter.plot([], [], marker="o", markersize=10, markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.5, linestyle="None")
    ax_scatter.set_title(f"labeled candidates{' | reverse' if args.reverse else ''}")
    ax_scatter.set_xlabel("candidate index")
    ax_scatter.set_ylabel(metric)
    if args.reverse:
        ax_hist.invert_xaxis()
        ax_scatter.invert_yaxis()
    annot = ax_scatter.annotate("", xy=(0, 0), xytext=(10, -46), textcoords="offset points", bbox={"boxstyle": "round", "fc": "white", "alpha": 1.0})
    annot.set_in_layout(False)
    annot.set_visible(False)
    selected = {"idx": 0, "spectrum_idx": 0, "candidate_pos": 0}
    spectrum_order: dict[tuple[int, int], int] = {}
    for item in data_rows:
        row = item["row"]
        spec_key = (int(row.get("y", -1)), int(row.get("x", -1)))
        if spec_key not in spectrum_order:
            spectrum_order[spec_key] = len(spectrum_order) + 1

    hist_marker = ax_hist.axvline(values[0], color="black", linestyle="-", linewidth=2.0, alpha=0.85)

    def _set_selected_index(index: int) -> None:
        idx = int(np.clip(index, 0, len(data_rows) - 1))
        selected["idx"] = idx
        row = data_rows[idx]["row"]
        spec_key = (int(row.get("y", -1)), int(row.get("x", -1)))
        selected["spectrum_idx"] = int(spectrum_keys.index(spec_key))
        selected["candidate_pos"] = int(spectrum_to_candidate_indices[spec_key].index(idx))

    def _tooltip(index: int) -> str:
        item = data_rows[index]
        row = item["row"]
        return "\n".join(
            [
                f"value={item['metric_value']:.4g}",
                f"label={item['label_text']}",
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
        ax_spec.plot(x_axis, sig, color="#1f77b4", linewidth=1.5)
        spectrum_rows = sorted(
            all_rows_by_spectrum.get((y, x), []),
            key=lambda r: (int(r.get("peak_index", -1)), int(r.get("start", -1)), int(r.get("end", -1))),
        )
        selected_peak = int(row.get("peak_index", -1))
        for spec_row in spectrum_rows:
            peak = int(spec_row.get("peak_index", -1))
            start = int(spec_row.get("start", peak))
            end = int(spec_row.get("end", peak))
            value = _safe_float(spec_row.get(metric))
            color = _threshold_color(value, float(thr_state["low"]), float(thr_state["high"]), bool(args.reverse))
            if 0 <= start < len(x_axis) and 0 <= end < len(x_axis):
                ax_spec.axvspan(x_axis[start], x_axis[end], color=color, alpha=0.06)
            if 0 <= peak < len(x_axis):
                ax_spec.axvline(x_axis[peak], color=color, linestyle="--", linewidth=(2.0 if peak == selected_peak else 1.1), alpha=0.95)
        ax_spec.set_title(
            f"spectrum {spectrum_order.get((y, x), 1)}/{max(1, len(spectrum_order))} | "
            f"source(y={row.get('source_y', row.get('y'))}, x={row.get('source_x', row.get('x'))}) | "
            f"compact(y={y}, x={x})"
        )
        ax_spec.set_xlabel("wavenumber")
        ax_spec.set_ylabel("intensity")
        sel_marker.set_data([xs[index]], [ys[index]])
        hist_marker.set_xdata([item["metric_value"], item["metric_value"]])

    def _sync_threshold_artists() -> None:
        low_line.set_xdata([thr_state["low"], thr_state["low"]])
        high_line.set_xdata([thr_state["high"], thr_state["high"]])
        low_h.set_ydata([thr_state["low"], thr_state["low"]])
        high_h.set_ydata([thr_state["high"], thr_state["high"]])
        fig.canvas.draw_idle()

    def on_move(event) -> None:
        if event.inaxes is not ax_scatter:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return
        contains, info = scat.contains(event)
        ind = info.get("ind", [])
        if (not contains) or ind is None or len(ind) == 0:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return
        idx = int(ind[0])
        annot.xy = (xs[idx], ys[idx])
        annot.set_text(_tooltip(idx))
        annot.set_visible(True)
        fig.canvas.draw_idle()

    def on_pick(event) -> None:
        if event.artist is not scat:
            return
        ind = getattr(event, "ind", [])
        if ind is None or len(ind) == 0:
            return
        _set_selected_index(int(ind[0]))
        _draw_spectrum(selected["idx"])
        fig.canvas.draw_idle()

    low_box = TextBox(ax_low_box, " ", initial=f"{thr_state['low']:.6g}")
    high_box = TextBox(ax_high_box, " ", initial=f"{thr_state['high']:.6g}")

    def _submit_box(which: str, text: str) -> None:
        try:
            value = float(text)
        except Exception:
            return
        thr_state[which] = value
        _draw_spectrum(selected["idx"])
        _sync_threshold_artists()

    def on_key(event) -> None:
        key = str(event.key).lower()
        if key in {"left", "right", "up", "down"}:
            step = 10 if key in {"up", "down"} else 1
            delta = -step if key in {"left", "up"} else step
            selected["spectrum_idx"] = int(np.clip(selected["spectrum_idx"] + delta, 0, len(spectrum_keys) - 1))
            spec_key = spectrum_keys[selected["spectrum_idx"]]
            spec_candidates = spectrum_to_candidate_indices[spec_key]
            selected["candidate_pos"] = int(np.clip(selected["candidate_pos"], 0, len(spec_candidates) - 1))
            _set_selected_index(spec_candidates[selected["candidate_pos"]])
        elif key == "c":
            spec_key = spectrum_keys[selected["spectrum_idx"]]
            spec_candidates = spectrum_to_candidate_indices[spec_key]
            selected["candidate_pos"] = (selected["candidate_pos"] + 1) % len(spec_candidates)
            _set_selected_index(spec_candidates[selected["candidate_pos"]])
        else:
            return
        _draw_spectrum(selected["idx"])
        fig.canvas.draw_idle()

    low_box.on_submit(lambda text: _submit_box("low", text))
    high_box.on_submit(lambda text: _submit_box("high", text))
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("key_press_event", on_key)
    _set_selected_index(0)
    _draw_spectrum(0)
    _sync_threshold_artists()
    plt.show()


if __name__ == "__main__":
    main()
