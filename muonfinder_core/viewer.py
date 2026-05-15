from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons
import numpy as np

if __package__ in {None, ""}:
    import sys

    _THIS_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _THIS_DIR.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from muonfinder_core.cache import load_viewer_cache
    from muonfinder_core.config import load_config
    from muonfinder_core.metrics import EDGE_ALL_LEVELS_ASC, EDGE_DENSE_LEVELS_ASC
    from muonfinder_core.plotting import candidate_status_color
    from muonfinder_core.utils import metric_float, to_contiguous_spans
else:
    from .cache import load_viewer_cache
    from .config import load_config
    from .metrics import EDGE_ALL_LEVELS_ASC, EDGE_DENSE_LEVELS_ASC
    from .plotting import candidate_status_color
    from .utils import metric_float, to_contiguous_spans


CHECKBOX_ORDER = [
    "located muon",
    "raw",
    "corrected",
    "dilation",
    "erosion",
    "opening",
    "top-hat",
    "gradient",
    "spike peaks",
    "spike edges",
    "spike bands",
    "noise reference",
    "PCE",
    "EDGE",
    "dilation contacts",
    "erosion contacts",
    "despike chords",
    "noise filter",
    "metrics",
]

OVERLAY_KEY_BY_LABEL = {
    "dilation": "dilation",
    "erosion": "erosion",
    "opening": "opening",
    "top-hat": "top_hat",
    "gradient": "gradient",
}


def _config_cache_path(config_path: Path) -> Path:
    cfg = load_config(config_path)
    return Path(str(cfg.paths["viewer_cache_path"]))


def _x_from_index(x_axis: np.ndarray, idx: float) -> float:
    xp = np.arange(int(x_axis.size), dtype=float)
    return float(np.interp(float(idx), xp, np.asarray(x_axis, dtype=float)))


def _dedup_legend(handles: list[Any], labels: list[str]) -> tuple[list[Any], list[str]]:
    seen: set[str] = set()
    out_h: list[Any] = []
    out_l: list[str] = []
    for handle, label in zip(handles, labels):
        if not label or label in seen:
            continue
        seen.add(label)
        out_h.append(handle)
        out_l.append(label)
    return out_h, out_l


def show_cache(cache: dict[str, Any]) -> None:
    x_axis = np.asarray(cache["x_axis"], dtype=float)
    spectra = np.asarray(cache["spectra"], dtype=float)
    corrected = np.asarray(cache["corrected_spectra"], dtype=float)
    score_map = np.asarray(cache["score_map"], dtype=float)
    metadata = dict(cache.get("metadata", {}))
    candidate_rows = [dict(row) for row in cache.get("candidate_records", [])]
    small_rows = [dict(row) for row in cache.get("small_morphology", [])]
    overlays = cache.get("overlays", {})
    chords = [dict(row) for row in cache.get("despike_chords", [])]
    despike_stages = [dict(row) for row in cache.get("despike_stages", [])]
    active_profile = str(metadata.get("decision_profile", "ss4")).strip().lower()
    morph_windows = sorted(int(v) for v in metadata.get("morphology_windows", sorted(overlays.get("dilation", {}).keys())))
    if not morph_windows:
        morph_windows = [3]

    rows_by_pixel: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        rows_by_pixel.setdefault((int(row["y"]), int(row["x"])), []).append(row)
    for rows in rows_by_pixel.values():
        rows.sort(key=lambda row: (int(row.get("peak_index", -1)), int(row.get("start", -1)), int(row.get("end", -1))))
    small_by_pixel = {(int(row["y"]), int(row["x"])): row for row in small_rows}
    coord_map = {
        (int(row["compact_y"]), int(row["compact_x"])): (int(row["source_y"]), int(row["source_x"]))
        for row in cache.get("coord_map", [])
    }
    accepted_offsets = np.asarray(
        [
            [int(row["x"]), int(row["y"])]
            for row in candidate_rows
            if str(row.get("primary_active_decision", "non_spike")) == "spike"
        ],
        dtype=float,
    )

    H, W = score_map.shape
    current = {"y": 0, "x": 0, "morph_idx": 0, "stage_idx": 0}
    frozen = {"state": False}
    spectrum_home = {"xlim": None, "ylim": None}
    if H > 1 or W > 1:
        iy, ix = np.unravel_index(int(np.nanargmax(score_map)), score_map.shape)
        current["y"] = int(iy)
        current["x"] = int(ix)

    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.30, 1.45],
        height_ratios=[1.0, 0.24],
        left=0.04,
        right=0.985,
        top=0.90,
        bottom=0.06,
        wspace=0.12,
        hspace=0.16,
    )
    ax_map = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[0, 1])
    chk_grid = gs[1, 1].subgridspec(1, 3, wspace=0.20)
    ax_chk_blocks = [fig.add_subplot(chk_grid[0, i]) for i in range(3)]
    for ax in ax_chk_blocks:
        ax.set_xticks([])
        ax.set_yticks([])

    states = {name: (name in {"located muon", "raw", "corrected"}) for name in CHECKBOX_ORDER}
    checks = []
    block_size = int(np.ceil(len(CHECKBOX_ORDER) / 3))
    for block_index, ax_chk in enumerate(ax_chk_blocks):
        start = block_index * block_size
        stop = min(len(CHECKBOX_ORDER), start + block_size)
        labels = CHECKBOX_ORDER[start:stop]
        actives = [states[label] for label in labels]
        chk = CheckButtons(ax_chk, labels=labels, actives=actives)
        for txt in chk.labels:
            txt.set_fontsize(11)
        checks.append(chk)

    def current_window() -> int:
        return int(morph_windows[current["morph_idx"] % len(morph_windows)])

    def current_stage_index() -> int:
        stage_count = max(1, len(despike_stages))
        return int(current["stage_idx"] % stage_count)

    def current_rows() -> list[dict[str, Any]]:
        return rows_by_pixel.get((int(current["y"]), int(current["x"])), [])

    map_im = ax_map.imshow(score_map, cmap="viridis", origin="upper", interpolation="nearest")
    located_scatter = ax_map.scatter([], [], s=26, c="#d62728", marker="s", linewidths=0.0, alpha=0.90)
    cursor_marker, = ax_map.plot(
        [current["x"]],
        [current["y"]],
        marker="s",
        markersize=8.0,
        markerfacecolor="none",
        markeredgecolor="white",
        markeredgewidth=1.6,
        linestyle="None",
    )
    ax_map.set_title("score map", fontsize=13)
    ax_map.set_xlabel("x (pixel)", fontsize=11)
    ax_map.set_ylabel("y (pixel)", fontsize=11)
    ax_map.tick_params(labelsize=10)
    fig.colorbar(map_im, ax=ax_map, fraction=0.046, pad=0.04)

    def _set_suptitle() -> None:
        compact = (int(current["y"]), int(current["x"]))
        source = coord_map.get(compact, compact)
        stage_idx = current_stage_index()
        stage_count = max(1, len(despike_stages))
        fig.suptitle(
            f"spectrum @ compact(y={compact[0]}, x={compact[1]}) -> "
            f"source(y={source[0]}, x={source[1]}) | "
            f"despike stage {stage_idx}/{stage_count - 1} (a/x) | morph window {current_window()} (z/c)",
            fontsize=13,
            fontweight="bold",
            y=0.965,
        )

    def _update_map_artists() -> None:
        cursor_marker.set_data([current["x"]], [current["y"]])
        if states["located muon"] and accepted_offsets.size:
            located_scatter.set_offsets(accepted_offsets)
            located_scatter.set_visible(True)
        else:
            located_scatter.set_offsets(np.empty((0, 2), dtype=float))
            located_scatter.set_visible(False)

    def _plot_overlay_line(label: str) -> None:
        overlay_key = OVERLAY_KEY_BY_LABEL[label]
        window = current_window()
        if window not in overlays.get(overlay_key, {}):
            return
        colors = {
            "dilation": "#ff7f0e",
            "erosion": "#222222",
            "opening": "#9467bd",
            "top-hat": "#8c564b",
            "gradient": "#e377c2",
        }
        ax_spec.plot(
            x_axis,
            np.asarray(overlays[overlay_key][window][current["y"], current["x"], :], dtype=float),
            color=colors[label],
            linewidth=1.8,
            alpha=0.95,
            label=f"{label} w={window}",
        )

    def _edge_debug(row: dict[str, Any]) -> dict[str, Any]:
        debug = row.get("edge_debug", {})
        return debug if isinstance(debug, dict) else {}

    def _context_bounds(row: dict[str, Any]) -> tuple[int, int]:
        debug = _edge_debug(row)
        left = debug.get("edge_context_left")
        right = debug.get("edge_context_right")
        if left is not None and right is not None:
            return int(left), int(right)
        return int(row.get("start", 0)), int(row.get("end", 0))

    def _contact_context_bounds(row: dict[str, Any], pad: int = 4) -> tuple[int, int]:
        left = max(0, int(row.get("start", 0)) - int(pad))
        right = min(len(x_axis) - 1, int(row.get("end", 0)) + int(pad))
        return left, right

    def _draw_pce_overlay(rows: list[dict[str, Any]]) -> None:
        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        first = True
        for row in rows:
            if str(row.get("candidate_noise_prefilter_status", "")) == "rejected_noise":
                continue
            debug = row.get("pce_t98_debug", {})
            if not isinstance(debug, dict):
                continue
            x_rel = [int(v) for v in debug.get("curve_x_rel", [])]
            y_vals = np.asarray(debug.get("curve_y", []), dtype=float)
            if not x_rel or y_vals.size != len(x_rel):
                continue
            start = int(row.get("start", 0))
            x_plot_idx = np.asarray([start + int(v) for v in x_rel], dtype=int)
            if np.any(x_plot_idx < 0) or np.any(x_plot_idx >= len(x_axis)):
                continue
            ax_spec.axvspan(x_axis[int(x_plot_idx[0])], x_axis[int(x_plot_idx[-1])], color="#666666", alpha=0.08, zorder=0)
            line, = ax_spec.plot(x_axis[x_plot_idx], y_vals, color="black", linewidth=1.4, alpha=0.95, zorder=2, label="curvature")
            apex_idx_rel = int(debug.get("apex_idx_rel", 1))
            chosen_idx_rel = int(debug.get("chosen_idx_rel", 1))
            base_idx_rel = int(debug.get("base_idx_rel", 1))
            neg_idx_rel = debug.get("negative_idx_rel")
            local_left_rel = int(debug.get("local_left_idx_rel", apex_idx_rel))
            local_right_rel = int(debug.get("local_right_idx_rel", apex_idx_rel))
            def _plot_rel(rel_idx: int | None, marker: str, color: str, size: float, zorder: float, fill: bool = False) -> Any | None:
                if rel_idx is None:
                    return None
                rel = int(rel_idx)
                pos = rel - 1
                if not (0 <= pos < y_vals.size):
                    return None
                return ax_spec.plot(
                    [x_axis[start + rel]],
                    [y_vals[pos]],
                    marker=marker,
                    color=color,
                    markersize=size,
                    linestyle="None",
                    markerfacecolor=(color if fill else "none"),
                    markeredgewidth=1.35,
                    zorder=zorder,
                )[0]
            apex_handle = _plot_rel(apex_idx_rel, "x", "#111111", 7.8, 6.0)
            base_handle = _plot_rel(base_idx_rel, "+", "#555555", 9.0, 5.2)
            neg_handle = _plot_rel((None if neg_idx_rel is None else int(neg_idx_rel)), "X", "#1f77b4", 7.4, 5.4)
            chosen_handle = _plot_rel(chosen_idx_rel, (5, 2, 0), "#d62728", 10.5, 6.2)
            ll = start + local_left_rel
            rr = start + local_right_rel
            if 0 <= ll < len(x_axis) and 0 <= rr < len(x_axis) and rr >= ll:
                ax_spec.axvspan(x_axis[ll], x_axis[rr], color="#999999", alpha=0.06, zorder=1)
            if first:
                legend_handles.append(line)
                legend_labels.append("curvature")
                for handle, label in (
                    (apex_handle, "apex"),
                    (base_handle, "global support"),
                    (neg_handle, "negative support"),
                    (chosen_handle, "chosen PCE point (t098)"),
                ):
                    if handle is not None:
                        legend_handles.append(handle)
                        legend_labels.append(label)
                first = False
        ax_spec._pce_legend_handles = legend_handles  # type: ignore[attr-defined]
        ax_spec._pce_legend_labels = legend_labels  # type: ignore[attr-defined]

    def _draw_edge_overlay(rows: list[dict[str, Any]], raw_sig: np.ndarray) -> None:
        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        first = True
        for row in rows:
            if str(row.get("candidate_noise_prefilter_status", "")) == "rejected_noise":
                continue
            debug = _edge_debug(row)
            if not debug:
                continue
            edge_value = metric_float(row, "recdw_sum_0_90_raman_veto_evidence_signed")
            if not np.isfinite(edge_value):
                continue
            ml = int(debug.get("edge_context_left", row.get("start", 0)))
            mr = int(debug.get("edge_context_right", row.get("end", 0)))
            if 0 <= ml < len(x_axis) and 0 <= mr < len(x_axis) and mr >= ml:
                ax_spec.axvspan(x_axis[ml], x_axis[mr], color="#fb6a4a", alpha=0.05)
            segments = []
            colors = []
            support_x = []
            support_y = []
            base_levels = []
            for item in debug.get("edge_selected_levels", []):
                if not isinstance(item, dict):
                    continue
                percent = int(item.get("percent", -1))
                if percent == 0:
                    base_levels.append(item)
                    continue
                if percent not in EDGE_DENSE_LEVELS_ASC:
                    continue
                left_cross = metric_float(item, "left_cross")
                right_cross = metric_float(item, "right_cross")
                level_y = metric_float(item, "level_value")
                if not (np.isfinite(left_cross) and np.isfinite(right_cross) and np.isfinite(level_y)):
                    continue
                lx = _x_from_index(x_axis, left_cross)
                rx = _x_from_index(x_axis, right_cross)
                segments.append([(lx, level_y), (rx, level_y)])
                colors.append((0.95, 0.25, 0.15, 0.18 + 0.50 * (percent / 100.0)))
                support_x.extend([lx, rx])
                support_y.extend([level_y, level_y])
            for item in base_levels:
                left_cross = metric_float(item, "left_cross")
                right_cross = metric_float(item, "right_cross")
                level_y = metric_float(item, "level_value")
                if np.isfinite(left_cross) and np.isfinite(right_cross) and np.isfinite(level_y):
                    ax_spec.plot(
                        [_x_from_index(x_axis, left_cross), _x_from_index(x_axis, right_cross)],
                        [level_y, level_y],
                        color="#fb6a4a",
                        linestyle=":",
                        linewidth=1.1,
                    )
            if segments:
                ax_spec.add_collection(LineCollection(segments, colors=colors, linewidths=1.35))
                ax_spec.scatter(support_x, support_y, s=16, c="#fb6a4a", zorder=4)
            if first:
                if segments:
                    legend_handles.append(Line2D([0], [0], color="#fb6a4a", linewidth=1.5))
                    legend_labels.append("EDGE levels 5..90")
                if base_levels:
                    legend_handles.append(Line2D([0], [0], color="#fb6a4a", linestyle=":", linewidth=1.2))
                    legend_labels.append("EDGE level 0")
                if support_x:
                    legend_handles.append(Line2D([0], [0], marker="o", color="#fb6a4a", linestyle="None", markersize=5.0))
                    legend_labels.append("EDGE support points")
            if bool(debug.get("edge_foot_search_triggered")) and str(debug.get("edge_foot_search_status", "")) in {"searched", "unresolved", "context_limited"}:
                peak = int(row.get("peak_index", 0))
                if 0 <= peak < len(x_axis):
                    ax_spec.text(
                        x_axis[peak],
                        raw_sig[peak],
                        "EDGE foot search",
                        fontsize=8,
                        color="#fb6a4a",
                        ha="left",
                        va="bottom",
                    )
            for item in debug.get("edge_rejected_foots", []):
                if not isinstance(item, dict):
                    continue
                idx = int(item.get("index", -1))
                ratio = metric_float(item, "r")
                if not (0 <= idx < len(x_axis)):
                    continue
                yv = float(raw_sig[idx])
                ax_spec.plot([x_axis[idx]], [yv], marker="v", color="#d62728", markersize=7.5, linestyle="None", zorder=5)
                ax_spec.axvline(x_axis[idx], color="#d62728", linestyle=":", linewidth=0.9, alpha=0.8)
                if np.isfinite(ratio):
                    ax_spec.text(x_axis[idx], yv, f"r={ratio:.1f}", color="#d62728", fontsize=8, ha="center", va="top")
            if first and debug.get("edge_rejected_foots"):
                legend_handles.append(Line2D([0], [0], marker="v", color="#d62728", linestyle="None", markersize=6.0))
                legend_labels.append("rejected foot")
            if first and legend_handles:
                first = False
        if legend_handles:
            ax_spec._edge_legend_handles = legend_handles  # type: ignore[attr-defined]
            ax_spec._edge_legend_labels = legend_labels  # type: ignore[attr-defined]

    def _draw_contacts(rows: list[dict[str, Any]], indices: list[int], marker: str, color: str, size: float) -> None:
        points: set[int] = set()
        for row in rows:
            left, right = _contact_context_bounds(row)
            for idx in indices:
                ii = int(idx)
                if left <= ii <= right:
                    points.add(ii)
        for idx in sorted(points):
            ax_spec.plot([x_axis[idx]], [raw_sig[idx]], marker=marker, color=color, markersize=size, linestyle="None")

    def _draw_noise_filter(rows: list[dict[str, Any]], raw_sig: np.ndarray) -> None:
        for row in rows:
            left_foot = row.get("candidate_noise_left_foot")
            right_foot = row.get("candidate_noise_right_foot")
            apex = row.get("candidate_noise_apex")
            chord_y = metric_float(row, "candidate_noise_chord_y_at_apex")
            if any(v is None for v in (left_foot, right_foot, apex)):
                continue
            li = int(left_foot)
            ri = int(right_foot)
            ai = int(apex)
            if not (0 <= li < len(raw_sig) and 0 <= ri < len(raw_sig) and 0 <= ai < len(raw_sig)) or ri <= li:
                continue
            status = str(row.get("candidate_noise_prefilter_status", ""))
            color = "#5b5b5b" if status == "rejected_noise" else "#6a3d9a"
            linestyle = "--" if status == "rejected_noise" else "-"
            ax_spec.plot([x_axis[li], x_axis[ri]], [raw_sig[li], raw_sig[ri]], color=color, linestyle=linestyle, linewidth=1.2)
            if np.isfinite(chord_y):
                ax_spec.plot([x_axis[ai], x_axis[ai]], [float(chord_y), float(raw_sig[ai])], color=color, linestyle=linestyle, linewidth=1.4)
            ratio = metric_float(row, "candidate_noise_height_ratio")
            height = metric_float(row, "candidate_noise_height_above_chord")
            parts = []
            if np.isfinite(height):
                parts.append(f"h={height:.0f}")
            if np.isfinite(ratio):
                parts.append(f"r={ratio:.1f}")
            if parts:
                ax_spec.text(x_axis[ai], raw_sig[ai], " | ".join(parts), color=color, fontsize=10, va="bottom", ha="center")

    def _draw_metrics(rows: list[dict[str, Any]], raw_sig: np.ndarray) -> None:
        metric_rows = [row for row in rows if str(row.get("candidate_noise_prefilter_status", "")) != "rejected_noise"]
        metric_rows.sort(key=lambda row: int(row.get("peak_index", 0)))
        if not metric_rows:
            return
        x_min, x_max = ax_spec.get_xlim()
        y_min, y_max = ax_spec.get_ylim()
        y_span = max(1e-9, float(y_max - y_min))
        renderer = fig.canvas.get_renderer()
        x_span = max(1e-9, float(x_max - x_min))
        placed_bboxes = []
        for row in metric_rows:
            peak = int(row.get("peak_index", 0))
            if not (0 <= peak < len(raw_sig)):
                continue
            ss1 = metric_float(row, "spike_score_v1")
            pce = metric_float(row, "pce_negpref_t098_evidence_signed")
            if not np.isfinite(pce):
                pce = metric_float(row, "pce")
            edge = metric_float(row, "recdw_sum_0_90_raman_veto_evidence_signed")
            finite_parts: list[str] = []
            if np.isfinite(ss1):
                finite_parts.append(f"ss1={ss1:.3g}")
            if np.isfinite(pce):
                finite_parts.append(f"pce={pce:.3g}")
            if np.isfinite(edge):
                finite_parts.append(f"edge={edge:.3g}")
            if not finite_parts:
                continue
            color = candidate_status_color(row, active_profile)
            x_pos = float(x_axis[peak])
            label = "\n".join(finite_parts + ([str(row.get("primary_active_reason", ""))] if row.get("primary_active_reason") else []))
            y_peak = float(raw_sig[peak])
            x_text = float(np.clip(x_pos + 0.010 * x_span, x_min + 0.02 * x_span, x_max - 0.20 * x_span))
            y_pos = float(np.clip(y_peak + 0.045 * y_span, y_min + 0.03 * y_span, y_max - 0.03 * y_span))
            text = ax_spec.annotate(
                label,
                xy=(x_pos, y_peak),
                xytext=(x_text, y_pos),
                textcoords="data",
                ha="left",
                va="bottom",
                fontsize=10,
                color=color,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": color, "linewidth": 0.9},
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7, "alpha": 0.55, "shrinkA": 2, "shrinkB": 2},
            )
            if renderer is None:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
            best_xy = (x_text, y_pos)
            for step in range(24):
                text.set_position((x_text, y_pos))
                bbox = text.get_window_extent(renderer=renderer).expanded(1.03, 1.08)
                axes_bbox = ax_spec.get_window_extent(renderer=renderer)
                inside = (
                    bbox.x0 >= axes_bbox.x0 + 2
                    and bbox.x1 <= axes_bbox.x1 - 2
                    and bbox.y0 >= axes_bbox.y0 + 2
                    and bbox.y1 <= axes_bbox.y1 - 2
                )
                overlap = any(bbox.overlaps(prev) for prev in placed_bboxes)
                if inside and not overlap:
                    best_xy = (x_text, y_pos)
                    placed_bboxes.append(bbox)
                    break
                if step % 2 == 0:
                    y_pos += 0.07 * y_span
                else:
                    y_pos -= 0.09 * y_span
                if step % 4 == 3:
                    x_text += 0.014 * x_span
                if step % 6 == 5:
                    x_text -= 0.020 * x_span
                x_text = float(np.clip(x_text, x_min + 0.02 * x_span, x_max - 0.20 * x_span))
                y_pos = float(np.clip(y_pos, y_min + 0.02 * y_span, y_max - 0.02 * y_span))
            text.set_position(best_xy)

    def _contact_context_spans(rows: list[dict[str, Any]], pad: int = 4) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for row in rows:
            left, right = _contact_context_bounds(row, pad=pad)
            spans.append((int(left), int(right)))
        indices: list[int] = []
        for left, right in spans:
            indices.extend(range(left, right + 1))
        return to_contiguous_spans(sorted(set(indices)))

    raw_sig = np.asarray(spectra[current["y"], current["x"], :], dtype=float)

    def _draw_spectrum() -> None:
        nonlocal raw_sig
        ax_spec.clear()
        ax_spec._pce_legend_handles = []  # type: ignore[attr-defined]
        ax_spec._pce_legend_labels = []  # type: ignore[attr-defined]
        ax_spec._edge_legend_handles = []  # type: ignore[attr-defined]
        ax_spec._edge_legend_labels = []  # type: ignore[attr-defined]
        rows = current_rows()
        raw_sig = np.asarray(spectra[current["y"], current["x"], :], dtype=float)
        if states["corrected"]:
            ax_spec.plot(x_axis, raw_sig, color="#d62728", linewidth=1.7, label="raw")
            ax_spec.plot(
                x_axis,
                np.asarray(corrected[current["y"], current["x"], :], dtype=float),
                color="#2ca02c",
                linewidth=1.7,
                label="corrected",
            )
        elif states["raw"]:
            ax_spec.plot(x_axis, raw_sig, color="#1f77b4", linewidth=1.7, label="raw")

        for label in ("dilation", "erosion", "opening", "top-hat", "gradient"):
            if states[label]:
                _plot_overlay_line(label)

        if states["spike bands"]:
            for row in rows:
                ax_spec.axvspan(x_axis[int(row["start"])], x_axis[int(row["end"])], color="#2ca02c", alpha=0.10)
        if states["spike edges"]:
            for row in rows:
                ax_spec.axvline(x_axis[int(row["start"])], color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.95)
                ax_spec.axvline(x_axis[int(row["end"])], color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.95)
        if states["spike peaks"]:
            for row in rows:
                color = candidate_status_color(row, active_profile)
                ax_spec.axvline(x_axis[int(row["peak_index"])], color=color, linestyle="--", linewidth=1.5)

        morph_row = small_by_pixel.get((int(current["y"]), int(current["x"])), {})
        if states["dilation contacts"] or states["erosion contacts"] or states["despike chords"]:
            for left, right in _contact_context_spans(rows, pad=4):
                li = max(0, min(int(left), len(x_axis) - 1))
                ri = max(0, min(int(right), len(x_axis) - 1))
                if ri >= li:
                    ax_spec.axvspan(x_axis[li], x_axis[ri], color="#17becf", alpha=0.14)
        if states["noise reference"]:
            for left, right in morph_row.get("noise_reference_spans", []):
                li = int(max(0, left))
                ri = int(min(len(raw_sig) - 1, right))
                if ri >= li:
                    ax_spec.plot(x_axis[li : ri + 1], raw_sig[li : ri + 1], color="black", linewidth=2.2)
            noise_line = None
            for row in rows:
                dbg = _edge_debug(row)
                src = dbg.get("edge_noise_source")
                val = metric_float(dbg, "edge_noise_value")
                if src and np.isfinite(val):
                    noise_line = f"noise={val:.1f} ({src})"
                    break
            if noise_line:
                ax_spec.text(
                    0.015,
                    0.98,
                    noise_line,
                    transform=ax_spec.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "black", "linewidth": 0.7},
                )

        if states["PCE"]:
            _draw_pce_overlay(rows)
        if states["EDGE"]:
            _draw_edge_overlay(rows, raw_sig)

        if states["dilation contacts"]:
            _draw_contacts(rows, morph_row.get("dilation_contacts", []), "^", "#ff7f0e", 8.5)
        if states["erosion contacts"]:
            _draw_contacts(rows, morph_row.get("erosion_contacts", []), "o", "#111111", 7.5)

        if states["despike chords"]:
            stage_id = None
            if despike_stages:
                stage_id = str(despike_stages[current_stage_index()].get("stage_id", ""))
            for chord in chords:
                if int(chord.get("y", -1)) != int(current["y"]) or int(chord.get("x", -1)) != int(current["x"]):
                    continue
                if stage_id is not None and str(chord.get("stage_id", "")) != stage_id:
                    continue
                li = int(chord["left"])
                ri = int(chord["right"])
                ax_spec.plot([x_axis[li], x_axis[ri]], [float(chord["y_left"]), float(chord["y_right"])], color="#17becf", linewidth=1.7)

        ax_spec.set_xlabel("wavenumber", fontsize=11)
        ax_spec.set_ylabel("intensity", fontsize=11)
        ax_spec.set_title("spectrum", fontsize=12)
        ax_spec.tick_params(labelsize=10)
        ax_spec.relim()
        ax_spec.autoscale_view()
        if states["noise filter"]:
            ax_spec.relim()
            ax_spec.autoscale_view()
        xlim = ax_spec.get_xlim()
        ylim = ax_spec.get_ylim()
        y_span = max(1e-9, float(ylim[1] - ylim[0]))
        spectrum_home["xlim"] = xlim
        spectrum_home["ylim"] = (float(ylim[0] - 0.03 * y_span), float(ylim[1] + 0.18 * y_span))
        ax_spec.set_xlim(*spectrum_home["xlim"])
        ax_spec.set_ylim(*spectrum_home["ylim"])
        if states["noise filter"]:
            _draw_noise_filter(rows, raw_sig)
        if states["metrics"]:
            _draw_metrics(rows, raw_sig)
        handles1, labels1 = ax_spec.get_legend_handles_labels()
        handles2 = list(getattr(ax_spec, "_pce_legend_handles", []))
        labels2 = list(getattr(ax_spec, "_pce_legend_labels", []))
        handles3 = list(getattr(ax_spec, "_edge_legend_handles", []))
        labels3 = list(getattr(ax_spec, "_edge_legend_labels", []))
        handles, labels = _dedup_legend(handles1 + handles2 + handles3, labels1 + labels2 + labels3)
        if handles:
            ax_spec.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.92)

    def update() -> None:
        _update_map_artists()
        _draw_spectrum()
        _set_suptitle()
        fig.canvas.draw_idle()

    def on_toggle(label: str) -> None:
        states[str(label)] = not states[str(label)]
        update()

    for chk in checks:
        chk.on_clicked(on_toggle)

    def on_move(event) -> None:
        if frozen["state"]:
            return
        if event.inaxes is not ax_map or event.xdata is None or event.ydata is None:
            return
        x = int(np.clip(round(event.xdata), 0, W - 1))
        y = int(np.clip(round(event.ydata), 0, H - 1))
        if x == int(current["x"]) and y == int(current["y"]):
            return
        current["x"] = x
        current["y"] = y
        update()

    def on_click(event) -> None:
        if event.inaxes is not ax_map or event.xdata is None or event.ydata is None:
            return
        if int(getattr(event, "button", 0)) != 3:
            return
        current["x"] = int(np.clip(round(event.xdata), 0, W - 1))
        current["y"] = int(np.clip(round(event.ydata), 0, H - 1))
        frozen["state"] = not frozen["state"]
        update()

    def on_key(event) -> None:
        key = str(event.key).lower()
        if key == "z":
            current["morph_idx"] = (current["morph_idx"] - 1) % len(morph_windows)
            update()
            return
        if key == "c":
            current["morph_idx"] = (current["morph_idx"] + 1) % len(morph_windows)
            update()
            return
        if key == "a":
            current["stage_idx"] = (current_stage_index() - 1) % max(1, len(despike_stages))
            update()
            return
        if key == "x":
            current["stage_idx"] = (current_stage_index() + 1) % max(1, len(despike_stages))
            update()
            return
        if key == "home":
            if spectrum_home["xlim"] is not None and spectrum_home["ylim"] is not None:
                ax_spec.set_xlim(*spectrum_home["xlim"])
                ax_spec.set_ylim(*spectrum_home["ylim"])
                fig.canvas.draw_idle()
            return
        if frozen["state"] and key in {"left", "right", "up", "down"}:
            dx = -1 if key == "left" else (1 if key == "right" else 0)
            dy = -1 if key == "up" else (1 if key == "down" else 0)
            current["x"] = int(np.clip(int(current["x"]) + dx, 0, W - 1))
            current["y"] = int(np.clip(int(current["y"]) + dy, 0, H - 1))
            update()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="MuonFinder core viewer")
    parser.add_argument("--config", type=Path, default=Path("config_core.json"), help="Core config JSON; used to resolve viewer_cache_path.")
    parser.add_argument("--cache", type=Path, default=None, help="Optional explicit viewer cache path; overrides config.")
    args = parser.parse_args()
    cache_path = Path(args.cache) if args.cache is not None else _config_cache_path(Path(args.config))
    show_cache(load_viewer_cache(cache_path))


if __name__ == "__main__":
    main()
