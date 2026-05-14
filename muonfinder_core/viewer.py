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
    from muonfinder_core.plotting import candidate_status_color
    from muonfinder_core.utils import metric_float
else:
    from .cache import load_viewer_cache
    from .config import load_config
    from .plotting import candidate_status_color
    from .utils import metric_float


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


def _compute_pce_overlay_debug(signal: np.ndarray, row: dict[str, Any]) -> dict[str, Any]:
    start = int(row.get("start", 0))
    end = int(row.get("end", 0))
    peak = int(row.get("peak_index", 0))
    n = int(signal.size)
    start = int(np.clip(start, 0, max(0, n - 1)))
    end = int(np.clip(end, 0, max(0, n - 1)))
    if end < start:
        start, end = end, start
    segment = np.asarray(signal[start : end + 1], dtype=float)
    if segment.size < 3:
        return {}
    d2 = np.diff(segment)
    if d2.size == 0:
        return {}
    peak_rel = int(np.clip(peak - start - 1, 0, d2.size - 1))
    local_radius = max(1, int(np.ceil(0.20 * d2.size)))
    local_left = max(0, peak_rel - local_radius)
    local_right = min(d2.size - 1, peak_rel + local_radius)
    local_slice = d2[local_left : local_right + 1]
    global_min_idx = int(np.argmin(d2))
    global_max_idx = int(np.argmax(d2))
    local_min_idx = int(local_left + np.argmin(local_slice))
    local_max_idx = int(local_left + np.argmax(local_slice))
    chosen_idx = local_min_idx if d2[local_min_idx] <= d2[global_min_idx] else global_min_idx
    return {
        "segment_left": int(start),
        "segment_right": int(end),
        "d2": d2,
        "peak_rel": int(peak_rel),
        "global_min_idx": int(global_min_idx),
        "global_max_idx": int(global_max_idx),
        "local_left_idx": int(local_left),
        "local_right_idx": int(local_right),
        "local_min_idx": int(local_min_idx),
        "local_max_idx": int(local_max_idx),
        "chosen_idx": int(chosen_idx),
    }


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
    ax_pce = ax_spec.twinx()
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

    overlay_diag_cache: dict[tuple[str, str], dict[str, Any]] = {}

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
            "erosion": "#1f77b4",
            "opening": "#9467bd",
            "top-hat": "#8c564b",
            "gradient": "#bcbd22",
        }
        ax_spec.plot(
            x_axis,
            np.asarray(overlays[overlay_key][window][current["y"], current["x"], :], dtype=float),
            color=colors[label],
            linewidth=1.8,
            alpha=0.95,
            label=f"{label} w={window}",
        )

    def _diag_cache_key(kind: str, row: dict[str, Any]) -> tuple[str, str]:
        return kind, str(row.get("candidate_id", ""))

    def _get_pce_debug(row: dict[str, Any], gradient_signal: np.ndarray) -> dict[str, Any]:
        key = _diag_cache_key("pce", row)
        cached = overlay_diag_cache.get(key)
        if cached is not None:
            return cached
        out = _compute_pce_overlay_debug(gradient_signal, row)
        overlay_diag_cache[key] = out
        return out

    def _edge_debug(row: dict[str, Any]) -> dict[str, Any]:
        debug = row.get("edge_debug", {})
        return debug if isinstance(debug, dict) else {}

    def _context_bounds(row: dict[str, Any]) -> tuple[int, int]:
        debug = _edge_debug(row)
        selected = debug.get("edge_selected_context", {})
        if isinstance(selected, dict):
            left = int(selected.get("measurement_left", row.get("start", 0)))
            right = int(selected.get("measurement_right", row.get("end", 0)))
            return left, right
        return int(row.get("start", 0)), int(row.get("end", 0))

    def _contact_context_bounds(row: dict[str, Any], pad: int = 4) -> tuple[int, int]:
        left = max(0, int(row.get("start", 0)) - int(pad))
        right = min(len(x_axis) - 1, int(row.get("end", 0)) + int(pad))
        return left, right

    def _draw_pce_overlay(rows: list[dict[str, Any]], gradient_signal: np.ndarray) -> None:
        ax_pce.clear()
        ax_pce.set_visible(True)
        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        first = True
        for row in rows:
            debug = _get_pce_debug(row, gradient_signal)
            d2 = np.asarray(debug.get("d2", []), dtype=float)
            if d2.size == 0:
                continue
            start = int(debug.get("segment_left", row.get("start", 0)))
            x_vals = x_axis[start + 1 : start + 1 + d2.size]
            if x_vals.size != d2.size:
                continue
            line, = ax_pce.plot(x_vals, d2, color="black", linewidth=1.5, alpha=0.85, label="PCE t98")
            peak_rel = int(debug.get("peak_rel", 0))
            global_min = int(debug.get("global_min_idx", 0))
            local_min = int(debug.get("local_min_idx", global_min))
            chosen = int(debug.get("chosen_idx", global_min))
            apex_handle = ax_pce.plot([x_vals[peak_rel]], [d2[peak_rel]], marker="x", color="#d62728", markersize=8.0, linestyle="None")[0]
            global_handle = ax_pce.plot([x_vals[global_min]], [d2[global_min]], marker="o", color="#1f77b4", markersize=6.8, linestyle="None")[0]
            local_handle = ax_pce.plot([x_vals[local_min]], [d2[local_min]], marker="^", color="#ff7f0e", markersize=7.0, linestyle="None")[0]
            chosen_handle = ax_pce.plot([x_vals[chosen]], [d2[chosen]], marker="D", color="#111111", markersize=6.4, linestyle="None")[0]
            ll = int(debug.get("local_left_idx", local_min))
            rr = int(debug.get("local_right_idx", local_min))
            if 0 <= ll < d2.size and 0 <= rr < d2.size and rr >= ll:
                ax_pce.axvspan(x_vals[ll], x_vals[rr], color="#444444", alpha=0.05)
            if first:
                legend_handles.extend([line, apex_handle, global_handle, local_handle, chosen_handle])
                legend_labels.extend(
                    [
                        "PCE t98",
                        "apex",
                        "global curvature support",
                        "local curvature support",
                        "chosen PCE point",
                    ]
                )
                first = False
        ax_pce.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9, alpha=0.8)
        ax_pce.set_ylabel("PCE curvature", fontsize=10, color="#333333")
        ax_pce.tick_params(axis="y", labelsize=9, colors="#333333")
        if legend_handles:
            handles, labels = _dedup_legend(legend_handles, legend_labels)
            ax_pce.legend(handles, labels, loc="upper left", fontsize=9, framealpha=0.92)

    def _draw_edge_overlay(rows: list[dict[str, Any]], raw_sig: np.ndarray) -> None:
        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        first = True
        for row in rows:
            debug = _edge_debug(row)
            if not debug:
                continue
            selected = debug.get("edge_selected_context", {})
            if isinstance(selected, dict):
                ml = int(selected.get("measurement_left", row.get("start", 0)))
                mr = int(selected.get("measurement_right", row.get("end", 0)))
                if 0 <= ml < len(x_axis) and 0 <= mr < len(x_axis) and mr >= ml:
                    ax_spec.axvspan(x_axis[ml], x_axis[mr], color="#fb6a4a", alpha=0.05)
            guard_passed = bool(debug.get("edge_guard_passed", str(debug.get("reason", "")).strip().lower() == "ok"))
            if not guard_passed:
                peak = int(row.get("peak_index", 0))
                if 0 <= peak < len(x_axis):
                    ax_spec.text(
                        x_axis[peak],
                        raw_sig[peak],
                        f"edge guard: {debug.get('edge_guard_reason', debug.get('reason', 'rejected'))}",
                        fontsize=9,
                        color="#fb6a4a",
                        ha="left",
                        va="bottom",
                    )
                continue
            segments = []
            colors = []
            support_x = []
            support_y = []
            for item in debug.get("edge_selected_levels", []):
                if not isinstance(item, dict):
                    continue
                left_cross = metric_float(item, "left_cross")
                right_cross = metric_float(item, "right_cross")
                level_y = metric_float(item, "level_value")
                percent = int(item.get("percent", 0))
                if not (np.isfinite(left_cross) and np.isfinite(right_cross) and np.isfinite(level_y)):
                    continue
                lx = _x_from_index(x_axis, left_cross)
                rx = _x_from_index(x_axis, right_cross)
                segments.append([(lx, level_y), (rx, level_y)])
                colors.append((0.95, 0.25, 0.15, 0.18 + 0.50 * (percent / 100.0)))
                support_x.extend([lx, rx])
                support_y.extend([level_y, level_y])
            if segments:
                ax_spec.add_collection(LineCollection(segments, colors=colors, linewidths=1.35))
                ax_spec.scatter(support_x, support_y, s=16, c="#fb6a4a", zorder=4)
                base_levels = [item for item in debug.get("edge_selected_levels", []) if isinstance(item, dict) and int(item.get("percent", -1)) == 0]
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
                if first:
                    legend_handles.extend(
                        [
                            Line2D([0], [0], color="#fb6a4a", linewidth=1.5),
                            Line2D([0], [0], color="#fb6a4a", linestyle=":", linewidth=1.2),
                            Line2D([0], [0], marker="o", color="#fb6a4a", linestyle="None", markersize=5.0),
                        ]
                    )
                    legend_labels.extend(["EDGE levels 5..90", "EDGE level 0", "EDGE support points"])
                    first = False
        if legend_handles:
            handles, labels = _dedup_legend(legend_handles, legend_labels)
            ax_spec.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.92)

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
        for idx, row in enumerate(rows):
            peak = int(row.get("peak_index", 0))
            if not (0 <= peak < len(raw_sig)):
                continue
            color = candidate_status_color(row, active_profile)
            y_pos = float(raw_sig[peak]) + (idx % 2) * 18.0
            label = (
                f"ss1={metric_float(row, 'spike_score_v1'):.3g}\n"
                f"pce={metric_float(row, 'pce_negpref_t098_evidence_signed'):.3g}\n"
                f"edge={metric_float(row, 'recdw_sum_0_90_raman_veto_evidence_signed'):.3g}\n"
                f"{row.get('primary_active_reason', '')}"
            )
            ax_spec.text(
                x_axis[peak],
                y_pos,
                label,
                color=color,
                fontsize=10,
                ha="left",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.74, "edgecolor": color, "linewidth": 0.9},
            )
            ax_spec.axvline(x_axis[peak], color=color, linestyle=":", linewidth=1.0, alpha=0.65)

    raw_sig = np.asarray(spectra[current["y"], current["x"], :], dtype=float)

    def _draw_spectrum() -> None:
        nonlocal raw_sig
        ax_spec.clear()
        ax_pce.clear()
        ax_pce.set_visible(False)
        rows = current_rows()
        raw_sig = np.asarray(spectra[current["y"], current["x"], :], dtype=float)
        gradient_signal = np.asarray(overlays["gradient"][current_window()][current["y"], current["x"], :], dtype=float)
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
            bounds = [_contact_context_bounds(row) for row in rows]
            if bounds:
                left = min(v[0] for v in bounds)
                right = max(v[1] for v in bounds)
                left = max(0, min(left, len(x_axis) - 1))
                right = max(0, min(right, len(x_axis) - 1))
                if right >= left:
                    ax_spec.axvspan(x_axis[left], x_axis[right], color="#17becf", alpha=0.05)
        if states["noise reference"]:
            for left, right in morph_row.get("noise_reference_spans", []):
                li = int(max(0, left))
                ri = int(min(len(raw_sig) - 1, right))
                if ri >= li:
                    ax_spec.plot(x_axis[li : ri + 1], raw_sig[li : ri + 1], color="black", linewidth=2.2)

        if states["PCE"]:
            _draw_pce_overlay(rows, gradient_signal)
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

        if states["noise filter"]:
            _draw_noise_filter(rows, raw_sig)
        if states["metrics"]:
            _draw_metrics(rows, raw_sig)

        ax_spec.set_xlabel("wavenumber", fontsize=11)
        ax_spec.set_ylabel("intensity", fontsize=11)
        ax_spec.set_title("spectrum", fontsize=12)
        ax_spec.tick_params(labelsize=10)
        handles1, labels1 = ax_spec.get_legend_handles_labels()
        handles2, labels2 = ax_pce.get_legend_handles_labels()
        handles, labels = _dedup_legend(handles1 + handles2, labels1 + labels2)
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
