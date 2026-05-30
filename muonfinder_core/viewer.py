from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import json

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

    from muonfinder_core.cap_metrics import join_extra_feature_rows, load_extra_feature_rows
    from muonfinder_core.cache import load_viewer_cache
    from muonfinder_core.config import load_config
    from muonfinder_core.despike import get_contact_context_bounds, load_despike_bundle
    from muonfinder_core.metrics import EDGE_ALL_LEVELS_ASC, EDGE_DENSE_LEVELS_ASC
    from muonfinder_core.plotting import candidate_status_color
    from muonfinder_core.ss6_decision import SS6_BRANCH_DEFINITIONS, SS6_KNOWN_BRANCHES
    from muonfinder_core.utils import metric_float, to_contiguous_spans
else:
    from .cap_metrics import join_extra_feature_rows, load_extra_feature_rows
    from .cache import load_viewer_cache
    from .config import load_config
    from .despike import get_contact_context_bounds, load_despike_bundle
    from .metrics import EDGE_ALL_LEVELS_ASC, EDGE_DENSE_LEVELS_ASC
    from .plotting import candidate_status_color
    from .ss6_decision import SS6_BRANCH_DEFINITIONS, SS6_KNOWN_BRANCHES
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
    "pre-EDGE",
    "EDGE",
    "dilation contacts",
    "erosion contacts",
    "despike chords",
    "despike metrics",
    "noise filter",
    "Experimental metrics",
    "metrics",
]

OVERLAY_KEY_BY_LABEL = {
    "dilation": "dilation",
    "erosion": "erosion",
    "opening": "opening",
    "top-hat": "top_hat",
    "gradient": "gradient",
}

SS6_BRANCH_MARKER_FACECOLOR = "#ff8c00"
SS6_BRANCH_MARKER_EDGECOLOR = "#111111"
ACTIVE_CASE_HIGHLIGHT_COLOR = "#ff8c00"


def _fmt_compact_value(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{float(value):.4g}"


def _x_from_index(x_axis: np.ndarray, idx: float) -> float:
    xp = np.arange(int(x_axis.size), dtype=float)
    return float(np.interp(float(idx), xp, np.asarray(x_axis, dtype=float)))


def _has_finite_experimental_metric(row: dict[str, Any], columns: list[str]) -> bool:
    for col in columns:
        value = metric_float(row, col)
        if np.isfinite(value):
            return True
        raw = row.get(col)
        if isinstance(raw, str) and raw.strip():
            return True
    return False


def _show_experimental_metric_row(row: dict[str, Any], columns: list[str]) -> bool:
    return (
        str(row.get("candidate_noise_prefilter_status", "")).strip() != "rejected_noise"
        and _has_finite_experimental_metric(row, columns)
    )


def _experimental_peak_text(row: dict[str, Any], x_axis: np.ndarray) -> str:
    try:
        peak_idx = int(row.get("peak_index", -1))
    except Exception:
        peak_idx = -1
    if 0 <= peak_idx < int(x_axis.size):
        try:
            peak_pos = float(x_axis[peak_idx])
        except Exception:
            peak_pos = float("nan")
        if np.isfinite(peak_pos):
            return rf"$\bf{{peak\ {peak_pos:.1f}}}$"
    return rf"$\bf{{peak\_idx\ {peak_idx}}}$"


def _experimental_metric_label(column: str, aliases: dict[str, str]) -> str:
    alias = str(aliases.get(column, "")).strip()
    if alias:
        return alias
    if "_" not in str(column) and len(str(column)) <= 12:
        return str(column)
    parts = [part for part in str(column).strip().split("_") if part]
    acronym = ""
    for part in parts:
        for ch in part:
            if ch.isalpha():
                acronym += ch.lower()
                break
        else:
            if part[0].isdigit():
                acronym += part[0]
    return acronym or str(column)


def _experimental_metric_text(row: dict[str, Any], column: str) -> str | None:
    value = metric_float(row, column)
    if np.isfinite(value):
        return f"{value:.4g}"
    raw = row.get(column)
    if raw is None:
        return None
    text = str(raw).strip()
    return text if text else None


def metric_bool(row: dict[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan", "none", "null"}:
            return bool(default)
        if text in {"1", "true", "yes"}:
            return True
        if text in {"0", "false", "no"}:
            return False
    numeric = metric_float(row, key)
    if not np.isfinite(numeric):
        return bool(default)
    return int(numeric) == 1


def _metric_int(row: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = metric_float(row, key)
    if not np.isfinite(value):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _viewer_active_decision(row: dict[str, Any], active_profile: str) -> tuple[str, str]:
    if active_profile == "ss6":
        accept = row.get("ss6_accept")
        try:
            accept_int = int(float(accept))
        except Exception:
            accept_int = -1
        branch = str(row.get("ss6_branch", "")).strip()
        if accept_int == 1:
            return "spike", branch
        if accept_int == 0 and branch:
            return "non_spike", branch
        return "unknown", "ss6_missing"
    decision_key = "ss5_decision" if active_profile == "ss5" else "ss4_decision"
    decision = str(row.get("primary_active_decision", row.get(decision_key, "non_spike"))).strip()
    reason = str(row.get("primary_active_reason", row.get(f"{active_profile}_reason", ""))).strip()
    return decision, reason


def _viewer_candidate_color(row: dict[str, Any], active_profile: str) -> str:
    if active_profile != "ss6":
        return candidate_status_color(row, active_profile)
    status = str(row.get("candidate_noise_prefilter_status", ""))
    if status == "rejected_noise":
        return "#7f7f7f"
    decision, _reason = _viewer_active_decision(row, active_profile)
    if decision == "spike":
        return "#d62728"
    if decision == "non_spike":
        return "#1f77b4"
    return "#9467bd"


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


def _compatible_signal_cube(candidate: Any, reference_shape: tuple[int, ...]) -> bool:
    arr = np.asarray(candidate)
    return tuple(arr.shape) == tuple(reference_shape)


def _filter_despike_rows_for_shape(
    rows: list[dict[str, Any]],
    *,
    h: int,
    w: int,
    n: int,
    y_key: str,
    x_key: str,
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    dropped = 0
    index_keys = (
        "original_peak_index",
        "detected_peak_index",
        "context_left",
        "context_right",
        "tested_left",
        "tested_right",
        "left_erosion_contact",
        "right_erosion_contact",
        "dilation_contact",
        "left_anchor",
        "right_anchor",
        "cell_left",
        "cell_right",
        "context_ss1_peak_index",
    )
    for row in rows:
        try:
            y = int(row.get(y_key, -1))
            x = int(row.get(x_key, -1))
        except Exception:
            dropped += 1
            continue
        if not (0 <= y < h and 0 <= x < w):
            dropped += 1
            continue
        bad_index = False
        for key in index_keys:
            value = row.get(key, "")
            if value in {"", None}:
                continue
            try:
                idx = int(value)
            except Exception:
                bad_index = True
                break
            if not (0 <= idx < n):
                bad_index = True
                break
        if bad_index:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def show_cache(cache: dict[str, Any], cfg: Any | None = None) -> None:
    x_axis = np.asarray(cache["x_axis"], dtype=float)
    spectra = np.asarray(cache["spectra"], dtype=float)
    corrected = np.asarray(cache["corrected_spectra"], dtype=float)
    score_map = np.asarray(cache["score_map"], dtype=float)
    metadata = dict(cache.get("metadata", {}))
    candidate_rows = [dict(row) for row in cache.get("candidate_records", [])]
    small_rows = [dict(row) for row in cache.get("small_morphology", [])]
    overlays = cache.get("overlays", {})
    chords = [dict(row) for row in cache.get("despike_chords", [])]
    despike_attempts: list[dict[str, Any]] = []
    active_profile = str(getattr(cfg, "decision_profile", metadata.get("decision_profile", "ss4"))).strip().lower()
    viewer_cfg = dict(getattr(cfg, "viewer", {}) if cfg is not None else {})
    experimental_cfg = dict(getattr(cfg, "experimental_features", {}) if cfg is not None else {})
    ss6_cfg = dict(getattr(cfg, "ss6", {}) if cfg is not None else {})
    ss6_metric_names = dict(ss6_cfg.get("metric_names", {}))
    despike_cfg = dict(getattr(cfg, "despike", {}) if cfg is not None else {})
    experimental_columns = [str(col) for col in experimental_cfg.get("viewer_columns", []) if str(col).strip()]
    ss6_columns = [str(col) for col in ss6_cfg.get("viewer_columns", []) if str(col).strip()]
    display_columns = list(dict.fromkeys(experimental_columns + ss6_columns))
    experimental_label_aliases = {
        str(key): str(value)
        for key, value in dict(experimental_cfg.get("viewer_label_aliases", {})).items()
        if str(key).strip() and str(value).strip()
    }
    for key, value in dict(ss6_cfg.get("viewer_label_aliases", {})).items():
        if str(key).strip() and str(value).strip():
            experimental_label_aliases[str(key)] = str(value)
    experimental_state = {"loaded": False, "message": "Experimental features not loaded."}
    load_messages: list[str] = []
    loaded_any = False
    exp_path_raw = str(experimental_cfg.get("features_path", "")).strip()
    if exp_path_raw:
        exp_path = Path(exp_path_raw)
        if exp_path.exists():
            extra_rows, _extra_columns = load_extra_feature_rows(exp_path)
            join_info = join_extra_feature_rows(candidate_rows, extra_rows)
            loaded_any = True
            load_messages.append(f"exp matched {join_info['matched_rows']} / {join_info['loaded_rows']}")
            print(f"experimental features path: {exp_path}")
            print(f"experimental features rows loaded: {join_info['loaded_rows']}")
            print(f"experimental features rows matched to candidates: {join_info['matched_rows']}")
        else:
            load_messages.append("exp missing")
    needs_ss6 = (
        bool(ss6_columns)
        or any(col.startswith("ss6_") for col in experimental_columns)
        or (active_profile == "ss6" and bool(ss6_cfg.get("auto_load_in_viewer", True)))
    )
    ss6_path_raw = str(ss6_cfg.get("decisions_path", "")).strip()
    if needs_ss6 and ss6_path_raw:
        ss6_path = Path(ss6_path_raw)
        if ss6_path.exists():
            ss6_rows, _ss6_columns = load_extra_feature_rows(ss6_path)
            join_info = join_extra_feature_rows(candidate_rows, ss6_rows)
            loaded_any = True
            load_messages.append(f"ss6 matched {join_info['matched_rows']} / {join_info['loaded_rows']}")
            print(f"ss6 decisions path: {ss6_path}")
            print(f"ss6 decision rows loaded: {join_info['loaded_rows']}")
            print(f"ss6 rows matched to candidates: {join_info['matched_rows']}")
            branch_counts_loaded: dict[str, int] = {}
            for row in candidate_rows:
                branch = str(row.get("ss6_branch", "")).strip()
                if branch:
                    branch_counts_loaded[branch] = branch_counts_loaded.get(branch, 0) + 1
            unique_branches = sorted(branch_counts_loaded.keys())
            print(f"ss6 unique branches: {unique_branches}")
            print(f"ss6 branch counts: {branch_counts_loaded}")
            unknown_branches = [branch for branch in unique_branches if branch not in SS6_KNOWN_BRANCHES]
            if unknown_branches:
                print(f"warning: ss6 decisions contain unknown branches: {unknown_branches}")
        else:
            load_messages.append("ss6 missing")
            if active_profile == "ss6":
                print("decision_profile is ss6, but ss6_decisions.csv was not found.")
                print("Run: python -m muonfinder_core.compute_ss6_decisions --config muonfinder_core/config_core.json")
    finite_metric_rows = sum(1 for row in candidate_rows if _show_experimental_metric_row(row, display_columns))
    print(f"experimental rows with finite/displayable configured metrics: {finite_metric_rows}")
    print(f"experimental viewer columns: {display_columns}")
    if loaded_any:
        experimental_state = {"loaded": True, "message": "; ".join(load_messages)}
    despike_path_raw = str(despike_cfg.get("corrected_path", "")).strip()
    if despike_path_raw:
        despike_path = Path(despike_path_raw)
        if despike_path.exists():
            despike_bundle = load_despike_bundle(despike_path)
            corrected_candidate = np.asarray(despike_bundle.get("corrected_spectra", corrected), dtype=float)
            print(f"despike corrected path: {despike_path}")
            if _compatible_signal_cube(corrected_candidate, tuple(spectra.shape)):
                corrected = corrected_candidate
                raw_chords = [dict(row) for row in despike_bundle.get("despike_chords", [])]
                raw_attempts = [dict(row) for row in despike_bundle.get("despike_attempt_rows", [])]
                h, w, n = map(int, spectra.shape)
                chords, dropped_chords = _filter_despike_rows_for_shape(raw_chords, h=h, w=w, n=n, y_key="y", x_key="x")
                despike_attempts, dropped_attempts = _filter_despike_rows_for_shape(raw_attempts, h=h, w=w, n=n, y_key="compact_y", x_key="compact_x")
                print(f"despike chords loaded: {len(chords)}")
                print(f"despike attempts loaded: {len(despike_attempts)}")
                if dropped_chords:
                    print(f"[viewer] ignored {dropped_chords} despike chord rows outside viewer cache shape")
                if dropped_attempts:
                    print(f"[viewer] ignored {dropped_attempts} despike attempt rows outside viewer cache shape")
            else:
                print("[viewer] ignored despike corrected data because shape does not match viewer cache")
                print(f"viewer cache shape: {tuple(spectra.shape)}")
                print(f"despike corrected shape: {tuple(corrected_candidate.shape)}")
                print(f"despike corrected path: {despike_path}")
        else:
            print("despike corrected file not found. Run compute_despike first.")
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
            if _viewer_active_decision(row, active_profile)[0] == "spike"
        ],
        dtype=float,
    )
    finite_map = score_map[np.isfinite(score_map)]
    map_vmin = None
    map_vmax = None
    if finite_map.size:
        try:
            p_lo, p_hi = viewer_cfg.get("map_color_percentiles", [5, 95])
            map_vmin = float(np.nanpercentile(finite_map, float(p_lo)))
            map_vmax = float(np.nanpercentile(finite_map, float(p_hi)))
        except Exception:
            map_vmin = float(np.nanmin(finite_map))
            map_vmax = float(np.nanmax(finite_map))
        if not (np.isfinite(map_vmin) and np.isfinite(map_vmax) and map_vmax > map_vmin):
            try:
                map_vmin = float(np.nanmin(finite_map))
                map_vmax = float(np.nanmax(finite_map))
            except Exception:
                map_vmin = None
                map_vmax = None

    H, W = score_map.shape
    current = {"y": 0, "x": 0, "morph_idx": 0, "chord_idx": 0, "active_case_candidate_id": "", "active_case_peak_index": -1}
    frozen = {"state": False}
    spectrum_home = {"xlim": None, "ylim": None}
    preserve_spec_limits = {"state": False}
    cycle_state = {"overlay_key": "", "index": 0}
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
    ss6_grid = gs[1, 0].subgridspec(2, 1, height_ratios=[0.28, 0.72], hspace=0.04)
    ax_ss6_info = fig.add_subplot(ss6_grid[0, 0])
    ax_ss6_info.set_xticks([])
    ax_ss6_info.set_yticks([])
    ss6_block_grid = ss6_grid[1, 0].subgridspec(1, 3, wspace=0.16)
    ax_ss6_blocks = [fig.add_subplot(ss6_block_grid[0, i]) for i in range(3)]
    for ax in ax_ss6_blocks:
        ax.set_xticks([])
        ax.set_yticks([])
    chk_grid = gs[1, 1].subgridspec(1, 3, wspace=0.20)
    ax_chk_blocks = [fig.add_subplot(chk_grid[0, i]) for i in range(3)]
    for ax in ax_chk_blocks:
        ax.set_xticks([])
        ax.set_yticks([])

    states = {name: (name in {"located muon", "raw", "corrected"}) for name in CHECKBOX_ORDER}
    located_count = int(len(accepted_offsets)) if accepted_offsets.size else 0
    display_label_map = {
        name: (f"located muon ({located_count})" if name == "located muon" and located_count > 0 else name)
        for name in CHECKBOX_ORDER
    }
    display_to_state = {label: state for state, label in display_label_map.items()}
    checks = []
    block_size = int(np.ceil(len(CHECKBOX_ORDER) / 3))
    for block_index, ax_chk in enumerate(ax_chk_blocks):
        start = block_index * block_size
        stop = min(len(CHECKBOX_ORDER), start + block_size)
        state_labels = CHECKBOX_ORDER[start:stop]
        labels = [display_label_map[label] for label in state_labels]
        actives = [states[label] for label in state_labels]
        chk = CheckButtons(ax_chk, labels=labels, actives=actives)
        for txt in chk.labels:
            txt.set_fontsize(11)
        checks.append(chk)

    def current_window() -> int:
        return int(morph_windows[current["morph_idx"] % len(morph_windows)])

    def current_rows() -> list[dict[str, Any]]:
        return rows_by_pixel.get((int(current["y"]), int(current["x"])), [])

    def current_active_case_row() -> dict[str, Any] | None:
        active_candidate_id = str(current.get("active_case_candidate_id", "")).strip()
        active_peak_index = int(current.get("active_case_peak_index", -1))
        for row in current_rows():
            if active_candidate_id and str(row.get("candidate_id", "")).strip() == active_candidate_id:
                return row
            if active_peak_index >= 0 and int(row.get("peak_index", -1)) == active_peak_index:
                return row
        return None

    def current_spectrum_chords() -> list[dict[str, Any]]:
        return [
            dict(chord)
            for chord in chords
            if int(chord.get("y", -1)) == int(current["y"]) and int(chord.get("x", -1)) == int(current["x"])
        ]

    def current_spectrum_attempts() -> list[dict[str, Any]]:
        rows = [
            dict(item)
            for item in despike_attempts
            if int(item.get("compact_y", -1)) == int(current["y"]) and int(item.get("compact_x", -1)) == int(current["x"])
        ]
        active = [
            row
            for row in rows
            if str(row.get("status", "")).strip() != "no_local_ss6_positive_candidate_remaining_in_any_context"
        ]
        active.sort(
            key=lambda item: (
                int(item.get("attempt_sequence", 0) or 0),
                int(item.get("stage_index", 0) or 0),
                int(item.get("attempt_index_within_stage", 0) or 0),
                int(item.get("detected_peak_index", -1) or -1),
            )
        )
        return active

    def current_attempt_index() -> int:
        spectrum_attempts = current_spectrum_attempts()
        if not spectrum_attempts:
            return 0
        return int(current["chord_idx"] % len(spectrum_attempts))

    def current_attempt() -> dict[str, Any] | None:
        spectrum_attempts = current_spectrum_attempts()
        if not spectrum_attempts:
            return None
        return spectrum_attempts[current_attempt_index()]

    def active_overlay_cycle_key() -> str:
        selected_branch = ss6_branch_state["selected"]
        if selected_branch is not None:
            return f"ss6_branch:{selected_branch}"
        if states["located muon"]:
            return "located_muon"
        return ""

    def _unique_cycle_pixels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        first_by_pixel: dict[tuple[int, int], dict[str, Any]] = {}
        for row in rows:
            key = (int(row.get("y", -1)), int(row.get("x", -1)))
            if key not in first_by_pixel:
                first_by_pixel[key] = dict(row)
        return sorted(
            first_by_pixel.values(),
            key=lambda row: (int(row.get("y", -1)), int(row.get("x", -1)), int(row.get("peak_index", -1))),
        )

    def get_active_overlay_cycle_rows() -> tuple[str, list[dict[str, Any]]]:
        selected_branch = ss6_branch_state["selected"]
        if selected_branch is not None:
            rows = [
                dict(row)
                for row in candidate_rows
                if str(row.get("ss6_branch", "")).strip() == str(selected_branch)
            ]
            return f"ss6_branch:{selected_branch}", _unique_cycle_pixels(rows)
        if states["located muon"]:
            rows = [
                dict(row)
                for row in candidate_rows
                if _viewer_active_decision(row, active_profile)[0] == "spike"
            ]
            return "located_muon", _unique_cycle_pixels(rows)
        return "", []

    def _reset_overlay_cycle_if_needed() -> None:
        overlay_key = active_overlay_cycle_key()
        if cycle_state["overlay_key"] != overlay_key:
            cycle_state["overlay_key"] = overlay_key
            cycle_state["index"] = 0

    def jump_to_case(case_row: dict[str, Any]) -> None:
        current["x"] = int(case_row.get("x", current["x"]))
        current["y"] = int(case_row.get("y", current["y"]))
        current["chord_idx"] = 0
        current["active_case_candidate_id"] = str(case_row.get("candidate_id", "")).strip()
        current["active_case_peak_index"] = int(case_row.get("peak_index", -1))

    def cycle_active_overlay_case(step: int) -> None:
        overlay_key, rows = get_active_overlay_cycle_rows()
        if not overlay_key:
            return
        if not rows:
            return
        if cycle_state["overlay_key"] != overlay_key:
            cycle_state["overlay_key"] = overlay_key
            cycle_state["index"] = 0
        else:
            cycle_state["index"] = int((int(cycle_state["index"]) + int(step)) % len(rows))
        row = rows[int(cycle_state["index"])]
        jump_to_case(row)

    map_im = ax_map.imshow(score_map, cmap="viridis", origin="upper", interpolation="nearest", vmin=map_vmin, vmax=map_vmax)
    located_scatter = ax_map.scatter([], [], s=26, c="#d62728", marker="s", linewidths=0.0, alpha=0.90)
    ss6_branch_scatter = ax_map.scatter(
        [],
        [],
        s=48,
        c=SS6_BRANCH_MARKER_FACECOLOR,
        marker="s",
        linewidths=0.8,
        edgecolors=SS6_BRANCH_MARKER_EDGECOLOR,
        alpha=0.92,
        zorder=5,
    )
    cursor_marker, = ax_map.plot(
        [current["x"]],
        [current["y"]],
        marker="s",
        markersize=8.0,
        markerfacecolor="none",
        markeredgecolor="white",
        markeredgewidth=1.6,
        linestyle="None",
        zorder=8,
    )
    ax_map.set_title("score map", fontsize=13)
    ax_map.set_xlabel("x (pixel)", fontsize=11)
    ax_map.set_ylabel("y (pixel)", fontsize=11)
    ax_map.tick_params(labelsize=10)
    if bool(viewer_cfg.get("show_map_colorbar", False)):
        fig.colorbar(map_im, ax=ax_map, fraction=0.046, pad=0.04)

    branch_counts: dict[str, int] = {}
    branch_pixels: dict[str, np.ndarray] = {}
    for row in candidate_rows:
        branch = str(row.get("ss6_branch", "")).strip()
        if not branch:
            continue
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
    for branch in list(branch_counts.keys()):
        coords = sorted({(int(row["x"]), int(row["y"])) for row in candidate_rows if str(row.get("ss6_branch", "")).strip() == branch})
        branch_pixels[branch] = np.asarray(coords, dtype=float) if coords else np.empty((0, 2), dtype=float)
    unknown_branch_names = sorted(branch for branch in branch_counts if branch not in SS6_KNOWN_BRANCHES)
    ss6_branch_state = {"selected": None, "message": ""}

    def _branch_label(branch_name: str) -> str:
        return f"{branch_name} ({int(branch_counts.get(branch_name, 0))})"

    accepted_names = [str(item["name"]) for item in SS6_BRANCH_DEFINITIONS if str(item.get("category", "")) == "accepted"]
    rejected_names = [str(item["name"]) for item in SS6_BRANCH_DEFINITIONS if str(item.get("category", "")) != "accepted"]
    ss6_panel_groups = [
        [("none / off", None)] + [(_branch_label(name), name) for name in accepted_names],
        [(_branch_label(name), name) for name in rejected_names],
        [(f"other: {branch} ({int(branch_counts.get(branch, 0))})", branch) for branch in unknown_branch_names],
    ]
    ss6_label_to_branch: dict[str, str | None] = {}
    ss6_button_index: dict[str, tuple[Any, int]] = {}
    ss6_controls: list[Any] = []
    ss6_updating = {"state": False}
    for ax, group in zip(ax_ss6_blocks, ss6_panel_groups):
        labels = [label for label, _branch in group] or [""]
        actives = [label == "none / off" for label in labels]
        chk = CheckButtons(ax, labels=labels, actives=actives)
        ss6_controls.append(chk)
        for idx, (label, branch) in enumerate(group):
            ss6_label_to_branch[label] = branch
            ss6_button_index[label] = (chk, idx)
            txt = chk.labels[idx]
            txt.set_fontsize(8.4)
            if branch is None:
                txt.set_fontweight("bold")
            elif int(branch_counts.get(branch, 0)) == 0:
                txt.set_color("#9a9a9a")
        if not group:
            for txt in chk.labels:
                txt.set_text("")

    def _set_ss6_info(text: str) -> None:
        ax_ss6_info.clear()
        ax_ss6_info.set_xticks([])
        ax_ss6_info.set_yticks([])
        ax_ss6_info.text(
            0.01,
            0.92,
            "SS6 branch overlay",
            transform=ax_ss6_info.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        ax_ss6_info.text(
            0.01,
            0.52,
            text,
            transform=ax_ss6_info.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            color="#333333",
        )
        ax_ss6_info.text(
            0.01,
            0.12,
            "h/j: overlay spectra  |  a/x: despike attempts",
            transform=ax_ss6_info.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            color="#555555",
        )

    def _apply_ss6_branch_selection(branch_name: str | None) -> None:
        ss6_branch_state["selected"] = branch_name
        cycle_state["overlay_key"] = ""
        cycle_state["index"] = 0
        if branch_name is None:
            ss6_branch_state["message"] = "Overlay off."
        elif int(branch_counts.get(branch_name, 0)) <= 0:
            ss6_branch_state["message"] = "No candidates for selected SS6 branch."
        else:
            ss6_branch_state["message"] = f"{branch_name}: {int(branch_counts.get(branch_name, 0))} candidate rows"
        _set_ss6_info(ss6_branch_state["message"])

    def _set_ss6_branch_buttons(active_branch: str | None) -> None:
        ss6_updating["state"] = True
        try:
            for label, (chk, idx) in ss6_button_index.items():
                should_be_on = (active_branch is None and label == "none / off") or (ss6_label_to_branch.get(label) == active_branch and active_branch is not None)
                current_state = bool(chk.get_status()[idx])
                if current_state != should_be_on:
                    chk.set_active(idx)
        finally:
            ss6_updating["state"] = False

    def _on_ss6_branch_toggle(label: str) -> None:
        if ss6_updating["state"]:
            return
        branch = ss6_label_to_branch.get(str(label))
        _set_ss6_branch_buttons(branch)
        _apply_ss6_branch_selection(branch)
        update()

    for chk in ss6_controls:
        chk.on_clicked(_on_ss6_branch_toggle)
    if ss6_path_raw and not Path(ss6_path_raw).exists():
        _set_ss6_info("SS6 decisions file not found. Run compute_ss6_decisions first.")
    else:
        _apply_ss6_branch_selection(None)

    def _set_suptitle() -> None:
        compact = (int(current["y"]), int(current["x"]))
        source = coord_map.get(compact, compact)
        spectrum_attempts = current_spectrum_attempts()
        attempt = current_attempt()
        attempt_count = len(spectrum_attempts)
        attempt_index = current_attempt_index() + 1 if attempt_count else 0
        attempt_window = int(despike_cfg.get("morph_window", 3))
        fig.suptitle(
            f"spectrum @ compact(y={compact[0]}, x={compact[1]}) -> "
            f"source(y={source[0]}, x={source[1]}) | "
            f"despike attempt {attempt_index}/{attempt_count} (a/x) | morph window {attempt_window}",
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
        selected_branch = ss6_branch_state["selected"]
        branch_points = branch_pixels.get(selected_branch, np.empty((0, 2), dtype=float)) if selected_branch is not None else np.empty((0, 2), dtype=float)
        if selected_branch is not None and branch_points.size:
            ss6_branch_scatter.set_offsets(branch_points)
            ss6_branch_scatter.set_visible(True)
        else:
            ss6_branch_scatter.set_offsets(np.empty((0, 2), dtype=float))
            ss6_branch_scatter.set_visible(False)

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

    def _contact_context_bounds(row: dict[str, Any], pad: int | None = None) -> tuple[int, int]:
        use_pad = int(despike_cfg.get("despike_context_window_pad", 0) if pad is None else pad)
        return get_contact_context_bounds(row, len(x_axis), pad=use_pad)

    def _draw_pce_overlay(rows: list[dict[str, Any]]) -> None:
        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        first = True

        def _draw_single_pce_debug(debug: dict[str, Any], start: int, context_left: int | None = None, context_right: int | None = None, peak_idx: int | None = None) -> None:
            nonlocal first
            x_rel = [int(v) for v in debug.get("curve_x_rel", [])]
            y_vals = np.asarray(debug.get("curve_y", []), dtype=float)
            if not x_rel or y_vals.size != len(x_rel):
                return
            x_plot_idx = np.asarray([start + int(v) for v in x_rel], dtype=int)
            if np.any(x_plot_idx < 0) or np.any(x_plot_idx >= len(x_axis)):
                return
            if context_left is not None and context_right is not None:
                li = int(max(0, context_left))
                ri = int(min(len(x_axis) - 1, context_right))
                if ri >= li:
                    ax_spec.axvspan(x_axis[li], x_axis[ri], color="#666666", alpha=0.08, zorder=0)
            else:
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
                abs_idx = start + rel
                if not (0 <= abs_idx < len(x_axis)):
                    return None
                return ax_spec.plot(
                    [x_axis[abs_idx]],
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
            if peak_idx is not None and 0 <= int(peak_idx) < len(x_axis):
                ax_spec.axvline(x_axis[int(peak_idx)], color="#d62728", linestyle=":", linewidth=1.1, alpha=0.8, zorder=3)
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
        attempt = current_attempt()
        if attempt is not None:
            debug = attempt.get("pce_t98_debug_local", {})
            if isinstance(debug, dict) and debug:
                _draw_single_pce_debug(
                    debug,
                    start=int(attempt.get("tested_left", attempt.get("cell_left", 0))),
                    context_left=int(attempt.get("context_left", attempt.get("tested_left", 0))),
                    context_right=int(attempt.get("context_right", attempt.get("tested_right", 0))),
                    peak_idx=int(attempt.get("detected_peak_index", -1)),
                )
                ax_spec._pce_legend_handles = legend_handles  # type: ignore[attr-defined]
                ax_spec._pce_legend_labels = legend_labels  # type: ignore[attr-defined]
                return
        for row in rows:
            if str(row.get("candidate_noise_prefilter_status", "")) == "rejected_noise":
                continue
            debug = row.get("pce_t98_debug", {})
            if not isinstance(debug, dict):
                continue
            _draw_single_pce_debug(debug, start=int(row.get("start", 0)))
        ax_spec._pce_legend_handles = legend_handles  # type: ignore[attr-defined]
        ax_spec._pce_legend_labels = legend_labels  # type: ignore[attr-defined]

    def _draw_pre_edge_overlay(rows: list[dict[str, Any]], raw_sig: np.ndarray) -> None:
        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        status_style = {
            "ok": {"color": "#4c78a8", "linestyle": "-", "alpha": 0.32},
            "transient": {"color": "#7f7f7f", "linestyle": "--", "alpha": 0.45},
            "neighbor_structure": {"color": "#f58518", "linestyle": "-", "alpha": 0.72},
            "lost": {"color": "#d62728", "linestyle": ":", "alpha": 0.78},
            "context_boundary": {"color": "#6f4c9b", "linestyle": "-.", "alpha": 0.62},
        }
        first = True
        for row in rows:
            if str(row.get("candidate_noise_prefilter_status", "")) == "rejected_noise":
                continue
            debug = _edge_debug(row)
            pre_levels = debug.get("edge_pre_levels", [])
            if not isinstance(pre_levels, list) or not pre_levels:
                continue
            ml = int(debug.get("edge_context_left", row.get("start", 0)))
            mr = int(debug.get("edge_context_right", row.get("end", 0)))
            if 0 <= ml < len(x_axis) and 0 <= mr < len(x_axis) and mr >= ml:
                ax_spec.axvspan(x_axis[ml], x_axis[mr], color="#1f78b4", alpha=0.05)
            apex_idx = int(debug.get("edge_apex_index", row.get("peak_index", 0)))
            apex_x = x_axis[apex_idx] if 0 <= apex_idx < len(x_axis) else None
            for item in pre_levels:
                if not isinstance(item, dict):
                    continue
                level_y = metric_float(item, "level_value")
                left_cross = metric_float(item, "apex_left")
                right_cross = metric_float(item, "apex_right")
                status = str(item.get("component_status", "ok"))
                style = status_style.get(status, status_style["ok"])
                if np.isfinite(left_cross) and np.isfinite(right_cross) and np.isfinite(level_y):
                    lx = _x_from_index(x_axis, left_cross)
                    rx = _x_from_index(x_axis, right_cross)
                    ax_spec.plot(
                        [lx, rx],
                        [level_y, level_y],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=1.1 if not bool(item.get("selected_for_foot")) else 1.8,
                        alpha=style["alpha"],
                    )
                    ax_spec.scatter([lx, rx], [level_y, level_y], s=12, c=style["color"], alpha=min(1.0, style["alpha"] + 0.2), zorder=4)
                elif apex_x is not None and np.isfinite(level_y):
                    ax_spec.plot(
                        [apex_x],
                        [level_y],
                        marker="x",
                        color=style["color"],
                        markersize=5.5,
                        linestyle="None",
                        alpha=min(1.0, style["alpha"] + 0.15),
                        zorder=5,
                    )
                if bool(item.get("neighbor_detected")) and np.isfinite(level_y):
                    neighbor_r = metric_float(item, "neighbor_r")
                    marker_x = _x_from_index(x_axis, right_cross) if np.isfinite(right_cross) else apex_x
                    if marker_x is not None:
                        ax_spec.plot(
                            [marker_x],
                            [level_y],
                            marker="o" if status != "neighbor_structure" else "s",
                            color=style["color"],
                            markersize=4.6 if status != "neighbor_structure" else 5.6,
                            linestyle="None",
                            zorder=5,
                        )
                        if status == "neighbor_structure" and np.isfinite(neighbor_r):
                            ax_spec.text(marker_x, level_y, f"r={neighbor_r:.1f}", fontsize=8, color=style["color"], ha="left", va="bottom")
            foot_left = metric_float(debug, "edge_selected_foot_left")
            foot_right = metric_float(debug, "edge_selected_foot_right")
            foot_value = metric_float(debug, "edge_selected_foot_value")
            foot_index = debug.get("edge_selected_foot_index")
            foot_status = str(debug.get("edge_selected_foot_status", "ok"))
            foot_color = "#f58518" if foot_status == "neighbor_structure_stop" else "#1f78b4"
            if np.isfinite(foot_left) and np.isfinite(foot_right) and np.isfinite(foot_value):
                ax_spec.plot(
                    [_x_from_index(x_axis, foot_left), _x_from_index(x_axis, foot_right)],
                    [foot_value, foot_value],
                    color=foot_color,
                    linestyle=":",
                    linewidth=2.0,
                    alpha=0.95,
                )
            if foot_index is not None:
                foot_idx = int(foot_index)
                if 0 <= foot_idx < len(x_axis):
                    ax_spec.plot(
                        [x_axis[foot_idx]],
                        [raw_sig[foot_idx]],
                        marker="D",
                        color=foot_color,
                        markersize=6.5,
                        linestyle="None",
                        zorder=6,
                    )
            if first:
                legend_handles.extend(
                    [
                        Line2D([0], [0], color="#4c78a8", linewidth=1.2),
                        Line2D([0], [0], color="#7f7f7f", linestyle="--", linewidth=1.2),
                        Line2D([0], [0], color="#f58518", linewidth=1.2),
                        Line2D([0], [0], color="#d62728", linestyle=":", linewidth=1.2),
                        Line2D([0], [0], color="#6f4c9b", linestyle="-.", linewidth=1.2),
                        Line2D([0], [0], color="#1f78b4", linestyle=":", linewidth=1.8),
                    ]
                )
                legend_labels.extend(
                    [
                        "pre-EDGE ok",
                        "pre-EDGE transient/noise",
                        "neighbor-structure stop",
                        "pre-EDGE lost",
                        "pre-EDGE context boundary",
                        "selected foot/base",
                    ]
                )
                first = False
        if legend_handles:
            ax_spec._pre_edge_legend_handles = legend_handles  # type: ignore[attr-defined]
            ax_spec._pre_edge_legend_labels = legend_labels  # type: ignore[attr-defined]

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
            if first and legend_handles:
                first = False
        if legend_handles:
            ax_spec._edge_legend_handles = legend_handles  # type: ignore[attr-defined]
            ax_spec._edge_legend_labels = legend_labels  # type: ignore[attr-defined]

    def _draw_contacts(rows: list[dict[str, Any]], indices: list[int], marker: str, color: str, size: float) -> None:
        points: set[int] = set()
        rows = [row for row in rows if _viewer_active_decision(row, active_profile)[0] == "spike"]
        for row in rows:
            if str(row.get("candidate_noise_prefilter_status", "")).strip() == "rejected_noise":
                continue
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
        active_case = current_active_case_row()
        active_candidate_id = str(active_case.get("candidate_id", "")).strip() if active_case is not None else ""
        metric_rows.sort(
            key=lambda row: (
                0 if active_candidate_id and str(row.get("candidate_id", "")).strip() == active_candidate_id else 1,
                int(row.get("peak_index", 0)),
            )
        )
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
            pce_field = str(ss6_metric_names.get("pce", "pce_negpref_t098_evidence_signed")).strip() or "pce_negpref_t098_evidence_signed"
            pce = metric_float(row, pce_field)
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
            color = _viewer_candidate_color(row, active_profile)
            is_active_case = active_candidate_id and str(row.get("candidate_id", "")).strip() == active_candidate_id
            if is_active_case:
                color = ACTIVE_CASE_HIGHLIGHT_COLOR
            x_pos = float(x_axis[peak])
            _decision, reason = _viewer_active_decision(row, active_profile)
            label = "\n".join(finite_parts + ([reason] if reason else []))
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

    def _draw_experimental_metrics(rows: list[dict[str, Any]]) -> None:
        if not display_columns:
            ax_spec.text(
                0.015,
                0.97,
                "Experimental viewer_columns not configured",
                transform=ax_spec.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                family="monospace",
                bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "#666666", "linewidth": 0.8},
            )
            return
        if not experimental_state["loaded"]:
            ax_spec.text(
                0.015,
                0.97,
                str(experimental_state["message"]),
                transform=ax_spec.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                family="monospace",
                bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "#666666", "linewidth": 0.8},
            )
            return
        display_rows = sorted(
            [row for row in rows if _show_experimental_metric_row(row, display_columns)],
            key=lambda item: int(item.get("peak_index", -1)),
        )
        if not display_rows:
            ax_spec.text(
                0.015,
                0.97,
                "No finite experimental metrics for this spectrum.",
                transform=ax_spec.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                family="monospace",
                bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "#666666", "linewidth": 0.8},
            )
            return
        blocks: list[str] = []
        for row in display_rows:
            block_lines = [_experimental_peak_text(row, x_axis)]
            for col in display_columns:
                text = _experimental_metric_text(row, col)
                if text is not None:
                    block_lines.append(f"{_experimental_metric_label(col, experimental_label_aliases)}: {text}")
            if len(block_lines) > 1:
                blocks.append("\n".join(block_lines))
        if not blocks:
            ax_spec.text(
                0.015,
                0.97,
                "No finite experimental metrics for this spectrum.",
                transform=ax_spec.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                family="monospace",
                bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "#666666", "linewidth": 0.8},
            )
            return
        ax_spec.text(
            0.015,
            0.97,
            "\n\n".join(blocks),
            transform=ax_spec.transAxes,
            ha="left",
            va="top",
            fontsize=8.7,
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#555555", "linewidth": 0.8},
        )

    def _contact_context_spans(rows: list[dict[str, Any]], pad: int | None = None) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for row in rows:
            left, right = _contact_context_bounds(row, pad=pad)
            spans.append((int(left), int(right)))
        indices: list[int] = []
        for left, right in spans:
            indices.extend(range(left, right + 1))
        return to_contiguous_spans(sorted(set(indices)))

    def _draw_active_profile_summary(rows: list[dict[str, Any]]) -> None:
        if not bool(viewer_cfg.get("show_candidate_status_summary_box", False)):
            return
        if active_profile != "ss6" or not rows:
            return
        lines: list[str] = []
        for row in rows[:5]:
            peak_text = _experimental_peak_text(row, x_axis).replace("$\\bf{", "").replace("}$", "").replace("\\_", "_").replace("\\ ", " ")
            branch = str(row.get("ss6_branch", "")).strip() or "unknown"
            try:
                accept_text = str(int(float(row.get("ss6_accept", np.nan))))
            except Exception:
                accept_text = "?"
            lines.append(f"ss6={accept_text} {branch} {peak_text}")
        if len(rows) > 5:
            lines.append(f"+{len(rows) - 5} more")
        ax_spec.text(
            0.015,
            0.90,
            "\n".join(lines),
            transform=ax_spec.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "#666666", "linewidth": 0.8},
        )

    def _draw_despike_metrics_box(attempt: dict[str, Any] | None) -> None:
        if not states["despike metrics"] or attempt is None:
            return
        detected_peak = int(attempt.get("detected_peak_index", -1) or -1)
        peak_pos = _x_from_index(x_axis, detected_peak) if detected_peak >= 0 else float("nan")
        lines = []
        if np.isfinite(peak_pos):
            lines.append(f"peak {peak_pos:.1f}")
        original_branch = str(attempt.get("original_ss6_branch", "")).strip()
        local_branch = str(attempt.get("local_ss6_branch", attempt.get("ss6_local_branch", ""))).strip()
        attempt_type = str(attempt.get("attempt_type", "")).strip()
        attempt_kind = str(attempt.get("attempt_kind", attempt_type)).strip()
        status = str(attempt.get("status", "")).strip()
        if attempt_type == "parent":
            if original_branch:
                lines.append(f"ss6 {original_branch}")
        elif attempt_kind == "mask_residual_cleanup":
            lines.append("mask residual cleanup")
        else:
            if status == "context_ss1_low":
                lines.append("local rejected")
            elif local_branch:
                lines.append(f"local ss6 {local_branch}")
        for key in ("ss1", "pce", "edge", "eel", "resid"):
            val = metric_float(attempt, key)
            if np.isfinite(val):
                lines.append(f"{key}={val:.3g}")
        despike_noise = metric_float(attempt, "despike_noise_value")
        if np.isfinite(despike_noise):
            lines.append(f"noise={despike_noise:.3g}")
        if status:
            if status == "corrected_parent":
                lines.append("corrected")
            elif status == "corrected_iterative_local_ss6":
                lines.append("corrected")
            elif status == "context_ss1_low":
                lines.append("reason=context_ss1_low")
            elif status == "corrected_mask_residual_cleanup":
                resid_h = metric_float(attempt, "residual_height")
                resid_hz = metric_float(attempt, "residual_height_noise_z")
                if np.isfinite(resid_hz):
                    lines.append(f"hchord={resid_hz:.3g}")
                if np.isfinite(resid_h):
                    lines.append(f"h={resid_h:.3g}")
                lines.append("corrected")
            elif status == "rejected_by_despike_noise_height":
                lines.append("rejected: noise height")
                hchord = metric_float(attempt, "height_above_chord_noise_z")
                if np.isfinite(hchord):
                    lines.append(f"hchord={hchord:.3g}")
            elif status == "rejected_mask_residual_below_noise_height":
                lines.append("rejected: mask residual")
                resid_hz = metric_float(attempt, "residual_height_noise_z")
                resid_h = metric_float(attempt, "residual_height")
                if np.isfinite(resid_hz):
                    lines.append(f"hchord={resid_hz:.3g}")
                if np.isfinite(resid_h):
                    lines.append(f"h={resid_h:.3g}")
            elif status == "local_missing_metric":
                missing = str(attempt.get("skipped_reason", "")).strip()
                lines.append(f"missing: {missing}" if missing else "missing metric")
            elif status.startswith("skipped_"):
                lines.append(f"skipped: {status.removeprefix('skipped_').replace('_', ' ')}")
            else:
                lines.append(status.replace("_", " "))
        if metric_bool(attempt, "chord_support_adjustment_applied"):
            lines.append("support adjusted")
        ax_spec.text(
            0.015,
            0.86,
            "\n".join(lines),
            transform=ax_spec.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#17becf", "linewidth": 0.8},
        )

    raw_sig = np.asarray(spectra[current["y"], current["x"], :], dtype=float)

    def _draw_spectrum() -> None:
        nonlocal raw_sig
        prev_xlim = ax_spec.get_xlim() if preserve_spec_limits["state"] else None
        prev_ylim = ax_spec.get_ylim() if preserve_spec_limits["state"] else None
        ax_spec.clear()
        ax_spec._pce_legend_handles = []  # type: ignore[attr-defined]
        ax_spec._pce_legend_labels = []  # type: ignore[attr-defined]
        ax_spec._pre_edge_legend_handles = []  # type: ignore[attr-defined]
        ax_spec._pre_edge_legend_labels = []  # type: ignore[attr-defined]
        ax_spec._edge_legend_handles = []  # type: ignore[attr-defined]
        ax_spec._edge_legend_labels = []  # type: ignore[attr-defined]
        rows = current_rows()
        raw_sig = np.asarray(spectra[current["y"], current["x"], :], dtype=float)
        corrected_sig = np.asarray(corrected[current["y"], current["x"], :], dtype=float)
        if states["corrected"]:
            ax_spec.plot(x_axis, raw_sig, color="#d62728", linewidth=1.7, label="raw")
            ax_spec.plot(
                x_axis,
                corrected_sig,
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
            active_case = current_active_case_row()
            active_candidate_id = str(active_case.get("candidate_id", "")).strip() if active_case is not None else ""
            for row in rows:
                color = _viewer_candidate_color(row, active_profile)
                is_active_case = active_candidate_id and str(row.get("candidate_id", "")).strip() == active_candidate_id
                line_color = ACTIVE_CASE_HIGHLIGHT_COLOR if is_active_case else color
                line_width = 2.2 if is_active_case else 1.5
                ax_spec.axvline(x_axis[int(row["peak_index"])], color=line_color, linestyle="--", linewidth=line_width)
                if is_active_case:
                    ax_spec.plot(
                        [x_axis[int(row["peak_index"])]],
                        [raw_sig[int(row["peak_index"])]],
                        marker="o",
                        color=ACTIVE_CASE_HIGHLIGHT_COLOR,
                        markersize=5.5,
                        linestyle="None",
                        zorder=6,
                    )

        morph_row = small_by_pixel.get((int(current["y"]), int(current["x"])), {})
        if states["dilation contacts"] or states["erosion contacts"]:
            for left, right in _contact_context_spans(rows):
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
                    noise_line = f"noise = {_fmt_compact_value(val)} ({src})"
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
        if states["pre-EDGE"]:
            _draw_pre_edge_overlay(rows, raw_sig)
        if states["EDGE"]:
            _draw_edge_overlay(rows, raw_sig)

        if states["dilation contacts"]:
            _draw_contacts(rows, morph_row.get("dilation_contacts", []), "^", "#ff7f0e", 8.5)
        if states["erosion contacts"]:
            _draw_contacts(rows, morph_row.get("erosion_contacts", []), "o", "#111111", 7.5)

        if states["despike chords"]:
            attempt = current_attempt()
            if attempt is not None:
                ctx_left = _metric_int(attempt, "fixed_context_left", _metric_int(attempt, "context_left"))
                ctx_right = _metric_int(attempt, "fixed_context_right", _metric_int(attempt, "context_right"))
                cell_left = _metric_int(attempt, "cell_left", _metric_int(attempt, "tested_left"))
                cell_right = _metric_int(attempt, "cell_right", _metric_int(attempt, "tested_right"))
                left_anchor = _metric_int(attempt, "left_anchor", _metric_int(attempt, "left_erosion_contact"))
                right_anchor = _metric_int(attempt, "right_anchor", _metric_int(attempt, "right_erosion_contact"))
                detected_peak = _metric_int(attempt, "detected_peak_index", _metric_int(attempt, "dilation_contact"))
                mask_left = _metric_int(attempt, "mask_interval_start")
                mask_right = _metric_int(attempt, "mask_interval_end")
                status = str(attempt.get("status", ""))
                if mask_left is not None and mask_right is not None:
                    pli = int(mask_left)
                    pri = int(mask_right)
                    if 0 <= pli < len(x_axis) and 0 <= pri < len(x_axis) and pri >= pli:
                        ax_spec.axvspan(x_axis[pli], x_axis[pri], color="#ffb347", alpha=0.12)
                if ctx_left is not None and ctx_right is not None:
                    pli = int(ctx_left)
                    pri = int(ctx_right)
                    if 0 <= pli < len(x_axis) and 0 <= pri < len(x_axis) and pri >= pli:
                        ax_spec.axvspan(x_axis[pli], x_axis[pri], color="#bdbdbd", alpha=0.10)
                try:
                    context_eros = [int(v) for v in json.loads(str(attempt.get("context_erosion_contacts", "[]")))]
                except Exception:
                    context_eros = []
                try:
                    context_dils = [int(v) for v in json.loads(str(attempt.get("context_dilation_contacts", "[]")))]
                except Exception:
                    context_dils = []
                if context_dils:
                    pts = [idx for idx in context_dils if 0 <= idx < len(x_axis)]
                    if pts:
                        ax_spec.scatter(x_axis[pts], raw_sig[pts], marker="^", s=34, c="#ff7f0e", alpha=0.8, zorder=4)
                if context_eros:
                    pts = [idx for idx in context_eros if 0 <= idx < len(x_axis)]
                    if pts:
                        ax_spec.scatter(x_axis[pts], raw_sig[pts], marker="o", s=22, c="#111111", alpha=0.75, zorder=4)
                if cell_left is not None and cell_right is not None:
                    tli = int(cell_left)
                    tri = int(cell_right)
                    if 0 <= tli < len(x_axis) and 0 <= tri < len(x_axis) and tri >= tli:
                        ax_spec.axvspan(x_axis[tli], x_axis[tri], color="#17becf", alpha=0.14)
                if detected_peak is not None:
                    dpi = int(detected_peak)
                    if 0 <= dpi < len(x_axis):
                        ax_spec.plot([x_axis[dpi]], [raw_sig[dpi]], marker="x", color="#d62728", markersize=7.0, linestyle="None")
                if left_anchor is not None and right_anchor is not None:
                    li = int(left_anchor)
                    ri = int(right_anchor)
                    if 0 <= li < len(x_axis) and 0 <= ri < len(x_axis) and ri > li:
                        if status.startswith("corrected_"):
                            yl = float(raw_sig[li])
                            yr = float(raw_sig[ri])
                            ax_spec.plot([x_axis[li], x_axis[ri]], [yl, yr], color="#17becf", linewidth=1.9, label="despike chord")
                        else:
                            yl = float(raw_sig[li])
                            yr = float(raw_sig[ri])
                            ax_spec.plot([x_axis[li], x_axis[ri]], [yl, yr], color="#17becf", linewidth=1.2, linestyle="--", alpha=0.85, label="tested chord")
                        ax_spec.plot([x_axis[li]], [yl], marker="o", color="#17becf", markersize=5.5, linestyle="None")
                        ax_spec.plot([x_axis[ri]], [yr], marker="o", color="#17becf", markersize=5.5, linestyle="None")
        _draw_despike_metrics_box(current_attempt())

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
        if preserve_spec_limits["state"] and prev_xlim is not None and prev_ylim is not None:
            ax_spec.set_xlim(*prev_xlim)
            ax_spec.set_ylim(*prev_ylim)
        else:
            ax_spec.set_xlim(*spectrum_home["xlim"])
            ax_spec.set_ylim(*spectrum_home["ylim"])
        if states["noise filter"]:
            _draw_noise_filter(rows, raw_sig)
        if states["metrics"]:
            _draw_metrics(rows, raw_sig)
        if states["Experimental metrics"]:
            _draw_experimental_metrics(rows)
        _draw_active_profile_summary(rows)
        handles1, labels1 = ax_spec.get_legend_handles_labels()
        handles2 = list(getattr(ax_spec, "_pce_legend_handles", []))
        labels2 = list(getattr(ax_spec, "_pce_legend_labels", []))
        handles3 = list(getattr(ax_spec, "_pre_edge_legend_handles", []))
        labels3 = list(getattr(ax_spec, "_pre_edge_legend_labels", []))
        handles4 = list(getattr(ax_spec, "_edge_legend_handles", []))
        labels4 = list(getattr(ax_spec, "_edge_legend_labels", []))
        handles, labels = _dedup_legend(handles1 + handles2 + handles3 + handles4, labels1 + labels2 + labels3 + labels4)
        if handles:
            ax_spec.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.92)

    def update() -> None:
        _reset_overlay_cycle_if_needed()
        _update_map_artists()
        _draw_spectrum()
        _set_suptitle()
        preserve_spec_limits["state"] = False
        fig.canvas.draw_idle()

    def on_toggle(label: str) -> None:
        state_key = display_to_state.get(str(label), str(label))
        states[state_key] = not states[state_key]
        preserve_spec_limits["state"] = True
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
        current["chord_idx"] = 0
        current["active_case_candidate_id"] = ""
        current["active_case_peak_index"] = -1
        update()

    def on_click(event) -> None:
        if event.inaxes is not ax_map or event.xdata is None or event.ydata is None:
            return
        if int(getattr(event, "button", 0)) != 3:
            return
        current["x"] = int(np.clip(round(event.xdata), 0, W - 1))
        current["y"] = int(np.clip(round(event.ydata), 0, H - 1))
        current["chord_idx"] = 0
        current["active_case_candidate_id"] = ""
        current["active_case_peak_index"] = -1
        frozen["state"] = not frozen["state"]
        update()

    def on_key(event) -> None:
        key = str(event.key).lower()
        if key == "a":
            spectrum_attempts = current_spectrum_attempts()
            if spectrum_attempts:
                current["chord_idx"] = (current_attempt_index() - 1) % len(spectrum_attempts)
                preserve_spec_limits["state"] = True
            update()
            return
        if key == "x":
            spectrum_attempts = current_spectrum_attempts()
            if spectrum_attempts:
                current["chord_idx"] = (current_attempt_index() + 1) % len(spectrum_attempts)
                preserve_spec_limits["state"] = True
            update()
            return
        if key == "h":
            preserve_spec_limits["state"] = False
            cycle_active_overlay_case(-1)
            update()
            return
        if key == "j":
            preserve_spec_limits["state"] = False
            cycle_active_overlay_case(1)
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
            current["chord_idx"] = 0
            current["active_case_candidate_id"] = ""
            current["active_case_peak_index"] = -1
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
    cfg = load_config(Path(args.config))
    cache_path = Path(args.cache) if args.cache is not None else Path(str(cfg.paths["viewer_cache_path"]))
    show_cache(load_viewer_cache(cache_path), cfg=cfg)


if __name__ == "__main__":
    main()
