from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _threshold_map(cfg: dict[str, Any]) -> dict[str, list[tuple[str, float]]]:
    return {
        "ss6_ss1": [("ss1_gate", float(cfg["ss1_gate"]))],
        "ss6_pce": [
            ("pce_dead_max", float(cfg["pce_dead_max"])),
            ("pce_gray_min", float(cfg["pce_gray_min"])),
            ("pce_gray_max", float(cfg["pce_gray_max"])),
            ("pce_strong_min", float(cfg["pce_strong_min"])),
        ],
        "ss6_edge": [
            ("edge_spike_max", float(cfg["edge_spike_max"])),
            ("pce_dead_edge_spike_max", float(cfg["pce_dead_edge_spike_max"])),
            ("pce_gray_soft_edge_max", float(cfg["pce_gray_soft_edge_max"])),
        ],
        "ss6_eel": [
            ("eel_spike_max", float(cfg["eel_spike_max"])),
            ("pce_gray_delta_eel_max", float(cfg["pce_gray_delta_eel_max"])),
            ("pce_gray_high_pce_eel_max", float(cfg["pce_gray_high_pce_eel_max"])),
            ("pce_gray_soft_eel_max", float(cfg["pce_gray_soft_eel_max"])),
        ],
        "ss6_edge_eel_delta": [
            ("edge_eel_delta_min", float(cfg["edge_eel_delta_min"])),
            ("pce_gray_delta_min", float(cfg["pce_gray_delta_min"])),
        ],
        "ss6_resid": [
            ("resid_rescue_min", float(cfg["resid_rescue_min"])),
            ("resid_strong_min", float(cfg["resid_strong_min"])),
            ("pce_gray_high_pce_resid_min", float(cfg["pce_gray_high_pce_resid_min"])),
            ("pce_gray_high_resid_min", float(cfg["pce_gray_high_resid_min"])),
            ("low_pce_double_edge_resid_min", float(cfg["low_pce_double_edge_resid_min"])),
        ],
    }


def _write_branch_counts_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        branch = str(row.get("ss6_branch", "")).strip()
        counts[branch] = counts.get(branch, 0) + 1
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ss6_branch", "count"])
        for branch, count in sorted(counts.items()):
            writer.writerow([branch, count])


def _write_rejected_near_thresholds_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if int(float(row.get("ss6_accept", 0) or 0)) != 0:
            continue
        pce = _safe_float(row.get("ss6_pce"))
        edge = _safe_float(row.get("ss6_edge"))
        eel = _safe_float(row.get("ss6_eel"))
        resid = _safe_float(row.get("ss6_resid"))
        if (np.isfinite(pce) and pce >= 0.55) or (
            np.isfinite(edge)
            and np.isfinite(eel)
            and edge <= -0.55
            and eel <= -0.55
        ) or (np.isfinite(resid) and 5.0 <= resid < 9.0):
            selected.append(row)
    if not selected:
        return
    fieldnames = [
        "source_y",
        "source_x",
        "compact_y",
        "compact_x",
        "peak_index",
        "candidate_id",
        "ss6_branch",
        "ss6_reason",
        "ss6_ss1",
        "ss6_pce",
        "ss6_edge",
        "ss6_eel",
        "ss6_resid",
        "ss6_edge_eel_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def generate_ss6_histograms(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[str]:
    out_dir = Path(str(cfg["histograms_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    bins = int(max(5, cfg.get("histogram_bins", 40)))
    thresholds = _threshold_map(cfg)
    files: list[str] = []
    metrics = [
        "ss6_ss1",
        "ss6_pce",
        "ss6_edge",
        "ss6_eel",
        "ss6_edge_eel_delta",
        "ss6_resid",
    ]
    accepted_mask = np.asarray([int(_safe_float(row.get("ss6_accept"))) == 1 for row in rows], dtype=bool)
    for metric in metrics:
        values = np.asarray([_safe_float(row.get(metric)) for row in rows], dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        accepted_vals = values[finite & accepted_mask]
        rejected_vals = values[finite & ~accepted_mask]
        combined = values[finite]
        lo = float(np.min(combined))
        hi = float(np.max(combined))
        if not np.isfinite(lo) or not np.isfinite(hi):
            plt.close(fig)
            continue
        if hi <= lo:
            hi = lo + 1.0
        hist_bins = np.linspace(lo, hi, bins)
        if rejected_vals.size:
            ax.hist(rejected_vals, bins=hist_bins, color="#1f77b4", alpha=0.55, label="rejected")
        if accepted_vals.size:
            ax.hist(accepted_vals, bins=hist_bins, color="#d62728", alpha=0.55, label="accepted")
        seen: set[str] = set()
        for label, threshold in thresholds.get(metric, []):
            if not np.isfinite(threshold) or label in seen:
                continue
            seen.add(label)
            ax.axvline(threshold, color="#333333", linestyle="--", linewidth=1.1, alpha=0.9, label=label)
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_ylabel("count")
        handles, labels = ax.get_legend_handles_labels()
        dedup_h = []
        dedup_l = []
        used: set[str] = set()
        for handle, label in zip(handles, labels):
            if label in used:
                continue
            used.add(label)
            dedup_h.append(handle)
            dedup_l.append(label)
        if dedup_h:
            ax.legend(dedup_h, dedup_l, fontsize=8)
        fig.tight_layout()
        out_path = out_dir / f"hist_{metric}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        files.append(str(out_path))

    branch_counts: dict[str, int] = {}
    for row in rows:
        branch = str(row.get("ss6_branch", "")).strip()
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
    if branch_counts:
        fig, ax = plt.subplots(figsize=(9.0, 5.5))
        labels = list(sorted(branch_counts.keys()))
        counts = [branch_counts[label] for label in labels]
        ax.barh(labels, counts, color="#4c78a8", alpha=0.9)
        ax.set_xlabel("count")
        ax.set_title("ss6 branch counts")
        fig.tight_layout()
        out_path = out_dir / "ss6_branch_counts.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        files.append(str(out_path))

    branch_csv = out_dir / "ss6_branch_counts.csv"
    _write_branch_counts_csv(branch_csv, rows)
    files.append(str(branch_csv))
    rejected_csv = out_dir / "ss6_rejected_near_thresholds.csv"
    _write_rejected_near_thresholds_csv(rejected_csv, rows)
    if rejected_csv.exists():
        files.append(str(rejected_csv))
    return files
