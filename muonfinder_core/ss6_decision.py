from __future__ import annotations

from typing import Any

import numpy as np

from .experimental_common import build_join_row, is_noise_rejected


DEFAULT_SS6_CONFIG: dict[str, Any] = {
    "enabled": False,
    "decisions_path": "outputs_core/ss6_decisions.csv",
    "summary_path": "outputs_core/ss6_decisions_summary.json",
    "ss1_gate": 0.95,
    "pce_strong_min": 0.75,
    "pce_dead_max": -0.999,
    "pce_gray_min": 0.10,
    "pce_gray_max": 0.75,
    "edge_spike_max": -0.50,
    "eel_spike_max": -0.50,
    "edge_eel_delta_min": 0.50,
    "pce_dead_edge_spike_max": -0.45,
    "pce_gray_delta_eel_max": -0.40,
    "pce_gray_delta_min": 0.50,
    "pce_gray_delta_resid_min": 8.5,
    "pce_gray_high_pce_min": 0.60,
    "pce_gray_high_pce_eel_max": -0.60,
    "pce_gray_high_pce_resid_min": 6.0,
    "pce_gray_high_resid_min": 12.0,
    "pce_gray_soft_edge_max": -0.45,
    "pce_gray_soft_eel_max": -0.45,
    "low_pce_double_edge_edge_max": -0.60,
    "low_pce_double_edge_eel_max": -0.60,
    "low_pce_double_edge_resid_min": 5.5,
    "resid_rescue_min": 8.5,
    "resid_strong_min": 20.0,
    "require_noise_kept": True,
    "save_histograms": True,
    "histograms_dir": "outputs_core/ss6_histograms",
    "histogram_bins": 40,
    "metric_names": {
        "ss1": "spike_score_v1",
        "pce": "pce_negpref_t098_evidence_signed",
        "edge": "recdw_sum_0_90_raman_veto_evidence_signed",
        "eel": "exp_edge_legacy_evidence_signed_modernnorm",
        "resid": "exp_resid3_height_noise_z",
    },
    "viewer_columns": [],
    "viewer_label_aliases": {
        "ss6_accept": "s6",
        "ss6_branch": "s6b",
        "ss6_edge_eel_delta": "eed",
        "ss6_resid": "r3",
        "ss6_eel": "eel",
        "ss6_edge": "edge",
    },
}


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def ss6_defaults(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return _deep_merge(DEFAULT_SS6_CONFIG, dict(config or {}))


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _base_output(row: dict[str, Any]) -> dict[str, Any]:
    out = build_join_row(row)
    out.update(
        {
            "ss6_accept": 0,
            "ss6_branch": "",
            "ss6_reason": "",
            "ss6_ss1": np.nan,
            "ss6_pce": np.nan,
            "ss6_edge": np.nan,
            "ss6_eel": np.nan,
            "ss6_resid": np.nan,
            "ss6_edge_eel_delta": np.nan,
            "ss6_ss1_gate_pass": 0,
            "ss6_pce_strong_flag": 0,
            "ss6_pce_dead_flag": 0,
            "ss6_pce_gray_flag": 0,
            "ss6_edge_spike_like_flag": 0,
            "ss6_eel_spike_like_flag": 0,
            "ss6_resid_rescue_flag": 0,
            "ss6_delta_spike_on_structure_flag": 0,
            "ss6_pce_dead_edge_relaxed_flag": 0,
            "ss6_pce_gray_delta_rescue_flag": 0,
            "ss6_pce_gray_high_pce_eel_rescue_flag": 0,
            "ss6_pce_gray_high_resid_soft_edge_rescue_flag": 0,
            "ss6_low_pce_double_edge_rescue_flag": 0,
        }
    )
    return out


def compute_ss6_row(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cfg = ss6_defaults(config)
    metric_names = dict(cfg.get("metric_names", {}))
    out = _base_output(row)

    if bool(cfg.get("require_noise_kept", True)):
        rejected, _field, _status = is_noise_rejected(row)
        if rejected is True:
            out["ss6_branch"] = "noise_rejected"
            out["ss6_reason"] = "noise_rejected"
            return out

    values = {name: _safe_float(row.get(metric_names.get(name, ""))) for name in ("ss1", "pce", "edge", "eel", "resid")}
    missing = [name for name, value in values.items() if not np.isfinite(value)]
    if missing:
        out["ss6_branch"] = "missing_metric"
        out["ss6_reason"] = "missing_" + ",".join(missing)
        return out

    edge_eel_delta = float(values["edge"] - values["eel"])
    out["ss6_ss1"] = float(values["ss1"])
    out["ss6_pce"] = float(values["pce"])
    out["ss6_edge"] = float(values["edge"])
    out["ss6_eel"] = float(values["eel"])
    out["ss6_resid"] = float(values["resid"])
    out["ss6_edge_eel_delta"] = float(edge_eel_delta)

    ss1_gate = float(cfg["ss1_gate"])
    pce_strong_min = float(cfg["pce_strong_min"])
    pce_dead_max = float(cfg["pce_dead_max"])
    pce_gray_min = float(cfg["pce_gray_min"])
    pce_gray_max = float(cfg["pce_gray_max"])
    edge_spike_max = float(cfg["edge_spike_max"])
    eel_spike_max = float(cfg["eel_spike_max"])
    delta_min = float(cfg["edge_eel_delta_min"])
    pce_dead_edge_spike_max = float(cfg["pce_dead_edge_spike_max"])
    pce_gray_delta_eel_max = float(cfg["pce_gray_delta_eel_max"])
    pce_gray_delta_min = float(cfg["pce_gray_delta_min"])
    pce_gray_delta_resid_min = float(cfg["pce_gray_delta_resid_min"])
    pce_gray_high_pce_min = float(cfg["pce_gray_high_pce_min"])
    pce_gray_high_pce_eel_max = float(cfg["pce_gray_high_pce_eel_max"])
    pce_gray_high_pce_resid_min = float(cfg["pce_gray_high_pce_resid_min"])
    pce_gray_high_resid_min = float(cfg["pce_gray_high_resid_min"])
    pce_gray_soft_edge_max = float(cfg["pce_gray_soft_edge_max"])
    pce_gray_soft_eel_max = float(cfg["pce_gray_soft_eel_max"])
    low_pce_double_edge_edge_max = float(cfg["low_pce_double_edge_edge_max"])
    low_pce_double_edge_eel_max = float(cfg["low_pce_double_edge_eel_max"])
    low_pce_double_edge_resid_min = float(cfg["low_pce_double_edge_resid_min"])
    resid_rescue_min = float(cfg["resid_rescue_min"])
    resid_strong_min = float(cfg["resid_strong_min"])

    ss1_gate_pass = bool(values["ss1"] >= ss1_gate)
    pce_strong_flag = bool(values["pce"] >= pce_strong_min)
    pce_dead_flag = bool(values["pce"] <= pce_dead_max)
    pce_gray_flag = bool(pce_gray_min <= values["pce"] < pce_gray_max)
    edge_spike_flag = bool(values["edge"] <= edge_spike_max)
    eel_spike_flag = bool(values["eel"] <= eel_spike_max)
    resid_rescue_flag = bool(values["resid"] >= resid_rescue_min)
    delta_flag = bool(edge_eel_delta >= delta_min)

    out["ss6_ss1_gate_pass"] = int(ss1_gate_pass)
    out["ss6_pce_strong_flag"] = int(pce_strong_flag)
    out["ss6_pce_dead_flag"] = int(pce_dead_flag)
    out["ss6_pce_gray_flag"] = int(pce_gray_flag)
    out["ss6_edge_spike_like_flag"] = int(edge_spike_flag)
    out["ss6_eel_spike_like_flag"] = int(eel_spike_flag)
    out["ss6_resid_rescue_flag"] = int(resid_rescue_flag)
    out["ss6_delta_spike_on_structure_flag"] = int(delta_flag)
    out["ss6_pce_dead_edge_relaxed_flag"] = int(bool(values["edge"] <= pce_dead_edge_spike_max))
    out["ss6_pce_gray_delta_rescue_flag"] = int(
        bool(values["resid"] >= pce_gray_delta_resid_min)
        and bool(edge_eel_delta >= pce_gray_delta_min)
        and bool(values["eel"] <= pce_gray_delta_eel_max)
    )
    out["ss6_pce_gray_high_pce_eel_rescue_flag"] = int(
        bool(values["pce"] >= pce_gray_high_pce_min)
        and bool(values["eel"] <= pce_gray_high_pce_eel_max)
        and bool(values["resid"] >= pce_gray_high_pce_resid_min)
    )
    out["ss6_pce_gray_high_resid_soft_edge_rescue_flag"] = int(
        bool(values["pce"] >= pce_gray_high_pce_min)
        and bool(values["resid"] >= pce_gray_high_resid_min)
        and (bool(values["edge"] <= pce_gray_soft_edge_max) or bool(values["eel"] <= pce_gray_soft_eel_max))
    )
    out["ss6_low_pce_double_edge_rescue_flag"] = int(
        bool(values["edge"] <= low_pce_double_edge_edge_max)
        and bool(values["eel"] <= low_pce_double_edge_eel_max)
        and bool(values["resid"] >= low_pce_double_edge_resid_min)
    )

    if not ss1_gate_pass:
        out["ss6_branch"] = "ss1_low"
        out["ss6_reason"] = "ss1_below_gate"
        return out

    if pce_strong_flag:
        out["ss6_accept"] = 1
        out["ss6_branch"] = "pce_strong"
        out["ss6_reason"] = "ss1_gate_pce_strong"
        return out

    if pce_dead_flag:
        dead_edge_relaxed_flag = bool(values["edge"] <= pce_dead_edge_spike_max)
        dead_accept = resid_rescue_flag and eel_spike_flag and (dead_edge_relaxed_flag or delta_flag)
        if dead_accept:
            out["ss6_accept"] = 1
            out["ss6_branch"] = "pce_dead_rescue"
            out["ss6_reason"] = "resid_eel_edge_or_delta"
        else:
            failures: list[str] = []
            if not resid_rescue_flag:
                failures.append("resid_low")
            if not eel_spike_flag:
                failures.append("eel_not_spike_like")
            if not (dead_edge_relaxed_flag or delta_flag):
                failures.append("edge_not_spike_like_and_delta_low")
            out["ss6_branch"] = "pce_dead_reject"
            out["ss6_reason"] = "+".join(failures) if failures else "pce_dead_reject"
        return out

    if pce_gray_flag:
        gray_spike_on_structure = (
            bool(values["resid"] >= pce_gray_delta_resid_min)
            and bool(edge_eel_delta >= pce_gray_delta_min)
            and bool(values["eel"] <= pce_gray_delta_eel_max)
        )
        gray_high_pce_eel_rescue = (
            bool(values["pce"] >= pce_gray_high_pce_min)
            and bool(values["eel"] <= pce_gray_high_pce_eel_max)
            and bool(values["resid"] >= pce_gray_high_pce_resid_min)
        )
        gray_high_resid_soft_edge_rescue = (
            bool(values["pce"] >= pce_gray_high_pce_min)
            and bool(values["resid"] >= pce_gray_high_resid_min)
            and (bool(values["edge"] <= pce_gray_soft_edge_max) or bool(values["eel"] <= pce_gray_soft_eel_max))
        )
        gray_accept = resid_rescue_flag and (edge_spike_flag or eel_spike_flag)
        if gray_spike_on_structure:
            out["ss6_accept"] = 1
            out["ss6_branch"] = "pce_gray_spike_on_structure"
            out["ss6_reason"] = "resid_delta_relaxed_eel"
        elif gray_high_pce_eel_rescue:
            out["ss6_accept"] = 1
            out["ss6_branch"] = "pce_gray_high_pce_eel_rescue"
            out["ss6_reason"] = "high_gray_pce_strong_eel_medium_resid"
        elif gray_high_resid_soft_edge_rescue:
            out["ss6_accept"] = 1
            out["ss6_branch"] = "pce_gray_high_resid_soft_edge_rescue"
            out["ss6_reason"] = "high_gray_pce_high_resid_soft_edge_or_eel"
        elif gray_accept:
            out["ss6_accept"] = 1
            out["ss6_branch"] = "pce_gray_rescue"
            out["ss6_reason"] = "resid_edge_or_eel"
        else:
            failures = []
            if not resid_rescue_flag:
                failures.append("resid_low")
            if not (edge_spike_flag or eel_spike_flag):
                failures.append("edge_and_eel_not_spike_like")
            out["ss6_branch"] = "pce_gray_reject"
            out["ss6_reason"] = "+".join(failures) if failures else "pce_gray_reject"
        return out

    low_pce_accept = values["pce"] > pce_dead_max and values["pce"] < pce_gray_min
    if low_pce_accept:
        double_edge_rescue = (
            bool(values["edge"] <= low_pce_double_edge_edge_max)
            and bool(values["eel"] <= low_pce_double_edge_eel_max)
            and bool(values["resid"] >= low_pce_double_edge_resid_min)
        )
        rescue = bool(values["resid"] >= resid_strong_min) and eel_spike_flag and (edge_spike_flag or delta_flag)
        if double_edge_rescue:
            out["ss6_accept"] = 1
            out["ss6_branch"] = "low_pce_double_edge_rescue"
            out["ss6_reason"] = "low_pce_edge_and_eel_strong_medium_resid"
        elif rescue:
            out["ss6_accept"] = 1
            out["ss6_branch"] = "low_pce_strong_rescue"
            out["ss6_reason"] = "resid_strong_eel_edge_or_delta"
        else:
            failures = []
            if not bool(values["resid"] >= resid_strong_min):
                failures.append("resid_not_strong")
            if not eel_spike_flag:
                failures.append("eel_not_spike_like")
            if not (edge_spike_flag or delta_flag):
                failures.append("edge_not_spike_like_and_delta_low")
            out["ss6_branch"] = "low_pce_reject"
            out["ss6_reason"] = "+".join(failures) if failures else "low_pce_reject"
        return out

    out["ss6_branch"] = "low_pce_reject"
    out["ss6_reason"] = "pce_not_rescued"
    return out
