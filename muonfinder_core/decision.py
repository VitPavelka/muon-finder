from __future__ import annotations

from typing import Any

import numpy as np


def _is_finite_number(value: Any) -> bool:
    try:
        v = float(value)
    except Exception:
        return False
    return bool(np.isfinite(v))


def compute_ss4(
    ss: float,
    pce: float,
    edge: float,
    *,
    ss_blue_max: float,
    ss_red_min: float,
    pce_red_min: float,
    edge_red_max: float,
    pce_dead_zone_enabled: bool,
    pce_dead_zone_low: float,
    pce_dead_zone_high: float,
    missing_policy: str,
) -> dict[str, Any]:
    def _edge_zone(value: Any) -> str:
        if not _is_finite_number(value):
            return "missing"
        return "red" if float(value) <= float(edge_red_max) else "blue"

    def _payload(score: float, decision: str, reason: str, ss_zone: str, pce_zone: str, edge_zone: str) -> dict[str, Any]:
        return {
            "ss4": float(score),
            "ss4_decision": str(decision),
            "ss4_reason": str(reason),
            "ss4_ss_zone": str(ss_zone),
            "ss4_pce_zone": str(pce_zone),
            "ss4_edge_zone": str(edge_zone),
            "ss4_rve_zone": str(edge_zone),
            "ss_zone": str(ss_zone),
            "pce_zone": str(pce_zone),
            "edge_zone": str(edge_zone),
        }

    if not _is_finite_number(ss):
        pce_zone = "missing" if not _is_finite_number(pce) else ("red" if float(pce) >= float(pce_red_min) else "blue")
        return _payload(float("nan"), str(missing_policy), "review_missing", "missing", pce_zone, _edge_zone(edge))
    ss_v = float(ss)
    ss_zone = "blue" if ss_v < float(ss_blue_max) else ("red" if ss_v > float(ss_red_min) else "orange")
    pce_ok = _is_finite_number(pce)
    edge_ok = _is_finite_number(edge)
    pce_zone = "missing" if not pce_ok else ("red" if float(pce) >= float(pce_red_min) else "blue")
    edge_zone = _edge_zone(edge)
    if ss_zone == "blue":
        return _payload(0.0, "non_spike", "ss_blue", ss_zone, pce_zone, edge_zone)
    if not pce_ok:
        return _payload(float("nan"), str(missing_policy), "review_missing", ss_zone, pce_zone, edge_zone)
    if pce_zone == "red":
        reason = "ss_orange_pce_red" if ss_zone == "orange" else "ss_red_pce_red"
        return _payload(1.0, "spike", reason, ss_zone, pce_zone, edge_zone)
    if (
        pce_dead_zone_enabled
        and ss_zone == "orange"
        and pce_zone == "blue"
        and float(pce_dead_zone_low) <= float(pce) <= float(pce_dead_zone_high)
    ):
        return _payload(0.0, "non_spike", "ss_orange_pce_dead_zone_reject", ss_zone, pce_zone, edge_zone)
    if not edge_ok:
        return _payload(float("nan"), str(missing_policy), "review_missing", ss_zone, pce_zone, edge_zone)
    if edge_zone == "red":
        reason = "ss_orange_pce_blue_edge_red" if ss_zone == "orange" else "ss_red_pce_blue_edge_red"
        return _payload(1.0, "spike", reason, ss_zone, pce_zone, edge_zone)
    reason = "ss_orange_pce_blue_edge_blue" if ss_zone == "orange" else "ss_red_pce_blue_edge_blue"
    return _payload(0.0, "non_spike", reason, ss_zone, pce_zone, edge_zone)


def compute_ss5(
    ss1: float,
    pce: float,
    edge: float,
    *,
    ss1_threshold: float,
    pce_spike_min: float,
    edge_spike_max: float,
) -> dict[str, Any]:
    ss1_vote = bool(_is_finite_number(ss1) and float(ss1) >= float(ss1_threshold))
    pce_vote = bool(_is_finite_number(pce) and float(pce) >= float(pce_spike_min))
    edge_vote = bool(_is_finite_number(edge) and float(edge) <= float(edge_spike_max))
    if not ss1_vote:
        score = 0.0
        decision = "non_spike"
        reason = "ss5_ss1_below_threshold"
    elif pce_vote and edge_vote:
        score = 1.0
        decision = "spike"
        reason = "ss5_ss1_pce_edge_red"
    elif pce_vote:
        score = 1.0
        decision = "spike"
        reason = "ss5_ss1_pce_red"
    elif edge_vote:
        score = 1.0
        decision = "spike"
        reason = "ss5_ss1_edge_red"
    else:
        score = 0.0
        decision = "non_spike"
        reason = "ss5_ss1_red_pce_edge_blue"
    return {
        "spike_score_v5": float(score),
        "ss5": float(score),
        "ss5_decision": str(decision),
        "ss5_reason": str(reason),
        "ss5_ss1_vote": float(ss1_vote),
        "ss5_pce_vote": float(pce_vote),
        "ss5_edge_vote": float(edge_vote),
        "ss5_ss1_threshold": float(ss1_threshold),
        "ss5_pce_spike_min": float(pce_spike_min),
        "ss5_edge_spike_max": float(edge_spike_max),
    }
