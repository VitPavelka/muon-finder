from __future__ import annotations

from typing import Any


def candidate_status_color(row: dict[str, Any], active_profile: str) -> str:
    status = str(row.get("candidate_noise_prefilter_status", ""))
    if status == "rejected_noise":
        return "#7f7f7f"
    decision_key = "ss5_decision" if active_profile == "ss5" else "ss4_decision"
    decision = str(row.get("primary_active_decision", row.get(decision_key, "non_spike")))
    return "#d62728" if decision == "spike" else "#1f77b4"
