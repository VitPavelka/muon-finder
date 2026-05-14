from __future__ import annotations

"""
Temporary isolated adapter for the exact active legacy primary formulas.

The clean core still uses these functions to preserve scientific equivalence
while the formulas are being ported fully into `muonfinder_core`.
All direct legacy imports should go through this module.
"""

from feature_discrimination import (  # type: ignore
    compute_edge_width_metrics,
    compute_peak_curvature_features,
    compute_spike_score_v2_features,
    estimate_background_mad,
)

__all__ = [
    "compute_edge_width_metrics",
    "compute_peak_curvature_features",
    "compute_spike_score_v2_features",
    "estimate_background_mad",
]
