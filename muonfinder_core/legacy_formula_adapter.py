from __future__ import annotations

"""
Temporary isolated adapter for the exact active legacy primary formulas.

The clean core still uses these functions to preserve scientific equivalence
while the formulas are being ported fully into `muonfinder_core`.
All direct legacy imports should go through this module.
"""

from feature_discrimination import (  # type: ignore
    CURVATURE_NEGPREF_LOCAL_RADIUS,
    compute_curvature_negpref_diagnostics,
    compute_peak_curvature_features,
    estimate_background_mad,
)

__all__ = [
    "CURVATURE_NEGPREF_LOCAL_RADIUS",
    "compute_curvature_negpref_diagnostics",
    "compute_peak_curvature_features",
    "estimate_background_mad",
]
