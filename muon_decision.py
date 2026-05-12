from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from feature_discrimination import (
	CONTEXT_PADS,
	compute_context_window_experiment_features,
	compute_gws_diagnostics,
	compute_gws_feature_bundle_from_diagnostics,
	compute_local_shape_features,
	compute_multiscale_tophat_features,
	compute_spike_score_v2_features,
	estimate_background_mad,
	filter_multiscale_tophat_export_features,
)


# Empirical decision profile calibrated on the current reference dataset.
# Keep these thresholds centralized so they can be adjusted without changing
# the rule implementation.
SS_LOW = 0.95
SS_HIGH = 1.0

PCE_LOW = 0.5
PCE_HIGH = 0.5

GWS_LOW = 0.0
GWS_HIGH = 0.95

GWS_AUTO_MIN_SS = 0.985

MUON_RULE_V3_SCORE_LOW = 0.25
MUON_RULE_V3_SCORE_HIGH = 0.675
MUON_RULE_V3_VALUE_LOW = 0.5
MUON_RULE_V3_VALUE_HIGH = 1.5

MUON_RULE_V3_SCORE_BY_REASON: Dict[str, float] = {
	"ss_red": 1.00,
	"ss_orange_pce_support": 0.95,
	"ss_orange_gws_red_high_ss_rescue": 0.85,
	"ss_orange_pce_blue_gws_orange_or_low_ss_red_review": 0.50,
	"ss_orange_pce_blue_gws_blue_reject": 0.00,
	"ss_blue_reject": 0.00,
}

MUON_RULE_V3_COLOR_BY_DECISION: Dict[str, str] = {
	"auto_muon": "#d62728",
	"maybe_muon": "#ff7f0e",
	"no_muon": "#1f77b4",
}

MUON_RULE_V3_DECISION_VALUE: Dict[str, int] = {
	"no_muon": 0,
	"maybe_muon": 1,
	"auto_muon": 2,
}

SPIKE_SCORE_V4_COLOR_BY_DECISION: Dict[str, str] = {
	"muon": "#d62728",
	"non_muon": "#1f77b4",
	"review": "#ff7f0e",
}


@dataclass(frozen=True)
class MuonRuleV3Result:
	decision: str
	reason: str
	score: float
	ss_category: str
	pce_category: str
	gws_category: str


def classify_three_zone(value: float, low: float, high: float) -> str:
	"""
	Return:
	- 'blue' if value < low
	- 'orange' if low <= value < high
	- 'red' if value >= high
	"""
	try:
		v = float(value)
	except Exception:
		return "blue"
	if not np.isfinite(v):
		return "blue"
	if v < float(low):
		return "blue"
	if v < float(high):
		return "orange"
	return "red"


def _is_finite_number(value: Any) -> bool:
	try:
		v = float(value)
	except Exception:
		return False
	return bool(np.isfinite(v))


def compute_ss4(
		ss: float,
		pce: float,
		rve: float,
		*,
		ss_blue_max: float = 0.95,
		ss_red_min: float = 0.9999,
		pce_red_min: float = 0.4,
		rve_red_max: float = -0.1,
		missing_policy: str = "review",
) -> Dict[str, Any]:
	"""
	Experimental ss4 rule.

	SpikeScore is evaluated first, PCE confirms sharp curvature, and RVE is
	reversed: lower/more negative RVE supports spike-like behavior.
	"""
	def _rve_zone(value: Any) -> str:
		if not _is_finite_number(value):
			return "missing"
		return "red" if float(value) <= float(rve_red_max) else "blue"

	def _score_payload(score: float, decision: str, reason: str, ss_zone: str, pce_zone: str, rve_zone: str) -> Dict[str, Any]:
		out = {
			"ss4": float(score),
			"ss4_decision": str(decision),
			"ss4_reason": str(reason),
			"ss4_ss_zone": str(ss_zone),
			"ss4_pce_zone": str(pce_zone),
			"ss4_rve_zone": str(rve_zone),
			"ss4_is_ss_blue": float(ss_zone == "blue"),
			"ss4_is_ss_orange": float(ss_zone == "orange"),
			"ss4_is_ss_red": float(ss_zone == "red"),
			"ss4_is_pce_red": float(pce_zone == "red"),
			"ss4_is_rve_red": float(rve_zone == "red"),
			"ss4_is_raman_veto": float(rve_zone == "blue"),
		}
		return out

	if not _is_finite_number(ss):
		pce_zone = "missing" if not _is_finite_number(pce) else ("red" if float(pce) >= float(pce_red_min) else "blue")
		return _score_payload(float("nan"), "review", "review_missing", "missing", pce_zone, _rve_zone(rve))

	ss_v = float(ss)
	pce_ok = _is_finite_number(pce)
	rve_ok = _is_finite_number(rve)
	ss_zone = "blue" if ss_v < float(ss_blue_max) else ("red" if ss_v > float(ss_red_min) else "orange")
	pce_zone = "missing" if not pce_ok else ("red" if float(pce) >= float(pce_red_min) else "blue")
	rve_zone = _rve_zone(rve)

	if ss_zone == "blue":
		return _score_payload(0.0, "non_spike", "ss_blue", ss_zone, pce_zone, rve_zone)
	if not pce_ok:
		return _score_payload(float("nan"), "review", "review_missing", ss_zone, pce_zone, rve_zone)
	if pce_zone == "red":
		reason = "ss_orange_pce_red" if ss_zone == "orange" else "ss_red_pce_red"
		return _score_payload(1.0, "spike", reason, ss_zone, pce_zone, rve_zone)
	if not rve_ok:
		return _score_payload(float("nan"), "review", "review_missing", ss_zone, pce_zone, rve_zone)
	if rve_zone == "red":
		reason = "ss_orange_pce_blue_rve_red" if ss_zone == "orange" else "ss_red_pce_blue_rve_red"
		return _score_payload(1.0, "spike", reason, ss_zone, pce_zone, rve_zone)
	reason = "ss_orange_pce_blue_rve_blue" if ss_zone == "orange" else "ss_red_pce_blue_rve_blue"
	return _score_payload(0.0, "non_spike", reason, ss_zone, pce_zone, rve_zone)


def compute_spike_score_v4_three_friends(*args: Any, **kwargs: Any) -> Dict[str, Any]:
	"""Backward-compatible wrapper for the old helper name."""
	if "edge_red_min" in kwargs and "rve_red_max" not in kwargs:
		kwargs["rve_red_max"] = kwargs.pop("edge_red_min")
	return compute_ss4(*args, **kwargs)


def annotate_feature_dict_with_spike_score_v4(
		features: Mapping[str, Any],
		*,
		rve_feature: str = "recdw_sum_0_90_raman_veto_evidence_signed",
		edge_feature: Optional[str] = None,
		ss_blue_max: float = 0.95,
		ss_red_min: float = 0.9999,
		pce_red_min: float = 0.4,
		rve_red_max: float = -0.1,
		edge_red_min: Optional[float] = None,
		missing_policy: str = "review",
) -> Dict[str, Any]:
	if edge_feature is not None:
		rve_feature = str(edge_feature)
	if edge_red_min is not None:
		rve_red_max = float(edge_red_min)
	ss = float(features.get("spike_score_v1", np.nan))
	pce = float(features.get("pce_negpref_t098_evidence_signed", np.nan))
	rve = float(features.get(str(rve_feature), np.nan))
	out = compute_ss4(
		ss,
		pce,
		rve,
		ss_blue_max=ss_blue_max,
		ss_red_min=ss_red_min,
		pce_red_min=pce_red_min,
		rve_red_max=rve_red_max,
		missing_policy=missing_policy,
	)
	out["ss4_rve_feature"] = str(rve_feature)
	out["ss4_rve_value"] = rve
	return out


def classify_muon_rule_v3(
		spike_score_v1: float,
		pce_negpref_t098_evidence_signed: float,
		gws_evidence_signed: float,
) -> MuonRuleV3Result:
	ss_category = classify_three_zone(spike_score_v1, SS_LOW, SS_HIGH)
	pce_category = classify_three_zone(pce_negpref_t098_evidence_signed, PCE_LOW, PCE_HIGH)
	gws_category = classify_three_zone(gws_evidence_signed, GWS_LOW, GWS_HIGH)

	if ss_category == "blue":
		reason = "ss_blue_reject"
		decision = "no_muon"
	elif ss_category == "red":
		reason = "ss_red"
		decision = "auto_muon"
	else:
		if pce_category != "blue":
			reason = "ss_orange_pce_support"
			decision = "auto_muon"
		elif gws_category == "red" and float(spike_score_v1) >= float(GWS_AUTO_MIN_SS):
			reason = "ss_orange_gws_red_high_ss_rescue"
			decision = "auto_muon"
		elif gws_category != "blue":
			reason = "ss_orange_pce_blue_gws_orange_or_low_ss_red_review"
			decision = "maybe_muon"
		else:
			reason = "ss_orange_pce_blue_gws_blue_reject"
			decision = "no_muon"

	return MuonRuleV3Result(
		decision=decision,
		reason=reason,
		score=float(MUON_RULE_V3_SCORE_BY_REASON.get(reason, 0.0)),
		ss_category=ss_category,
		pce_category=pce_category,
		gws_category=gws_category,
	)


def annotate_feature_dict_with_muon_rule_v3(features: Mapping[str, Any]) -> Dict[str, Any]:
	ss = float(features.get("spike_score_v1", np.nan))
	pce = float(features.get("pce_negpref_t098_evidence_signed", np.nan))
	gws = float(features.get("gws_evidence_signed", np.nan))
	result = classify_muon_rule_v3(ss, pce, gws)
	return {
		"muon_rule_v3_decision": result.decision,
		"muon_rule_v3_reason": result.reason,
		"muon_rule_v3_value": int(MUON_RULE_V3_DECISION_VALUE.get(result.decision, 0)),
		"muon_rule_v3_score": float(result.score),
		"ss_category_v3": result.ss_category,
		"pce_category_v3": result.pce_category,
		"gws_category_v3": result.gws_category,
	}


def get_muon_rule_v3_metric_thresholds(metric_name: str) -> Optional[tuple[float, float, bool]]:
	name = str(metric_name).strip()
	if name == "muon_rule_v3_score":
		return (float(MUON_RULE_V3_SCORE_LOW), float(MUON_RULE_V3_SCORE_HIGH), False)
	if name == "muon_rule_v3_value":
		return (float(MUON_RULE_V3_VALUE_LOW), float(MUON_RULE_V3_VALUE_HIGH), False)
	return None


def compute_minimal_muon_features(
		raw_signal: np.ndarray,
		gradient_signal: Optional[np.ndarray],
		start: int,
		end: int,
		peak_index: int,
) -> Dict[str, float]:
	"""
	Compute only the existing feature subset needed by muon_rule_v3.

	This intentionally reuses the established feature helpers and does not
	introduce any new feature family.
	"""
	raw = np.asarray(raw_signal, dtype=float)
	grad = np.asarray(gradient_signal, dtype=float) if gradient_signal is not None else np.array([], dtype=float)
	n = int(raw.size)
	if n <= 2:
		return {
			"spike_score_v1": float("nan"),
			"pce_negpref_t098_evidence_signed": float("nan"),
			"gws_evidence_signed": float("nan"),
		}

	a = int(np.clip(start, 0, n - 1))
	b = int(np.clip(end, 0, n - 1))
	p = int(np.clip(peak_index, 0, n - 1))
	if not (a < p < b):
		return {
			"spike_score_v1": float("nan"),
			"pce_negpref_t098_evidence_signed": float("nan"),
			"gws_evidence_signed": float("nan"),
		}

	sig_spec = grad if grad.size == n else raw
	segment = sig_spec[a:b + 1]
	if segment.size < 3:
		return {
			"spike_score_v1": float("nan"),
			"pce_negpref_t098_evidence_signed": float("nan"),
			"gws_evidence_signed": float("nan"),
		}

	rise = np.diff(segment)[: max(1, p - a)]
	fall = np.diff(segment)[max(1, p - a):]
	rise_slope = float(np.max(rise)) if rise.size else 0.0
	fall_slope = float(np.min(fall)) if fall.size else 0.0

	bg_mad = estimate_background_mad(sig_spec, a, b)

	rise_slope_z = float(rise_slope / bg_mad)
	fall_slope_z = float(abs(fall_slope / bg_mad))
	sr = float(np.tanh(rise_slope_z / 6.0))
	sf = float(np.tanh(fall_slope_z / 6.0))

	features: Dict[str, float] = {
		"rise_slope_z": rise_slope_z,
		"fall_slope_z": fall_slope_z,
		"spike_score_v1": float(0.50 * sr + 0.50 * sf),
	}

	raw_seg_local = raw[a:b + 1]
	p_rel = int(np.clip(p - a, 0, raw_seg_local.size - 1))
	features.update(
		compute_local_shape_features(
			raw_seg_local,
			p_rel,
			core_radius=1,
			center_radius=2,
			shoulder_inner=3,
			shoulder_outer=8,
			curvature_search_radius=3,
		)
	)

	wide_l = max(0, a - (b - a + 1))
	wide_r = min(n - 1, b + (b - a + 1))
	raw_wide = raw[wide_l:wide_r + 1]
	p_wide = int(np.clip(p - wide_l, 0, raw_wide.size - 1))
	gradient_wide = grad[wide_l:wide_r + 1] if grad.size == n else np.array([], dtype=float)
	if gradient_wide.size:
		gradient_tophat = compute_multiscale_tophat_features(gradient_wide, bg_mad)
		gradient_tophat_prefixed = {
			(key.replace("mt_", "gmt_", 1) if key.startswith("mt_") else key): value
			for key, value in gradient_tophat.items()
		}
		features.update(filter_multiscale_tophat_export_features(gradient_tophat_prefixed, prefix="gmt"))
	if grad.size == n:
		shared_gws_diag = compute_gws_diagnostics(
			grad,
			raw_signal=raw,
			apex_idx=p,
			detection_left=a,
			detection_right=b,
			bg_mad=bg_mad,
		)
		features.update(compute_gws_feature_bundle_from_diagnostics(shared_gws_diag))
	features.update(
		compute_context_window_experiment_features(
			raw,
			(grad if grad.size == n else np.zeros_like(raw)),
			apex_idx=p,
			detection_left=a,
			detection_right=b,
			bg_mad=bg_mad,
			context_pads=CONTEXT_PADS,
			curvature_search_radius=3,
		)
	)
	features.update(compute_spike_score_v2_features(features))
	return {
		"spike_score_v1": float(features.get("spike_score_v1", np.nan)),
		"pce_negpref_t098_evidence_signed": float(features.get("pce_negpref_t098_evidence_signed", np.nan)),
		"gws_evidence_signed": float(features.get("gws_evidence_signed", np.nan)),
		"gmt_width_soft_d1_mean": float(features.get("gmt_width_soft_d1_mean", np.nan)),
		"gws_support01": float(features.get("gws_support01", np.nan)),
		"gmt_width_soft_d1_mean_ctx10": float(features.get("gmt_width_soft_d1_mean_ctx10", np.nan)),
		"gmt_width_soft_slope_ctx10": float(features.get("gmt_width_soft_slope_ctx10", np.nan)),
		"gws_support01_ctx10": float(features.get("gws_support01_ctx10", np.nan)),
		"gws_evidence_signed_ctx10": float(features.get("gws_evidence_signed_ctx10", np.nan)),
	}


def classify_segment_with_muon_rule_v3(
		raw_signal: np.ndarray,
		gradient_signal: Optional[np.ndarray],
		start: int,
		end: int,
		peak_index: int,
) -> Dict[str, Any]:
	features = compute_minimal_muon_features(
		raw_signal=raw_signal,
		gradient_signal=gradient_signal,
		start=start,
		end=end,
		peak_index=peak_index,
	)
	out: Dict[str, Any] = dict(features)
	out.update(annotate_feature_dict_with_muon_rule_v3(features))
	return out
