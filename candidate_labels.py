from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple


CandidateKey = Tuple[int, int, int]

KNOWN_LABEL_CLASSES = {"muon", "raman", "noise", "unknown"}
TERNARY_CLASSES = ("muon", "raman", "noise")
LABEL_CLASS_COLORS = {
	"muon": "#d62728",
	"raman": "#2ca02c",
	"noise": "#1f77b4",
	"unknown": "#7f7f7f",
}


def candidate_key_from_row(row: dict) -> CandidateKey:
	return (int(row["y"]), int(row["x"]), int(row["peak_index"]))


def normalize_label_class(value: object) -> str:
	text = str(value if value is not None else "").strip().lower()
	if text in {"", "none", "nan", "unlabeled", "clear"}:
		return "unknown"
	if text in {"signal", "spectral", "raman_like", "raman-like"}:
		return "raman"
	if text in {"background", "bg"}:
		return "noise"
	if text in KNOWN_LABEL_CLASSES:
		return text
	return "unknown"


def parse_optional_binary_label(value: object) -> Optional[int]:
	text = str(value if value is not None else "").strip().lower()
	if text in {"", "none", "nan", "unknown"}:
		return None
	if text in {"1", "true", "t", "yes", "y", "muon"}:
		return 1
	if text in {"0", "false", "f", "no", "n", "nonmuon", "non-muon", "raman", "noise"}:
		return 0
	try:
		return 1 if int(float(text)) else 0
	except Exception:
		return None


def binary_from_label_class(label_class: str) -> Optional[int]:
	cls = normalize_label_class(label_class)
	if cls == "muon":
		return 1
	if cls in {"raman", "noise"}:
		return 0
	return None


def label_class_from_row(row: dict) -> str:
	"""Return ternary class without assuming old is_muon=0 means noise."""
	if "label_class" in row and str(row.get("label_class", "")).strip():
		return normalize_label_class(row.get("label_class"))
	lbl = parse_optional_binary_label(row.get("is_muon"))
	if lbl == 1:
		return "muon"
	return "unknown"


def load_binary_labels(path: Path) -> Dict[CandidateKey, int]:
	"""Load legacy muon/non-muon labels; this preserves old is_muon behavior."""
	out: Dict[CandidateKey, int] = {}
	with Path(path).open("r", encoding="utf-8", newline="") as f:
		for row in csv.DictReader(f):
			key = candidate_key_from_row(row)
			lbl = parse_optional_binary_label(row.get("is_muon"))
			if lbl is None:
				lbl = binary_from_label_class(label_class_from_row(row))
			if lbl is not None:
				out[key] = int(lbl)
	return out


def load_label_classes(
		path: Path,
		*,
		include_unknown: bool = False,
		unlabeled_as_noise: bool = False,
) -> Dict[CandidateKey, str]:
	"""Load ternary labels; unknown rows are excluded unless explicitly requested."""
	out: Dict[CandidateKey, str] = {}
	with Path(path).open("r", encoding="utf-8", newline="") as f:
		for row in csv.DictReader(f):
			key = candidate_key_from_row(row)
			cls = label_class_from_row(row)
			if cls == "unknown" and bool(unlabeled_as_noise):
				cls = "noise"
			if cls in TERNARY_CLASSES or bool(include_unknown):
				out[key] = cls
	return out
