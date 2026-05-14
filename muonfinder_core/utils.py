from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import json
import numpy as np


def json_clean(value: Any) -> Any:
    if is_dataclass(value):
        return json_clean(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [json_clean(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    return value


def dumps_json(value: Any) -> str:
    return json.dumps(json_clean(value), ensure_ascii=False)


def loads_json(text: str, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def metric_float(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        try:
            value = float(row.get(key, np.nan))
        except Exception:
            value = np.nan
        if np.isfinite(value):
            return value
    return float("nan")


def to_contiguous_spans(indices: Sequence[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    spans: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        iv = int(value)
        if iv == prev + 1:
            prev = iv
            continue
        spans.append((start, prev))
        start = iv
        prev = iv
    spans.append((start, prev))
    return spans


def flatten_dict_keys(rows: Iterable[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(str(key) for key in row.keys())
    return sorted(keys)


def human_text_key(text: str) -> list[object]:
    import re

    parts = re.split(r"(\d+)", text)
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def nearest_row_by_peak(rows: Sequence[dict[str, Any]], peak_index: int) -> dict[str, Any] | None:
    if not rows:
        return None
    scored: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        try:
            peak = int(row.get("peak_index", -1))
        except Exception:
            continue
        scored.append((abs(peak - int(peak_index)), row))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    return scored[0][1]
