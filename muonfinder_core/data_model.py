from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class SpectrumDataset:
    path: Path
    x_axis: np.ndarray
    spectra: np.ndarray
    xpos: Optional[np.ndarray] = None
    ypos: Optional[np.ndarray] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateSegment:
    y: int
    x: int
    peak_index: int
    start: int
    end: int
    peak_height: float
    area: float

    @property
    def candidate_id(self) -> str:
        return f"candidate:{self.y}:{self.x}:{self.peak_index}:{self.start}:{self.end}"


@dataclass(frozen=True)
class BackgroundNoiseEstimate:
    method: str
    value: float


@dataclass(frozen=True)
class DespikeChord:
    chord_id: str
    stage_id: str
    y: int
    x: int
    left: int
    right: int
    method: str
    y_left: float
    y_right: float


@dataclass(frozen=True)
class DespikeStage:
    stage_id: str
    name: str
    description: str


@dataclass(frozen=True)
class CorrectionResult:
    corrected_spectra: np.ndarray
    stages: list[DespikeStage]
    chords: list[DespikeChord]


@dataclass(frozen=True)
class PipelineArtifacts:
    dataset: SpectrumDataset
    x_axis: np.ndarray
    spectra: np.ndarray
    corrected_spectra: np.ndarray
    score_map: np.ndarray
    candidate_mask: np.ndarray
    overlays: dict[str, dict[int, np.ndarray]]
    candidates_by_pixel: dict[tuple[int, int], list[CandidateSegment]]
    candidate_records_by_pixel: dict[tuple[int, int], list[dict[str, Any]]]
    source_coords_map: dict[tuple[int, int], tuple[int, int]]
    small_morphology_by_pixel: dict[tuple[int, int], dict[str, Any]]
    despike_stages: list[DespikeStage]
    despike_chords: list[DespikeChord]
    metadata: dict[str, Any]
