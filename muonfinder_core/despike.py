from __future__ import annotations

from .data_model import CorrectionResult, DespikeChord, DespikeStage

import numpy as np


def build_placeholder_correction(raw_spectra: np.ndarray) -> CorrectionResult:
    return CorrectionResult(
        corrected_spectra=np.asarray(raw_spectra, dtype=np.float32).copy(),
        stages=[DespikeStage(stage_id="stage:0", name="preview", description="Raw passthrough placeholder")],
        chords=[],
    )
