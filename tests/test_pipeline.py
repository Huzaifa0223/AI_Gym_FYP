"""
tests.test_pipeline — ScoringPipeline composition.

Uses real VideoProcessor + MediaPipe against synthetic solid-colour videos
(no pose detected). The synthetic clips never trigger landmark detection,
so 0 reps are emitted — exercising the "no reps detected" headline path
plus the threshold pre-flight, MLAggregator graceful-degradation, and
CNN+LSTM-null fusion all in one trip.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest

from core.cnn_lstm_scorer import NullFormQualityScorer
from core.pipeline import ScoringPipeline
from core.rep_counter import RepEvent
from core.threshold_provider import ThresholdProvider, ThresholdsNotFoundError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FPS = 30
_WIDTH = 160
_HEIGHT = 120


def _write_solid_video(path: Path, *, seconds: float = 1.0) -> bytes:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, _FPS, (_WIDTH, _HEIGHT))
    if not writer.isOpened():
        pytest.skip("MJPG VideoWriter unavailable in this OpenCV build")
    try:
        for i in range(int(seconds * _FPS)):
            shade = 20 + (i % 200)
            writer.write(np.full((_HEIGHT, _WIDTH, 3), shade, dtype=np.uint8))
    finally:
        writer.release()
    return path.read_bytes()


@pytest.fixture
def synthetic_video_bytes(tmp_path: Path) -> bytes:
    return _write_solid_video(tmp_path / "synthetic.avi")


# ---------------------------------------------------------------------------
# Happy path — no reps but the pipeline composes cleanly
# ---------------------------------------------------------------------------

class TestNoRepsHeadline:
    def test_solid_video_produces_zero_rep_headline(
        self, synthetic_video_bytes: bytes,
    ) -> None:
        # Production threshold provider points at the committed seed.
        pipeline = ScoringPipeline()
        result = pipeline.score(synthetic_video_bytes, "bicep_curl", "adult")

        assert result.headline.rep_count == 0
        assert result.headline.score == 0
        assert result.headline.form_feedback == "No reps detected"
        assert result.headline.source_breakdown == {
            "rules": 0, "rf": None, "cnn_lstm": None,
        }
        assert result.average_score is None
        assert result.frames_processed > 0
        assert result.frames_with_landmarks == 0
        assert result.per_rep == []


# ---------------------------------------------------------------------------
# Threshold pre-flight — ThresholdsNotFoundError surfaces unchanged
# ---------------------------------------------------------------------------

class TestThresholdsPreFlight:
    def test_missing_thresholds_raises(self, tmp_path: Path) -> None:
        # tmp_path has no per-exercise subfolders, so the provider can't
        # find a thresholds file for any exercise.
        pipeline = ScoringPipeline(
            threshold_provider=ThresholdProvider(exercises_dir=tmp_path),
        )
        with pytest.raises(ThresholdsNotFoundError):
            # video bytes never read — pre-flight raises first
            pipeline.score(b"any-bytes", "bicep_curl", "adult")


# ---------------------------------------------------------------------------
# CNN+LSTM injection — non-null scorer contributes to source_breakdown
# ---------------------------------------------------------------------------

class _ConstantScorer:
    """FormQualityScorer that returns a constant score for every rep."""

    def __init__(self, value: int) -> None:
        self._value = value

    def score(self, rep: RepEvent) -> int | None:
        return self._value


class TestCnnLstmInjection:
    def test_null_scorer_produces_none_in_breakdown(
        self, synthetic_video_bytes: bytes,
    ) -> None:
        # Default pipeline uses NullFormQualityScorer; even with no reps,
        # the cnn_lstm key in source_breakdown should stay None.
        pipeline = ScoringPipeline(cnn_lstm_scorer=NullFormQualityScorer())
        result = pipeline.score(synthetic_video_bytes, "bicep_curl", "adult")
        assert result.headline.source_breakdown["cnn_lstm"] is None


# ---------------------------------------------------------------------------
# Construction smoke
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self) -> None:
        pipeline = ScoringPipeline()
        assert pipeline is not None

    def test_with_injected_components(self) -> None:
        pipeline = ScoringPipeline(
            threshold_provider=ThresholdProvider(),
            cnn_lstm_scorer=_ConstantScorer(75),
        )
        assert pipeline is not None
