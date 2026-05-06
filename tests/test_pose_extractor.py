"""
tests.test_pose_extractor — protocol smoke + StubPoseExtractor sequence.

The :class:`MediaPipePoseExtractor` is exercised only at construction time —
its ``extract`` method opens a real video and runs the model, which the
test suite avoids per the "no binary fixtures" project constraint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from core.schemas import LandmarkFrame
from training.pose_extractor import (
    MediaPipePoseExtractor,
    PoseExtractor,
    StubPoseExtractor,
)


def _lf(frame_id: int) -> LandmarkFrame:
    """A trivially-valid LandmarkFrame for sequence-order assertions."""
    return LandmarkFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 33,
        landmarks=tuple((0.5, 0.5, 0.0) for _ in range(33)),
        visibility=tuple(0.95 for _ in range(33)),
        user_height_cm=170.0,
    )


# ---------------------------------------------------------------------------
# MediaPipePoseExtractor — construction smoke only
# ---------------------------------------------------------------------------

class TestMediaPipePoseExtractorSmoke:
    def test_constructs_without_invoking_model(self) -> None:
        # Must succeed even with the model-instantiation path unreached.
        extractor = MediaPipePoseExtractor()
        assert extractor is not None

    def test_constructs_with_custom_params(self) -> None:
        extractor = MediaPipePoseExtractor(
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        assert extractor is not None

    def test_satisfies_pose_extractor_protocol(self) -> None:
        # PoseExtractor is @runtime_checkable, so isinstance works.
        assert isinstance(MediaPipePoseExtractor(), PoseExtractor)


# ---------------------------------------------------------------------------
# StubPoseExtractor — deterministic test stub
# ---------------------------------------------------------------------------

class TestStubPoseExtractor:
    def test_returns_canned_sequence_in_order(self) -> None:
        frames: list[LandmarkFrame | None] = [_lf(0), _lf(1), _lf(2)]
        stub = StubPoseExtractor(frames)
        result = list(stub.extract(Path("ignored.mp4")))
        assert len(result) == 3
        assert result[0] is not None and result[0].frame_id == 0
        assert result[1] is not None and result[1].frame_id == 1
        assert result[2] is not None and result[2].frame_id == 2

    def test_passes_through_none_entries(self) -> None:
        frames: list[LandmarkFrame | None] = [_lf(0), None, _lf(2), None]
        stub = StubPoseExtractor(frames)
        result = list(stub.extract(Path("ignored.mp4")))
        assert result == frames

    def test_video_path_is_ignored(self) -> None:
        # Same canned sequence regardless of the path argument.
        stub = StubPoseExtractor([_lf(0)])
        a = list(stub.extract(Path("a.mp4")))
        b = list(stub.extract(Path("b.mp4")))
        assert len(a) == 1
        assert len(b) == 1
        assert a[0] is not None and b[0] is not None
        assert a[0].frame_id == b[0].frame_id

    def test_satisfies_pose_extractor_protocol(self) -> None:
        assert isinstance(StubPoseExtractor([]), PoseExtractor)
