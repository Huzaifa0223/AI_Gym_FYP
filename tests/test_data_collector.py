"""
tests.test_data_collector — ThresholdCalibrator end-to-end.

Uses :class:`StubPoseExtractor` so no real video is needed. Test scenarios
cover the happy path, the inconclusive verdict, and dropped-frame handling
per ``docs/architecture_3pillar.md`` §1 / spec brief.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from core.schemas import HeuristicThresholds, LandmarkFrame
from training.data_collector import ThresholdCalibrator
from training.pose_extractor import StubPoseExtractor


# ---------------------------------------------------------------------------
# Frame fabrication
# ---------------------------------------------------------------------------

def _frame_with_elbow_angle(elbow_deg: float, frame_id: int = 0) -> LandmarkFrame:
    """Build a LandmarkFrame whose right-arm geometry produces *elbow_deg*.

    The shoulder→elbow→wrist triangle is constructed so that the angle at
    the elbow equals *elbow_deg*. Other landmarks (right hip, left side) are
    set so the calibrator's required-visibility check passes.
    """
    theta = math.radians(elbow_deg)
    # Right shoulder at (0.5, 0.5), right elbow at (0.6, 0.5).
    # ba = shoulder - elbow = (-0.1, 0); pick |bc| = 0.1 to keep both arm
    # segments equal-length, and place wrist so angle at elbow = θ.
    shoulder = (0.5, 0.5, 0.0)
    elbow = (0.6, 0.5, 0.0)
    wrist = (
        0.6 - 0.1 * math.cos(theta),
        0.5 + 0.1 * math.sin(theta),
        0.0,
    )
    hip = (0.5, 0.6, 0.0)  # below shoulder

    landmarks: list[tuple[float, float, float]] = [(0.5, 0.5, 0.0)] * 33
    landmarks[11] = (0.4, 0.5, 0.0)  # left shoulder (anchor for normalization)
    landmarks[12] = shoulder
    landmarks[14] = elbow
    landmarks[16] = wrist
    landmarks[23] = (0.4, 0.6, 0.0)  # left hip
    landmarks[24] = hip

    return LandmarkFrame(
        frame_id=frame_id,
        timestamp_ms=int(frame_id * 1000.0 / 30.0),
        landmarks=tuple(landmarks),
        visibility=tuple(0.95 for _ in range(33)),
        user_height_cm=170.0,
    )


def _smooth_curl_video(num_frames: int = 60, reps: int = 3) -> list[LandmarkFrame | None]:
    """Sine-wave elbow trajectory between 60° and 140° — low-CV good form."""
    frames: list[LandmarkFrame | None] = []
    for i in range(num_frames):
        phase = (i / num_frames) * reps * 2 * math.pi
        angle = 100.0 + 40.0 * math.sin(phase)
        frames.append(_frame_with_elbow_angle(angle, frame_id=i))
    return frames


def _chaotic_curl_video(num_frames: int = 60, seed: int = 42) -> list[LandmarkFrame | None]:
    """Random elbow angles in [60°, 140°] — high-CV bad form."""
    rng = random.Random(seed)
    frames: list[LandmarkFrame | None] = []
    for i in range(num_frames):
        angle = rng.uniform(60.0, 140.0)
        frames.append(_frame_with_elbow_angle(angle, frame_id=i))
    return frames


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_unsupported_exercise_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported exercise"):
            ThresholdCalibrator("squat")  # not in _SUPPORTED_EXERCISES


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_status_ok_with_differentiating_thresholds(self, tmp_path: Path) -> None:
        good = _smooth_curl_video(num_frames=60)
        bad = _chaotic_curl_video(num_frames=60)
        calibrator = ThresholdCalibrator(
            "bicep_curl",
            extractor=StubPoseExtractor(good + bad),
            output_dir=tmp_path,
        )
        # The stub yields good then bad consecutively — but calibrator calls
        # extract() once per video. Use two separate stubs instead.
        good_extractor = StubPoseExtractor(good)
        bad_extractor = StubPoseExtractor(bad)

        # Re-build with a stub that supports two videos. Since the calibrator
        # calls extractor.extract(path) twice (once per video), we need a
        # stub that returns different sequences per call. The simplest path
        # is two independent calibrator runs against fresh stubs — but the
        # calibrator API expects one extractor for both videos.
        #
        # Solution: a thin two-call stub.
        class _TwoCallStub:
            def __init__(self, first: list, second: list) -> None:
                self._calls = [first, second]

            def extract(self, video_path):  # type: ignore[no-untyped-def]
                yield from self._calls.pop(0)

        calibrator = ThresholdCalibrator(
            "bicep_curl",
            extractor=_TwoCallStub(good, bad),
            output_dir=tmp_path,
        )
        thresholds, log, status = calibrator.calibrate(
            Path("good.mp4"), Path("bad.mp4"),
        )

        assert status == "ok"
        assert isinstance(thresholds, HeuristicThresholds)
        assert thresholds.exercise == "bicep_curl"
        assert thresholds.rom_band[0] < thresholds.rom_band[1]
        # All four sources tagged derived_from_video per Stage 3 brief.
        assert set(thresholds.sources.values()) == {"derived_from_video"}
        # At least one threshold differentiated good from bad.
        assert log["differentiating_thresholds"]
        assert log["status"] == "ok"

    def test_writes_both_json_files(self, tmp_path: Path) -> None:
        good = _smooth_curl_video(60)
        bad = _chaotic_curl_video(60)

        class _TwoCallStub:
            def __init__(self, first: list, second: list) -> None:
                self._calls = [first, second]

            def extract(self, video_path):  # type: ignore[no-untyped-def]
                yield from self._calls.pop(0)

        calibrator = ThresholdCalibrator(
            "bicep_curl",
            extractor=_TwoCallStub(good, bad),
            output_dir=tmp_path,
        )
        thresholds, log, _ = calibrator.calibrate(
            Path("good.mp4"), Path("bad.mp4"),
        )
        thresholds_path, log_path = calibrator.write(thresholds, log)

        assert thresholds_path.exists()
        assert log_path.exists()

        with thresholds_path.open() as f:
            saved = json.load(f)
        assert saved["exercise"] == "bicep_curl"
        assert "rom_band" in saved
        assert saved["sources"]["rom_band"] == "derived_from_video"

        with log_path.open() as f:
            saved_log = json.load(f)
        assert saved_log["status"] == "ok"
        assert saved_log["good_video"]["valid_frame_count"] == 60


# ---------------------------------------------------------------------------
# Inconclusive
# ---------------------------------------------------------------------------

class TestInconclusive:
    def test_bad_inside_good_band_returns_inconclusive(
        self, tmp_path: Path,
    ) -> None:
        # Good and bad are *both* smooth — their stats overlap so no
        # threshold differentiates.
        good = _smooth_curl_video(60)
        bad_same = _smooth_curl_video(60)

        class _TwoCallStub:
            def __init__(self, first: list, second: list) -> None:
                self._calls = [first, second]

            def extract(self, video_path):  # type: ignore[no-untyped-def]
                yield from self._calls.pop(0)

        calibrator = ThresholdCalibrator(
            "bicep_curl",
            extractor=_TwoCallStub(good, bad_same),
            output_dir=tmp_path,
        )
        thresholds, log, status = calibrator.calibrate(
            Path("good.mp4"), Path("bad.mp4"),
        )

        assert status == "inconclusive"
        assert log["status"] == "inconclusive"
        assert log["differentiating_thresholds"] == []
        assert any("differentiate" in w for w in log["warnings"])

    def test_too_few_good_frames_returns_inconclusive(
        self, tmp_path: Path,
    ) -> None:
        # Good has fewer than _MIN_VALID_FRAMES (30) — fails the gate even
        # if differentiation against bad would otherwise succeed.
        good = _smooth_curl_video(20)  # below 30
        bad = _chaotic_curl_video(60)

        class _TwoCallStub:
            def __init__(self, first: list, second: list) -> None:
                self._calls = [first, second]

            def extract(self, video_path):  # type: ignore[no-untyped-def]
                yield from self._calls.pop(0)

        calibrator = ThresholdCalibrator(
            "bicep_curl",
            extractor=_TwoCallStub(good, bad),
            output_dir=tmp_path,
        )
        _, log, status = calibrator.calibrate(
            Path("good.mp4"), Path("bad.mp4"),
        )
        assert status == "inconclusive"
        assert any("valid frames" in w for w in log["warnings"])


# ---------------------------------------------------------------------------
# Dropped frames
# ---------------------------------------------------------------------------

class TestDroppedFrames:
    def test_none_frames_counted_as_dropped(self, tmp_path: Path) -> None:
        good_with_drops: list[LandmarkFrame | None] = []
        for i, lf in enumerate(_smooth_curl_video(60)):
            good_with_drops.append(lf)
            if i % 5 == 0:
                good_with_drops.append(None)  # 12 dropped frames inserted

        bad = _chaotic_curl_video(60)

        class _TwoCallStub:
            def __init__(self, first: list, second: list) -> None:
                self._calls = [first, second]

            def extract(self, video_path):  # type: ignore[no-untyped-def]
                yield from self._calls.pop(0)

        calibrator = ThresholdCalibrator(
            "bicep_curl",
            extractor=_TwoCallStub(good_with_drops, bad),
            output_dir=tmp_path,
        )
        _, log, _ = calibrator.calibrate(
            Path("good.mp4"), Path("bad.mp4"),
        )

        good_stats = log["good_video"]
        assert good_stats["frame_count"] == 60 + 12  # total emitted
        assert good_stats["valid_frame_count"] == 60
        assert good_stats["dropped_frame_count"] == 12
