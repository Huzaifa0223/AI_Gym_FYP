"""
tests.test_video_processor — pytest suite for core.video_processor.

The tests generate synthetic videos with ``cv2.VideoWriter`` at runtime so
that no binary fixtures ever land in the repository.  The synthetic clips
contain solid-colour frames, which intentionally never trigger MediaPipe
pose detection — this suite validates the *video-to-features pipeline
plumbing*, not pose accuracy.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.rep_counter import FrameFeatures
from core.video_processor import (
    NoFramesExtractedError,
    ProcessedFrame,
    UnsupportedExerciseError,
    VideoDecodeError,
    VideoProcessor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FPS = 30
_WIDTH = 160
_HEIGHT = 120


def _write_solid_video(path: Path, *, seconds: float = 2.0, fps: int = _FPS) -> int:
    """Write a solid-colour video to *path* and return the frame count.

    Uses the MJPG codec in an ``.avi`` container because it is bundled with
    OpenCV on every platform (no system ffmpeg/ffprobe required).
    """
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (_WIDTH, _HEIGHT))
    if not writer.isOpened():
        pytest.skip("MJPG VideoWriter unavailable in this OpenCV build")

    n_frames = int(round(seconds * fps))
    # Slight brightness gradient across frames so every frame is distinct;
    # MediaPipe still won't find a human in a solid gradient, which is
    # exactly what we want.
    try:
        for i in range(n_frames):
            shade = 20 + (i % 200)
            frame = np.full((_HEIGHT, _WIDTH, 3), shade, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return n_frames


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """A 2-second 30-fps solid video on disk."""
    path = tmp_path / "synthetic.avi"
    _write_solid_video(path, seconds=2.0, fps=_FPS)
    assert path.exists() and path.stat().st_size > 0
    return path


@pytest.fixture
def synthetic_video_bytes(synthetic_video: Path) -> bytes:
    return synthetic_video.read_bytes()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConstruction:
    """1. VideoProcessor instantiates cleanly; close() is idempotent."""

    def test_instantiate_and_close_idempotent(self) -> None:
        vp = VideoProcessor('bicep_curl')
        vp.close()
        vp.close()  # second close must be a no-op, not an error

    def test_unknown_exercise_raises(self) -> None:
        with pytest.raises(UnsupportedExerciseError):
            VideoProcessor('juggling')


class TestDecodeFailure:
    """3. process_bytes on non-video input raises VideoDecodeError."""

    def test_garbage_bytes_raise_decode_error(self) -> None:
        with VideoProcessor('bicep_curl') as vp:
            with pytest.raises(VideoDecodeError):
                # The generator must be consumed for the error to fire
                list(vp.process_bytes(b"not a video at all"))


class TestTimestampsAndFrameCount:
    """4 & 7. Timestamps monotonically increase; frame count matches writer."""

    def test_timestamps_and_count(self, synthetic_video_bytes: bytes) -> None:
        with VideoProcessor('bicep_curl') as vp:
            frames = list(vp.process_bytes(synthetic_video_bytes))

        expected = int(round(2.0 * _FPS))
        # Container/codec rounding can shave a frame or two; allow ±2.
        assert abs(len(frames) - expected) <= 2, (
            f"expected ~{expected} frames, got {len(frames)}"
        )

        # Monotonic non-decreasing timestamps, starting at 0.
        assert frames[0].timestamp == pytest.approx(0.0)
        for a, b in zip(frames, frames[1:]):
            assert b.timestamp >= a.timestamp


class TestNoLandmarksOnSolidFrames:
    """5. Solid-colour frames produce landmarks_detected=False."""

    def test_all_frames_report_no_landmarks(
        self, synthetic_video_bytes: bytes
    ) -> None:
        with VideoProcessor('bicep_curl') as vp:
            frames = list(vp.process_bytes(synthetic_video_bytes))

        assert len(frames) > 0
        assert all(isinstance(f, ProcessedFrame) for f in frames)
        assert all(f.landmarks_detected is False for f in frames)
        assert all(f.primary_angle == 0.0 for f in frames)
        assert all(f.features == FrameFeatures(0.0, 0.0, 0.0) for f in frames)


class TestContextManagerReleasesPose:
    """6. Context manager releases the MediaPipe pose on exit."""

    def test_exit_closes_pose(self, synthetic_video_bytes: bytes) -> None:
        vp = VideoProcessor('bicep_curl')
        with vp as ctx:
            assert ctx is vp
            # Drain at least one frame to prove the processor was usable.
            list(ctx.process_bytes(synthetic_video_bytes))

        # After __exit__, the processor must refuse further work.
        with pytest.raises(RuntimeError):
            list(vp.process_file(Path("does-not-matter.avi")))


class TestAllSupportedExercises:
    """Extra — every registered exercise works end-to-end on the same clip."""

    @pytest.mark.parametrize("exercise", ['bicep_curl', 'bent_over_row', 'push_up'])
    def test_all_exercises_produce_frames(
        self, exercise: str, synthetic_video_bytes: bytes
    ) -> None:
        with VideoProcessor(exercise) as vp:
            frames = list(vp.process_bytes(synthetic_video_bytes))
        assert len(frames) > 0


class TestEmptyVideo:
    """8. A zero-frame video raises NoFramesExtractedError.

    Creating a reliably-zero-frame file across codecs is flaky, so we mark
    xfail on failure rather than failing the suite.
    """

    @pytest.mark.xfail(
        reason="Zero-frame video synthesis is codec-dependent on Windows",
        strict=False,
    )
    def test_empty_video_raises(self, tmp_path: Path) -> None:
        # Open a writer, release it immediately without writing any frames.
        path = tmp_path / "empty.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(str(path), fourcc, _FPS, (_WIDTH, _HEIGHT))
        writer.release()
        if not path.exists() or path.stat().st_size == 0:
            pytest.skip("Writer did not produce a file header")

        with VideoProcessor('bicep_curl') as vp:
            with pytest.raises(NoFramesExtractedError):
                list(vp.process_file(path))
