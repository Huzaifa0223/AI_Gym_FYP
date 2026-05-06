"""
training.pose_extractor — Pose extraction abstraction for offline calibration.

Defines the :class:`PoseExtractor` protocol consumed by the calibrator and
ships two implementations:

* :class:`MediaPipePoseExtractor` — production. Wraps
  ``mediapipe.solutions.pose`` and yields one :class:`LandmarkFrame` per
  decoded video frame, or ``None`` when the pose model returned no
  landmarks for that frame.
* :class:`StubPoseExtractor` — dependency-injection seam for tests.
  Returns a hard-coded sequence; ``video_path`` is ignored.

This is the dependency boundary that keeps the calibrator's tests free
from real video fixtures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from core.schemas import LandmarkFrame


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class PoseExtractor(Protocol):
    """Yield per-frame :class:`LandmarkFrame` (or ``None``) from a video file."""

    def extract(self, video_path: Path) -> Iterator[LandmarkFrame | None]: ...


# ---------------------------------------------------------------------------
# Production implementation — MediaPipe + OpenCV
# ---------------------------------------------------------------------------

class MediaPipePoseExtractor:
    """Production extractor backed by MediaPipe Pose + OpenCV decoding.

    Construction is intentionally cheap — the MediaPipe Pose instance is
    **not** instantiated until :meth:`extract` is called, so unit tests can
    construct ``MediaPipePoseExtractor`` without pulling the model into
    memory.

    Args:
        model_complexity:         MediaPipe model complexity (0/1/2).
        min_detection_confidence: MediaPipe detection threshold.
        min_tracking_confidence:  MediaPipe tracking threshold.
    """

    def __init__(
        self,
        *,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._model_complexity = model_complexity
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence

    def extract(self, video_path: Path) -> Iterator[LandmarkFrame | None]:
        """Decode *video_path* and yield one frame per decoded image.

        Yields ``None`` for frames where MediaPipe returned no landmarks so
        the caller can decide whether to drop or pad them.

        Raises:
            RuntimeError: If OpenCV cannot open the video.
        """
        # Lazy imports keep the constructor model-free.
        import cv2  # type: ignore[import-not-found]
        import mediapipe as mp  # type: ignore[import-not-found]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self._model_complexity,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        try:
            frame_id = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks is None:
                    yield None
                else:
                    landmarks: list[tuple[float, float, float]] = []
                    visibility: list[float] = []
                    for lm in results.pose_landmarks.landmark:
                        landmarks.append(
                            (float(lm.x), float(lm.y), float(lm.z))
                        )
                        visibility.append(float(lm.visibility))
                    yield LandmarkFrame(
                        frame_id=frame_id,
                        timestamp_ms=int(frame_id * 1000.0 / fps),
                        landmarks=tuple(landmarks),
                        visibility=tuple(visibility),
                        user_height_cm=None,
                    )

                frame_id += 1
        finally:
            cap.release()
            pose.close()


# ---------------------------------------------------------------------------
# Test stub — no I/O, no model
# ---------------------------------------------------------------------------

class StubPoseExtractor:
    """Test stub returning a canned sequence; ``video_path`` is ignored.

    Args:
        frames: The exact sequence to yield. ``None`` entries simulate
                frames where MediaPipe found no pose.
    """

    def __init__(self, frames: list[LandmarkFrame | None]) -> None:
        self._frames = list(frames)

    def extract(self, video_path: Path) -> Iterator[LandmarkFrame | None]:
        yield from self._frames


__all__ = [
    "PoseExtractor",
    "MediaPipePoseExtractor",
    "StubPoseExtractor",
]
