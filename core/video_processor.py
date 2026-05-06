"""
core.video_processor — Decode an uploaded video into an iterator of per-frame
:class:`~core.rep_counter.FrameFeatures`.

The processor bridges raw video bytes and the downstream rep-counter / form-scorer
pipeline.  For every decoded frame it runs MediaPipe Pose once, extracts the
exercise-specific primary/secondary/tertiary joint angles (matching the exact
formulas used by ``exercises/*.py`` and ``core/rep_counter.py``), and yields a
:class:`ProcessedFrame`.

Design notes
------------
* **Single responsibility** — no rep counting, no form scoring, no REST glue.
  The output is a stream of landmark-derived features, nothing else.
* **Windows-compatible decoding** — OpenCV's ``VideoCapture`` needs a filesystem
  path, so :meth:`VideoProcessor.process_bytes` writes the upload to a
  :func:`tempfile.NamedTemporaryFile` and removes it in a ``finally`` block.
* **Graceful landmark failure** — a frame that MediaPipe cannot label is still
  emitted with ``landmarks_detected=False`` so the caller can decide whether
  to count, skip, or report it.  Swallowing such frames silently would hide
  occlusion problems upstream.
* **Resource hygiene** — the MediaPipe ``Pose`` instance owns ~30 MB of native
  memory.  :meth:`close` releases it; the class also implements the context
  manager protocol so callers never have to remember.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import mediapipe as mp
import numpy as np

from core.rep_counter import FrameFeatures
from utils.angle_utils import calculate_angle_3d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UnsupportedExerciseError(ValueError):
    """Raised when the requested exercise has no registered feature extractor."""


class VideoDecodeError(RuntimeError):
    """Raised when OpenCV cannot open the supplied video."""


class NoFramesExtractedError(RuntimeError):
    """Raised when a video opens successfully but yields zero decoded frames."""


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProcessedFrame:
    """One frame of landmark-derived features.

    Attributes:
        timestamp:          Seconds since the start of the video
                            (``frame_index / fps``).
        primary_angle:      The exercise's primary joint angle in degrees;
                            ``0.0`` when pose detection failed this frame.
        features:           Three-element feature vector matching the schema
                            consumed by :class:`~core.ml_aggregator.MLAggregator`.
        landmarks_detected: ``True`` iff MediaPipe returned pose landmarks.
    """
    timestamp: float
    primary_angle: float
    features: FrameFeatures
    landmarks_detected: bool


# ---------------------------------------------------------------------------
# Feature extractors — one per supported exercise
# ---------------------------------------------------------------------------
#
# Every extractor mirrors the angle formulas already used elsewhere in the
# codebase so that the live API (``main.py``) and the offline video pipeline
# produce identical features for identical poses.
#
# MediaPipe Pose landmark ids used throughout:
#   11/12: left/right shoulder        13/14: left/right elbow
#   15/16: left/right wrist           23/24: left/right hip
#   25/26: left/right knee            27/28: left/right ankle
# The existing exercise modules all use the right-side landmarks (12/14/16/...).


def _xyz(lm: object) -> np.ndarray:
    """Return a landmark's ``(x, y, z)`` as a numpy array."""
    return np.array([lm.x, lm.y, lm.z], dtype=float)  # type: ignore[attr-defined]


def _extract_bicep_features(landmarks: list) -> FrameFeatures:
    """Compute bicep-curl features.

    * primary   — shoulder(12) → elbow(14) → wrist(16): elbow flexion angle
    * secondary — hip(24) → shoulder(12) → elbow(14): upper-arm deviation from
                  the torso line (proxy for elbow drift / torso lean).
    * tertiary  — hip(24) → shoulder(12) → wrist(16): total arm deviation
                  from the body line (matches ``bicep_curl.py``'s trained
                  feature).
    """
    shoulder = _xyz(landmarks[12])
    elbow = _xyz(landmarks[14])
    wrist = _xyz(landmarks[16])
    hip = _xyz(landmarks[24])

    primary = calculate_angle_3d(shoulder, elbow, wrist)
    secondary = calculate_angle_3d(hip, shoulder, elbow)
    tertiary = calculate_angle_3d(hip, shoulder, wrist)
    return FrameFeatures(
        primary_angle=float(primary),
        secondary_angle=float(secondary),
        tertiary_angle=float(tertiary),
    )


def _extract_back_features(landmarks: list) -> FrameFeatures:
    """Compute back-row features (identical geometry to ``back_exercise.py``).

    * primary   — shoulder(12) → elbow(14) → wrist(16): elbow bend at pull.
    * secondary — hip(24) → shoulder(12) → elbow(14): torso/back lean.
    * tertiary  — hip(24) → shoulder(12) → wrist(16): full-arm deviation
                  from the body line (matches training feature schema).
    """
    shoulder = _xyz(landmarks[12])
    elbow = _xyz(landmarks[14])
    wrist = _xyz(landmarks[16])
    hip = _xyz(landmarks[24])

    primary = calculate_angle_3d(shoulder, elbow, wrist)
    secondary = calculate_angle_3d(hip, shoulder, elbow)
    tertiary = calculate_angle_3d(hip, shoulder, wrist)
    return FrameFeatures(
        primary_angle=float(primary),
        secondary_angle=float(secondary),
        tertiary_angle=float(tertiary),
    )


def _extract_chest_features(landmarks: list) -> FrameFeatures:
    """Compute push-up features (identical geometry to ``chest_exercise.py``).

    * primary   — shoulder(12) → elbow(14) → wrist(16): elbow flexion depth.
    * secondary — shoulder(12) → hip(24) → knee(26): body/plank alignment
                  (values close to 180° indicate a straight plank).
    * tertiary  — hip(24) → knee(26) → ankle(28): leg straightness.
    """
    shoulder = _xyz(landmarks[12])
    elbow = _xyz(landmarks[14])
    wrist = _xyz(landmarks[16])
    hip = _xyz(landmarks[24])
    knee = _xyz(landmarks[26])
    ankle = _xyz(landmarks[28])

    primary = calculate_angle_3d(shoulder, elbow, wrist)
    secondary = calculate_angle_3d(shoulder, hip, knee)
    tertiary = calculate_angle_3d(hip, knee, ankle)
    return FrameFeatures(
        primary_angle=float(primary),
        secondary_angle=float(secondary),
        tertiary_angle=float(tertiary),
    )


_FEATURE_EXTRACTORS: dict[str, Callable[[list], FrameFeatures]] = {
    'bicep_curl': _extract_bicep_features,
    'bent_over_row': _extract_back_features,
    'push_up': _extract_chest_features,
}


# ---------------------------------------------------------------------------
# VideoProcessor
# ---------------------------------------------------------------------------

_DEFAULT_FPS = 30.0  # fallback when CAP_PROP_FPS is missing or zero


class VideoProcessor:
    """Decode a video and emit per-frame :class:`ProcessedFrame` records.

    Args:
        exercise:                 Exercise key registered in
                                  :data:`_FEATURE_EXTRACTORS`.
        model_complexity:         MediaPipe model complexity (0/1/2).
                                  Matches the live ``api_backend`` default.
        min_detection_confidence: MediaPipe detection threshold.
        min_tracking_confidence:  MediaPipe tracking threshold.

    Raises:
        UnsupportedExerciseError: If *exercise* has no feature extractor.
    """

    def __init__(
        self,
        exercise: str,
        *,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if exercise not in _FEATURE_EXTRACTORS:
            raise UnsupportedExerciseError(
                f"Unsupported exercise {exercise!r}. "
                f"Supported: {sorted(_FEATURE_EXTRACTORS)}"
            )

        self._exercise = exercise
        self._extractor = _FEATURE_EXTRACTORS[exercise]
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._closed = False

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> VideoProcessor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- Resource release ----------------------------------------------------

    def close(self) -> None:
        """Release the MediaPipe Pose instance. Idempotent."""
        if self._closed:
            return
        try:
            self._pose.close()
        except Exception:
            logger.exception("MediaPipe Pose.close() raised — ignoring")
        finally:
            self._closed = True

    # -- Public iteration API -----------------------------------------------

    def process_bytes(self, video_bytes: bytes) -> Iterator[ProcessedFrame]:
        """Decode *video_bytes* from memory and iterate over frames.

        A temporary file is required because OpenCV's ``VideoCapture`` cannot
        read from a Python buffer directly on Windows.  The file is deleted
        in a ``finally`` block regardless of success.

        Args:
            video_bytes: Full contents of an uploaded video file.

        Yields:
            :class:`ProcessedFrame` per decoded frame.

        Raises:
            VideoDecodeError:       The video could not be opened.
            NoFramesExtractedError: The video opened but decoded zero frames.
        """
        # delete=False so we can close the handle before OpenCV opens it;
        # Windows refuses concurrent reads on an unclosed NamedTemporaryFile.
        tmp = tempfile.NamedTemporaryFile(suffix='.video', delete=False)
        try:
            tmp.write(video_bytes)
            tmp.flush()
            tmp.close()
            yield from self.process_file(Path(tmp.name))
        finally:
            try:
                Path(tmp.name).unlink(missing_ok=True)
            except OSError:
                logger.exception("Failed to delete temp video %s", tmp.name)

    def process_file(self, video_path: Path) -> Iterator[ProcessedFrame]:
        """Decode the file at *video_path* and iterate over frames.

        Args:
            video_path: Path to a decodable video file.

        Yields:
            :class:`ProcessedFrame` per decoded frame.

        Raises:
            VideoDecodeError:       The video could not be opened.
            NoFramesExtractedError: The video opened but decoded zero frames.
        """
        if self._closed:
            raise RuntimeError("VideoProcessor has been closed")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoDecodeError(
                f"OpenCV could not open video: {video_path}"
            )

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0.0:
                logger.warning(
                    "CAP_PROP_FPS reported %.2f for %s — falling back to %.1f",
                    fps, video_path, _DEFAULT_FPS,
                )
                fps = _DEFAULT_FPS

            frame_index = 0
            yielded_any = False

            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break

                timestamp = frame_index / fps
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self._pose.process(frame_rgb)

                if results.pose_landmarks is None:
                    yield ProcessedFrame(
                        timestamp=timestamp,
                        primary_angle=0.0,
                        features=FrameFeatures(0.0, 0.0, 0.0),
                        landmarks_detected=False,
                    )
                else:
                    features = self._extractor(results.pose_landmarks.landmark)
                    yield ProcessedFrame(
                        timestamp=timestamp,
                        primary_angle=features.primary_angle,
                        features=features,
                        landmarks_detected=True,
                    )

                frame_index += 1
                yielded_any = True
        finally:
            cap.release()

        if not yielded_any:
            raise NoFramesExtractedError(
                f"Video at {video_path} opened but produced zero frames"
            )
