"""
utils.normalization — Body-segment-length normalization for landmarks.

Computes a per-frame scalar that downstream consumers use to scale landmark
coordinates into unit-less form. Priority:

1. ``user_height_cm`` if present on the :class:`LandmarkFrame`.
2. Otherwise, the mean of shoulder-width and hip-width derived from
   visible landmarks (visibility ≥ 0.5).

When fewer than the four required body-anchor landmarks (left shoulder,
right shoulder, left hip, right hip) clear the visibility threshold,
:class:`InsufficientLandmarksError` is raised so the caller can drop the
frame instead of silently producing a meaningless unit.

Note on dimensional consistency
-------------------------------
The two priority paths return values in different scales: ``user_height_cm``
is in real-world centimetres (~170), while shoulder/hip-width is in the
landmark coordinate system (typically ≈ 0.15 in normalised [0, 1] coords).
Consumers must apply the same division to both paths consistently.
Body-segment-relative math (joint angles, ratios) is scale-invariant, so
the asymmetry does not affect downstream rule engines or the calibrator's
threshold derivation.
"""
from __future__ import annotations

import math
from typing import Final

from core.schemas import LandmarkFrame


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_MIN_LANDMARK_VISIBILITY: Final[float] = 0.5

# MediaPipe Pose landmark indices for the body-anchor fallback.
_LEFT_SHOULDER: Final[int] = 11
_RIGHT_SHOULDER: Final[int] = 12
_LEFT_HIP: Final[int] = 23
_RIGHT_HIP: Final[int] = 24

_REQUIRED_ANCHORS: Final[tuple[int, ...]] = (
    _LEFT_SHOULDER,
    _RIGHT_SHOULDER,
    _LEFT_HIP,
    _RIGHT_HIP,
)


class InsufficientLandmarksError(ValueError):
    """Raised when normalisation cannot derive a unit from the frame.

    Triggered when ``user_height_cm`` is ``None`` **and** fewer than the
    four required body-anchor landmarks (both shoulders + both hips)
    have visibility ≥ 0.5.
    """


def compute_body_segment_unit(lf: LandmarkFrame) -> float:
    """Return the per-frame normalisation unit for *lf*.

    See module docstring for the priority ordering and dimensional caveat.

    Args:
        lf: The frame to compute a unit for.

    Returns:
        A positive scalar representing the body-segment unit length.

    Raises:
        InsufficientLandmarksError: When falling back to body-segment
            widths and any of the four required anchor landmarks
            (indices 11, 12, 23, 24) has visibility < 0.5.
    """
    if lf.user_height_cm is not None:
        return float(lf.user_height_cm)

    if not all(lf.visibility[i] >= _MIN_LANDMARK_VISIBILITY
               for i in _REQUIRED_ANCHORS):
        raise InsufficientLandmarksError(
            "Need both shoulders (11, 12) and both hips (23, 24) at "
            f"visibility >= {_MIN_LANDMARK_VISIBILITY:.1f} to derive the "
            "shoulder/hip-width fallback unit."
        )

    sx1, sy1, _ = lf.landmarks[_LEFT_SHOULDER]
    sx2, sy2, _ = lf.landmarks[_RIGHT_SHOULDER]
    hx1, hy1, _ = lf.landmarks[_LEFT_HIP]
    hx2, hy2, _ = lf.landmarks[_RIGHT_HIP]

    shoulder_width = math.hypot(sx2 - sx1, sy2 - sy1)
    hip_width = math.hypot(hx2 - hx1, hy2 - hy1)
    return (shoulder_width + hip_width) / 2.0


__all__ = ["InsufficientLandmarksError", "compute_body_segment_unit"]
