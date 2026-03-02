"""
utils.py - Shared coordinate and angle utilities for AI Gym Trainer.

Keep all pose-math helpers here so exercise modules stay focused on logic.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe Pose Landmark IDs (quick reference)
# ---------------------------------------------------------------------------
# 11 - Left Shoulder      12 - Right Shoulder
# 13 - Left Elbow         14 - Right Elbow
# 15 - Left Wrist         16 - Right Wrist
# 23 - Left Hip           24 - Right Hip
# 25 - Left Knee          26 - Right Knee
# 27 - Left Ankle         28 - Right Ankle
# ---------------------------------------------------------------------------


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate the angle (in degrees) at joint B formed by points A-B-C.

    Uses atan2 so the result is always in the range [0, 180] degrees,
    matching the convention used by MediaPipe-based exercise detectors.

    Args:
        a: (x, y) or (x, y, z) coordinates of the first point  (e.g. Shoulder).
        b: (x, y) or (x, y, z) coordinates of the vertex point (e.g. Elbow).
        c: (x, y) or (x, y, z) coordinates of the third point  (e.g. Wrist).

    Returns:
        Angle in degrees at point B, clamped to [0, 180].

    Example:
        >>> shoulder = np.array([0.5, 0.3])
        >>> elbow    = np.array([0.5, 0.5])
        >>> wrist    = np.array([0.7, 0.5])
        >>> calculate_angle(shoulder, elbow, wrist)
        90.0

    Suggested thresholds (bicep curl example):
        - "Up"   state : angle <  30°
        - "Down" state : angle > 160°
    """
    a = np.array(a[:2], dtype=float)  # use only x, y
    b = np.array(b[:2], dtype=float)
    c = np.array(c[:2], dtype=float)

    # Vectors from vertex B
    ba = a - b
    bc = c - b

    # atan2 of the cross product and dot product gives the signed angle;
    # abs() + degrees conversion gives an unsigned [0, 180] result.
    angle_rad = math.atan2(
        abs(ba[0] * bc[1] - ba[1] * bc[0]),  # |cross product| (sine component)
        float(np.dot(ba, bc)),                 # dot product      (cosine component)
    )

    return math.degrees(angle_rad)


def calculate_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate the angle (in degrees) at joint B formed by points A-B-C.

    Uses the law of cosines on full 3D (x, y, z) coordinates, matching the
    convention used by BackExercise and ChestExercise where MediaPipe provides
    a meaningful Z depth component.

    Args:
        a: (x, y, z) coordinates of the first point  (e.g. Shoulder).
        b: (x, y, z) coordinates of the vertex point (e.g. Elbow).
        c: (x, y, z) coordinates of the third point  (e.g. Wrist).

    Returns:
        Angle in degrees at point B, clamped to [0, 180].

    Example:
        >>> shoulder = np.array([0.0, 0.0, 0.0])
        >>> elbow    = np.array([1.0, 0.0, 0.0])
        >>> wrist    = np.array([1.0, 1.0, 0.0])
        >>> calculate_angle_3d(shoulder, elbow, wrist)
        90.0
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))
