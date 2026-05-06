"""
tests.test_normalization — body-segment-length normalisation.

See ``utils/normalization.py`` for the priority ordering. Tests cover:

* the height-priority path (``user_height_cm`` returned directly),
* the shoulder/hip-width fallback,
* the fail-loud path when too few visible landmarks are present.
"""
from __future__ import annotations

import math

import pytest

from core.schemas import LandmarkFrame
from utils.normalization import (
    InsufficientLandmarksError,
    compute_body_segment_unit,
)


def _lf(
    *,
    visibility_overrides: dict[int, float] | None = None,
    user_height_cm: float | None = None,
) -> LandmarkFrame:
    """Construct a 33-landmark frame with shoulders + hips at known positions."""
    landmarks: list[tuple[float, float, float]] = [(0.5, 0.5, 0.0)] * 33
    # Place the four anchors at known coordinates so widths are predictable.
    landmarks[11] = (0.40, 0.30, 0.0)  # left shoulder
    landmarks[12] = (0.60, 0.30, 0.0)  # right shoulder  -> width 0.20
    landmarks[23] = (0.42, 0.60, 0.0)  # left hip
    landmarks[24] = (0.58, 0.60, 0.0)  # right hip       -> width 0.16
    visibility = [0.95] * 33
    if visibility_overrides:
        for idx, vis in visibility_overrides.items():
            visibility[idx] = vis
    return LandmarkFrame(
        frame_id=0,
        timestamp_ms=0,
        landmarks=tuple(landmarks),
        visibility=tuple(visibility),
        user_height_cm=user_height_cm,
    )


class TestHeightPriority:
    def test_returns_user_height_when_provided(self) -> None:
        unit = compute_body_segment_unit(_lf(user_height_cm=170.0))
        assert unit == 170.0

    def test_height_priority_ignores_visibility(self) -> None:
        # Even when no anchor landmark is visible, height short-circuits.
        unit = compute_body_segment_unit(
            _lf(
                visibility_overrides={11: 0.0, 12: 0.0, 23: 0.0, 24: 0.0},
                user_height_cm=180.0,
            ),
        )
        assert unit == 180.0


class TestShoulderHipFallback:
    def test_mean_of_shoulder_and_hip_width(self) -> None:
        unit = compute_body_segment_unit(_lf())
        # shoulder width 0.20, hip width 0.16, mean = 0.18
        assert unit == pytest.approx(0.18)

    def test_low_visibility_anchor_raises(self) -> None:
        # Drop one shoulder below the visibility threshold.
        with pytest.raises(InsufficientLandmarksError):
            compute_body_segment_unit(
                _lf(visibility_overrides={11: 0.1}),
            )

    def test_all_anchors_invisible_raises(self) -> None:
        with pytest.raises(InsufficientLandmarksError):
            compute_body_segment_unit(
                _lf(visibility_overrides={
                    11: 0.0, 12: 0.0, 23: 0.0, 24: 0.0,
                }),
            )

    def test_returns_positive_scalar(self) -> None:
        unit = compute_body_segment_unit(_lf())
        assert unit > 0.0
        assert math.isfinite(unit)
