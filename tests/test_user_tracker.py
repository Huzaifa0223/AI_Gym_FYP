"""
tests.test_user_tracker — primary-user lock semantics: acquisition,
hysteresis, lock release, and reset behaviour.

LandmarkFrames are fabricated so their visible-landmark bbox spans a known
normalised range. The default frame size is 1920×1080 (area 2,073,600 px),
so with the default ``min_bbox_area_ratio = 0.05`` the area gate is
103,680 px.

  LARGE_PRIMARY     → 0.4 × 0.9   = 0.36 frame   = 746,496 px (above gate)
  LARGER_CHALLENGER → 0.9 × 0.96  = 0.864 frame  = 1,791,590 px (above gate, larger)
  TINY              → 0.05 × 0.01 = 0.0005 frame = 1,037 px (below gate)
"""
from __future__ import annotations

import pytest

from core.config import PrimaryUserConfig
from core.schemas import LandmarkFrame
from core.user_tracker import PrimaryUserTracker


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

LARGE_PRIMARY = ((0.30, 0.05), (0.70, 0.95))
LARGER_CHALLENGER = ((0.05, 0.02), (0.95, 0.98))
TINY = ((0.45, 0.45), (0.50, 0.46))


def _make_lf(
    *,
    bbox_norm: tuple[tuple[float, float], tuple[float, float]],
    visibility: float = 0.95,
    frame_id: int = 0,
) -> LandmarkFrame:
    """Build a 33-landmark LandmarkFrame whose visible bbox spans *bbox_norm*.

    Two corner landmarks pin the bbox to the requested rectangle; the
    remaining 31 sit at the centre so visible-set min/max are exactly the
    corners regardless of the visibility threshold.
    """
    (x1, y1), (x2, y2) = bbox_norm
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    landmarks: list[tuple[float, float, float]] = []
    for i in range(33):
        if i == 0:
            landmarks.append((x1, y1, 0.0))
        elif i == 1:
            landmarks.append((x2, y2, 0.0))
        else:
            landmarks.append((cx, cy, 0.0))
    return LandmarkFrame(
        frame_id=frame_id,
        timestamp_ms=frame_id * 33,
        landmarks=tuple(landmarks),
        visibility=tuple(visibility for _ in range(33)),
        user_height_cm=170.0,
    )


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

class TestAcquisition:
    def test_single_qualifying_candidate_acquires(self) -> None:
        tracker = PrimaryUserTracker()
        lf = _make_lf(bbox_norm=LARGE_PRIMARY)
        result = tracker.track([lf])
        assert result is not None
        assert result.track_id == 1
        assert result.frames_since_acquired == 0
        assert result.bbox_area_px > 100_000

    def test_two_candidates_largest_acquires(self) -> None:
        tracker = PrimaryUserTracker()
        smaller = _make_lf(bbox_norm=LARGE_PRIMARY)
        larger = _make_lf(bbox_norm=LARGER_CHALLENGER)
        result = tracker.track([smaller, larger])
        assert result is not None
        assert result.track_id == 1
        assert result.bbox_area_px > 1_500_000

    def test_below_threshold_returns_none(self) -> None:
        tracker = PrimaryUserTracker()
        tiny = _make_lf(bbox_norm=TINY)
        assert tracker.track([tiny]) is None

    def test_empty_candidates_returns_none(self) -> None:
        tracker = PrimaryUserTracker()
        assert tracker.track([]) is None

    def test_low_visibility_landmarks_excluded_from_bbox(self) -> None:
        """Visibility < 0.5 candidates do not contribute to the bbox.

        A frame whose only visible (≥ 0.5) landmarks fall inside the TINY
        rectangle should fail the area gate even when invisible landmarks
        sit at the frame corners.
        """
        # Construct a frame: 2 corner landmarks at the frame edges with
        # near-zero visibility, 31 centre landmarks at high visibility.
        landmarks = [
            (0.0, 0.0, 0.0),  # would inflate bbox if counted
            (1.0, 1.0, 0.0),  # ditto
        ] + [(0.475, 0.455, 0.0)] * 31  # all visible, single point
        visibility = (0.1, 0.1) + tuple(0.95 for _ in range(31))
        lf = LandmarkFrame(
            frame_id=0,
            timestamp_ms=0,
            landmarks=tuple(landmarks),
            visibility=visibility,
            user_height_cm=170.0,
        )
        tracker = PrimaryUserTracker()
        # Visible landmarks form a zero-area point → fails area gate.
        assert tracker.track([lf]) is None


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------

class TestHysteresis:
    def test_hold_against_short_challenger(self) -> None:
        """Three frames of challenger pressure must not transfer the lock."""
        tracker = PrimaryUserTracker()
        primary = _make_lf(bbox_norm=LARGE_PRIMARY)

        first = tracker.track([primary])
        assert first is not None
        assert first.track_id == 1

        challenger = _make_lf(bbox_norm=LARGER_CHALLENGER)
        for i in range(3):
            r = tracker.track([primary, challenger])
            assert r is not None
            assert r.track_id == 1, f"track_id changed prematurely at frame {i + 2}"

        # Challenger disappears — primary becomes largest again.
        r = tracker.track([primary])
        assert r is not None
        assert r.track_id == 1

    def test_transfer_after_full_streak(self) -> None:
        """Challenger sustained for ≥ bbox_hysteresis_frames triggers transfer."""
        tracker = PrimaryUserTracker()  # default hysteresis = 5
        primary = _make_lf(bbox_norm=LARGE_PRIMARY)
        challenger = _make_lf(bbox_norm=LARGER_CHALLENGER)

        r = tracker.track([primary])
        assert r is not None
        assert r.track_id == 1

        # 4 challenger frames → streaks 1..4, no transfer.
        for i in range(4):
            r = tracker.track([primary, challenger])
            assert r is not None
            assert r.track_id == 1, f"transfer too early at challenger frame {i + 1}"

        # 5th challenger frame → streak hits hysteresis, lock transfers.
        r = tracker.track([primary, challenger])
        assert r is not None
        assert r.track_id == 2
        assert r.frames_since_acquired == 0

    def test_challenger_break_resets_streak(self) -> None:
        """A challenger that disappears mid-streak does not accumulate.

        After 3 challenger frames, 1 frame with the challenger absent, then
        3 more challenger frames, the streak should not have crossed
        ``bbox_hysteresis_frames=5``.
        """
        tracker = PrimaryUserTracker()
        primary = _make_lf(bbox_norm=LARGE_PRIMARY)
        challenger = _make_lf(bbox_norm=LARGER_CHALLENGER)

        tracker.track([primary])  # acquire

        for _ in range(3):
            tracker.track([primary, challenger])

        # Challenger disappears for one frame — streak should reset.
        r = tracker.track([primary])
        assert r is not None
        assert r.track_id == 1

        # 3 more frames of challenger — streak rebuilds from 1, max 3.
        for _ in range(3):
            r = tracker.track([primary, challenger])
            assert r is not None
            assert r.track_id == 1, "transfer happened despite streak break"


# ---------------------------------------------------------------------------
# Lock release
# ---------------------------------------------------------------------------

class TestLockRelease:
    def test_release_after_disappearance(self) -> None:
        """Primary missing for ≥ 30 frames releases the lock."""
        tracker = PrimaryUserTracker()  # default lock_release_after_frames = 30
        primary = _make_lf(bbox_norm=LARGE_PRIMARY)

        r = tracker.track([primary])
        assert r is not None
        assert r.track_id == 1

        # 30 frames of disappearance — lock releases on frame 30.
        for _ in range(30):
            assert tracker.track([]) is None

        # Frame 31: fresh acquisition with new track_id.
        r = tracker.track([primary])
        assert r is not None
        assert r.track_id == 2
        assert r.frames_since_acquired == 0

    def test_grace_period_recovers_lock(self) -> None:
        """Primary returning within the grace period preserves track_id."""
        tracker = PrimaryUserTracker()
        primary = _make_lf(bbox_norm=LARGE_PRIMARY)

        r = tracker.track([primary])
        assert r is not None
        assert r.track_id == 1

        # 28 missing frames (grace ends at 30).
        for _ in range(28):
            assert tracker.track([]) is None

        r = tracker.track([primary])
        assert r is not None
        assert r.track_id == 1


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state_and_track_id(self) -> None:
        """After reset, the next acquisition restarts at track_id == 1."""
        tracker = PrimaryUserTracker()
        primary = _make_lf(bbox_norm=LARGE_PRIMARY)

        r = tracker.track([primary])
        assert r is not None
        assert r.track_id == 1

        # Force a transfer to take track_id past 1.
        challenger = _make_lf(bbox_norm=LARGER_CHALLENGER)
        for _ in range(5):
            tracker.track([primary, challenger])
        # After hysteresis, track_id has advanced past 1.

        tracker.reset()

        r = tracker.track([primary])
        assert r is not None
        assert r.track_id == 1
        assert r.frames_since_acquired == 0


# ---------------------------------------------------------------------------
# frames_since_acquired + identity matching
# ---------------------------------------------------------------------------

class TestFramesSinceAcquired:
    def test_increments_each_observed_frame(self) -> None:
        tracker = PrimaryUserTracker()
        primary = _make_lf(bbox_norm=LARGE_PRIMARY)
        for i in range(5):
            r = tracker.track([primary])
            assert r is not None
            assert r.frames_since_acquired == i


class TestIdentityMatching:
    def test_small_movement_keeps_lock(self) -> None:
        """Primary shifts slightly but retains identity via IoU ≥ 0.3."""
        tracker = PrimaryUserTracker()
        first_lf = _make_lf(bbox_norm=((0.30, 0.05), (0.70, 0.95)))
        r1 = tracker.track([first_lf])
        assert r1 is not None
        assert r1.track_id == 1

        # Small shift — IoU stays high.
        second_lf = _make_lf(bbox_norm=((0.32, 0.07), (0.72, 0.97)))
        r2 = tracker.track([second_lf])
        assert r2 is not None
        assert r2.track_id == r1.track_id
        assert r2.frames_since_acquired == 1
