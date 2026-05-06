"""
core.user_tracker — Primary-user selection with bbox-area thresholding,
IoU-based identity matching, and hysteresis-driven lock transfer.

The tracker consumes a list of :class:`LandmarkFrame` candidates per call
(currently length-1 from MediaPipe Pose; multi-pose feeders such as YOLO
person-crops or MediaPipe Tasks ``PoseLandmarker`` can supply longer lists
without code changes here) and returns a :class:`PrimaryUserBox` describing
the locked primary user, or ``None`` when no primary can be selected.

State machine
-------------
The tracker maintains four pieces of internal state:

* ``_current_primary``  — the active lock as a :class:`PrimaryUserBox`,
                          or ``None`` when released.
* ``_next_track_id``    — monotonic counter, starts at 1, increments on
                          every fresh acquisition, never reused.
                          :meth:`reset` returns it to 1.
* ``_challenger_streak`` / ``_challenger_bbox`` — challenger identity
  tracking. A *challenger* is the largest above-threshold candidate that
  is **not** the current primary. The streak increments only while the
  challenger's identity persists (IoU ≥ 0.3 between consecutive
  challenger bboxes); a break in identity resets it to 1, and the
  current primary regaining largest area resets it to 0.
* ``_frames_since_seen`` — counter that increments each frame the
  current primary fails to IoU-match any candidate. Triggers lock
  release when it reaches
  :attr:`~core.config.PrimaryUserConfig.lock_release_after_frames`.

Acquisition rules
-----------------
* No active lock + qualifying candidate(s) → largest is acquired,
  ``frames_since_acquired = 0``, ``track_id`` advances.
* Active lock retained when matched primary remains the largest.
* Active lock retained but challenger streak grows when matched
  primary is not the largest.
* Lock transfers when ``challenger_streak >=
  PrimaryUserConfig.bbox_hysteresis_frames``: the challenger acquires a
  fresh ``track_id``.
* Lock releases after
  ``PrimaryUserConfig.lock_release_after_frames`` consecutive
  un-matched frames; subsequent acquisitions begin from scratch.

Returns ``None`` whenever the current primary is not observed this
frame (grace period) and whenever no qualifying candidate is available
at all. The lock state persists internally during the grace period so
the same primary can be re-acquired without advancing the
``track_id``.
"""
from __future__ import annotations

import logging
from typing import Final

from core.config import PrimaryUserConfig
from core.schemas import LandmarkFrame, PrimaryUserBox

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hard-coded thresholds — not exposed via PrimaryUserConfig
# ---------------------------------------------------------------------------

_IOU_MATCH_THRESHOLD: Final[float] = 0.3
"""Minimum bbox IoU between consecutive frames to count as the same identity.

Used both for primary tracking (does this frame's candidate match the locked
primary?) and challenger identity (is the largest non-primary candidate the
same person we saw last frame?). Centroid-distance fallbacks are intentionally
omitted: IoU is more robust on the size scales we care about."""

_MIN_LANDMARK_VISIBILITY: Final[float] = 0.5
"""Landmarks below this MediaPipe visibility score are excluded from bbox
derivation so a single low-confidence point cannot inflate the bbox."""


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

_BBox = tuple[float, float, float, float]
_Qualified = tuple[_BBox, float]  # (bbox_xyxy_pixels, area_px)


class PrimaryUserTracker:
    """Single-session, single-instance primary-user selector.

    Args:
        cfg:      Hysteresis / area-gate parameters. Defaults to a fresh
                  :class:`PrimaryUserConfig`.
        frame_w:  Frame width in pixels. Used by the area-ratio gate.
        frame_h:  Frame height in pixels. Used by the area-ratio gate.
    """

    def __init__(
        self,
        cfg: PrimaryUserConfig = PrimaryUserConfig(),
        frame_w: int = 1920,
        frame_h: int = 1080,
    ) -> None:
        self._cfg = cfg
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._frame_area = float(frame_w * frame_h)

        # Lock state
        self._current_primary: PrimaryUserBox | None = None
        self._next_track_id: int = 1

        # Hysteresis state
        self._challenger_streak: int = 0
        self._challenger_bbox: _BBox | None = None

        # Release state
        self._frames_since_seen: int = 0

    # -- Public API ----------------------------------------------------------

    def track(self, candidates: list[LandmarkFrame]) -> PrimaryUserBox | None:
        """Process one frame's candidates and return the primary lock state.

        Returns ``None`` when:

        * No candidate clears the area-ratio gate **and** no active lock
          exists; or
        * The active lock primary is not IoU-matched in this frame's
          candidates (grace period). The lock is retained internally
          until ``lock_release_after_frames`` is reached.

        Args:
            candidates: Per-frame candidate poses. Order is irrelevant
                — selection is by bbox area + IoU identity.

        Returns:
            A :class:`PrimaryUserBox` describing the locked primary, or
            ``None`` when no primary is observable this frame.
        """
        qualified = self._qualify_candidates(candidates)
        largest = max(qualified, key=lambda q: q[1]) if qualified else None

        if self._current_primary is None:
            return self._handle_no_active_lock(largest)

        matched = self._match_by_iou(self._current_primary.bbox_xyxy, qualified)

        if matched is None:
            return self._handle_primary_not_observed()

        return self._handle_primary_observed(matched, largest)

    def reset(self) -> None:
        """Clear all state.

        Subsequent acquisitions restart at ``track_id == 1`` — the monotonic
        counter is reset to 1 so test-harness sequences are predictable.
        """
        self._current_primary = None
        self._next_track_id = 1
        self._challenger_streak = 0
        self._challenger_bbox = None
        self._frames_since_seen = 0

    # -- Frame qualification -------------------------------------------------

    def _qualify_candidates(
        self, candidates: list[LandmarkFrame]
    ) -> list[_Qualified]:
        """Compute bbox + area for each candidate; filter by area threshold."""
        out: list[_Qualified] = []
        min_area = self._cfg.min_bbox_area_ratio * self._frame_area
        for lf in candidates:
            bb = self._bbox_from_landmarks(lf)
            if bb is None:
                continue
            bbox, area = bb
            if area >= min_area:
                out.append((bbox, area))
        return out

    def _bbox_from_landmarks(
        self, lf: LandmarkFrame
    ) -> tuple[_BBox, float] | None:
        """Derive a pixel-coord bbox from landmarks with visibility ≥ 0.5.

        Multiplies normalised landmark coordinates by the configured frame
        dimensions to produce pixel-space (x1, y1, x2, y2) and area.
        Returns ``None`` when no landmark clears the visibility threshold.
        """
        xs: list[float] = []
        ys: list[float] = []
        for (x, y, _z), vis in zip(lf.landmarks, lf.visibility):
            if vis >= _MIN_LANDMARK_VISIBILITY:
                xs.append(x * self._frame_w)
                ys.append(y * self._frame_h)
        if not xs:
            return None
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return (x1, y1, x2, y2), (x2 - x1) * (y2 - y1)

    # -- State transitions ---------------------------------------------------

    def _handle_no_active_lock(
        self, largest: _Qualified | None
    ) -> PrimaryUserBox | None:
        if largest is None:
            return None
        return self._acquire_new_lock(largest)

    def _handle_primary_not_observed(self) -> None:
        self._frames_since_seen += 1
        if self._frames_since_seen >= self._cfg.lock_release_after_frames:
            self._release_lock()
        return None

    def _handle_primary_observed(
        self,
        matched: _Qualified,
        largest: _Qualified | None,
    ) -> PrimaryUserBox:
        assert self._current_primary is not None
        self._frames_since_seen = 0
        matched_bbox, matched_area = matched

        if largest is matched:
            # Primary is itself the largest — challenger pressure clears.
            self._challenger_streak = 0
            self._challenger_bbox = None
        else:
            # Some other candidate is larger; track the challenger.
            assert largest is not None
            challenger_bbox = largest[0]
            same_challenger = (
                self._challenger_bbox is not None
                and self._iou(challenger_bbox, self._challenger_bbox)
                >= _IOU_MATCH_THRESHOLD
            )
            if same_challenger:
                self._challenger_streak += 1
            else:
                self._challenger_streak = 1
            self._challenger_bbox = challenger_bbox

            if self._challenger_streak >= self._cfg.bbox_hysteresis_frames:
                # Transfer the lock to the challenger.
                return self._acquire_new_lock(largest)

        # Lock retained — refresh bbox and increment frames_since_acquired.
        new = PrimaryUserBox(
            track_id=self._current_primary.track_id,
            bbox_xyxy=matched_bbox,
            bbox_area_px=matched_area,
            frames_since_acquired=self._current_primary.frames_since_acquired + 1,
        )
        self._current_primary = new
        return new

    # -- Lock lifecycle ------------------------------------------------------

    def _acquire_new_lock(self, qualified: _Qualified) -> PrimaryUserBox:
        bbox, area = qualified
        track_id = self._next_track_id
        self._next_track_id += 1

        new = PrimaryUserBox(
            track_id=track_id,
            bbox_xyxy=bbox,
            bbox_area_px=area,
            frames_since_acquired=0,
        )
        self._current_primary = new
        self._challenger_streak = 0
        self._challenger_bbox = None
        self._frames_since_seen = 0
        logger.debug("Lock acquired: track_id=%d, area=%.0f px", track_id, area)
        return new

    def _release_lock(self) -> None:
        logger.debug(
            "Lock released after %d unobserved frames",
            self._frames_since_seen,
        )
        self._current_primary = None
        self._challenger_streak = 0
        self._challenger_bbox = None
        self._frames_since_seen = 0

    # -- Identity matching ---------------------------------------------------

    def _match_by_iou(
        self,
        prev_bbox: _BBox,
        qualified: list[_Qualified],
    ) -> _Qualified | None:
        """Return the candidate with highest IoU vs *prev_bbox*, or ``None``.

        ``None`` is returned when no candidate's IoU clears
        :data:`_IOU_MATCH_THRESHOLD`, signalling the primary is not observed.
        """
        best: _Qualified | None = None
        best_iou = 0.0
        for q in qualified:
            iou = self._iou(prev_bbox, q[0])
            if iou > best_iou:
                best_iou = iou
                best = q
        if best_iou >= _IOU_MATCH_THRESHOLD:
            return best
        return None

    @staticmethod
    def _iou(a: _BBox, b: _BBox) -> float:
        """Standard axis-aligned intersection-over-union for xyxy bboxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        a_area = (ax2 - ax1) * (ay2 - ay1)
        b_area = (bx2 - bx1) * (by2 - by1)
        union = a_area + b_area - inter
        if union <= 0.0:
            return 0.0
        return inter / union


__all__ = ["PrimaryUserTracker"]
