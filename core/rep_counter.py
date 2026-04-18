"""
core.rep_counter — Production-grade repetition counter driven by a 5-state
finite state machine with hysteresis and duration debouncing.

Geometric rationale
-------------------
Every supported exercise has a *primary joint angle* that monotonically
decreases during the eccentric (lowering) phase and increases during the
concentric (lifting) phase of a single repetition.  The FSM tracks the
direction of that angle through five states::

        IDLE ──► DESCENDING ──► BOTTOM ──► ASCENDING ──► TOP ─┐
                    ▲                                          │
                    └──────────────────────────────────────────┘

Hysteresis bands around the top and bottom thresholds prevent jitter near
boundary angles from causing spurious state transitions.  Duration gates
reject oscillations faster than ``min_rep_duration_s`` (e.g. sensor noise)
and stalls longer than ``max_rep_duration_s`` (e.g. the user paused).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from config import RepCounterConfig, REP_COUNTER_CONFIGS
from utils.angle_utils import calculate_angle_3d

if TYPE_CHECKING:
    pass  # reserved for future type-only imports

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

class RepState(Enum):
    """Five states of the rep-counter FSM."""
    IDLE = auto()
    DESCENDING = auto()
    BOTTOM = auto()
    ASCENDING = auto()
    TOP = auto()


@dataclass(frozen=True)
class FrameFeatures:
    """Per-frame feature vector matching the trained model schema.

    The three columns correspond to ``feature_cols`` stored in every
    trained ``.pkl`` model: ``['primary_angle', 'secondary_angle',
    'tertiary_angle']``.
    """
    primary_angle: float
    secondary_angle: float
    tertiary_angle: float


@dataclass(frozen=True)
class RepEvent:
    """Immutable record emitted when a single repetition completes.

    Attributes:
        rep_number:    Monotonically increasing rep index (1-based).
        start_ts:      Monotonic timestamp at rep start (entering DESCENDING).
        end_ts:        Monotonic timestamp at rep end (entering TOP).
        duration_s:    ``end_ts - start_ts``.
        peak_angle:    Maximum primary angle sampled during the rep (top).
        trough_angle:  Minimum primary angle sampled during the rep (bottom).
        trajectory:    Every primary-angle sample captured between start and end.
        exercise_name: E.g. ``"bicep"``, ``"back"``, ``"chest"``.
        age_group:     E.g. ``"adult"``, ``"children"``, ``"senior"``.
    """
    rep_number: int
    start_ts: float
    end_ts: float
    duration_s: float
    peak_angle: float
    trough_angle: float
    trajectory: list[float]
    exercise_name: str
    age_group: str
    feature_history: tuple[FrameFeatures, ...] = ()


# ---------------------------------------------------------------------------
# Abstract base counter
# ---------------------------------------------------------------------------

class BaseRepCounter(ABC):
    """Abstract base class for exercise-specific rep counters.

    Subclasses implement :meth:`extract_primary_angle` to pull the relevant
    joint angle from a MediaPipe landmark list.  All FSM logic, hysteresis,
    and duration gating live in this base class.
    """

    def __init__(
        self,
        config: RepCounterConfig,
        exercise_name: str,
        age_group: str,
    ) -> None:
        self._config = config
        self._exercise_name = exercise_name
        self._age_group = age_group

        # FSM bookkeeping
        self._state: RepState = RepState.IDLE
        self._rep_count: int = 0

        # Per-rep accumulators (reset each rep)
        self._rep_start_ts: float = 0.0
        self._trajectory: list[float] = []
        self._peak_angle: float = 0.0
        self._trough_angle: float = 360.0
        self._feature_buffer: list[FrameFeatures] = []

    # -- Public properties ---------------------------------------------------

    @property
    def rep_count(self) -> int:
        """Total number of completed (and accepted) repetitions."""
        return self._rep_count

    @property
    def state(self) -> RepState:
        """Current FSM state."""
        return self._state

    # -- Abstract method -----------------------------------------------------

    @abstractmethod
    def extract_primary_angle(self, pose_landmarks: list) -> float:
        """Return the primary joint angle (degrees) from *pose_landmarks*.

        Args:
            pose_landmarks: MediaPipe-style landmark list indexed by landmark
                id.  Each element must expose ``.x``, ``.y``, ``.z`` floats.

        Returns:
            Angle in degrees at the primary joint.
        """

    # -- Concrete API --------------------------------------------------------

    def update(
        self,
        pose_landmarks: list,
        timestamp: float,
        *,
        features: FrameFeatures | None = None,
    ) -> RepEvent | None:
        """Feed one frame into the FSM.

        Args:
            pose_landmarks: Full MediaPipe landmark list for this frame.
            timestamp:      Monotonic time of this frame (seconds).
            features:       Optional per-frame feature vector.

        Returns:
            A :class:`RepEvent` on the exact frame that closes a rep, or
            ``None`` otherwise.
        """
        angle = self.extract_primary_angle(pose_landmarks)
        return self._advance_fsm(angle, timestamp, features=features)

    def update_angle(
        self,
        angle: float,
        timestamp: float,
        *,
        features: FrameFeatures | None = None,
    ) -> RepEvent | None:
        """Feed a pre-computed angle into the FSM (useful for testing).

        Identical to :meth:`update` but skips landmark extraction.

        Args:
            angle:     Primary joint angle in degrees.
            timestamp: Monotonic time of this frame (seconds).
            features:  Optional per-frame feature vector.  When provided,
                       accumulated into the rep's ``feature_history``.
        """
        return self._advance_fsm(angle, timestamp, features=features)

    def reset(self) -> None:
        """Zero the rep count and restore the FSM to IDLE."""
        self._state = RepState.IDLE
        self._rep_count = 0
        self._rep_start_ts = 0.0
        self._trajectory = []
        self._peak_angle = 0.0
        self._trough_angle = 360.0
        self._feature_buffer = []

    # -- FSM internals -------------------------------------------------------

    def _advance_fsm(
        self,
        angle: float,
        timestamp: float,
        *,
        features: FrameFeatures | None = None,
    ) -> RepEvent | None:
        """Run one FSM tick and return a RepEvent if a rep just closed."""
        cfg = self._config
        top_enter = cfg.top_threshold - cfg.hysteresis
        bottom_enter = cfg.bottom_threshold + cfg.hysteresis

        state = self._state

        # -- IDLE: waiting for the first downward movement -------------------
        if state is RepState.IDLE:
            if angle < top_enter:
                self._begin_rep(angle, timestamp, features)
                self._state = RepState.DESCENDING
                logger.debug("IDLE -> DESCENDING @ %.1f deg", angle)

        # -- DESCENDING: angle is falling toward the bottom ------------------
        elif state is RepState.DESCENDING:
            self._record_sample(angle, features)
            if angle <= bottom_enter:
                self._state = RepState.BOTTOM
                logger.debug("DESCENDING -> BOTTOM @ %.1f deg", angle)

        # -- BOTTOM: angle is at or below the bottom threshold ---------------
        elif state is RepState.BOTTOM:
            self._record_sample(angle, features)
            if angle > bottom_enter:
                self._state = RepState.ASCENDING
                logger.debug("BOTTOM -> ASCENDING @ %.1f deg", angle)

        # -- ASCENDING: angle is rising toward the top -----------------------
        elif state is RepState.ASCENDING:
            self._record_sample(angle, features)
            if angle >= top_enter:
                event = self._try_close_rep(timestamp)
                self._state = RepState.TOP
                logger.debug("ASCENDING -> TOP @ %.1f deg", angle)
                return event  # RepEvent or None if duration-rejected

        # -- TOP: waiting for the next descent or staying at top -------------
        elif state is RepState.TOP:
            if angle < top_enter:
                self._begin_rep(angle, timestamp, features)
                self._state = RepState.DESCENDING
                logger.debug("TOP -> DESCENDING @ %.1f deg", angle)

        return None

    def _begin_rep(
        self,
        angle: float,
        timestamp: float,
        features: FrameFeatures | None = None,
    ) -> None:
        """Initialise accumulators for a new rep."""
        self._rep_start_ts = timestamp
        self._trajectory = [angle]
        self._peak_angle = angle
        self._trough_angle = angle
        self._feature_buffer = [features] if features is not None else []

    def _record_sample(
        self,
        angle: float,
        features: FrameFeatures | None = None,
    ) -> None:
        """Append an angle sample and update peak/trough."""
        self._trajectory.append(angle)
        if angle > self._peak_angle:
            self._peak_angle = angle
        if angle < self._trough_angle:
            self._trough_angle = angle
        if features is not None:
            self._feature_buffer.append(features)

    def _try_close_rep(self, end_ts: float) -> RepEvent | None:
        """Validate duration and emit a RepEvent, or discard the rep."""
        duration = end_ts - self._rep_start_ts

        if duration < self._config.min_rep_duration_s:
            logger.debug("Rep rejected: too fast (%.2fs < %.2fs)",
                         duration, self._config.min_rep_duration_s)
            return None

        if duration > self._config.max_rep_duration_s:
            logger.debug("Rep rejected: too slow (%.2fs > %.2fs)",
                         duration, self._config.max_rep_duration_s)
            return None

        self._rep_count += 1

        event = RepEvent(
            rep_number=self._rep_count,
            start_ts=self._rep_start_ts,
            end_ts=end_ts,
            duration_s=duration,
            peak_angle=self._peak_angle,
            trough_angle=self._trough_angle,
            trajectory=list(self._trajectory),
            exercise_name=self._exercise_name,
            age_group=self._age_group,
            feature_history=tuple(self._feature_buffer),
        )
        logger.info("Rep %d closed: %.2fs, peak=%.1f, trough=%.1f",
                     event.rep_number, event.duration_s,
                     event.peak_angle, event.trough_angle)
        return event


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------

class BicepRepCounter(BaseRepCounter):
    """Rep counter for bicep curls.

    Primary angle: shoulder(12) -> elbow(14) -> wrist(16).
    """

    def __init__(self, age_group: str = 'adult') -> None:
        super().__init__(
            config=REP_COUNTER_CONFIGS['bicep'],
            exercise_name='bicep',
            age_group=age_group,
        )

    def extract_primary_angle(self, pose_landmarks: list) -> float:
        shoulder = np.array([pose_landmarks[12].x, pose_landmarks[12].y, pose_landmarks[12].z])
        elbow = np.array([pose_landmarks[14].x, pose_landmarks[14].y, pose_landmarks[14].z])
        wrist = np.array([pose_landmarks[16].x, pose_landmarks[16].y, pose_landmarks[16].z])
        return calculate_angle_3d(shoulder, elbow, wrist)


class BackRepCounter(BaseRepCounter):
    """Rep counter for back rows.

    Primary angle: shoulder(12) -> elbow(14) -> wrist(16).
    """

    def __init__(self, age_group: str = 'adult') -> None:
        super().__init__(
            config=REP_COUNTER_CONFIGS['back'],
            exercise_name='back',
            age_group=age_group,
        )

    def extract_primary_angle(self, pose_landmarks: list) -> float:
        shoulder = np.array([pose_landmarks[12].x, pose_landmarks[12].y, pose_landmarks[12].z])
        elbow = np.array([pose_landmarks[14].x, pose_landmarks[14].y, pose_landmarks[14].z])
        wrist = np.array([pose_landmarks[16].x, pose_landmarks[16].y, pose_landmarks[16].z])
        return calculate_angle_3d(shoulder, elbow, wrist)


class ChestRepCounter(BaseRepCounter):
    """Rep counter for push-ups.

    Primary angle: shoulder(12) -> elbow(14) -> wrist(16).
    """

    def __init__(self, age_group: str = 'adult') -> None:
        super().__init__(
            config=REP_COUNTER_CONFIGS['chest'],
            exercise_name='chest',
            age_group=age_group,
        )

    def extract_primary_angle(self, pose_landmarks: list) -> float:
        shoulder = np.array([pose_landmarks[12].x, pose_landmarks[12].y, pose_landmarks[12].z])
        elbow = np.array([pose_landmarks[14].x, pose_landmarks[14].y, pose_landmarks[14].z])
        wrist = np.array([pose_landmarks[16].x, pose_landmarks[16].y, pose_landmarks[16].z])
        return calculate_angle_3d(shoulder, elbow, wrist)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_COUNTER_CLASSES: dict[str, type[BaseRepCounter]] = {
    'bicep': BicepRepCounter,
    'back': BackRepCounter,
    'chest': ChestRepCounter,
}


def make_rep_counter(exercise: str, age_group: str = 'adult') -> BaseRepCounter:
    """Create the correct rep counter for *exercise*.

    Args:
        exercise:  One of ``'bicep'``, ``'back'``, ``'chest'``.
        age_group: One of ``'children'``, ``'adult'``, ``'senior'``.

    Returns:
        A fully initialised :class:`BaseRepCounter` subclass instance.

    Raises:
        ValueError: If *exercise* is not recognised.
    """
    cls = _COUNTER_CLASSES.get(exercise)
    if cls is None:
        raise ValueError(
            f"Unknown exercise {exercise!r}. "
            f"Supported: {sorted(_COUNTER_CLASSES)}"
        )
    return cls(age_group=age_group)
