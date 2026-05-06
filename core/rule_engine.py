"""
core.rule_engine — Deterministic form-violation detection using biomechanical
rules applied to a completed :class:`~core.rep_counter.RepEvent`.

Each rule inspects the trajectory (angle samples over time) and emits a
:class:`FormViolation` if a threshold is breached.  Rules are composable:
exercise-specific engines inherit from :class:`BaseRuleEngine` and combine
universal helpers with exercise-specific checks.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

from core.rep_counter import RepEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormViolation:
    """A single form defect detected by the rule engine.

    Attributes:
        name:        Machine-readable violation id (e.g. ``'rom_short'``).
        severity:    ``'minor'`` (5-point penalty) or ``'major'`` (15-point).
        message:     Human-readable corrective cue.
        frame_range: ``(start, end)`` indices into the trajectory, or ``None``
                     if the violation spans the entire rep.
        penalty:     Score deduction in ``[0, 100]``.
    """
    name: str
    severity: Literal['minor', 'major']
    message: str
    frame_range: tuple[int, int] | None
    penalty: float


# ---------------------------------------------------------------------------
# Abstract base engine
# ---------------------------------------------------------------------------

class BaseRuleEngine(ABC):
    """Abstract base class for per-exercise rule engines."""

    @abstractmethod
    def evaluate(self, rep: RepEvent) -> list[FormViolation]:
        """Return all violations detected in *rep*."""

    # -- Universal helpers ---------------------------------------------------

    @staticmethod
    def _check_rom(rep: RepEvent, min_range: float = 80.0) -> FormViolation | None:
        """Check range of motion (peak - trough).

        A full rep should cover at least *min_range* degrees.  Partial reps
        that barely cross the hysteresis thresholds are flagged.

        Args:
            rep:       Completed RepEvent.
            min_range: Minimum acceptable ROM in degrees.

        Returns:
            A ``'major'`` violation if ROM is insufficient, else ``None``.
        """
        rom = rep.peak_angle - rep.trough_angle
        if rom < min_range:
            return FormViolation(
                name='rom_short',
                severity='major',
                message=f"Partial rep — ROM {rom:.0f}° is below {min_range:.0f}° minimum",
                frame_range=None,
                penalty=15.0,
            )
        return None

    @staticmethod
    def _check_speed(
        rep: RepEvent,
        min_s: float = 1.0,
        max_s: float = 5.0,
    ) -> FormViolation | None:
        """Check rep duration against tempo bounds.

        Very fast reps often indicate momentum-cheating.  Very slow reps
        may indicate a stall or loss of control.

        Args:
            rep:   Completed RepEvent.
            min_s: Minimum acceptable duration in seconds.
            max_s: Maximum acceptable duration in seconds.

        Returns:
            A ``'minor'`` violation if too fast, ``'major'`` if too slow,
            else ``None``.
        """
        if rep.duration_s < min_s:
            return FormViolation(
                name='too_fast',
                severity='minor',
                message=f"Rep too fast ({rep.duration_s:.1f}s < {min_s:.1f}s) — control the negative",
                frame_range=None,
                penalty=5.0,
            )
        if rep.duration_s > max_s:
            return FormViolation(
                name='too_slow',
                severity='major',
                message=f"Rep too slow ({rep.duration_s:.1f}s > {max_s:.1f}s) — maintain tempo",
                frame_range=None,
                penalty=15.0,
            )
        return None

    @staticmethod
    def _check_tempo_stability(
        rep: RepEvent,
        max_cv: float = 0.6,
    ) -> FormViolation | None:
        """Check smoothness of the angle trajectory via coefficient of variation.

        The CV of frame-to-frame angle deltas measures how jerky the movement
        is.  A smooth rep has consistent delta magnitudes (low CV); a jerky
        rep has wildly varying deltas (high CV).

        Args:
            rep:    Completed RepEvent.
            max_cv: Maximum acceptable CV of angle deltas.

        Returns:
            A ``'minor'`` violation if CV exceeds *max_cv*, else ``None``.
        """
        if len(rep.trajectory) < 3:
            return None

        deltas = np.abs(np.diff(rep.trajectory))
        mean_delta = float(np.mean(deltas))
        if mean_delta < 1e-6:
            return None  # constant signal — no tempo to evaluate

        cv = float(np.std(deltas) / mean_delta)
        if cv > max_cv:
            return FormViolation(
                name='jerky_tempo',
                severity='minor',
                message=f"Uneven tempo (CV={cv:.2f} > {max_cv:.2f}) — smooth the motion",
                frame_range=None,
                penalty=5.0,
            )
        return None

    @staticmethod
    def _check_tempo_symmetry(
        rep: RepEvent,
        *,
        min_ratio: float = 0.7,
        max_ratio: float = 1.5,
    ) -> FormViolation | None:
        """Check that descending and ascending phases have similar duration.

        Under the uniform-frame-rate assumption (~30 fps from MediaPipe),
        the fraction of trajectory samples before the trough approximates
        the eccentric-phase time fraction.  Schoenfeld (2010) and NSCA
        guidelines recommend controlled eccentric and concentric phases
        with roughly symmetric timing.

        Args:
            rep:       Completed RepEvent.
            min_ratio: Minimum acceptable descending/ascending sample ratio.
            max_ratio: Maximum acceptable descending/ascending sample ratio.

        Returns:
            A ``'major'`` violation if the phase ratio is outside bounds,
            else ``None``.
        """
        if len(rep.trajectory) < 4:
            return None

        trough_idx = int(np.argmin(rep.trajectory))
        descending_samples = trough_idx + 1
        ascending_samples = len(rep.trajectory) - trough_idx

        if descending_samples < 2 or ascending_samples < 2:
            return None

        ratio = descending_samples / ascending_samples

        if ratio < min_ratio:
            return FormViolation(
                name='tempo_asymmetry',
                severity='major',
                message=(
                    f"Eccentric phase too fast (ratio={ratio:.2f} < {min_ratio:.2f}) "
                    f"— slow the lowering phase"
                ),
                frame_range=None,
                penalty=15.0,
            )
        if ratio > max_ratio:
            return FormViolation(
                name='tempo_asymmetry',
                severity='major',
                message=(
                    f"Concentric phase too fast (ratio={ratio:.2f} > {max_ratio:.2f}) "
                    f"— control the lifting phase"
                ),
                frame_range=None,
                penalty=15.0,
            )
        return None


# ---------------------------------------------------------------------------
# Concrete engines
# ---------------------------------------------------------------------------

class BicepRuleEngine(BaseRuleEngine):
    """Rule engine for bicep curls.

    Applies universal ROM, speed, and tempo checks.  Exercise-specific
    secondary-angle rules (elbow drift, torso swing) will be added once
    ``feature_history`` is reliably populated during live sessions.
    """

    def evaluate(self, rep: RepEvent) -> list[FormViolation]:
        violations: list[FormViolation] = []
        for check in (
            self._check_rom(rep, min_range=80.0),
            self._check_speed(rep, min_s=1.0, max_s=5.0),
            self._check_tempo_stability(rep, max_cv=0.6),
        ):
            if check is not None:
                violations.append(check)
        return violations


class BackRuleEngine(BaseRuleEngine):
    """Rule engine for back rows (bent-over rows, cable rows).

    Applies universal ROM, speed, tempo, and symmetry checks plus two
    back-specific rules that inspect the secondary angle
    (hip→shoulder→elbow) from ``feature_history``.
    """

    def evaluate(self, rep: RepEvent) -> list[FormViolation]:
        violations: list[FormViolation] = []
        for check in (
            self._check_rom(rep, min_range=80.0),
            self._check_speed(rep, min_s=1.0, max_s=5.0),
            self._check_tempo_stability(rep, max_cv=0.6),
            self._check_tempo_symmetry(rep),
            self._check_scapular_retraction(rep),
            self._check_torso_stability(rep),
        ):
            if check is not None:
                violations.append(check)
        return violations

    @staticmethod
    def _check_scapular_retraction(
        rep: RepEvent,
        max_retraction_angle: float = 50.0,
    ) -> FormViolation | None:
        """Check that scapulae retract sufficiently during the pull.

        The secondary angle (hip→shoulder→elbow) should drop below
        *max_retraction_angle* at peak contraction, indicating the elbow
        has moved behind the torso line and the scapulae are squeezed
        together.  If the minimum secondary angle stays above this
        threshold, the pull is incomplete.

        Args:
            rep:                  Completed RepEvent.
            max_retraction_angle: Maximum acceptable minimum secondary angle.

        Returns:
            A ``'major'`` violation if retraction is insufficient,
            else ``None``.
        """
        if not rep.feature_history:
            return None

        secondary_angles = np.array(
            [f.secondary_angle for f in rep.feature_history]
        )
        min_secondary = float(np.min(secondary_angles))

        if min_secondary > max_retraction_angle:
            return FormViolation(
                name='scapular_retraction',
                severity='major',
                message=(
                    f"Incomplete scapular retraction — secondary angle "
                    f"never dropped below {max_retraction_angle:.0f}° "
                    f"(min was {min_secondary:.0f}°)"
                ),
                frame_range=None,
                penalty=15.0,
            )
        return None

    @staticmethod
    def _check_torso_stability(
        rep: RepEvent,
        max_std: float = 15.0,
    ) -> FormViolation | None:
        """Check that the torso stays stable during the row.

        High standard deviation of the secondary angle (hip→shoulder→elbow)
        across the rep indicates torso kipping or momentum-cheating rather
        than isolating the back muscles.

        Args:
            rep:     Completed RepEvent.
            max_std: Maximum acceptable standard deviation (degrees).

        Returns:
            A ``'major'`` violation if torso is unstable, else ``None``.
        """
        if not rep.feature_history:
            return None

        secondary_angles = np.array(
            [f.secondary_angle for f in rep.feature_history]
        )
        std = float(np.std(secondary_angles))

        if std > max_std:
            return FormViolation(
                name='torso_unstable',
                severity='major',
                message=(
                    f"Torso kipping detected — secondary angle std "
                    f"{std:.1f}° exceeds {max_std:.0f}° limit"
                ),
                frame_range=None,
                penalty=15.0,
            )
        return None


class ChestRuleEngine(BaseRuleEngine):
    """Rule engine for push-ups.

    Applies universal ROM (60° minimum), speed, tempo, and symmetry
    checks plus two push-up-specific rules that inspect the secondary
    angle (shoulder→hip→knee for plank integrity) and tertiary angle
    (hip→knee→ankle for leg alignment) from ``feature_history``.

    Note: the plank-integrity and leg-alignment checks are push-up
    biomechanics; they do **not** make sense for bench press (where the
    body is on the bench rather than in plank, and knees are typically
    bent). The rule-engine factory keys this class on ``'push_up'``.
    """

    def evaluate(self, rep: RepEvent) -> list[FormViolation]:
        violations: list[FormViolation] = []
        for check in (
            self._check_rom(rep, min_range=60.0),
            self._check_speed(rep, min_s=0.8, max_s=4.0),
            self._check_tempo_stability(rep, max_cv=0.6),
            self._check_tempo_symmetry(rep),
            self._check_plank_integrity(rep),
            self._check_leg_alignment(rep),
        ):
            if check is not None:
                violations.append(check)
        return violations

    @staticmethod
    def _check_plank_integrity(
        rep: RepEvent,
        min_mean_angle: float = 165.0,
    ) -> FormViolation | None:
        """Check that the body maintains a straight plank throughout the rep.

        The secondary angle (shoulder→hip→knee) should stay near 180° for
        a proper plank.  Any deviation — either hip sag (angle decreases)
        or hip pike (angle also decreases from the straight-line interior
        angle) — reduces this angle below *min_mean_angle*.

        Args:
            rep:            Completed RepEvent.
            min_mean_angle: Minimum acceptable mean secondary angle.

        Returns:
            A ``'major'`` violation if plank breaks, else ``None``.
        """
        if not rep.feature_history:
            return None

        secondary_angles = np.array(
            [f.secondary_angle for f in rep.feature_history]
        )
        mean_angle = float(np.mean(secondary_angles))

        if mean_angle < min_mean_angle:
            return FormViolation(
                name='plank_broken',
                severity='major',
                message=(
                    f"Plank integrity lost — mean body angle "
                    f"{mean_angle:.0f}° is below {min_mean_angle:.0f}° "
                    f"(hip sag or pike detected)"
                ),
                frame_range=None,
                penalty=15.0,
            )
        return None

    @staticmethod
    def _check_leg_alignment(
        rep: RepEvent,
        min_mean_angle: float = 170.0,
    ) -> FormViolation | None:
        """Check that legs stay straight during push-ups.

        The tertiary angle (hip→knee→ankle) should stay near 180° when
        the legs are locked.  A mean below *min_mean_angle* indicates
        bent knees, which reduces push-up effectiveness and shifts load
        off the chest.

        Args:
            rep:            Completed RepEvent.
            min_mean_angle: Minimum acceptable mean tertiary angle.

        Returns:
            A ``'major'`` violation if knees are bent, else ``None``.
        """
        if not rep.feature_history:
            return None

        tertiary_angles = np.array(
            [f.tertiary_angle for f in rep.feature_history]
        )
        mean_angle = float(np.mean(tertiary_angles))

        if mean_angle < min_mean_angle:
            return FormViolation(
                name='leg_bent',
                severity='major',
                message=(
                    f"Knees bent — mean leg angle {mean_angle:.0f}° is "
                    f"below {min_mean_angle:.0f}° (lock knees straight)"
                ),
                frame_range=None,
                penalty=15.0,
            )
        return None


class NullRuleEngine(BaseRuleEngine):
    """No-op rule engine for exercises without biomechanics rules.

    Always returns an empty violation list.  Used as fallback for
    unknown exercise types.
    """

    def evaluate(self, rep: RepEvent) -> list[FormViolation]:
        return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ENGINE_CLASSES: dict[str, type[BaseRuleEngine]] = {
    'bicep_curl': BicepRuleEngine,
    'bent_over_row': BackRuleEngine,
    'push_up': ChestRuleEngine,
}


def make_rule_engine(exercise: str) -> BaseRuleEngine:
    """Create the appropriate rule engine for *exercise*.

    Args:
        exercise: Exercise key (``'bicep_curl'``, ``'bent_over_row'``,
                  ``'push_up'``).

    Returns:
        The matching rule engine, or :class:`NullRuleEngine` for
        unrecognised exercises.
    """
    cls = _ENGINE_CLASSES.get(exercise)
    if cls is not None:
        return cls()

    logger.info(
        "No rule engine for %r — using NullRuleEngine",
        exercise,
    )
    return NullRuleEngine()
