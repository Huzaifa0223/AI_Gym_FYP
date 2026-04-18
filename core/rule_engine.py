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


class NullRuleEngine(BaseRuleEngine):
    """No-op rule engine for exercises without biomechanics rules yet.

    Always returns an empty violation list.  Used for ``back`` and ``chest``
    until exercise-specific rules are implemented.
    """

    def evaluate(self, rep: RepEvent) -> list[FormViolation]:
        return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ENGINE_CLASSES: dict[str, type[BaseRuleEngine]] = {
    'bicep': BicepRuleEngine,
}


def make_rule_engine(exercise: str) -> BaseRuleEngine:
    """Create the appropriate rule engine for *exercise*.

    Args:
        exercise: Exercise key (``'bicep'``, ``'back'``, ``'chest'``).

    Returns:
        A :class:`BicepRuleEngine` for ``'bicep'``, or a
        :class:`NullRuleEngine` for exercises without rules yet.
    """
    cls = _ENGINE_CLASSES.get(exercise)
    if cls is not None:
        return cls()

    logger.info(
        "No rule engine for %r — using NullRuleEngine (intentional: "
        "back/chest biomechanics rules not yet implemented)",
        exercise,
    )
    return NullRuleEngine()
