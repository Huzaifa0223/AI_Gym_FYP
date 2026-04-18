"""
core.form_scorer — Hybrid form scorer that combines ML prediction confidence
with deterministic rule-engine penalties.

Graceful degradation matrix
---------------------------

+------------------+------------------+------------------------------------+
| ML available?    | Rules available? | ``final_score`` computation        |
+==================+==================+====================================+
| Yes              | Yes              | ``clamp(ml_score - penalty, 0, 100)`` |
+------------------+------------------+------------------------------------+
| Yes              | No (NullEngine)  | ``ml_score`` (no penalty applied)  |
+------------------+------------------+------------------------------------+
| No               | Yes              | ``clamp(100 - penalty, 0, 100)``   |
+------------------+------------------+------------------------------------+
| No               | No               | ``NaN`` — both signals unavailable |
+------------------+------------------+------------------------------------+
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from core.ml_aggregator import (
    InsufficientDataError,
    MLAggregator,
    ModelNotAvailableError,
)
from core.rep_counter import RepEvent
from core.rule_engine import FormViolation, NullRuleEngine, make_rule_engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormScore:
    """Immutable, hashable result of scoring a single rep.

    Attributes:
        rep_number:      1-based rep index.
        exercise_name:   E.g. ``'bicep'``.
        age_group:       E.g. ``'adult'``.
        ml_score:        ML good-form score (0-100), ``NaN`` if unavailable.
        ml_confidence:   Mean max probability (0-1), ``NaN`` if unavailable.
        rule_penalty:    Sum of :class:`FormViolation` penalties.
        final_score:     Hybrid score after degradation logic.
        violations:      Tuple of detected violations (frozen for hashability).
        ml_available:    Whether ML scoring succeeded.
        rules_available: Whether the rule engine produced real results
                         (``False`` when :class:`NullRuleEngine` was used).
    """
    rep_number: int
    exercise_name: str
    age_group: str
    ml_score: float
    ml_confidence: float
    rule_penalty: float
    final_score: float
    violations: tuple[FormViolation, ...]
    ml_available: bool
    rules_available: bool


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class FormScorer:
    """Orchestrate ML aggregation and rule-engine evaluation for a single rep.

    Args:
        exercise:  Exercise key (``'bicep'``, ``'back'``, ``'chest'``).
        age_group: Age group key (``'children'``, ``'adult'``, ``'senior'``).
    """

    def __init__(self, exercise: str, age_group: str) -> None:
        self._exercise = exercise
        self._age_group = age_group

        # ML aggregator — may fail at init (missing model)
        self._ml: MLAggregator | None = None
        try:
            self._ml = MLAggregator(exercise, age_group)
        except ModelNotAvailableError:
            logger.warning(
                "ML model unavailable for %s/%s — ML scoring disabled",
                exercise, age_group,
            )

        # Rule engine — always succeeds (NullRuleEngine fallback)
        self._rule_engine = make_rule_engine(exercise)
        self._rules_available = not isinstance(self._rule_engine, NullRuleEngine)

    def score(self, rep: RepEvent) -> FormScore:
        """Score a completed rep with graceful degradation.

        Args:
            rep: A completed :class:`RepEvent`.

        Returns:
            A :class:`FormScore` capturing both ML and rule signals.
        """
        # -- ML stage --------------------------------------------------------
        ml_score = math.nan
        ml_confidence = math.nan
        ml_available = False

        if self._ml is not None:
            try:
                ml_score, ml_confidence = self._ml.score(rep)
                ml_available = True
            except InsufficientDataError:
                logger.warning(
                    "No feature_history for %s rep #%d — ML score unavailable",
                    self._exercise, rep.rep_number,
                )

        # -- Rule stage ------------------------------------------------------
        violations = self._rule_engine.evaluate(rep)
        rule_penalty = sum(v.penalty for v in violations)

        # -- Combine ---------------------------------------------------------
        final_score = self._compute_final(ml_score, ml_available, rule_penalty)

        return FormScore(
            rep_number=rep.rep_number,
            exercise_name=self._exercise,
            age_group=self._age_group,
            ml_score=ml_score,
            ml_confidence=ml_confidence,
            rule_penalty=rule_penalty,
            final_score=final_score,
            violations=tuple(violations),
            ml_available=ml_available,
            rules_available=self._rules_available,
        )

    @staticmethod
    def _compute_final(
        ml_score: float,
        ml_available: bool,
        rule_penalty: float,
    ) -> float:
        """Apply the degradation matrix to compute final_score."""
        if ml_available:
            return max(0.0, min(100.0, ml_score - rule_penalty))
        if rule_penalty > 0.0 or not math.isnan(rule_penalty):
            return max(0.0, min(100.0, 100.0 - rule_penalty))
        return math.nan
