"""
tests.test_form_scorer — pytest suite for core.form_scorer.

Tests cover all four graceful-degradation paths and FormScore immutability.
"""
from __future__ import annotations

import math

import pytest

from core.form_scorer import FormScore, FormScorer
from core.rep_counter import FrameFeatures, RepEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rep(
    exercise: str = 'bicep',
    age_group: str = 'adult',
    features: tuple[FrameFeatures, ...] = (),
    peak: float = 160.0,
    trough: float = 40.0,
    duration: float = 2.0,
) -> RepEvent:
    """Build a RepEvent with controlled parameters."""
    import numpy as np
    n = 60
    t = np.linspace(0, 2 * np.pi, n)
    mid = (peak + trough) / 2.0
    amp = (peak - trough) / 2.0
    trajectory = (mid + amp * np.cos(t)).tolist()
    return RepEvent(
        rep_number=1,
        start_ts=0.0,
        end_ts=duration,
        duration_s=duration,
        peak_angle=peak,
        trough_angle=trough,
        trajectory=trajectory,
        exercise_name=exercise,
        age_group=age_group,
        feature_history=features,
    )


def _good_features(n: int = 9) -> tuple[FrameFeatures, ...]:
    """Generate good-form features (low torso/arm deviation)."""
    angles = [150.0, 130.0, 100.0, 70.0, 50.0, 70.0, 100.0, 130.0, 150.0]
    return tuple(
        FrameFeatures(primary_angle=a, secondary_angle=10.0, tertiary_angle=20.0)
        for a in angles[:n]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullHybridPath:
    """1. ML + rules both work → full hybrid score."""

    def test_bicep_full_hybrid(self) -> None:
        scorer = FormScorer('bicep', 'adult')
        rep = _make_rep(features=_good_features())
        result = scorer.score(rep)

        assert result.ml_available is True
        assert result.rules_available is True
        assert 0.0 <= result.final_score <= 100.0
        assert not math.isnan(result.ml_score)
        assert not math.isnan(result.ml_confidence)
        assert isinstance(result.violations, tuple)


class TestMLOnlyDegradation:
    """2. Back exercise (NullRuleEngine) → final_score ≈ ml_score."""

    def test_back_ml_only(self) -> None:
        scorer = FormScorer('back', 'adult')
        rep = _make_rep(exercise='back', features=_good_features())
        result = scorer.score(rep)

        assert result.ml_available is True
        assert result.rules_available is False
        assert result.violations == ()
        assert result.rule_penalty == 0.0
        # final_score = clamp(ml_score - 0, 0, 100) = ml_score
        assert abs(result.final_score - result.ml_score) < 1e-6


class TestRulesOnlyDegradation:
    """3. Nonexistent age_group → ML unavailable, rules still work."""

    def test_bicep_rules_only(self) -> None:
        scorer = FormScorer('bicep', 'infant')
        rep = _make_rep(age_group='infant', features=_good_features())
        result = scorer.score(rep)

        assert result.ml_available is False
        assert result.rules_available is True
        assert math.isnan(result.ml_score)
        # final_score = 100 - rule_penalty
        expected = max(0.0, 100.0 - result.rule_penalty)
        assert abs(result.final_score - expected) < 1e-6


class TestBothFailDegradation:
    """4. Nonexistent age_group + back (NullEngine) → both fail."""

    def test_both_fail(self) -> None:
        scorer = FormScorer('back', 'infant')
        # No features → ML can't score even if model existed (it doesn't)
        rep = _make_rep(exercise='back', age_group='infant')
        result = scorer.score(rep)

        assert result.ml_available is False
        assert result.rules_available is False
        # final_score: no ML, NullRuleEngine gives 0 penalty → 100 - 0 = 100
        # (this is the rules-only degradation with zero penalty)
        assert result.final_score == 100.0 or math.isnan(result.final_score)


class TestFormScoreImmutability:
    """5. FormScore is frozen and hashable."""

    def test_frozen_and_hashable(self) -> None:
        scorer = FormScorer('bicep', 'adult')
        rep = _make_rep(features=_good_features())
        result = scorer.score(rep)

        # Frozen — assignment raises
        with pytest.raises(AttributeError):
            result.final_score = 0.0  # type: ignore[misc]

        # Hashable — can be added to a set
        s = {result}
        assert len(s) == 1
