"""
tests.test_ml_aggregator — pytest suite for core.ml_aggregator.

Tests 1-2 use the real trained model at data/models/bicep_adult.pkl.
Tests 3-5 use synthetic FrameFeatures to validate score boundaries.
"""
from __future__ import annotations

import pytest

from core.ml_aggregator import (
    InsufficientDataError,
    MLAggregator,
    ModelNotAvailableError,
)
from core.rep_counter import FrameFeatures, RepEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rep(
    features: tuple[FrameFeatures, ...] = (),
    exercise: str = 'bicep',
    age_group: str = 'adult',
) -> RepEvent:
    """Build a minimal RepEvent with controlled feature_history."""
    return RepEvent(
        rep_number=1,
        start_ts=0.0,
        end_ts=2.0,
        duration_s=2.0,
        peak_angle=160.0,
        trough_angle=40.0,
        trajectory=[160.0, 100.0, 40.0, 100.0, 160.0],
        exercise_name=exercise,
        age_group=age_group,
        feature_history=features,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMLAggregatorBasic:
    """1-2. Load real model and validate return types."""

    def test_score_returns_float_pair(self) -> None:
        agg = MLAggregator('bicep', 'adult')
        feats = tuple(
            FrameFeatures(primary_angle=a, secondary_angle=15.0, tertiary_angle=30.0)
            for a in [160.0, 120.0, 80.0, 40.0, 80.0, 120.0, 160.0]
        )
        rep = _make_rep(features=feats)
        ml_score, confidence = agg.score(rep)

        assert isinstance(ml_score, float)
        assert isinstance(confidence, float)
        assert 0.0 <= ml_score <= 100.0
        assert 0.0 <= confidence <= 1.0

    def test_missing_model_raises(self) -> None:
        with pytest.raises(ModelNotAvailableError):
            MLAggregator('bicep', 'infant')


class TestMLAggregatorEdgeCases:
    """3. Empty feature_history raises InsufficientDataError."""

    def test_empty_features_raises(self) -> None:
        agg = MLAggregator('bicep', 'adult')
        rep = _make_rep(features=())
        with pytest.raises(InsufficientDataError):
            agg.score(rep)


class TestMLAggregatorScoreDirection:
    """4-5. Good-form features should score higher than bad-form features."""

    def test_good_form_scores_high(self) -> None:
        agg = MLAggregator('bicep', 'adult')
        # Good form: moderate elbow angles, low torso deviation, low arm swing
        feats = tuple(
            FrameFeatures(primary_angle=a, secondary_angle=10.0, tertiary_angle=20.0)
            for a in [150.0, 130.0, 100.0, 70.0, 50.0, 70.0, 100.0, 130.0, 150.0]
        )
        rep = _make_rep(features=feats)
        ml_score, _ = agg.score(rep)
        assert ml_score > 50.0

    def test_bad_form_scores_low(self) -> None:
        agg = MLAggregator('bicep', 'adult')
        # Bad form per trained model: moderate secondary (~40°) + low tertiary
        # (~10°) indicates torso swing without compensating arm extension —
        # the model's primary bad-form signature.
        feats = tuple(
            FrameFeatures(primary_angle=a, secondary_angle=40.0, tertiary_angle=10.0)
            for a in [150.0, 130.0, 100.0, 70.0, 50.0, 70.0, 100.0, 130.0, 150.0]
        )
        rep = _make_rep(features=feats)
        ml_score, _ = agg.score(rep)
        assert ml_score < 50.0
