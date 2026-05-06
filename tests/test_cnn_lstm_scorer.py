"""
tests.test_cnn_lstm_scorer — Protocol satisfaction + null implementation.
"""
from __future__ import annotations

from core.cnn_lstm_scorer import FormQualityScorer, NullFormQualityScorer
from core.rep_counter import RepEvent


def _stub_rep() -> RepEvent:
    return RepEvent(
        rep_number=1,
        start_ts=0.0,
        end_ts=2.0,
        duration_s=2.0,
        peak_angle=160.0,
        trough_angle=40.0,
        trajectory=[160.0, 100.0, 40.0, 100.0, 160.0],
        exercise_name="bicep_curl",
        age_group="adult",
        feature_history=(),
    )


class TestNullFormQualityScorer:
    def test_returns_none(self) -> None:
        scorer = NullFormQualityScorer()
        assert scorer.score(_stub_rep()) is None

    def test_satisfies_protocol(self) -> None:
        # FormQualityScorer is @runtime_checkable.
        assert isinstance(NullFormQualityScorer(), FormQualityScorer)
