"""
tests.test_score_aggregator — three weight regimes, source breakdown,
clamping, feedback selection.
"""
from __future__ import annotations

import pytest

from core.config import (
    ScoreAggregatorWeights,
    WithCnnLstmWeights,
    WithoutCnnLstmWeights,
)
from core.schemas import HeadlineScore
from core.score_aggregator import aggregate


# ---------------------------------------------------------------------------
# All-three-present path
# ---------------------------------------------------------------------------

class TestAllThreePresent:
    def test_weighted_average_matches_hand_computed(self) -> None:
        # Defaults: rules 0.4, rf 0.3, cnn_lstm 0.3
        # rules=80, rf=70, cnn_lstm=90
        # → 80*0.4 + 70*0.3 + 90*0.3 = 32 + 21 + 27 = 80
        result = aggregate(
            rules_score=80,
            rf_score=70,
            cnn_lstm_score=90,
            rep_count=5,
            feedback_messages=[],
        )
        assert result.score == 80
        assert result.source_breakdown == {"rules": 80, "rf": 70, "cnn_lstm": 90}


# ---------------------------------------------------------------------------
# CNN+LSTM-None path
# ---------------------------------------------------------------------------

class TestCnnLstmAbsent:
    def test_uses_without_cnn_lstm_weights(self) -> None:
        # Defaults: rules 0.6, rf 0.4
        # rules=80, rf=60 → 80*0.6 + 60*0.4 = 48 + 24 = 72
        result = aggregate(
            rules_score=80,
            rf_score=60,
            cnn_lstm_score=None,
            rep_count=5,
            feedback_messages=[],
        )
        assert result.score == 72
        assert result.source_breakdown["cnn_lstm"] is None
        assert result.source_breakdown["rules"] == 80
        assert result.source_breakdown["rf"] == 60


# ---------------------------------------------------------------------------
# Both RF and CNN+LSTM None — rules-only
# ---------------------------------------------------------------------------

class TestRulesOnly:
    def test_final_equals_rules_score(self) -> None:
        result = aggregate(
            rules_score=73,
            rf_score=None,
            cnn_lstm_score=None,
            rep_count=3,
            feedback_messages=[],
        )
        assert result.score == 73
        assert result.source_breakdown == {"rules": 73, "rf": None, "cnn_lstm": None}


# ---------------------------------------------------------------------------
# Feedback selection
# ---------------------------------------------------------------------------

class TestFeedback:
    def test_empty_messages_yields_empty_string(self) -> None:
        result = aggregate(
            rules_score=80, rf_score=70, cnn_lstm_score=90,
            rep_count=1, feedback_messages=[],
        )
        assert result.form_feedback == ""

    def test_first_message_wins(self) -> None:
        result = aggregate(
            rules_score=80, rf_score=70, cnn_lstm_score=90,
            rep_count=1,
            feedback_messages=[
                "Solid technique",
                "Eccentric phase good",
                "Tempo stable",
            ],
        )
        assert result.form_feedback == "Solid technique"


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------

class TestClamping:
    def test_clamps_above_100(self) -> None:
        # Custom weights summing > 1.0 to force overflow scenario, even
        # though defaults always sum to 1.0. Pydantic would otherwise
        # reject the score.
        weights = ScoreAggregatorWeights(
            with_cnn_lstm=WithCnnLstmWeights(rules=2.0, rf=0.0, cnn_lstm=0.0),
            without_cnn_lstm=WithoutCnnLstmWeights(rules=1.0, rf=0.0),
        )
        result = aggregate(
            rules_score=80, rf_score=80, cnn_lstm_score=80,
            rep_count=1, feedback_messages=[], weights=weights,
        )
        # 80 * 2.0 = 160 -> clamped to 100
        assert result.score == 100

    def test_clamps_below_0(self) -> None:
        # Negative weight to force a negative weighted sum.
        weights = ScoreAggregatorWeights(
            with_cnn_lstm=WithCnnLstmWeights(rules=-1.0, rf=0.0, cnn_lstm=0.0),
            without_cnn_lstm=WithoutCnnLstmWeights(rules=1.0, rf=0.0),
        )
        result = aggregate(
            rules_score=50, rf_score=50, cnn_lstm_score=50,
            rep_count=1, feedback_messages=[], weights=weights,
        )
        assert result.score == 0


# ---------------------------------------------------------------------------
# rep_count passthrough
# ---------------------------------------------------------------------------

class TestRepCountPassthrough:
    @pytest.mark.parametrize("rep_count", [0, 1, 5, 100])
    def test_rep_count_passes_through(self, rep_count: int) -> None:
        result = aggregate(
            rules_score=70, rf_score=70, cnn_lstm_score=70,
            rep_count=rep_count, feedback_messages=[],
        )
        assert result.rep_count == rep_count


# ---------------------------------------------------------------------------
# Return-type smoke
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_headline_score(self) -> None:
        result = aggregate(
            rules_score=70, rf_score=70, cnn_lstm_score=70,
            rep_count=1, feedback_messages=["ok"],
        )
        assert isinstance(result, HeadlineScore)
