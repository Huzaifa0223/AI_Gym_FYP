"""
tests.test_config — validate the 3-pillar pipeline config dataclasses in
``core.config``.

Covers immutability and the spec-§1.2 invariant that score-aggregator
weight profiles each sum to 1.0 within a 1e-6 tolerance.
See ``docs/architecture_3pillar.md`` §1.2 / §6.
"""
from __future__ import annotations

import dataclasses

import pytest

from core.config import (
    LatencyBudget,
    PrimaryUserConfig,
    ScoreAggregatorWeights,
    WithCnnLstmWeights,
    WithoutCnnLstmWeights,
)


_FLOAT_EPSILON = 1e-6


# ---------------------------------------------------------------------------
# LatencyBudget
# ---------------------------------------------------------------------------

class TestLatencyBudget:
    def test_defaults(self) -> None:
        lb = LatencyBudget()
        assert lb.per_frame_ms_p95 == 90.0
        assert lb.hard_limit_ms == 100.0

    def test_frozen(self) -> None:
        lb = LatencyBudget()
        with pytest.raises(dataclasses.FrozenInstanceError):
            lb.per_frame_ms_p95 = 80.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PrimaryUserConfig
# ---------------------------------------------------------------------------

class TestPrimaryUserConfig:
    def test_defaults(self) -> None:
        cfg = PrimaryUserConfig()
        assert cfg.bbox_hysteresis_frames == 5
        assert cfg.lock_release_after_frames == 30
        assert cfg.min_bbox_area_ratio == 0.05

    def test_frozen(self) -> None:
        cfg = PrimaryUserConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.bbox_hysteresis_frames = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ScoreAggregatorWeights + inner profiles
# ---------------------------------------------------------------------------

class TestScoreAggregatorWeights:
    def test_outer_defaults(self) -> None:
        w = ScoreAggregatorWeights()
        assert w.with_cnn_lstm == WithCnnLstmWeights(
            rules=0.4, rf=0.3, cnn_lstm=0.3,
        )
        assert w.without_cnn_lstm == WithoutCnnLstmWeights(
            rules=0.6, rf=0.4,
        )

    def test_with_cnn_lstm_sums_to_one(self) -> None:
        w = WithCnnLstmWeights()
        total = w.rules + w.rf + w.cnn_lstm
        assert abs(total - 1.0) < _FLOAT_EPSILON, f"sum was {total}"

    def test_without_cnn_lstm_sums_to_one(self) -> None:
        w = WithoutCnnLstmWeights()
        total = w.rules + w.rf
        assert abs(total - 1.0) < _FLOAT_EPSILON, f"sum was {total}"

    def test_outer_frozen(self) -> None:
        w = ScoreAggregatorWeights()
        with pytest.raises(dataclasses.FrozenInstanceError):
            w.with_cnn_lstm = WithCnnLstmWeights()  # type: ignore[misc]

    def test_inner_with_frozen(self) -> None:
        w = WithCnnLstmWeights()
        with pytest.raises(dataclasses.FrozenInstanceError):
            w.rules = 0.5  # type: ignore[misc]

    def test_inner_without_frozen(self) -> None:
        w = WithoutCnnLstmWeights()
        with pytest.raises(dataclasses.FrozenInstanceError):
            w.rules = 0.5  # type: ignore[misc]
