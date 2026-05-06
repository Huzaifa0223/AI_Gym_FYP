"""
tests.test_rule_engine — pytest suite for core.rule_engine.

All tests construct synthetic RepEvent objects with controlled trajectory,
peak/trough, and duration to trigger or avoid each rule.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.rep_counter import RepEvent
from core.rule_engine import (
    BackRuleEngine,
    BicepRuleEngine,
    ChestRuleEngine,
    NullRuleEngine,
    make_rule_engine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rep(
    peak: float = 160.0,
    trough: float = 40.0,
    duration: float = 2.0,
    trajectory: list[float] | None = None,
) -> RepEvent:
    """Build a RepEvent with controlled geometry."""
    if trajectory is None:
        # Smooth sine-like trajectory from peak -> trough -> peak
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
        exercise_name='bicep_curl',
        age_group='adult',
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestROMRule:
    """1. BicepRuleEngine returns ROM violation when peak-trough < 80."""

    def test_short_rom_flagged(self) -> None:
        engine = BicepRuleEngine()
        rep = _make_rep(peak=120.0, trough=80.0)  # ROM = 40 < 80
        violations = engine.evaluate(rep)
        names = [v.name for v in violations]
        assert 'rom_short' in names

    def test_full_rom_clean(self) -> None:
        engine = BicepRuleEngine()
        rep = _make_rep(peak=160.0, trough=40.0)  # ROM = 120 >= 80
        violations = engine.evaluate(rep)
        names = [v.name for v in violations]
        assert 'rom_short' not in names


class TestSpeedRule:
    """2-3. Speed violations for too-fast and too-slow reps."""

    def test_too_fast_flagged(self) -> None:
        engine = BicepRuleEngine()
        rep = _make_rep(duration=0.5)  # < 1.0s
        violations = engine.evaluate(rep)
        names = [v.name for v in violations]
        assert 'too_fast' in names

    def test_too_slow_flagged(self) -> None:
        engine = BicepRuleEngine()
        rep = _make_rep(duration=6.0)  # > 5.0s
        violations = engine.evaluate(rep)
        names = [v.name for v in violations]
        assert 'too_slow' in names

    def test_normal_speed_clean(self) -> None:
        engine = BicepRuleEngine()
        rep = _make_rep(duration=2.0)  # 1.0 <= 2.0 <= 5.0
        violations = engine.evaluate(rep)
        speed_names = [v.name for v in violations if v.name in ('too_fast', 'too_slow')]
        assert len(speed_names) == 0


class TestTempoRule:
    """4. Tempo violation when CV of angle deltas > 0.6."""

    def test_jerky_trajectory_flagged(self) -> None:
        engine = BicepRuleEngine()
        # Alternate between big and tiny deltas → high CV
        trajectory = []
        for i in range(60):
            if i % 2 == 0:
                trajectory.append(160.0 - i * 2.0)
            else:
                trajectory.append(trajectory[-1] - 0.1)
        rep = _make_rep(trajectory=trajectory)
        violations = engine.evaluate(rep)
        names = [v.name for v in violations]
        assert 'jerky_tempo' in names


class TestCleanRep:
    """5. A well-formed rep produces no violations."""

    def test_no_violations(self) -> None:
        engine = BicepRuleEngine()
        rep = _make_rep(peak=160.0, trough=40.0, duration=2.0)
        violations = engine.evaluate(rep)
        assert len(violations) == 0


class TestNullRuleEngine:
    """6. NullRuleEngine always returns empty list."""

    def test_always_empty(self) -> None:
        engine = NullRuleEngine()
        rep = _make_rep(peak=100.0, trough=90.0, duration=0.1)
        violations = engine.evaluate(rep)
        assert violations == []


class TestFactory:
    """7. make_rule_engine returns correct types."""

    def test_bicep_returns_bicep_engine(self) -> None:
        assert isinstance(make_rule_engine('bicep_curl'), BicepRuleEngine)

    def test_back_returns_back_engine(self) -> None:
        assert isinstance(make_rule_engine('bent_over_row'), BackRuleEngine)

    def test_chest_returns_chest_engine(self) -> None:
        assert isinstance(make_rule_engine('push_up'), ChestRuleEngine)

    def test_unknown_returns_null_engine(self) -> None:
        assert isinstance(make_rule_engine('deadlift'), NullRuleEngine)
