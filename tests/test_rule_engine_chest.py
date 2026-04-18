"""
tests.test_rule_engine_chest — pytest suite for ChestRuleEngine.

All tests use synthetic RepEvent and FrameFeatures objects.
No MediaPipe imports required.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.rep_counter import FrameFeatures, RepEvent
from core.rule_engine import ChestRuleEngine, make_rule_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chest_rep(
    peak: float = 160.0,
    trough: float = 70.0,
    duration: float = 2.0,
    trajectory: list[float] | None = None,
    features: tuple[FrameFeatures, ...] = (),
) -> RepEvent:
    """Build a push-up RepEvent with controlled geometry."""
    if trajectory is None:
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
        exercise_name='chest',
        age_group='adult',
        feature_history=features,
    )


def _good_chest_features(n: int = 30) -> tuple[FrameFeatures, ...]:
    """Good-form push-up: straight plank (secondary ≈ 178°), straight legs
    (tertiary ≈ 175°)."""
    angles = np.linspace(160.0, 70.0, n // 2).tolist() + \
             np.linspace(70.0, 160.0, n - n // 2).tolist()
    return tuple(
        FrameFeatures(
            primary_angle=a,
            secondary_angle=178.0,   # near-perfect plank
            tertiary_angle=175.0,    # legs straight
        )
        for a in angles
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCleanChestRep:
    """1. Clean push-up → empty violations."""

    def test_no_violations(self) -> None:
        engine = ChestRuleEngine()
        rep = _make_chest_rep(features=_good_chest_features())
        violations = engine.evaluate(rep)
        assert len(violations) == 0


class TestPlankIntegrity:
    """2. Hip sag → plank violation."""

    def test_hip_sag_flagged(self) -> None:
        engine = ChestRuleEngine()
        # Mean secondary = 150° < 165° threshold
        feats = tuple(
            FrameFeatures(primary_angle=100.0, secondary_angle=150.0, tertiary_angle=175.0)
            for _ in range(30)
        )
        rep = _make_chest_rep(features=feats)
        violations = engine.evaluate(rep)
        names = [v.name for v in violations]
        assert 'plank_broken' in names


class TestLegAlignment:
    """3. Knee bend → leg alignment violation."""

    def test_knee_bend_flagged(self) -> None:
        engine = ChestRuleEngine()
        # Mean tertiary = 160° < 170° threshold
        feats = tuple(
            FrameFeatures(primary_angle=100.0, secondary_angle=178.0, tertiary_angle=160.0)
            for _ in range(30)
        )
        rep = _make_chest_rep(features=feats)
        violations = engine.evaluate(rep)
        names = [v.name for v in violations]
        assert 'leg_bent' in names


class TestPlankAndLegStack:
    """4. Both plank and leg faults in same rep → two violations."""

    def test_both_faults(self) -> None:
        engine = ChestRuleEngine()
        feats = tuple(
            FrameFeatures(primary_angle=100.0, secondary_angle=150.0, tertiary_angle=160.0)
            for _ in range(30)
        )
        rep = _make_chest_rep(features=feats)
        violations = engine.evaluate(rep)
        names = [v.name for v in violations]
        assert 'plank_broken' in names
        assert 'leg_bent' in names


class TestTempoAsymmetryChest:
    """5. Eccentric dropping → tempo asymmetry violation."""

    def test_eccentric_dropping(self) -> None:
        engine = ChestRuleEngine()
        # Trough at frame 5 out of 60 → ratio = 6/55 ≈ 0.11 < 0.7
        trajectory = list(np.linspace(160.0, 70.0, 5)) + \
                     list(np.linspace(70.0, 160.0, 55))
        rep = _make_chest_rep(trajectory=trajectory, features=_good_chest_features())
        violations = engine.evaluate(rep)
        names = [v.name for v in violations]
        assert 'tempo_asymmetry' in names


class TestPartialROMChest:
    """6. Partial ROM (peak-trough < 60°) → ROM violation."""

    def test_short_rom_flagged(self) -> None:
        engine = ChestRuleEngine()
        # ROM = 130 - 90 = 40 < 60
        rep = _make_chest_rep(peak=130.0, trough=90.0, features=_good_chest_features())
        violations = engine.evaluate(rep)
        names = [v.name for v in violations]
        assert 'rom_short' in names


class TestEmptyFeatureHistoryChest:
    """7. Empty feature_history doesn't crash plank/leg checks."""

    def test_no_crash_on_empty_features(self) -> None:
        engine = ChestRuleEngine()
        rep = _make_chest_rep(features=())
        violations = engine.evaluate(rep)
        feat_names = {'plank_broken', 'leg_bent'}
        assert all(v.name not in feat_names for v in violations)


class TestChestFactory:
    """8. Factory returns ChestRuleEngine for 'chest'."""

    def test_factory_chest(self) -> None:
        engine = make_rule_engine('chest')
        assert isinstance(engine, ChestRuleEngine)
