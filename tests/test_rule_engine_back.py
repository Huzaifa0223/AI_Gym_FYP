"""
tests.test_rule_engine_back — pytest suite for BackRuleEngine.

All tests use synthetic RepEvent and FrameFeatures objects.
No MediaPipe imports required.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.rep_counter import FrameFeatures, RepEvent
from core.rule_engine import BackRuleEngine, make_rule_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_back_rep(
    peak: float = 160.0,
    trough: float = 40.0,
    duration: float = 2.0,
    trajectory: list[float] | None = None,
    features: tuple[FrameFeatures, ...] = (),
) -> RepEvent:
    """Build a back-exercise RepEvent with controlled geometry."""
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
        exercise_name='back',
        age_group='adult',
        feature_history=features,
    )


def _good_back_features(n: int = 30) -> tuple[FrameFeatures, ...]:
    """Good-form back features: proper retraction (secondary dips to 30°),
    stable torso (low std), reasonable tertiary."""
    angles = np.linspace(150.0, 40.0, n // 2).tolist() + \
             np.linspace(40.0, 150.0, n - n // 2).tolist()
    return tuple(
        FrameFeatures(
            primary_angle=a,
            secondary_angle=30.0 + 5.0 * np.sin(i * 0.2),  # stable around 30°
            tertiary_angle=20.0,
        )
        for i, a in enumerate(angles)
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCleanBackRep:
    """1. Clean back rep → empty violations."""

    def test_no_violations(self) -> None:
        engine = BackRuleEngine()
        rep = _make_back_rep(features=_good_back_features())
        violations = engine.evaluate(rep)
        assert len(violations) == 0


class TestScapularRetraction:
    """2. Insufficient scapular retraction → violation."""

    def test_bad_retraction_flagged(self) -> None:
        engine = BackRuleEngine()
        # Secondary angle never drops below 70° (threshold is 50°)
        feats = tuple(
            FrameFeatures(primary_angle=100.0, secondary_angle=70.0, tertiary_angle=20.0)
            for _ in range(30)
        )
        rep = _make_back_rep(features=feats)
        violations = engine.evaluate(rep)
        names = [v.name for v in violations]
        assert 'scapular_retraction' in names


class TestTorsoStability:
    """3. Torso kipping → violation."""

    def test_unstable_torso_flagged(self) -> None:
        engine = BackRuleEngine()
        # High std in secondary angle → torso kipping
        rng = np.random.default_rng(42)
        feats = tuple(
            FrameFeatures(
                primary_angle=100.0,
                secondary_angle=30.0 + rng.normal(0, 25),  # std ≈ 25 > 15
                tertiary_angle=20.0,
            )
            for _ in range(60)
        )
        rep = _make_back_rep(features=feats)
        violations = engine.evaluate(rep)
        names = [v.name for v in violations]
        assert 'torso_unstable' in names


class TestTempoAsymmetryEccentricFast:
    """4. Eccentric dropping (trough heavily skewed toward start) → violation."""

    def test_eccentric_too_fast(self) -> None:
        engine = BackRuleEngine()
        # Trough at frame 5 out of 60 → ratio = 6/55 ≈ 0.11 < 0.7
        trajectory = list(np.linspace(160.0, 40.0, 5)) + \
                     list(np.linspace(40.0, 160.0, 55))
        rep = _make_back_rep(trajectory=trajectory, features=_good_back_features())
        violations = engine.evaluate(rep)
        names = [v.name for v in violations]
        assert 'tempo_asymmetry' in names
        # Check message direction
        asym_v = [v for v in violations if v.name == 'tempo_asymmetry'][0]
        assert 'lowering' in asym_v.message.lower() or 'eccentric' in asym_v.message.lower()


class TestTempoAsymmetryEccentricSlow:
    """5. Eccentric too slow (trough heavily skewed toward end) → violation."""

    def test_eccentric_too_slow(self) -> None:
        engine = BackRuleEngine()
        # Trough at frame 55 out of 60 → ratio = 56/5 ≈ 11.2 > 1.5
        trajectory = list(np.linspace(160.0, 40.0, 55)) + \
                     list(np.linspace(40.0, 160.0, 5))
        rep = _make_back_rep(trajectory=trajectory, features=_good_back_features())
        violations = engine.evaluate(rep)
        names = [v.name for v in violations]
        assert 'tempo_asymmetry' in names
        asym_v = [v for v in violations if v.name == 'tempo_asymmetry'][0]
        assert 'lifting' in asym_v.message.lower() or 'concentric' in asym_v.message.lower()


class TestMultipleFaultsStack:
    """6. Bad retraction + unstable torso → both violations in list."""

    def test_stacked_violations(self) -> None:
        engine = BackRuleEngine()
        # Secondary alternates between 60 and 100 → min=60 > 50 threshold
        # AND std ≈ 20 > 15 threshold → triggers both violations
        feats = tuple(
            FrameFeatures(
                primary_angle=100.0,
                secondary_angle=60.0 if i % 2 == 0 else 100.0,
                tertiary_angle=20.0,
            )
            for i in range(60)
        )
        rep = _make_back_rep(features=feats)
        violations = engine.evaluate(rep)
        names = [v.name for v in violations]
        assert 'scapular_retraction' in names
        assert 'torso_unstable' in names


class TestEmptyFeatureHistoryGraceful:
    """7. Empty feature_history doesn't crash retraction/stability checks."""

    def test_no_crash_on_empty_features(self) -> None:
        engine = BackRuleEngine()
        rep = _make_back_rep(features=())
        # Should not raise; retraction/stability just return None
        violations = engine.evaluate(rep)
        # No feature-based violations possible
        feat_names = {'scapular_retraction', 'torso_unstable'}
        assert all(v.name not in feat_names for v in violations)


class TestBackFactory:
    """8. Factory returns BackRuleEngine for 'back'."""

    def test_factory_back(self) -> None:
        engine = make_rule_engine('back')
        assert isinstance(engine, BackRuleEngine)
