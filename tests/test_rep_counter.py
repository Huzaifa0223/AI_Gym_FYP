"""
tests.test_rep_counter — pytest suite for core.rep_counter.

All tests feed pre-computed angle floats via ``update_angle()`` so that
MediaPipe is never imported.  Synthetic signals are generated with numpy.
"""
from __future__ import annotations

import numpy as np
import pytest

from config import RepCounterConfig
from core.rep_counter import (
    BackRepCounter,
    BaseRepCounter,
    BicepRepCounter,
    ChestRepCounter,
    RepEvent,
    RepState,
    make_rep_counter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_counter(
    top: float = 140.0,
    bottom: float = 60.0,
    hysteresis: float = 10.0,
    min_dur: float = 0.8,
    max_dur: float = 8.0,
) -> BaseRepCounter:
    """Return a BicepRepCounter with overridden config for controlled tests."""
    cfg = RepCounterConfig(
        top_threshold=top,
        bottom_threshold=bottom,
        hysteresis=hysteresis,
        min_rep_duration_s=min_dur,
        max_rep_duration_s=max_dur,
    )
    counter = BicepRepCounter.__new__(BicepRepCounter)
    BaseRepCounter.__init__(counter, config=cfg, exercise_name='bicep_curl', age_group='adult')
    return counter


def _sine_wave_angles(
    n_reps: int,
    samples_per_rep: int = 60,
    top_angle: float = 160.0,
    bottom_angle: float = 40.0,
) -> list[float]:
    """Generate a clean sine wave oscillating between *top_angle* and *bottom_angle*.

    Each full cycle (top -> bottom -> top) represents one rep.
    """
    total = n_reps * samples_per_rep
    t = np.linspace(0, n_reps * 2 * np.pi, total, endpoint=False)
    # cosine starts at +1 (top), dips to -1 (bottom)
    mid = (top_angle + bottom_angle) / 2.0
    amp = (top_angle - bottom_angle) / 2.0
    return (mid + amp * np.cos(t)).tolist()


def _feed_angles(
    counter: BaseRepCounter,
    angles: list[float],
    dt: float = 1.0 / 30.0,
) -> list[RepEvent]:
    """Push every angle through the counter and collect emitted events."""
    events: list[RepEvent] = []
    t = 0.0
    for a in angles:
        ev = counter.update_angle(a, t)
        if ev is not None:
            events.append(ev)
        t += dt
    return events


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestSineWaveReps:
    """1. A clean sine-wave signal with N cycles must produce exactly N reps."""

    @pytest.mark.parametrize("n_reps", [1, 3, 5, 10])
    def test_exact_rep_count(self, n_reps: int) -> None:
        counter = _make_counter(min_dur=0.0)  # disable duration gate
        angles = _sine_wave_angles(n_reps, samples_per_rep=60)
        events = _feed_angles(counter, angles)

        assert counter.rep_count == n_reps
        assert len(events) == n_reps
        for i, ev in enumerate(events, start=1):
            assert ev.rep_number == i
            assert ev.exercise_name == 'bicep_curl'
            assert ev.age_group == 'adult'
            assert ev.duration_s > 0


class TestJitterRejection:
    """2. Noise within the hysteresis band must not produce false reps."""

    def test_jitter_below_hysteresis(self) -> None:
        counter = _make_counter(hysteresis=10.0)
        rng = np.random.default_rng(42)
        # Hover around mid-range (100 deg) with +-8 deg jitter (< 10 hysteresis)
        angles = (100.0 + rng.uniform(-8, 8, size=300)).tolist()
        events = _feed_angles(counter, angles)

        assert counter.rep_count == 0
        assert len(events) == 0


class TestMinDurationRejection:
    """3. Oscillations faster than min_rep_duration_s are rejected."""

    def test_sub_min_duration(self) -> None:
        counter = _make_counter(min_dur=2.0)
        # 3 very fast reps at 30 fps, each cycle = 20 frames = 0.67s < 2.0s
        angles = _sine_wave_angles(3, samples_per_rep=20)
        events = _feed_angles(counter, angles, dt=1.0 / 30.0)

        assert counter.rep_count == 0
        assert len(events) == 0


class TestStuckFrame:
    """4. A constant angle (stuck sensor) produces zero reps."""

    def test_constant_angle(self) -> None:
        counter = _make_counter()
        angles = [90.0] * 300
        events = _feed_angles(counter, angles)

        assert counter.rep_count == 0
        assert len(events) == 0


class TestFactory:
    """5. Factory returns the correct subclass for each exercise."""

    @pytest.mark.parametrize("exercise,expected_cls", [
        ('bicep_curl', BicepRepCounter),
        ('bent_over_row', BackRepCounter),
        ('push_up', ChestRepCounter),
    ])
    def test_correct_subclass(self, exercise: str, expected_cls: type) -> None:
        counter = make_rep_counter(exercise, age_group='adult')
        assert isinstance(counter, expected_cls)
        assert isinstance(counter, BaseRepCounter)

    def test_unknown_exercise_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown exercise"):
            make_rep_counter('juggling')


class TestTrajectoryLength:
    """6. Trajectory length matches frames between rep start and end."""

    def test_trajectory_frame_count(self) -> None:
        counter = _make_counter(min_dur=0.0)
        angles = _sine_wave_angles(1, samples_per_rep=60)
        events = _feed_angles(counter, angles)

        assert len(events) == 1
        ev = events[0]
        # Trajectory starts when entering DESCENDING and collects samples
        # through DESCENDING, BOTTOM, ASCENDING — must have > 0 entries
        assert len(ev.trajectory) > 0
        # Peak should be >= top threshold area, trough <= bottom threshold area
        assert ev.peak_angle >= ev.trough_angle
        assert ev.trough_angle <= 60.0 + 15.0  # near bottom threshold


class TestReset:
    """7. reset() zeros the count and restores IDLE state."""

    def test_reset_clears_state(self) -> None:
        counter = _make_counter(min_dur=0.0)
        angles = _sine_wave_angles(3, samples_per_rep=60)
        _feed_angles(counter, angles)

        assert counter.rep_count == 3

        counter.reset()

        assert counter.rep_count == 0
        assert counter.state is RepState.IDLE

        # Can count again after reset
        events = _feed_angles(counter, angles)
        assert counter.rep_count == 3
        assert len(events) == 3


class TestIncompleteRomTurnaround:
    """8. Reps that never re-extend past the top threshold still count via the
    top-turnaround close — the real-world case the absolute-threshold FSM missed
    (it returned ~0 reps on 45/62 real clips)."""

    def test_incomplete_rom_reps_counted(self) -> None:
        # top_enter = 130, but the signal only tops out at 110 — full
        # re-extension never happens, so the old close path would yield ~0.
        counter = _make_counter(min_dur=0.0)
        angles = _sine_wave_angles(
            5, samples_per_rep=60, top_angle=110.0, bottom_angle=45.0,
        )
        events = _feed_angles(counter, angles)

        assert counter.rep_count >= 4          # turnaround recovers the reps
        assert counter.rep_count <= 6          # without over-segmenting
        # All closed below full re-extension — i.e. via the turnaround path.
        assert all(ev.peak_angle < 130.0 for ev in events)

    def test_clean_full_rom_still_exact(self) -> None:
        # Full-ROM reps must be unaffected by the turnaround addition.
        counter = _make_counter(min_dur=0.0)
        angles = _sine_wave_angles(4, samples_per_rep=60)  # 40..160, crosses 130
        events = _feed_angles(counter, angles)
        assert counter.rep_count == 4
        assert len(events) == 4
