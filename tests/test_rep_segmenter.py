"""
tests.test_rep_segmenter — synthetic-fixture tests for core.rep_segmenter.

Covers the batch find_peaks path (used by /api/score) and the streaming online
detector: clean, spike-noisy, shallow-ROM, fast (sub-refractory), slow, flat,
variable-amplitude (progressively shallower), batch/streaming sanity, and the
streaming cold-start (reps before the amplitude window fills). All signals are
numpy-generated; the existing FSM tests in test_rep_counter.py are untouched.
"""
from __future__ import annotations

import numpy as np

from config import RepSegmenterConfig
from core.rep_counter import FrameFeatures
from core.rep_segmenter import (
    StreamingRepSegmenter,
    count_reps_batch,
    segment_reps_batch,
    smooth_angles,
)

FPS = 30.0
# Fixed default params so these algorithm tests are independent of the deployed
# per-exercise tuning in REP_SEGMENTER_CONFIGS (which Stage C changed for bicep).
_CFG = RepSegmenterConfig()


def _reps(n: int, spr: int = 60, top: float = 160.0, bottom: float = 40.0) -> np.ndarray:
    """``n`` clean cosine reps (start at top, one bottom per rep)."""
    t = np.linspace(0.0, n * 2 * np.pi, n * spr, endpoint=False)
    mid, amp = (top + bottom) / 2.0, (top - bottom) / 2.0
    return mid + amp * np.cos(t)


def _ts(angles: np.ndarray) -> np.ndarray:
    return np.arange(angles.size) / FPS


def _stream_count(angles: np.ndarray, ts: np.ndarray) -> int:
    seg = StreamingRepSegmenter("bicep_curl", "adult", config=_CFG)
    for a, t in zip(angles, ts):
        seg.update_angle(float(a), float(t))
    return seg.rep_count


def _batch_count(angles: np.ndarray, ts: np.ndarray) -> int:
    return count_reps_batch(angles, ts, "bicep_curl", config=_CFG)


# ---------------------------------------------------------------------------

class TestCleanSignal:
    """A clean N-rep oscillation yields exactly N in both paths."""

    def test_batch_exact(self) -> None:
        ang = _reps(5)
        assert _batch_count(ang, _ts(ang)) == 5

    def test_streaming_exact(self) -> None:
        ang = _reps(5)
        assert _stream_count(ang, _ts(ang)) == 5


class TestSpikeNoise:
    """Impulsive ±25 deg spikes are rejected by the median filter."""

    def test_spikes_dont_inflate_count(self) -> None:
        ang = _reps(5).copy()
        ang[::17] += 25.0  # isolated single-frame spikes
        ts = _ts(ang)
        assert _batch_count(ang, ts) == 5
        assert abs(_stream_count(ang, ts) - 5) <= 1


class TestShallowRom:
    """A narrow ROM band (~30 deg) still counts — prominence is amplitude-relative."""

    def test_shallow_counts(self) -> None:
        ang = _reps(5, top=110.0, bottom=80.0)
        ts = _ts(ang)
        assert _batch_count(ang, ts) == 5
        assert abs(_stream_count(ang, ts) - 5) <= 1


class TestFastReps:
    """Reps closer than the refractory period are debounced (not all counted)."""

    def test_sub_refractory_debounced(self) -> None:
        ang = _reps(6, spr=12)  # 0.4 s/rep < 0.8 s refractory
        ts = _ts(ang)
        assert 2 <= _batch_count(ang, ts) < 6
        assert 1 <= _stream_count(ang, ts) < 6


class TestSlowReps:
    """Long reps still count — the segmenter has no max-duration gate (unlike the FSM)."""

    def test_slow_reps_counted(self) -> None:
        ang = _reps(3, spr=180)  # 6 s/rep
        ts = _ts(ang)
        assert _batch_count(ang, ts) == 3
        assert _stream_count(ang, ts) == 3


class TestFlatSignal:
    """A flat / near-constant signal yields zero reps."""

    def test_no_reps(self) -> None:
        rng = np.random.default_rng(0)
        ang = 100.0 + rng.uniform(-2.0, 2.0, size=300)
        ts = _ts(ang)
        assert _batch_count(ang, ts) == 0
        assert _stream_count(ang, ts) == 0


class TestVariableAmplitude:
    """Progressively shallower reps: deep reps count; reps below the amplitude-

    relative prominence are dropped. This SURFACES the partial-rep definition gap
    (Stage C reconciles it against human ground truth) rather than hiding it.
    """

    def test_shallow_partial_reps_below_cutoff_drop(self) -> None:
        segs = []
        for bottom in (40.0, 40.0, 40.0, 140.0, 140.0):  # 3 deep, 2 shallow
            mid, amp = (160.0 + bottom) / 2.0, (160.0 - bottom) / 2.0
            segs.append(mid + amp * np.cos(np.linspace(0.0, 2 * np.pi, 60, endpoint=False)))
        ang = np.concatenate(segs)
        # 3 deep reps are clearly above the amplitude-relative prominence; the 2
        # shallow (dip ~20 deg vs ~0.35*120 ~= 42) fall below it.
        assert 3 <= _batch_count(ang, _ts(ang)) <= 4


class TestBatchStreamingSanity:
    """On a clean constant-amplitude signal the two paths agree within ±1.

    NOT a guarantee: batch uses whole-clip amplitude, streaming a rolling 5 s
    window, so variable-amplitude clips can legitimately diverge.
    """

    def test_within_one(self) -> None:
        ang = _reps(7)
        ts = _ts(ang)
        assert abs(_batch_count(ang, ts) - _stream_count(ang, ts)) <= 1


class TestStreamingColdStart:
    """Reps that occur before the amplitude window fills still count (floor fallback)."""

    def test_opening_reps_count(self) -> None:
        ang = _reps(4, spr=30, top=160.0, bottom=60.0)  # 4 reps in 4 s (< 5 s window)
        ts = _ts(ang)
        assert _stream_count(ang, ts) == 4


class TestRepEventShape:
    """Batch RepEvents carry trajectory, feature_history, and identity fields."""

    def test_fields_populated(self) -> None:
        ang = _reps(3)
        ts = _ts(ang)
        feats = [FrameFeatures(float(a), 10.0, 20.0) for a in ang]
        events = segment_reps_batch(ang, ts, "bicep_curl", "adult", features=feats, config=_CFG)
        assert len(events) == 3
        ev = events[0]
        assert ev.exercise_name == "bicep_curl" and ev.age_group == "adult"
        assert len(ev.trajectory) > 0
        assert len(ev.feature_history) == len(ev.trajectory)
        assert ev.peak_angle >= ev.trough_angle


class TestSmoothHelper:
    """smooth_angles uses an odd kernel and preserves length."""

    def test_odd_kernel_and_length(self) -> None:
        ang = _reps(2)
        out = smooth_angles(ang, window=4)  # even -> coerced to odd internally
        assert out.size == ang.size
