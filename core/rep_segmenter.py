"""
core.rep_segmenter — smoothed, amplitude-relative, prominence-based rep counting.

A **rep is one curl-bottom**: a local *minimum* of the median-smoothed primary
joint angle, confirmed by a rebound of at least the (amplitude-relative)
prominence and separated from the previous rep by at least the refractory period.
This replaces the absolute top/bottom gates of the FSM (``core/rep_counter.py``),
which under-counted real video badly (the real ROM band ~80-120 deg rarely crosses
the textbook gates). Amplitude-relative prominence — a fraction of the signal's own
``p95 - p5`` amplitude — makes counting robust to that shifted, narrow band.

Two entry points share the **same median smoothing** so they stay comparable:

* :func:`segment_reps_batch` — the **batch** path for ``/api/score`` (whole video
  available). Uses :func:`scipy.signal.find_peaks` with **global** ``p5/p95``
  prominence and a ``distance`` = refractory. This is the production counting path
  for uploaded videos and the one Stage C validates against ground truth.
* :class:`StreamingRepSegmenter` — the **online** path for a future live/streaming
  feed. Per-frame ``update_angle`` with a rolling-median smoother, a rolling
  amplitude estimate (cold-start falls back to ``prominence_floor_deg`` until the
  window fills), and a refractory debounce.

Batch and streaming use **different amplitude windows** (whole-clip vs rolling), so
on variable-amplitude clips their counts can diverge — they are a *sanity check*
on synthetic constant-amplitude signals, **not** a guaranteed identity.

Median (not mean/Savitzky-Golay) smoothing is used because the observed jitter is
*impulsive* (single-frame spikes up to ~29 deg): a median rejects impulses without
smearing the rep oscillation.
"""
from __future__ import annotations

import logging
from collections import deque

import numpy as np
from scipy.signal import find_peaks, medfilt

from config import REP_SEGMENTER_CONFIGS, RepSegmenterConfig
from core.rep_counter import FrameFeatures, RepEvent

logger = logging.getLogger(__name__)

# Streaming smoothing buffers. The runtime window is min(_MAX_SMOOTH_FRAMES,
# round(smoothing_window_seconds * observed_fps)); the cap merely bounds memory.
_MAX_SMOOTH_FRAMES = 64   # raw-angle buffer cap (covers any plausible fps × window)
_FPS_EST_FRAMES = 5       # recent timestamps kept to estimate live fps (median of diffs)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _odd_window(window: int, n: int) -> int:
    """Largest odd median-filter kernel <= ``window`` and <= ``n`` (>= 1)."""
    w = max(1, min(int(window), n))
    if w % 2 == 0:
        w -= 1
    return max(1, w)


def smooth_angles(angles: np.ndarray, window: int) -> np.ndarray:
    """Median-filter *angles* with an odd window (impulsive-jitter rejection)."""
    angles = np.asarray(angles, dtype=float)
    if angles.size < 3:
        return angles
    return medfilt(angles, kernel_size=_odd_window(window, angles.size))


def _amplitude(values: np.ndarray) -> float:
    """Robust signal amplitude: 95th minus 5th percentile (degrees)."""
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def _prominence(amplitude: float, cfg: RepSegmenterConfig) -> float:
    """Amplitude-relative prominence with an absolute floor."""
    return max(cfg.prominence_floor_deg, cfg.prominence_frac * amplitude)


def _median_dt(timestamps: np.ndarray, fps_fallback: float) -> float:
    """Median frame interval (seconds) from *timestamps*, or 1/fps fallback."""
    if timestamps.size >= 2:
        dt = float(np.median(np.diff(timestamps)))
        if dt > 1e-6:
            return dt
    return 1.0 / fps_fallback


# ---------------------------------------------------------------------------
# Batch path — find_peaks over the whole clip (used by /api/score)
# ---------------------------------------------------------------------------

def _rep_windows(minima: np.ndarray, smoothed: np.ndarray) -> list[tuple[int, int]]:
    """Split the frame index range into one window per curl-bottom.

    Boundaries sit at the highest (most-extended) frame between consecutive
    minima, so each window spans roughly one top->bottom->top cycle and contains
    exactly one minimum.
    """
    n = smoothed.size
    if minima.size == 0:
        return []
    bounds = [0]
    for a, b in zip(minima[:-1], minima[1:]):
        bounds.append(int(a + np.argmax(smoothed[a:b + 1])))
    bounds.append(n - 1)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def segment_reps_batch(
    angles: np.ndarray,
    timestamps: np.ndarray,
    exercise: str,
    age_group: str,
    *,
    features: list[FrameFeatures] | None = None,
    config: RepSegmenterConfig | None = None,
) -> list[RepEvent]:
    """Detect every rep in a full angle trace and return :class:`RepEvent`s.

    The production counting path for ``/api/score``. Smooths, computes a global
    ``p5/p95`` amplitude, then ``find_peaks`` on the inverted signal with
    amplitude-relative prominence and a refractory ``distance``.

    Args:
        angles:      Per-frame primary joint angle (degrees), in clip order.
        timestamps:  Per-frame timestamps (seconds), same length as *angles*.
        exercise:    Exercise key (selects the :class:`RepSegmenterConfig`).
        age_group:   Age group key, carried onto each emitted :class:`RepEvent`.
        features:    Optional per-frame :class:`FrameFeatures`, same length as
                     *angles*; sliced into each rep's ``feature_history``.
        config:      Override config (defaults to ``REP_SEGMENTER_CONFIGS``).

    Returns:
        One :class:`RepEvent` per detected rep, in order.
    """
    cfg = config or REP_SEGMENTER_CONFIGS.get(exercise, RepSegmenterConfig())
    angles = np.asarray(angles, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    if angles.size < 3:
        return []

    dt = _median_dt(timestamps, cfg.fps_fallback)
    window_frames = max(1, round(cfg.smoothing_window_seconds / dt))
    smoothed = smooth_angles(angles, window_frames)
    prominence = _prominence(_amplitude(smoothed), cfg)
    distance = max(1, round(cfg.refractory_s / dt))

    minima, _ = find_peaks(-smoothed, prominence=prominence, distance=distance)

    events: list[RepEvent] = []
    for rep_no, (start, end) in enumerate(_rep_windows(minima, smoothed), start=1):
        seg = angles[start:end + 1]
        feats = tuple(features[start:end + 1]) if features is not None else ()
        events.append(RepEvent(
            rep_number=rep_no,
            start_ts=float(timestamps[start]),
            end_ts=float(timestamps[end]),
            duration_s=float(timestamps[end] - timestamps[start]),
            peak_angle=float(seg.max()),
            trough_angle=float(seg.min()),
            trajectory=seg.tolist(),
            exercise_name=exercise,
            age_group=age_group,
            feature_history=feats,
        ))
    logger.debug("batch segmenter: %d reps (prominence=%.1f, distance=%d)",
                 len(events), prominence, distance)
    return events


def count_reps_batch(
    angles: np.ndarray,
    timestamps: np.ndarray,
    exercise: str,
    age_group: str = "adult",
    *,
    config: RepSegmenterConfig | None = None,
) -> int:
    """Convenience: number of reps detected by :func:`segment_reps_batch`."""
    return len(segment_reps_batch(angles, timestamps, exercise, age_group, config=config))


# ---------------------------------------------------------------------------
# Streaming path — online prominence detector (used by the live feed)
# ---------------------------------------------------------------------------

class StreamingRepSegmenter:
    """Online rep counter: one :meth:`update_angle` call per frame.

    Mirrors the :class:`~core.rep_counter.BaseRepCounter` per-frame interface
    (``update_angle``/``reset``/``rep_count``) so it can drop into the live path.
    A rep is confirmed when the rolling-median angle, after descending at least
    ``prominence`` below a preceding peak, **rebounds** at least ``prominence``
    above its trough, with the refractory period satisfied.

    Cold start: until the rolling amplitude window spans ``amplitude_window_s``,
    prominence falls back to ``prominence_floor_deg`` (never an undefined
    ``p95 - p5``), so the opening reps of a live session still count.
    """

    def __init__(
        self,
        exercise: str,
        age_group: str = "adult",
        config: RepSegmenterConfig | None = None,
    ) -> None:
        self._exercise = exercise
        self._age_group = age_group
        self._cfg = config or REP_SEGMENTER_CONFIGS.get(exercise, RepSegmenterConfig())
        self.reset()

    # -- public API ----------------------------------------------------------

    @property
    def rep_count(self) -> int:
        """Total reps confirmed so far."""
        return self._rep_count

    def reset(self) -> None:
        """Clear all streaming state."""
        self._raw: deque[float] = deque(maxlen=_MAX_SMOOTH_FRAMES)
        self._recent_ts: deque[float] = deque(maxlen=_FPS_EST_FRAMES)  # for live-fps estimate
        self._amp: deque[tuple[float, float]] = deque()  # (ts, smoothed)
        self._peak_ref = -np.inf
        self._trough = np.inf
        self._descended = False
        self._last_rep_ts = -np.inf
        self._rep_count = 0
        self._cur_start_ts: float | None = None
        self._cur_traj: list[float] = []
        self._cur_feats: list[FrameFeatures] = []
        # Last-update telemetry (read by debug_snapshot; no effect on counting).
        self._last_smoothed = float("nan")
        self._last_observed_fps = float("nan")
        self._last_window_frames = 0
        self._last_amp = float("nan")
        self._last_prominence = float("nan")

    def update_angle(
        self,
        angle: float,
        timestamp: float,
        *,
        features: FrameFeatures | None = None,
    ) -> RepEvent | None:
        """Feed one frame; return a :class:`RepEvent` if a rep just closed."""
        self._raw.append(float(angle))
        self._recent_ts.append(float(timestamp))
        # Live smoothing window in *frames*, derived from the observed fps so the
        # SAME smoothing_window_seconds adapts across regimes (~7 frames @30fps
        # batch, ~1 @~5fps live). fps is estimated from the FIRST available
        # inter-frame interval (converges within ~2-3 frames) and is NOT gated on
        # the amplitude window filling. The conservative streaming_fps_fallback is
        # used only on frame 1, where the median over a single sample is a no-op
        # anyway — so cold-start never over-smooths the first rep.
        if len(self._recent_ts) >= 2:
            dt = float(np.median(np.diff(np.fromiter(self._recent_ts, dtype=float))))
            observed_fps = (1.0 / dt) if dt > 1e-6 else self._cfg.streaming_fps_fallback
        else:
            observed_fps = self._cfg.streaming_fps_fallback
        window_frames = max(1, round(self._cfg.smoothing_window_seconds * observed_fps))
        smoothed = float(np.median(list(self._raw)[-window_frames:]))

        # Rolling amplitude window; cold-start uses the floor until it fills.
        self._amp.append((timestamp, smoothed))
        while self._amp and timestamp - self._amp[0][0] > self._cfg.amplitude_window_s:
            self._amp.popleft()
        span = timestamp - self._amp[0][0]
        if span >= self._cfg.amplitude_window_s:
            amp = _amplitude(np.fromiter((s for _, s in self._amp), dtype=float))
        else:
            amp = 0.0  # cold start -> prominence == floor
        prominence = _prominence(amp, self._cfg)

        # Stash this frame's computed internals for opt-in live tracing.
        self._last_observed_fps = observed_fps
        self._last_window_frames = window_frames
        self._last_smoothed = smoothed
        self._last_amp = amp
        self._last_prominence = prominence

        # Per-rep accumulation.
        if self._cur_start_ts is None:
            self._cur_start_ts = timestamp
        self._cur_traj.append(float(angle))
        if features is not None:
            self._cur_feats.append(features)

        # Online prominence-minimum detection.
        if not self._descended:
            self._peak_ref = max(self._peak_ref, smoothed)
            if smoothed <= self._peak_ref - prominence:
                self._descended = True
                self._trough = smoothed
            return None

        self._trough = min(self._trough, smoothed)
        rebounded = smoothed >= self._trough + prominence
        refractory_ok = (timestamp - self._last_rep_ts) >= self._cfg.refractory_s
        if rebounded and refractory_ok:
            event = self._emit(timestamp)
            self._peak_ref = smoothed
            self._trough = np.inf
            self._descended = False
            self._last_rep_ts = timestamp
            self._cur_start_ts = timestamp
            self._cur_traj = [float(angle)]
            self._cur_feats = [features] if features is not None else []
            return event
        return None

    def debug_snapshot(self) -> dict[str, float | bool | int | None]:
        """Read-only view of the latest update's internals (for live tracing).

        Behaviour-neutral telemetry: the values the online detector computed on
        the most recent :meth:`update_angle` call, so a trace can show *why* a
        rep did or did not close (e.g. a near-zero ``window_frames`` reveals the
        smoothing collapsing at low fps). Non-finite sentinels (``±inf`` peak /
        trough before they are set, ``nan`` before the first frame) are mapped to
        ``None`` so the snapshot is strict-JSON-serialisable. Not part of the
        rep-counting contract.
        """
        def _finite(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        return {
            "smoothed": _finite(self._last_smoothed),
            "observed_fps": _finite(self._last_observed_fps),
            "window_frames": int(self._last_window_frames),
            "amplitude": _finite(self._last_amp),
            "prominence": _finite(self._last_prominence),
            "peak_ref": _finite(self._peak_ref),
            "trough": _finite(self._trough),
            "descended": bool(self._descended),
            "rep_count": int(self._rep_count),
        }

    # -- internals -----------------------------------------------------------

    def _emit(self, end_ts: float) -> RepEvent:
        """Build the :class:`RepEvent` for the just-completed rep window."""
        self._rep_count += 1
        seg = np.asarray(self._cur_traj, dtype=float)
        return RepEvent(
            rep_number=self._rep_count,
            start_ts=float(self._cur_start_ts if self._cur_start_ts is not None else end_ts),
            end_ts=float(end_ts),
            duration_s=float(end_ts - (self._cur_start_ts or end_ts)),
            peak_angle=float(seg.max()),
            trough_angle=float(seg.min()),
            trajectory=self._cur_traj.copy(),
            exercise_name=self._exercise,
            age_group=self._age_group,
            feature_history=tuple(self._cur_feats),
        )


__all__ = ["segment_reps_batch", "count_reps_batch", "StreamingRepSegmenter", "smooth_angles"]
