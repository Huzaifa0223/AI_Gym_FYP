"""
core.config — 3-pillar pipeline configuration objects.

Three frozen dataclasses live here:

* :class:`LatencyBudget`           — per-frame latency targets (spec §4).
* :class:`PrimaryUserConfig`       — primary-user-tracker hysteresis params.
* :class:`ScoreAggregatorWeights`  — score-fusion weights for the hybrid
                                     scoring path (see
                                     ``docs/architecture_3pillar.md`` §1.2 / §6).

Distinct from the project's existing root-level ``config.py``, which holds
RandomForest model paths and rep-counter thresholds. These objects are
3-pillar-pipeline-specific and intentionally namespaced under ``core/``.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Latency budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LatencyBudget:
    """Per-frame latency targets.

    Attributes:
        per_frame_ms_p95:  p95 budget for the synchronous per-frame path
                           (MediaPipe → user-tracker → rule engines →
                           MLAggregator → score aggregator).
        hard_limit_ms:     Absolute ceiling per spec §3 ("≤ 100 ms end-to-end
                           per frame"). Per-frame work that exceeds this
                           must be deferred to a thread-pool executor.
    """
    per_frame_ms_p95: float = 90.0
    hard_limit_ms: float = 100.0


# ---------------------------------------------------------------------------
# Primary-user tracker
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrimaryUserConfig:
    """Primary-user-tracker hysteresis parameters.

    Attributes:
        bbox_hysteresis_frames:    Frames a candidate must dominate before
                                   the lock can transfer to a new track.
                                   Prevents lock flicker between two
                                   similarly-sized people.
        lock_release_after_frames: Frames a locked track may stay missing
                                   before the lock is released to a new
                                   candidate. Tolerates brief MediaPipe
                                   detection drops without surrendering
                                   the lock.
        min_bbox_area_ratio:       The largest pose bbox must cover at
                                   least this fraction of the frame area
                                   to acquire the lock — prevents
                                   background bystanders from being
                                   acquired when the actual user steps
                                   out of frame.
    """
    bbox_hysteresis_frames: int = 5
    lock_release_after_frames: int = 30
    min_bbox_area_ratio: float = 0.05


# ---------------------------------------------------------------------------
# Score aggregator weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WithCnnLstmWeights:
    """Score weights when the CNN+LSTM signal is available (spec §1.2 / §6).

    Fields sum to 1.0.
    """
    rules: float = 0.4
    rf: float = 0.3
    cnn_lstm: float = 0.3


@dataclass(frozen=True)
class WithoutCnnLstmWeights:
    """Score weights when the CNN+LSTM signal is unavailable (spec §1.2 / §6).

    Fields sum to 1.0.
    """
    rules: float = 0.6
    rf: float = 0.4


@dataclass(frozen=True)
class ScoreAggregatorWeights:
    """Pair of weighting profiles used by the ``ScoreAggregator``.

    The aggregator selects :attr:`with_cnn_lstm` when the neural form scorer
    contributes a signal, otherwise :attr:`without_cnn_lstm`. The field
    values inside each inner dataclass sum to 1.0 (asserted in tests).
    """
    with_cnn_lstm: WithCnnLstmWeights = WithCnnLstmWeights()
    without_cnn_lstm: WithoutCnnLstmWeights = WithoutCnnLstmWeights()


__all__ = [
    "LatencyBudget",
    "PrimaryUserConfig",
    "WithCnnLstmWeights",
    "WithoutCnnLstmWeights",
    "ScoreAggregatorWeights",
]
