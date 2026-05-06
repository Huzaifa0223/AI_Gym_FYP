"""
core.cnn_lstm_scorer — Optional CNN+LSTM form-quality scoring seam.

Defines the :class:`FormQualityScorer` Protocol consumed by
:class:`core.pipeline.ScoringPipeline`, plus :class:`NullFormQualityScorer`
— the production stand-in until trained weights ship.

The scorer takes a completed :class:`RepEvent` and returns either an
integer score in ``[0, 100]`` or ``None`` to indicate "scoring is not
available for this rep." The pipeline forwards ``None`` to
:func:`core.score_aggregator.aggregate`, which then drops the CNN+LSTM
weight from the fusion math.

A real implementation would load a trained CNN+LSTM model (lazy import,
similar to :class:`core.object_detector.Yolov8NanoDetector`) and score
the rep's ``feature_history`` or trajectory. That work is out of Stage
5c scope — current ship is the protocol + null implementation so the
pipeline contract is stable.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.rep_counter import RepEvent


@runtime_checkable
class FormQualityScorer(Protocol):
    """Score a completed :class:`RepEvent` on form quality.

    Returns ``None`` when scoring is unavailable (no trained model,
    inference failed, insufficient data) — pipeline drops the CNN+LSTM
    contribution in that case.
    """

    def score(self, rep: RepEvent) -> int | None: ...


class NullFormQualityScorer:
    """Always returns ``None``.

    Production stand-in until a trained CNN+LSTM model ships.
    Constructing this class is intentionally side-effect-free so the
    pipeline can default to it without paying the cost of loading
    anything.
    """

    def score(self, rep: RepEvent) -> int | None:
        return None


__all__ = ["FormQualityScorer", "NullFormQualityScorer"]
