"""
core.live_session — per-session state for the live per-frame scoring path.

The batch :class:`~core.pipeline.ScoringPipeline` is whole-video only, so the
live camera feed (``POST /api/live-frame``) cannot reuse it. This module holds
the **stateful** pieces a live session needs, one instance per session:

* :class:`~core.rep_segmenter.StreamingRepSegmenter` — online rep counting,
  driven one frame at a time via :meth:`process_features`.
* :class:`~core.form_scorer.FormScorer` + :func:`~core.score_aggregator.aggregate`
  — run **only when a rep closes** (not per frame), fusing rules + RandomForest
  (+ CNN+LSTM when enabled) into a single 0-100 score, exactly as the batch
  headline does.

Cadence (intentional): ``counter`` and ``primary_angle`` update every frame;
``rep_quality`` and ``feedback`` update only on rep close and **hold last-known**
between reps. Before the first rep, ``rep_quality`` is ``None`` (the UI renders a
neutral "—", never a misleading 0%) and ``feedback`` is :data:`NEUTRAL_FEEDBACK`.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from config import RepSegmenterConfig
from core.cnn_lstm_scorer import FormQualityScorer, NullFormQualityScorer
from core.form_scorer import FormScorer
from core.rep_counter import FrameFeatures, RepEvent
from core.rep_segmenter import StreamingRepSegmenter
from core.score_aggregator import aggregate

logger = logging.getLogger(__name__)

#: Shown until the first rep closes (no score exists yet — not a 0%).
NEUTRAL_FEEDBACK = "Form score appears after your first rep"
#: Shown on a clean rep close with no violations.
GOOD_REP_FEEDBACK = "Good rep — keep going"


@dataclass(frozen=True)
class LiveFrameResult:
    """Outcome of feeding one frame to a :class:`LiveSession`.

    Attributes:
        counter:       Running rep count (every frame).
        primary_angle: This frame's primary joint angle (every frame).
        rep_quality:   Fused 0-100 score; ``None`` until the first rep closes,
                       then held between reps.
        feedback:      Coaching string; neutral pre-first-rep, then held.
        rep_closed:    ``True`` iff a rep closed on this frame.
    """
    counter: int
    primary_angle: float
    rep_quality: float | None
    feedback: str
    rep_closed: bool


class LiveSession:
    """Stateful live-scoring context for a single workout session.

    Args:
        exercise:        Exercise key (``'bicep_curl'`` / ``'bent_over_row'`` /
                         ``'push_up'``).
        age_group:       Age group key (``'children'`` / ``'adult'`` /
                         ``'senior'``).
        cnn_lstm_scorer: Optional CNN+LSTM scorer (defaults to the null scorer,
                         which returns ``None`` so the aggregator uses the
                         rules+RF weight profile).
        config:          Optional :class:`RepSegmenterConfig` override for the
                         streaming segmenter; ``None`` selects the per-exercise
                         entry in ``REP_SEGMENTER_CONFIGS``.
    """

    def __init__(
        self,
        exercise: str,
        age_group: str = "adult",
        *,
        cnn_lstm_scorer: FormQualityScorer | None = None,
        config: RepSegmenterConfig | None = None,
    ) -> None:
        self.exercise = exercise
        self.age_group = age_group
        self._segmenter = StreamingRepSegmenter(exercise, age_group, config=config)
        self._scorer = FormScorer(exercise, age_group)
        self._cnn: FormQualityScorer = cnn_lstm_scorer or NullFormQualityScorer()
        self.last_form_score: float | None = None
        self.last_feedback: str = NEUTRAL_FEEDBACK

    @property
    def rep_count(self) -> int:
        """Reps confirmed so far (held across no-pose frames)."""
        return self._segmenter.rep_count

    def process_features(
        self, features: FrameFeatures, timestamp: float
    ) -> LiveFrameResult:
        """Feed one frame of features; score the rep if one just closed.

        Args:
            features:  This frame's :class:`FrameFeatures` (from
                       :func:`core.video_processor.extract_features`).
            timestamp: Monotonic seconds; the segmenter uses only differences.

        Returns:
            A :class:`LiveFrameResult` with the per-frame and (held) per-rep
            fields.
        """
        event = self._segmenter.update_angle(
            features.primary_angle, timestamp, features=features
        )
        if event is not None:
            self._score_rep(event)
        return LiveFrameResult(
            counter=self._segmenter.rep_count,
            primary_angle=features.primary_angle,
            rep_quality=self.last_form_score,
            feedback=self.last_feedback,
            rep_closed=event is not None,
        )

    # -- internals -----------------------------------------------------------

    def _score_rep(self, event: RepEvent) -> None:
        """Score a just-closed rep and update the held score/feedback.

        Mirrors :meth:`core.pipeline.ScoringPipeline._build_headline` per-rep
        contribution logic so the live score matches the batch headline weighting
        (rules 0.6 / rf 0.4 with CNN+LSTM off; rules 0.4 / rf 0.3 / nn 0.3 when on).
        """
        score = self._scorer.score(event)
        rules_score = max(0, min(100, round(100.0 - score.rule_penalty)))
        rf_score = (
            int(round(score.ml_score))
            if score.ml_available and not math.isnan(score.ml_score)
            else None
        )
        cnn = self._cnn.score(event)
        cnn_score = int(cnn) if cnn is not None else None
        messages = [v.message for v in score.violations]

        headline = aggregate(
            rules_score=rules_score,
            rf_score=rf_score,
            cnn_lstm_score=cnn_score,
            rep_count=self._segmenter.rep_count,
            feedback_messages=messages,
        )
        self.last_form_score = float(headline.score)
        self.last_feedback = headline.form_feedback or GOOD_REP_FEEDBACK


__all__ = ["LiveSession", "LiveFrameResult", "NEUTRAL_FEEDBACK", "GOOD_REP_FEEDBACK"]
