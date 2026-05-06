"""
core.pipeline — End-to-end video scoring pipeline.

Composes the per-stage components shipped in earlier 3-pillar stages
into a single ``score(video_bytes, exercise, age_group) ->
PipelineResult`` call:

* :class:`core.video_processor.VideoProcessor` — frames → per-frame
  features (Stage 0/4 of the original audit).
* :func:`core.rep_counter.make_rep_counter` — frames → :class:`RepEvent`
  closures (Stage 1).
* :class:`core.form_scorer.FormScorer` — per-rep ML score + rule
  penalties (Stage 1, kept unchanged for the per-rep API contract).
* :class:`core.threshold_provider.ThresholdProvider` — fail-fast lookup
  to translate "no thresholds for this exercise" into a 422 at the
  HTTP layer (Stage 5b).
* :class:`core.cnn_lstm_scorer.FormQualityScorer` — optional per-rep
  CNN+LSTM score (Stage 5c seam; null until trained weights ship).
* :func:`core.score_aggregator.aggregate` — per-video headline fusion
  (Stage 5b).

This module owns the per-rep → per-video aggregation that produces
:class:`HeadlineScore`, leaving :class:`FormScorer` responsible only
for per-rep math. The split keeps the existing ``ScoreResponse.reps``
shape stable while ``ScoreResponse.headline`` is populated for the
first time.

The pipeline is **synchronous**. The FastAPI handler dispatches it via
``loop.run_in_executor`` so the event loop stays free; the executor
thread is the natural place to add CNN+LSTM ``predict()`` calls when a
real model lands.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from core.cnn_lstm_scorer import FormQualityScorer, NullFormQualityScorer
from core.config import ScoreAggregatorWeights
from core.form_scorer import FormScore, FormScorer
from core.rep_counter import RepEvent, make_rep_counter
from core.schemas import HeadlineScore
from core.score_aggregator import aggregate
from core.threshold_provider import ThresholdProvider
from core.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerRepScore:
    """Per-rep output, used by the API adapter to build ``RepResult``s."""
    event: RepEvent
    score: FormScore


@dataclass(frozen=True)
class PipelineResult:
    """End-to-end pipeline output.

    Attributes:
        per_rep:                List of (RepEvent, FormScore) pairs in
                                rep order — the API adapter converts
                                these into ``RepResult`` for
                                ``ScoreResponse.reps``.
        headline:               Per-video :class:`HeadlineScore`.
        average_score:          Mean of per-rep ``FormScore.final_score``
                                values that are not NaN; ``None`` when
                                no rep produced a valid score (preserves
                                existing ``ScoreResponse.average_score``
                                semantics).
        frames_processed:       Total decoded frames.
        frames_with_landmarks:  Subset of ``frames_processed`` for which
                                MediaPipe returned a pose.
        processing_time_s:      Wall-clock pipeline duration.
    """
    per_rep: list[PerRepScore]
    headline: HeadlineScore
    average_score: float | None
    frames_processed: int
    frames_with_landmarks: int
    processing_time_s: float


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ScoringPipeline:
    """Compose the per-stage components into a single video-to-headline call.

    Args:
        threshold_provider:  Used to fail-fast when an exercise has no
                             ``heuristic_thresholds.json`` /
                             ``seed_thresholds.json`` on disk. The
                             pipeline calls
                             :meth:`ThresholdProvider.load` *before*
                             touching the video so the API layer can
                             surface :class:`ThresholdsNotFoundError`
                             as a 422 instead of a generic 500.
        cnn_lstm_scorer:     Optional per-rep form-quality scorer. The
                             default :class:`NullFormQualityScorer`
                             always returns ``None``, which makes the
                             aggregator pick the ``without_cnn_lstm``
                             weight profile.
        weights:             Score-fusion weights. Defaults to spec §6
                             (``rules 0.4 / rf 0.3 / cnn_lstm 0.3`` when
                             all three present; ``rules 0.6 / rf 0.4``
                             otherwise).
    """

    def __init__(
        self,
        *,
        threshold_provider: ThresholdProvider | None = None,
        cnn_lstm_scorer: FormQualityScorer | None = None,
        weights: ScoreAggregatorWeights = ScoreAggregatorWeights(),
    ) -> None:
        self._threshold_provider = (
            threshold_provider if threshold_provider is not None
            else ThresholdProvider()
        )
        self._cnn_lstm_scorer: FormQualityScorer = (
            cnn_lstm_scorer if cnn_lstm_scorer is not None
            else NullFormQualityScorer()
        )
        self._weights = weights

    # -- Public API ----------------------------------------------------------

    def score(
        self,
        video_bytes: bytes,
        exercise: str,
        age_group: str,
    ) -> PipelineResult:
        """Run the full pipeline.

        Raises:
            ThresholdsNotFoundError: When neither
                ``heuristic_thresholds.json`` nor ``seed_thresholds.json``
                exists for *exercise*. The API adapter maps this to a
                422.
            UnsupportedExerciseError, VideoDecodeError,
                NoFramesExtractedError:  As propagated from
                :class:`VideoProcessor`. The API adapter maps these to
                400/500 as appropriate.
        """
        # 1. Pre-flight: ensure thresholds load. Fail fast before reading
        # the video bytes if the exercise has no thresholds shipped.
        self._threshold_provider.load(exercise)

        # 2. Run the per-frame loop, collect per-rep scores.
        start = time.monotonic()
        per_rep: list[PerRepScore] = []
        frames_processed = 0
        frames_with_landmarks = 0

        with VideoProcessor(exercise) as vp:
            counter = make_rep_counter(exercise, age_group)
            scorer = FormScorer(exercise, age_group)
            for pf in vp.process_bytes(video_bytes):
                frames_processed += 1
                if not pf.landmarks_detected:
                    continue
                frames_with_landmarks += 1
                event = counter.update_angle(
                    pf.primary_angle,
                    pf.timestamp,
                    features=pf.features,
                )
                if event is not None:
                    score = scorer.score(event)
                    per_rep.append(PerRepScore(event=event, score=score))

        processing_time_s = time.monotonic() - start

        # 3. Aggregate per-rep → per-video.
        headline = self._build_headline(per_rep)
        average_score = self._compute_average_score(per_rep)

        return PipelineResult(
            per_rep=per_rep,
            headline=headline,
            average_score=average_score,
            frames_processed=frames_processed,
            frames_with_landmarks=frames_with_landmarks,
            processing_time_s=processing_time_s,
        )

    # -- Aggregation ---------------------------------------------------------

    def _build_headline(self, per_rep: list[PerRepScore]) -> HeadlineScore:
        """Aggregate per-rep scores into a :class:`HeadlineScore`."""
        if not per_rep:
            # No reps detected — emit a defensible "0 score, no data"
            # headline. Aggregator's rules-only path returns a score of
            # *rules_score* (0 here), source_breakdown reflects the
            # all-None state.
            return aggregate(
                rules_score=0,
                rf_score=None,
                cnn_lstm_score=None,
                rep_count=0,
                feedback_messages=["No reps detected"],
                weights=self._weights,
            )

        per_rep_rules: list[int] = []
        per_rep_rf: list[int] = []
        per_rep_cnn: list[int] = []
        feedback_messages: list[str] = []

        for entry in per_rep:
            score = entry.score
            # Rules score per rep: 100 - rule_penalty, clamped.
            rules_score = max(0, min(100, round(100.0 - score.rule_penalty)))
            per_rep_rules.append(rules_score)

            if score.ml_available:
                per_rep_rf.append(int(round(score.ml_score)))

            cnn = self._cnn_lstm_scorer.score(entry.event)
            if cnn is not None:
                per_rep_cnn.append(int(cnn))

            for v in score.violations:
                feedback_messages.append(v.message)

        mean_rules = int(round(sum(per_rep_rules) / len(per_rep_rules)))
        mean_rf = (
            int(round(sum(per_rep_rf) / len(per_rep_rf)))
            if per_rep_rf else None
        )
        mean_cnn = (
            int(round(sum(per_rep_cnn) / len(per_rep_cnn)))
            if per_rep_cnn else None
        )

        return aggregate(
            rules_score=mean_rules,
            rf_score=mean_rf,
            cnn_lstm_score=mean_cnn,
            rep_count=len(per_rep),
            feedback_messages=feedback_messages,
            weights=self._weights,
        )

    @staticmethod
    def _compute_average_score(per_rep: list[PerRepScore]) -> float | None:
        """Mean of per-rep ``final_score``; ``None`` when no rep was scored.

        Preserves the existing ``ScoreResponse.average_score`` semantics
        from the old inline pipeline so the 8 existing /api/score tests
        continue to pass without changes.
        """
        import math
        valid = [
            r.score.final_score
            for r in per_rep
            if not math.isnan(r.score.final_score)
        ]
        return (sum(valid) / len(valid)) if valid else None


__all__ = ["PerRepScore", "PipelineResult", "ScoringPipeline"]
