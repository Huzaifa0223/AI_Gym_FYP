"""
core.score_aggregator — fuse rule-engine, RandomForest, and CNN+LSTM
scores into a single :class:`HeadlineScore`.

Three weight regimes selected by which inputs are non-``None``:

* **All three present** — uses :attr:`ScoreAggregatorWeights.with_cnn_lstm`
  (defaults: rules 0.4, rf 0.3, cnn_lstm 0.3).
* **CNN+LSTM None, RF present** — uses
  :attr:`ScoreAggregatorWeights.without_cnn_lstm` (defaults: rules 0.6,
  rf 0.4).
* **Both RF and CNN+LSTM None** — rules-only: ``final == rules_score``.
  No renormalisation; the rules signal is treated as authoritative.

``source_breakdown`` reflects exactly which inputs contributed: the
weighted contribution as an int for present inputs, ``None`` for absent
ones. The final score is clamped to ``[0, 100]`` defensively before
constructing :class:`HeadlineScore` — Pydantic also enforces the bound,
but rounding from a float weighted-sum can drift past it.

``form_feedback``: first message in *feedback_messages* wins; empty list
yields ``""``. Documented behaviour rather than an emergent property —
later stages may want to escalate severity-tagged messages, but Stage 5b
ships the simplest contract.

Pure function. No class, no state, no I/O.
"""
from __future__ import annotations

import logging

from core.config import ScoreAggregatorWeights
from core.schemas import HeadlineScore

logger = logging.getLogger(__name__)


def aggregate(
    rules_score: int,
    rf_score: int | None,
    cnn_lstm_score: int | None,
    rep_count: int,
    feedback_messages: list[str],
    weights: ScoreAggregatorWeights = ScoreAggregatorWeights(),
) -> HeadlineScore:
    """Fuse the three signals into a frontend-facing :class:`HeadlineScore`.

    Args:
        rules_score:        Heuristic rules score in ``[0, 100]``. Always
                            present.
        rf_score:           RandomForest score in ``[0, 100]`` or ``None``
                            when no model could be loaded.
        cnn_lstm_score:     CNN+LSTM score in ``[0, 100]`` or ``None``
                            when weights are missing or inference failed.
        rep_count:          Total reps detected — passes through unchanged.
        feedback_messages:  Optional list of human-readable feedback
                            strings. The first non-empty message wins;
                            empty list yields ``""``.
        weights:            Weight profiles for the two-signal and
                            three-signal regimes.

    Returns:
        A :class:`HeadlineScore` with ``score`` clamped to ``[0, 100]``,
        ``source_breakdown`` reflecting only the contributing inputs,
        and ``form_feedback`` set per the first-wins rule.
    """
    rules_present = True  # rules signal is always required to be present
    rf_present = rf_score is not None
    cnn_lstm_present = cnn_lstm_score is not None

    breakdown: dict[str, int | None] = {
        "rules": int(rules_score) if rules_present else None,
        "rf": int(rf_score) if rf_present else None,
        "cnn_lstm": int(cnn_lstm_score) if cnn_lstm_present else None,
    }

    if rf_present and cnn_lstm_present:
        w = weights.with_cnn_lstm
        weighted = (
            rules_score * w.rules
            + (rf_score or 0) * w.rf
            + (cnn_lstm_score or 0) * w.cnn_lstm
        )
    elif rf_present:
        w_no_nn = weights.without_cnn_lstm
        weighted = (
            rules_score * w_no_nn.rules
            + (rf_score or 0) * w_no_nn.rf
        )
    else:
        # Neither RF nor CNN+LSTM available. Rules-only — no renormalisation.
        weighted = float(rules_score)

    clamped = max(0, min(100, round(weighted)))

    form_feedback = feedback_messages[0] if feedback_messages else ""

    return HeadlineScore(
        rep_count=rep_count,
        form_feedback=form_feedback,
        score=clamped,
        source_breakdown=breakdown,
    )


__all__ = ["aggregate"]
