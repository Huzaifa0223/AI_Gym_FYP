"""
core.ml_aggregator — Aggregate per-frame ML predictions into a single rep-level
form score.

The trained Random Forest models classify each frame as good-form (1) or
bad-form (0) using three joint-angle features.  This module loads the model
once and scores an entire :class:`~core.rep_counter.RepEvent` by averaging
``predict_proba`` across all frames in its ``feature_history``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from pickle import UnpicklingError

import joblib
import numpy as np

from config import model_path
from core.rep_counter import FrameFeatures, RepEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ModelNotAvailableError(Exception):
    """Raised when the ``.pkl`` model file cannot be loaded."""


class InsufficientDataError(Exception):
    """Raised when a :class:`RepEvent` has no ``feature_history``."""


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class MLAggregator:
    """Load a trained ``.pkl`` model and score a completed rep.

    Args:
        exercise:  Exercise key (``'bicep_curl'``, ``'bent_over_row'``,
                   ``'push_up'``).
        age_group: Age group key (``'children'``, ``'adult'``, ``'senior'``).

    Raises:
        ModelNotAvailableError: If the model file does not exist or fails to
            load.
    """

    def __init__(self, exercise: str, age_group: str) -> None:
        self._exercise = exercise
        self._age_group = age_group
        self._model = self._load_model(model_path(exercise, age_group))

    # -- Public API ----------------------------------------------------------

    def score(self, rep: RepEvent) -> tuple[float, float]:
        """Score a completed rep using the trained model.

        Runs ``predict_proba`` on every frame in *rep.feature_history* and
        returns the mean good-form probability scaled to 0-100.

        Args:
            rep: A completed :class:`RepEvent` with ``feature_history``.

        Returns:
            ``(ml_score, confidence)`` where *ml_score* is in ``[0, 100]``
            and *confidence* is the mean of the max per-frame probability
            (in ``[0, 1]``).

        Raises:
            InsufficientDataError: If ``rep.feature_history`` is empty.
        """
        if not rep.feature_history:
            raise InsufficientDataError(
                f"RepEvent #{rep.rep_number} has no feature_history — "
                f"cannot compute ML score"
            )

        features_2d = np.array(
            [[f.primary_angle, f.secondary_angle, f.tertiary_angle]
             for f in rep.feature_history],
            dtype=np.float64,
        )

        # Trim or pad columns to match model's expected feature count
        n_expected = getattr(self._model, 'n_features_in_', 3)
        if features_2d.shape[1] > n_expected:
            features_2d = features_2d[:, :n_expected]

        probas = self._model.predict_proba(features_2d)  # shape (N, 2)

        # Class 1 = good form
        good_idx = min(1, probas.shape[1] - 1)
        good_probs = probas[:, good_idx]

        ml_score = float(np.mean(good_probs)) * 100.0
        confidence = float(np.mean(np.max(probas, axis=1)))

        logger.debug(
            "ML score for %s rep #%d: %.1f (confidence %.2f, %d frames)",
            self._exercise, rep.rep_number, ml_score, confidence,
            len(rep.feature_history),
        )
        return ml_score, confidence

    # -- Internals -----------------------------------------------------------

    @staticmethod
    def _load_model(path: Path) -> object:
        """Load a sklearn model from a ``.pkl`` file.

        The file may contain either a bare model or a dict with a
        ``'model'`` key (the project convention).
        """
        if not path.exists():
            raise ModelNotAvailableError(f"Model file not found: {path}")

        try:
            raw = joblib.load(path)
        except (UnpicklingError, EOFError, OSError, ValueError) as exc:
            raise ModelNotAvailableError(
                f"Failed to load model from {path}: {exc}"
            ) from exc

        if isinstance(raw, dict):
            model = raw.get('model')
            if model is None:
                raise ModelNotAvailableError(
                    f"Model dict at {path} has no 'model' key"
                )
            return model

        return raw
