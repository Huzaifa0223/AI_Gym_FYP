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

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from core.rep_counter import RepEvent

logger = logging.getLogger(__name__)


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


_MIN_FRAMES = 4  # reps shorter than this carry no usable trajectory to model


class CnnLstmModelNotAvailableError(RuntimeError):
    """Raised when the trained CNN+LSTM weights cannot be loaded."""


class CnnLstmFormScorer:
    """Real :class:`FormQualityScorer` backed by a trained CNN+LSTM.

    Torch is imported **lazily** in :meth:`__init__` so merely importing this
    module (for :class:`NullFormQualityScorer`) never pays the torch cost — the
    same pattern as :class:`core.object_detector.Yolov8NanoDetector`.

    Scores a rep's ``feature_history`` (per-frame primary/secondary/tertiary
    angles) as one temporal sequence resampled to a fixed window. Returns
    ``None`` when the rep is too short to carry a trajectory, so the aggregator
    drops the CNN+LSTM contribution for that rep instead of fusing noise.

    Args:
        weights_path: Path to the trained ``.pt`` state-dict produced by
            ``python -m training.train_cnn_lstm``.

    Raises:
        CnnLstmModelNotAvailableError: If the weights file is missing or fails
            to load. The FastAPI lifespan catches this and falls back to
            :class:`NullFormQualityScorer`, so the API never 500s.
    """

    def __init__(self, weights_path: Path) -> None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise CnnLstmModelNotAvailableError(
                f"CNN+LSTM weights not found: {weights_path}. "
                f"Run `python -m training.train_cnn_lstm`."
            )
        try:
            import torch  # lazy import — keep module-level import torch-free
            from core.cnn_lstm_model import FormQualityNet

            self._torch = torch
            model = FormQualityNet()
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            model.eval()
            self._model = model
        except CnnLstmModelNotAvailableError:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            # Any torch/load failure is non-fatal to the app: surface a typed
            # error so the lifespan can fall back to the null scorer.
            raise CnnLstmModelNotAvailableError(
                f"Failed to load CNN+LSTM weights from {weights_path}: {exc}"
            ) from exc
        logger.info("CnnLstmFormScorer loaded weights from %s", weights_path)

    def score(self, rep: RepEvent) -> int | None:
        from core.cnn_lstm_model import ANGLE_SCALE, SEQ_LEN, resample_sequence

        if len(rep.feature_history) < _MIN_FRAMES:
            return None
        seq = np.array(
            [[f.primary_angle, f.secondary_angle, f.tertiary_angle]
             for f in rep.feature_history],
            dtype=np.float64,
        )
        seq = resample_sequence(seq, SEQ_LEN) / ANGLE_SCALE
        torch = self._torch
        with torch.no_grad():
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, T, 3)
            prob = float(torch.sigmoid(self._model(x)).item())
        return int(round(max(0.0, min(1.0, prob)) * 100.0))


__all__ = [
    "FormQualityScorer",
    "NullFormQualityScorer",
    "CnnLstmFormScorer",
    "CnnLstmModelNotAvailableError",
]
