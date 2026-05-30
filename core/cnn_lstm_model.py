"""
core.cnn_lstm_model — PyTorch CNN+LSTM architecture for rep-level form scoring.

Imported **lazily** (only when a real :class:`CnnLstmFormScorer` is constructed
or the trainer runs) so that importing :mod:`core.cnn_lstm_scorer` — and
therefore the FastAPI app's :class:`NullFormQualityScorer` fallback — stays
torch-free and cheap.

The model consumes a fixed-length window of per-frame joint-angle features
(``SEQ_LEN`` timesteps × ``N_FEATURES`` channels) and emits a single logit for
P(good-form). A 1-D CNN extracts *local temporal* patterns across the angle
channels (velocity, smoothness, hitches); an LSTM integrates them over the
whole rep; a linear head maps the final hidden state to the logit.

The temporal pillar is deliberately complementary to the other two: the rule
engine and RandomForest see per-frame / aggregate angle thresholds, whereas the
CNN+LSTM sees *trajectory shape over time* — hesitations, jerky velocity, and
eccentric/concentric asymmetry that the other pillars cannot detect.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

SEQ_LEN: int = 32          # every rep is resampled to this many timesteps
N_FEATURES: int = 3        # primary, secondary, tertiary joint angles
ANGLE_SCALE: float = 180.0  # angles divided by this so inputs sit in ~[0, 1]


class FormQualityNet(nn.Module):
    """1-D CNN → LSTM → linear binary head. Tiny and CPU-friendly.

    Input  : ``(batch, SEQ_LEN, N_FEATURES)`` of angles already divided by
             ``ANGLE_SCALE``.
    Output : ``(batch,)`` raw logits — apply ``sigmoid`` for P(good-form).
             Training uses :class:`~torch.nn.BCEWithLogitsLoss` on the logit.
    """

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> Conv1d wants (B, C, T)
        z = self.conv(x.transpose(1, 2))   # (B, 32, T)
        z = z.transpose(1, 2)              # (B, T, 32)
        out, _ = self.lstm(z)              # (B, T, hidden)
        logit = self.head(out[:, -1, :])   # final timestep -> (B, 1)
        return logit.squeeze(-1)           # (B,)


def resample_sequence(seq: np.ndarray, target_len: int = SEQ_LEN) -> np.ndarray:
    """Linearly resample a ``(n, N_FEATURES)`` angle sequence to ``target_len``.

    Makes the model length-invariant: reps of any frame count map to a fixed
    ``SEQ_LEN`` window. Each feature channel is interpolated independently along
    time. A single-frame input is broadcast (degenerate but safe).

    Args:
        seq:        ``(n, N_FEATURES)`` array of per-frame angles (degrees).
        target_len: Output timestep count.

    Returns:
        ``(target_len, N_FEATURES)`` resampled array (float64).
    """
    seq = np.asarray(seq, dtype=np.float64)
    n = seq.shape[0]
    if n == target_len:
        return seq
    if n == 1:
        return np.repeat(seq, target_len, axis=0)
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, target_len)
    return np.stack(
        [np.interp(dst, src, seq[:, c]) for c in range(seq.shape[1])],
        axis=1,
    )


__all__ = ["FormQualityNet", "resample_sequence", "SEQ_LEN", "N_FEATURES", "ANGLE_SCALE"]
