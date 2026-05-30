"""
training.train_cnn_lstm — train the CNN+LSTM form-quality scorer.

Pillar-2 of the 3-pillar pipeline. Produces ``data/models/cnn_lstm_form_scorer.pt``
(a torch ``state_dict``), consumed by :class:`core.cnn_lstm_scorer.CnnLstmFormScorer`.

Data strategy (honest, see ``docs/ML_FACTS.md``)
-----------------------------------------------
* **Train on synthetic rep *sequences*.** Each sample is a time series of the
  three joint angles for one rep. "Good" reps are smooth, full-ROM, symmetric.
  "Bad" reps are biased toward **temporal** defects the rule engine and
  RandomForest cannot see — mid-rep hesitation, jerky velocity, and
  eccentric/concentric asymmetry — plus some spatial defects (partial ROM,
  swing / plank-break) so the model is not one-dimensional. This is what
  justifies an LSTM over the existing per-frame pillars.
* **Validate against real data.** After training we segment the 62 real
  good-form bicep clips (``data/training/bicep_adult_good_form_*.csv``) into
  per-rep sequences with the production rep counter and score them. We have no
  real *bad-form* data, so this measures only "does it recognise real good
  form?" — not accuracy. Reported honestly.

Run::

    python -m training.train_cnn_lstm
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import numpy as np
import torch
from scipy.signal import find_peaks
from torch import nn

import config
from core.cnn_lstm_model import ANGLE_SCALE, SEQ_LEN, FormQualityNet, resample_sequence
from utils.angle_utils import calculate_angle_3d

logger = logging.getLogger(__name__)

SEED: Final[int] = 42
SAMPLES_PER_CLASS_PER_EX: Final[int] = 1500
EPOCHS: Final[int] = 25
BATCH: Final[int] = 256
LR: Final[float] = 1e-3
GOOD_THRESHOLD: Final[int] = 60  # score at/above which a rep is "passing"
# Real-clip rep segmentation for validation — peak/turnaround based, NOT the
# production FSM (which under-segments real video: 0 reps on 45/62 clips because
# it only closes a rep when the arm re-extends past ~130 deg; see ML_FACTS sec 5).
# A curl = a primary-angle minimum (the curl bottom); window a fixed span around it.
REAL_BOTTOM_MAX: Final[float] = 80.0  # primary below this counts as a curl bottom
REAL_MIN_SEP: Final[int] = 12         # min frames between adjacent bottoms
REAL_HALF_WIN: Final[int] = 18        # half-window (frames) sampled around a bottom

# Per-exercise sequence priors (degrees), expressed as full-rep trajectories.
# Bicep good-form secondary/tertiary baselines are CALIBRATED to the 62 real
# good-form clips (measured: secondary mean~30, tertiary mean~53, ROM~105) so the
# synthetic "good" class matches what real MediaPipe produces — without this the
# model rejects real good reps (see docs/ML_FACTS.md sim-to-real note). back /
# push_up reuse the same-geometry baselines (no real data to calibrate against).
EXERCISE_SEQ: Final[dict[str, dict[str, float]]] = {
    "bicep_curl":    {"top": 158, "bottom": 45, "sec_good": 30,  "sec_bad": 85,  "ter_good": 53},
    "bent_over_row": {"top": 158, "bottom": 45, "sec_good": 32,  "sec_bad": 85,  "ter_good": 50},
    "push_up":       {"top": 170, "bottom": 80, "sec_good": 177, "sec_bad": 150, "ter_good": 177},
}
# Per-frame noise on the secondary/tertiary channels — wide, matching the large
# real-data spread (real good-form secondary std~21, tertiary std~28). Kept wide
# on purpose so the model leans on the *primary trajectory shape* (the temporal
# signal) rather than secondary magnitude, except for the explicit "swing" defect.
SEC_NOISE: Final[float] = 10.0
TER_NOISE: Final[float] = 15.0
# Primary-channel jitter. Real MediaPipe good-form reps show ~15 deg frame-to-frame
# std (measured on the 62 clips), so "good" must carry this much noise or real reps
# read as "jerky". The jerky DEFECT therefore has to be markedly noisier still.
PRIM_NOISE: Final[float] = 13.0
JERK_NOISE: Final[float] = 30.0

# Bad-form defect mix — weighted toward temporal defects (~70%).
DEFECTS: Final[tuple[tuple[str, float], ...]] = (
    ("hitch", 0.25),    # temporal: mid-rep hesitation / plateau
    ("jerky", 0.25),    # temporal: high-frequency velocity noise
    ("asym", 0.20),     # temporal: eccentric/concentric asymmetry
    ("partial", 0.15),  # spatial: reduced ROM
    ("swing", 0.15),    # spatial: secondary-angle drift (swing / plank-break)
)


# ---------------------------------------------------------------------------
# Synthetic sequence generation
# ---------------------------------------------------------------------------

def _good_sequence(cfg: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    """One smooth, full-ROM, symmetric rep → ``(n, 3)`` angle sequence."""
    n = int(rng.integers(28, 56))
    t = np.linspace(0.0, 1.0, n)
    mid = (cfg["top"] + cfg["bottom"]) / 2.0
    amp = (cfg["top"] - cfg["bottom"]) / 2.0
    primary = mid + amp * np.cos(2 * np.pi * t) + rng.normal(0, PRIM_NOISE, n)
    secondary = cfg["sec_good"] + rng.normal(0, SEC_NOISE, n)
    tertiary = cfg["ter_good"] + rng.normal(0, TER_NOISE, n)
    return np.column_stack([primary, secondary, tertiary])


def _bad_sequence(cfg: dict[str, float], defect: str, rng: np.random.Generator) -> np.ndarray:
    """One bad rep with the named defect → ``(n, 3)`` angle sequence."""
    n = int(rng.integers(28, 56))
    t = np.linspace(0.0, 1.0, n)
    mid = (cfg["top"] + cfg["bottom"]) / 2.0
    amp = (cfg["top"] - cfg["bottom"]) / 2.0
    secondary = cfg["sec_good"] + rng.normal(0, SEC_NOISE, n)
    tertiary = cfg["ter_good"] + rng.normal(0, TER_NOISE, n)

    if defect == "partial":                       # spatial: never reaches bottom
        primary = mid + amp * 0.45 * np.cos(2 * np.pi * t) + (amp * 0.45) + rng.normal(0, PRIM_NOISE, n)
    elif defect == "swing":                        # spatial: secondary drifts off baseline
        ramp = np.abs(np.sin(np.pi * t))
        secondary = cfg["sec_good"] + (cfg["sec_bad"] - cfg["sec_good"]) * ramp + rng.normal(0, SEC_NOISE, n)
        primary = mid + amp * np.cos(2 * np.pi * t) + rng.normal(0, PRIM_NOISE, n)
    elif defect == "jerky":                        # temporal: stuttering velocity
        primary = mid + amp * np.cos(2 * np.pi * t) + rng.normal(0, JERK_NOISE, n)
    elif defect == "asym":                         # temporal: fast down, slow up (warped phase)
        k = rng.uniform(1.8, 2.6)
        warped = np.where(t < 0.5, 0.5 * (2 * t) ** k, 1.0 - 0.5 * (2 * (1 - t)) ** (1.0 / k))
        primary = mid + amp * np.cos(2 * np.pi * warped) + rng.normal(0, PRIM_NOISE, n)
    else:                                          # "hitch" — temporal: mid-rep hold/plateau
        primary = mid + amp * np.cos(2 * np.pi * t) + rng.normal(0, PRIM_NOISE, n)
        c = int(rng.integers(int(0.3 * n), int(0.55 * n)))
        w = max(2, int(0.18 * n))
        primary[c:c + w] = primary[c]
    return np.column_stack([primary, secondary, tertiary])


def _pick_defect(rng: np.random.Generator) -> str:
    names = [d for d, _ in DEFECTS]
    probs = np.array([p for _, p in DEFECTS])
    return str(rng.choice(names, p=probs / probs.sum()))


def build_dataset(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate the pooled synthetic dataset → ``(X, y)``.

    ``X`` is ``(N, SEQ_LEN, 3)`` (each rep resampled to a fixed window and scaled
    by ``ANGLE_SCALE``); ``y`` is ``(N,)`` with 1 = good, 0 = bad.
    """
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for cfg in EXERCISE_SEQ.values():
        for _ in range(SAMPLES_PER_CLASS_PER_EX):
            xs.append(resample_sequence(_good_sequence(cfg, rng), SEQ_LEN))
            ys.append(1)
            xs.append(resample_sequence(_bad_sequence(cfg, _pick_defect(rng), rng), SEQ_LEN))
            ys.append(0)
    x = np.stack(xs).astype(np.float32) / ANGLE_SCALE
    y = np.array(ys, dtype=np.float32)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(x: np.ndarray, y: np.ndarray) -> tuple[FormQualityNet, float]:
    """Train :class:`FormQualityNet`; return ``(model, val_accuracy)``."""
    torch.manual_seed(SEED)
    n_val = int(0.2 * len(y))
    xt = torch.from_numpy(x[n_val:]); yt = torch.from_numpy(y[n_val:])
    xv = torch.from_numpy(x[:n_val]); yv = torch.from_numpy(y[:n_val])

    model = FormQualityNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        for i in range(0, len(yt), BATCH):
            xb, yb = xt[i:i + BATCH], yt[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                acc = ((torch.sigmoid(model(xv)) >= 0.5).float() == yv).float().mean().item()
            logger.info("epoch %2d/%d  val_acc=%.4f", epoch + 1, EPOCHS, acc)

    model.eval()
    with torch.no_grad():
        val_acc = ((torch.sigmoid(model(xv)) >= 0.5).float() == yv).float().mean().item()
    return model, val_acc


def _score_sequence(model: FormQualityNet, seq: np.ndarray) -> int:
    """Score one ``(n, 3)`` angle sequence → 0-100 (same path as the scorer)."""
    s = resample_sequence(seq, SEQ_LEN).astype(np.float32) / ANGLE_SCALE
    with torch.no_grad():
        prob = float(torch.sigmoid(model(torch.from_numpy(s).unsqueeze(0))).item())
    return int(round(prob * 100))


# ---------------------------------------------------------------------------
# Real-data validation — 62 good-form bicep clips
# ---------------------------------------------------------------------------

def validate_on_real(model: FormQualityNet) -> None:
    """Segment the real good-form bicep clips into reps and score them.

    No real bad-form data exists, so this measures *recognition of real good
    form only* — never reported as accuracy. Skips quietly if the CSV is absent.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas unavailable — skipping real-data validation")
        return

    matches = sorted(config.TRAINING_DIR.glob("bicep_adult_good_form_2026*.csv"))
    real = [p for p in matches if "synthetic" not in p.name]
    if not real:
        logger.warning("No real bicep good-form CSV found — skipping real validation")
        return

    df = pd.read_csv(real[0])
    if "augmentation" in df.columns:
        df = df[df["augmentation"] == "original"]
    if "video_file" not in df.columns:
        logger.warning("Real CSV has no 'video_file' column — skipping")
        return

    def _pt(row, i: int) -> np.ndarray:
        return np.array([row[f"landmark_{i}_x"], row[f"landmark_{i}_y"], row[f"landmark_{i}_z"]])

    scores: list[int] = []
    n_videos = 0
    for _, group in df.groupby("video_file"):
        n_videos += 1
        rows = [r for _, r in group.iterrows()]
        prim = np.array([calculate_angle_3d(_pt(r, 12), _pt(r, 14), _pt(r, 16)) for r in rows])
        sec = np.array([calculate_angle_3d(_pt(r, 24), _pt(r, 12), _pt(r, 14)) for r in rows])
        ter = np.array([calculate_angle_3d(_pt(r, 24), _pt(r, 12), _pt(r, 16)) for r in rows])
        # Curl bottoms = primary-angle minima — robust to incomplete ROM, unlike
        # the production FSM's fixed top-threshold close.
        minima, _ = find_peaks(-prim, height=-REAL_BOTTOM_MAX, distance=REAL_MIN_SEP)
        for m in minima:
            a, b = max(0, m - REAL_HALF_WIN), min(len(prim), m + REAL_HALF_WIN)
            if b - a < 6:
                continue
            seq = np.column_stack([prim[a:b], sec[a:b], ter[a:b]])
            scores.append(_score_sequence(model, seq))

    if not scores:
        logger.warning("Real validation: 0 reps segmented from %d videos", n_videos)
        return
    arr = np.array(scores)
    logger.info("=" * 60)
    logger.info("REAL-DATA validation (peak-segmented real reps; good-form only "
                "— recognition, not accuracy):")
    logger.info("  videos=%d  reps_segmented=%d", n_videos, len(arr))
    logger.info("  score mean=%.1f  median=%.1f  min=%d  max=%d",
                arr.mean(), np.median(arr), arr.min(), arr.max())
    logger.info("  %% reps scored >=%d (passing): %.1f%%",
                GOOD_THRESHOLD, 100.0 * (arr >= GOOD_THRESHOLD).mean())
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--output", type=Path,
                        default=config.MODELS_DIR / "cnn_lstm_form_scorer.pt")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    rng = np.random.default_rng(SEED)
    logger.info("Generating synthetic rep sequences...")
    x, y = build_dataset(rng)
    logger.info("dataset: %d sequences (%d good / %d bad), shape %s",
                len(y), int(y.sum()), int((1 - y).sum()), x.shape)

    model, val_acc = train(x, y)
    logger.info("synthetic held-out val accuracy: %.4f", val_acc)

    validate_on_real(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    logger.info("saved CNN+LSTM weights -> %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
