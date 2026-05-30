"""tools.diagnose_rep_counter — Phase-1 diagnostic for the rep-counting under-count.

Runs the **current, unchanged** rep-counter FSM over the real extracted bicep clips
and writes per-frame traces to ``diagnostics/`` (gitignored) plus a console table and
an aggregate verdict. This is read-only with respect to ``core/rep_counter.py`` and
``config.py`` — it only characterises *why* the counter under-counts before any fix is
designed.

Data source: the landmarks were already extracted for the 62 real bicep clips into
``data/training/bicep_adult_good_form_*.csv`` (grouped by ``video_file``); we reuse them
rather than re-running MediaPipe.

The "peak-estimate" column is a `scipy.signal.find_peaks` reference only — it is **not**
ground truth. True rep counts are a human input (see ``calibration/bicep_ground_truth.csv``).

Usage::

    python -m tools.diagnose_rep_counter
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from config import REP_COUNTER_CONFIGS
from core.rep_counter import make_rep_counter
from utils.angle_utils import calculate_angle_3d

FPS: float = 30.0
DIAG_DIR: Path = Path("diagnostics")
GT_PATH: Path = Path("calibration") / "bicep_ground_truth.csv"
# Reference peak-estimate params (NOT ground truth): a 5-frame smooth then find_peaks
# on the inverted angle (curl bottoms), requiring a clear dip and minimum spacing.
SMOOTH_WIN: int = 5
PEAK_DISTANCE: int = 15      # min frames between curl bottoms (~0.5 s @ 30 fps)
PEAK_PROMINENCE: float = 8.0  # deg dip depth to count as a curl bottom


def _find_real_csv() -> Path:
    """Return the newest real (non-synthetic) extracted bicep CSV."""
    matches = [
        Path(p) for p in glob.glob("data/training/bicep_adult_good_form_*.csv")
        if "synthetic" not in p
    ]
    if not matches:
        raise FileNotFoundError(
            "No real bicep CSV under data/training/bicep_adult_good_form_*.csv"
        )
    return sorted(matches)[-1]


def _slug(name: str, idx: int) -> str:
    """Filesystem-safe trace filename from a clip name with spaces/emojis."""
    stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(name).stem)[:40].strip("_")
    return f"{idx:02d}_{stem or 'clip'}"


def _primary_angles(group: pd.DataFrame) -> np.ndarray:
    """Per-frame elbow angle shoulder(12)→elbow(14)→wrist(16), in clip order."""
    def pt(row: pd.Series, i: int) -> np.ndarray:
        return np.array([row[f"landmark_{i}_x"], row[f"landmark_{i}_y"],
                         row[f"landmark_{i}_z"]], dtype=float)

    return np.asarray(
        [calculate_angle_3d(pt(r, 12), pt(r, 14), pt(r, 16)) for _, r in group.iterrows()],
        dtype=float,
    )


def _run_current_counter(angles: np.ndarray) -> tuple[int, list[tuple[float, float, str, bool]]]:
    """Feed the angle series through the CURRENT FSM; return (count, trace rows)."""
    counter = make_rep_counter("bicep_curl", "adult")
    rows: list[tuple[float, float, str, bool]] = []
    for i, a in enumerate(angles):
        event = counter.update_angle(float(a), i / FPS)
        rows.append((round(i / FPS, 4), round(float(a), 2), counter.state.name, event is not None))
    return counter.rep_count, rows


def _peak_estimate(angles: np.ndarray) -> int:
    """Reference rep estimate: smoothed curl-bottom peak count. NOT ground truth."""
    if angles.size < SMOOTH_WIN:
        return 0
    kernel = np.ones(SMOOTH_WIN) / SMOOTH_WIN
    smoothed = np.convolve(angles, kernel, mode="same")
    minima, _ = find_peaks(-smoothed, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    return int(minima.size)


def main() -> int:
    cfg = REP_COUNTER_CONFIGS["bicep_curl"]
    top_enter = cfg.top_threshold - cfg.hysteresis
    bottom_enter = cfg.bottom_threshold + cfg.hysteresis

    csv = _find_real_csv()
    df = pd.read_csv(csv)
    if "augmentation" in df.columns:
        df = df[df["augmentation"] == "original"]
    DIAG_DIR.mkdir(exist_ok=True)
    GT_PATH.parent.mkdir(exist_ok=True)

    print(f"\nSource: {csv}")
    print(f"Current bicep gates: top_enter={top_enter:.0f}, bottom_enter={bottom_enter:.0f}, "
          f"reversal_margin={cfg.reversal_margin:.0f}, dur=[{cfg.min_rep_duration_s},"
          f"{cfg.max_rep_duration_s}]s\n")
    print(f"{'#':>2} {'clip':40s} {'det':>3} {'est':>3} {'min':>4} {'med':>4} {'max':>4} "
          f"{'jit':>4} {'<bot':>5} {'>=top':>6}  note")

    gt_rows: list[tuple[str, str, str]] = []
    tot_det = tot_est = n_zero = n_no_bottom = n_no_top = 0
    for idx, (clip, group) in enumerate(df.groupby("video_file"), 1):
        ang = _primary_angles(group)
        if ang.size < 3:
            continue
        det, rows = _run_current_counter(ang)
        pd.DataFrame(rows, columns=["timestamp", "primary_angle", "fsm_state", "rep_fired"]) \
            .to_csv(DIAG_DIR / f"{_slug(clip, idx)}.csv", index=False)

        amin, amed, amax = float(ang.min()), float(np.median(ang)), float(ang.max())
        jit = float(np.std(np.diff(ang)))
        est = _peak_estimate(ang)
        reach_bot, reach_top = amin <= bottom_enter, amax >= top_enter

        notes = []
        if not reach_bot:
            notes.append("never<bottom_enter")
        if not reach_top:
            notes.append("never>=top_enter")
        if jit > 12.0:
            notes.append("noisy")
        if not notes:
            notes.append("ok")

        tot_det += det
        tot_est += est
        n_zero += int(det == 0)
        n_no_bottom += int(not reach_bot)
        n_no_top += int(not reach_top)
        gt_rows.append((str(clip), "", ""))
        print(f"{idx:>2} {Path(str(clip)).stem[:40]:40s} {det:>3} {est:>3} {amin:>4.0f} "
              f"{amed:>4.0f} {amax:>4.0f} {jit:>4.1f} {str(reach_bot):>5} {str(reach_top):>6}"
              f"  {','.join(notes)}")

    n = len(gt_rows)
    print(f"\n--- AGGREGATE over {n} real bicep clips ---")
    print(f"  current FSM detected : {tot_det} reps")
    print(f"  peak-estimate (ref)  : {tot_est} reps   (smoothed find_peaks; NOT ground truth)")
    print(f"  clips with 0 detected reps                      : {n_zero}/{n}")
    print(f"  clips whose angle NEVER reaches bottom_enter={bottom_enter:.0f} : {n_no_bottom}/{n}")
    print(f"  clips whose angle NEVER reaches top_enter={top_enter:.0f}    : {n_no_top}/{n}")

    pd.DataFrame(gt_rows, columns=["clip_filename", "true_reps", "notes"]).to_csv(
        GT_PATH, index=False)
    print(f"\nWrote ground-truth template (true_reps BLANK — human fills): {GT_PATH}")
    print(f"Per-frame traces: {DIAG_DIR}/*.csv ({n} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
