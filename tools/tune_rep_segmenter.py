"""tools.tune_rep_segmenter — Stage C: dedup, bucket, split, tune, held-out eval (bicep).

Methodology (per the reviewer's Stage-C constraints):
  1. DEDUP by md5 file hash; keep one representative per identical-video group so
     no duplicate can span the tune/held-out split (leakage guard).
  2. EXCLUDE bad-form demo clips. Bucket SIDE/BACK-angle clips separately — the
     shoulder-elbow-wrist angle from a non-frontal view is geometrically distorted,
     so a miss there is MediaPipe geometry, not the counter. FRONT is the primary
     metric.
  3. Split UNIQUE FRONT clips into tune/held-out (stratified single vs multi).
     Tune the batch segmenter (segment_reps_batch — the path /api/score runs) on
     the tune set only; report on held-out, broken out.
  4. Report exact-match AND within-+-1, separately for single-rep and multi-rep,
     with n per bucket. List held-out mismatches labelled partial-rep / real-miss.

Categories are derived (by hand) from calibration/bicep_ground_truth.csv notes and
printed for verification; true_reps are read from that file.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import RepSegmenterConfig
from core.rep_segmenter import count_reps_batch
from utils.angle_utils import calculate_angle_3d

FPS = 30.0
VID_DIR = Path("videos/bicep/adult/good_form")
GT_PATH = Path("calibration/bicep_ground_truth.csv")

# Categories read from the ground-truth notes (printed below for the reviewer).
BAD_FORM = {5, 8, 9, 30, 41, 53, 56}
SIDE = {11, 13, 18, 20, 31, 46, 49, 61, 62}
BACK = {6, 50, 57, 58, 59, 60}
PARTIAL_NOTE = {7, 12, 43, 51}     # notes mention half/partial/prep-move reps
NOTED_DUPS = [{14, 15, 16, 17}, {18, 19}, {20, 21, 22, 27, 28, 29},
              {23, 24, 25, 26}, {31, 32, 33, 34, 35, 36}, {37, 38, 39}]

GRID_WINDOWS = [5, 7, 9]
GRID_FRAC = [0.25, 0.35, 0.45]
GRID_FLOOR = [10.0, 12.0, 15.0, 18.0]
GRID_REFRACTORY = [0.8, 1.0, 1.2]


def _num(fname: str) -> int:
    return int(re.search(r"_(\d+)\.mp4", fname).group(1))


def _md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load_angles() -> dict[int, np.ndarray]:
    csv = sorted(p for p in Path("data/training").glob("bicep_adult_good_form_*.csv")
                 if "synthetic" not in p.name)[-1]
    df = pd.read_csv(csv)
    if "augmentation" in df.columns:
        df = df[df["augmentation"] == "original"]

    def pt(r: pd.Series, i: int) -> np.ndarray:
        return np.array([r[f"landmark_{i}_x"], r[f"landmark_{i}_y"], r[f"landmark_{i}_z"]])

    out: dict[int, np.ndarray] = {}
    for clip, g in df.groupby("video_file"):
        ang = np.array([calculate_angle_3d(pt(r, 12), pt(r, 14), pt(r, 16))
                        for _, r in g.iterrows()], dtype=float)
        out[_num(str(clip))] = ang
    return out


def _detect(angles: np.ndarray, cfg: RepSegmenterConfig) -> int:
    ts = np.arange(angles.size) / FPS
    return count_reps_batch(angles, ts, "bicep_curl", config=cfg)


def _metrics(nums: list[int], detected: dict[int, int], true: dict[int, float]) -> tuple[float, float, float]:
    if not nums:
        return (1.0, 1.0, 0.0)
    exact = np.mean([detected[n] == round(true[n]) for n in nums])
    within1 = np.mean([abs(detected[n] - true[n]) <= 1 for n in nums])
    mae = float(np.mean([abs(detected[n] - true[n]) for n in nums]))
    return float(within1), float(exact), mae


def main() -> int:
    gt = pd.read_csv(GT_PATH, dtype=str).fillna("")
    gt["num"] = gt["clip_filename"].map(_num)
    true = {int(r.num): float(r.true_reps) for r in gt.itertuples()}
    fname = {int(r.num): r.clip_filename for r in gt.itertuples()}

    angles = _load_angles()

    # 1. dedup ------------------------------------------------------------------
    groups: dict[str, list[int]] = {}
    for n, fn in fname.items():
        p = VID_DIR / fn
        if p.exists():
            groups.setdefault(_md5(p), []).append(n)
    hash_dups = sorted([sorted(v) for v in groups.values() if len(v) > 1])
    noted = sorted([sorted(s) for s in NOTED_DUPS])
    print("=== 1. DEDUP ===")
    print(f"  md5 file-hash identical groups: {hash_dups}")
    print(f"  notes-flagged 'same video'    : {noted}")
    print("  -> file hashes do NOT match the notes; the duplicate copies are re-encodes")
    print("     (byte-different, same content). Verifying by angle-series identity:")
    for g in NOTED_DUPS:
        members = sorted(g)
        base = angles[members[0]]
        diffs = []
        for m in members[1:]:
            a = angles[m]
            length = min(len(a), len(base))
            diffs.append(round(float(np.mean(np.abs(a[:length] - base[:length]))), 2)
                         if length else float("nan"))
        print(f"     group {members}: mean|angle-diff| vs {members[0]} = {diffs} deg")
    # Hash failed to find them, so dedup by the human-confirmed notes groups.
    dup_extras: set[int] = set()
    for g in NOTED_DUPS:
        dup_extras |= (g - {min(g)})
    reps = set(true) - dup_extras
    print(f"  -> deduped by notes: 62 clips -> {len(reps)} unique videos "
          f"(removed {len(dup_extras)} duplicate copies)")

    # 2. exclude bad-form, bucket by angle -------------------------------------
    unique = sorted(reps - BAD_FORM)
    cat = {n: ("side" if n in SIDE else "back" if n in BACK else "front") for n in unique}
    front = [n for n in unique if cat[n] == "front"]
    side = [n for n in unique if cat[n] == "side"]
    back = [n for n in unique if cat[n] == "back"]
    f_single = [n for n in front if true[n] < 2]
    f_multi = [n for n in front if true[n] >= 2]
    print("\n=== 2. SET (after dedup + bad-form exclusion) ===")
    print(f"  excluded bad-form: {sorted(BAD_FORM)}")
    print(f"  unique non-bad clips: {len(unique)}  |  FRONT {len(front)} "
          f"(single {len(f_single)}, multi {len(f_multi)})  SIDE {len(side)}  BACK {len(back)}")

    # 3. split unique FRONT into tune/held-out (stratified single/multi) --------
    rng = np.random.default_rng(42)
    def split(items: list[int], frac_tune: float = 0.6) -> tuple[list[int], list[int]]:
        idx = rng.permutation(len(items))
        k = round(len(items) * frac_tune)
        items = [items[i] for i in idx]
        return sorted(items[:k]), sorted(items[k:])
    tune_s, held_s = split(f_single)
    tune_m, held_m = split(f_multi)
    tune = tune_s + tune_m
    held = held_s + held_m
    print("\n=== 3. SPLIT (unique FRONT only; no duplicate spans the split) ===")
    print(f"  tune  n={len(tune)} (single {len(tune_s)}, multi {len(tune_m)})  clips={tune}")
    print(f"  held  n={len(held)} (single {len(held_s)}, multi {len(held_m)})  clips={held}")

    # 4. grid-search on TUNE FRONT (maximise within1, then exact, then -mae) -----
    best = None
    for w in GRID_WINDOWS:
        for fr in GRID_FRAC:
            for fl in GRID_FLOOR:
                for rf in GRID_REFRACTORY:
                    # grid is in frames; store as seconds (harness ts are @FPS).
                    cfg = RepSegmenterConfig(smoothing_window_seconds=w / FPS, prominence_frac=fr,
                                             prominence_floor_deg=fl, refractory_s=rf)
                    det = {n: _detect(angles[n], cfg) for n in tune}
                    w1, ex, mae = _metrics(tune, det, true)
                    score = (round(w1, 4), round(ex, 4), -round(mae, 4))
                    if best is None or score > best[0]:
                        best = (score, cfg)
    score, cfg = best
    print("\n=== 4. TUNED PARAMS (locked on tune set) ===")
    print(f"  smoothing_window_seconds={cfg.smoothing_window_seconds:.3f} "
          f"(~{round(cfg.smoothing_window_seconds * FPS)} frames @{FPS:.0f}fps)  "
          f"prominence_frac={cfg.prominence_frac}"
          f"  prominence_floor_deg={cfg.prominence_floor_deg}  refractory_s={cfg.refractory_s}")
    print(f"  tune-set front: within+-1={score[0]:.0%}  exact={score[1]:.0%}  MAE={-score[2]:.2f}")

    detected = {n: _detect(angles[n], cfg) for n in unique}

    # 5. HELD-OUT report, broken out -------------------------------------------
    def report(label: str, nums: list[int]) -> None:
        w1, ex, mae = _metrics(nums, detected, true)
        print(f"  {label:28s} n={len(nums):2d}  exact={ex:5.0%}  within+-1={w1:5.0%}  MAE={mae:.2f}")
    print("\n=== 5. HELD-OUT FRONT (PRIMARY METRIC) ===")
    report("single-rep", held_s)
    report("multi-rep  (the one that matters)", held_m)
    report("all front held-out", held)
    print("  -- context (all UNIQUE front, tune+held; larger n) --")
    report("front multi-rep (all unique)", f_multi)
    report("front single-rep (all unique)", f_single)
    print("\n=== 5b. SIDE / BACK buckets (SEPARATE — geometry-distorted, NOT tuned on) ===")
    report("side-angle (all unique)", side)
    report("back-angle (all unique)", back)

    # 6. held-out front mismatches, labelled -----------------------------------
    print("\n=== 6. HELD-OUT FRONT mismatches (detected != round(true)) ===")
    any_mm = False
    for n in held:
        det, tr = detected[n], true[n]
        if det == round(tr):
            continue
        any_mm = True
        if n in PARTIAL_NOTE or tr != round(tr):
            label = "partial-rep (definitional)"
        elif det > tr:
            label = "real-miss (over-count)"
        else:
            label = "real-miss (under-count)"
        print(f"  clip {n:2d}: true={tr} detected={det}  -> {label}")
    if not any_mm:
        print("  (none)")

    # 7. over-count check on single-rep front (the 99-vs-83 prep-move concern) ---
    over = [n for n in f_single if detected[n] > round(true[n])]
    print(f"\n=== 7. OVER-COUNT CHECK (front single-rep, all unique) ===")
    print(f"  over-counted single-rep clips: {len(over)}/{len(f_single)}  clips={over}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
