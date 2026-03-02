#!/usr/bin/env python3
"""
Synthetic Biomechanics Data Generator — AI Gym FYP
====================================================
Generates bad-form (and balanced good-form) training data based on
biomechanical principles. No video dataset required.

HOW IT WORKS
------------
For every sample we:
  1. Sample a (primary_angle, secondary_angle) pair from a biomechanically
     motivated distribution for the given exercise/age_group/form type.
  2. Use 2-D forward kinematics to place the four key MediaPipe joints
     (shoulder-12, elbow-14, wrist-16, hip-24, knee-26) so that the
     angles computed by train_universal.py exactly match those samples.
  3. Fill the remaining 29 joints from a base skeleton, with small random
     perturbations for realism.
  4. Write one row per sample in the exact column format expected by
     video_pipeline.py / train_universal.py.

ANGLE DEFINITIONS (matches train_universal.py)
-----------------------------------------------
BICEP  : primary  = angle(shoulder→elbow→wrist)  [at elbow,    pts 12-14-16]
         secondary = angle(hip→shoulder→elbow)    [at shoulder, pts 24-12-14]

BACK   : primary  = angle(shoulder→elbow→wrist)  [at elbow,    pts 12-14-16]
         secondary = angle(hip→shoulder→elbow)    [at shoulder, pts 24-12-14]

CHEST  : primary  = angle(shoulder→elbow→wrist)  [at elbow,    pts 12-14-16]
         secondary = angle(shoulder→hip→knee)     [at hip,      pts 12-24-26]

BIOMECHANICAL BAD-FORM RATIONALE
---------------------------------
BICEP
  elbow_swing   : elbow rotates forward (secondary >> normal), primary OK
  incomplete_rom: primary range never reaches full extension/curl
  torso_lean    : secondary moderately elevated (hip-shoulder-elbow shifts)

BACK  (modelled as lat-pulldown / bent-over row)
  arms_only     : secondary stays small (person too upright, no back lean)
  excessive_lean: secondary too large (dangerous forward lean)
  incomplete_pull: primary range limited (never completes the pull)

CHEST (modelled as push-up / chest press)
  sagging_hips  : secondary too small (body not straight, hips drop)
  raised_hips   : secondary moderately small but different range (piking)
  partial_depth : primary never reaches low enough (shallow push-up)

OUTPUT
------
data/training/{exercise}_{age_group}_good_form_synthetic_{ts}.csv
data/training/{exercise}_{age_group}_bad_form_synthetic_{ts}.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import copy

# ─────────────────────────────────────────────────────────────────────────────
# ANGLE DISTRIBUTION CONFIGS
# Each entry: {'primary': (low, high), 'secondary': (low, high)}
# Angles are sampled uniformly within [low, high] and then gaussian-jittered.
# ─────────────────────────────────────────────────────────────────────────────

ANGLE_CONFIGS = {
    'bicep': {
        # PRIMARY  = elbow angle (50° = fully curled, 170° = arm down)
        # SECONDARY = hip→shoulder→elbow deviation from torso
        #   Good form: elbow stays at side → secondary ≈ 8–22°
        #   Elbow swing: elbow rotates forward → secondary ≈ 35–68°
        #   Torso lean: body momentum used → secondary ≈ 25–48°
        #   Incomplete ROM: arm never fully extends/curls → primary 90–145°
        'good_form': {
            'children': {'primary': (60,  160), 'secondary': (8,  22)},
            'adult':    {'primary': (50,  170), 'secondary': (8,  22)},
            'senior':   {'primary': (70,  155), 'secondary': (10, 25)},
        },
        'bad_form': {
            'elbow_swing': {
                'children': {'primary': (55,  165), 'secondary': (35, 65)},
                'adult':    {'primary': (50,  170), 'secondary': (35, 68)},
                'senior':   {'primary': (65,  158), 'secondary': (32, 60)},
            },
            'incomplete_rom': {
                'children': {'primary': (95,  145), 'secondary': (8,  22)},
                'adult':    {'primary': (90,  140), 'secondary': (8,  22)},
                'senior':   {'primary': (100, 148), 'secondary': (10, 25)},
            },
            'torso_lean': {
                'children': {'primary': (65,  158), 'secondary': (25, 45)},
                'adult':    {'primary': (60,  165), 'secondary': (25, 48)},
                'senior':   {'primary': (72,  152), 'secondary': (22, 42)},
            },
        },
    },

    'back': {
        # PRIMARY  = elbow angle (arm extended = 170°, fully pulled = 30°)
        # SECONDARY = hip→shoulder→elbow  (measures torso lean / back engagement)
        #   Good form: appropriate lean → secondary ≈ 50–100°
        #   Arms only: person too upright, no back → secondary ≈ 8–30°
        #   Excessive lean: dangerous forward lean → secondary ≈ 118–148°
        #   Incomplete pull: primary limited → 80–170° (never completes)
        'good_form': {
            'children': {'primary': (40,  160), 'secondary': (55,  95)},
            'adult':    {'primary': (30,  170), 'secondary': (50,  100)},
            'senior':   {'primary': (50,  155), 'secondary': (58,  95)},
        },
        'bad_form': {
            'arms_only': {
                'children': {'primary': (50,  160), 'secondary': (8,  28)},
                'adult':    {'primary': (40,  170), 'secondary': (8,  28)},
                'senior':   {'primary': (55,  155), 'secondary': (10, 30)},
            },
            'excessive_lean': {
                'children': {'primary': (35,  158), 'secondary': (118, 145)},
                'adult':    {'primary': (30,  168), 'secondary': (118, 148)},
                'senior':   {'primary': (45,  153), 'secondary': (115, 140)},
            },
            'incomplete_pull': {
                'children': {'primary': (85,  160), 'secondary': (55,  95)},
                'adult':    {'primary': (80,  170), 'secondary': (50,  100)},
                'senior':   {'primary': (90,  155), 'secondary': (58,  95)},
            },
        },
    },

    'chest': {
        # PRIMARY  = elbow angle (push-up: top = 180°, bottom = 50°)
        # SECONDARY = shoulder→hip→knee  (body alignment in plank)
        #   Good form: straight body → secondary ≈ 168–180°
        #   Sagging hips: hips drop toward floor → secondary ≈ 128–162°
        #   Raised hips (pike): hips too high → secondary ≈ 108–136°
        #   Partial depth: never goes low enough → primary 92–180°
        'good_form': {
            'children': {'primary': (70,  170), 'secondary': (168, 180)},
            'adult':    {'primary': (50,  180), 'secondary': (170, 180)},
            'senior':   {'primary': (80,  165), 'secondary': (166, 180)},
        },
        'bad_form': {
            'sagging_hips': {
                'children': {'primary': (70,  170), 'secondary': (133, 162)},
                'adult':    {'primary': (50,  180), 'secondary': (128, 160)},
                'senior':   {'primary': (80,  165), 'secondary': (135, 163)},
            },
            'raised_hips': {
                'children': {'primary': (70,  170), 'secondary': (112, 136)},
                'adult':    {'primary': (50,  180), 'secondary': (108, 133)},
                'senior':   {'primary': (80,  165), 'secondary': (114, 138)},
            },
            'partial_depth': {
                'children': {'primary': (100, 170), 'secondary': (168, 180)},
                'adult':    {'primary': (92,  180), 'secondary': (170, 180)},
                'senior':   {'primary': (105, 165), 'secondary': (166, 180)},
            },
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# BASE SKELETON — resting upright position for all 33 MediaPipe landmarks
# Key joints (12, 14, 16, 24, 26) are OVERWRITTEN by the kinematics below.
# ─────────────────────────────────────────────────────────────────────────────

BASE_SKELETON = {
    0:  (0.500, 0.080, -0.100),
    1:  (0.480, 0.060, -0.120),
    2:  (0.470, 0.060, -0.120),
    3:  (0.460, 0.070, -0.120),
    4:  (0.520, 0.060, -0.120),
    5:  (0.530, 0.060, -0.120),
    6:  (0.540, 0.070, -0.120),
    7:  (0.450, 0.100, -0.050),
    8:  (0.550, 0.100, -0.050),
    9:  (0.480, 0.120, -0.080),
    10: (0.520, 0.120, -0.080),
    11: (0.400, 0.280,  0.000),  # left_shoulder
    12: (0.600, 0.280,  0.000),  # right_shoulder  ← overwritten
    13: (0.350, 0.480,  0.000),  # left_elbow
    14: (0.650, 0.480,  0.000),  # right_elbow     ← overwritten
    15: (0.330, 0.680,  0.000),  # left_wrist
    16: (0.670, 0.680,  0.000),  # right_wrist     ← overwritten
    17: (0.320, 0.720,  0.020),
    18: (0.680, 0.720,  0.020),
    19: (0.310, 0.700,  0.020),
    20: (0.690, 0.700,  0.020),
    21: (0.320, 0.700,  0.020),
    22: (0.680, 0.700,  0.020),
    23: (0.430, 0.620,  0.000),  # left_hip
    24: (0.570, 0.620,  0.000),  # right_hip       ← overwritten (bicep/back)
    25: (0.430, 0.820,  0.000),  # left_knee
    26: (0.570, 0.820,  0.000),  # right_knee      ← overwritten (chest)
    27: (0.430, 1.000,  0.000),
    28: (0.570, 1.000,  0.000),
    29: (0.420, 1.020,  0.020),
    30: (0.580, 1.020,  0.020),
    31: (0.430, 1.040, -0.020),
    32: (0.570, 1.040, -0.020),
}

# Limb lengths in normalised image coords (adult baseline)
_L = {'upper_arm': 0.200, 'forearm': 0.180, 'thigh': 0.220}

# Age-group limb-length scale factors
AGE_SCALE = {'children': 0.85, 'adult': 1.00, 'senior': 0.95}

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rot2d(v: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate 2-D vector v by angle_deg (CCW in standard math coords)."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c * v[0] - s * v[1],
                     s * v[0] + c * v[1]])


def _calc_angle_2d(a, b, c) -> float:
    """Angle in degrees at point b between vectors b→a and b→c (2-D)."""
    ba = np.array(a[:2]) - np.array(b[:2])
    bc = np.array(c[:2]) - np.array(b[:2])
    n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if n_ba < 1e-6 or n_bc < 1e-6:
        return 90.0
    cos_val = np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def _place_child_joint(center, reference, target_angle_deg, limb_len):
    """
    Return a 2-D point C such that angle(reference, center, C) ≈ target_angle_deg
    and distance(center, C) = limb_len.

    Strategy: rotate the (center → reference) direction by +target_angle_deg.
    This is equivalent to the forward-kinematics used in all three exercises.
    """
    ref_dir = np.array(reference[:2]) - np.array(center[:2])
    norm = np.linalg.norm(ref_dir)
    if norm < 1e-6:
        ref_dir = np.array([1.0, 0.0])
    else:
        ref_dir = ref_dir / norm
    child_dir = _rot2d(ref_dir, target_angle_deg)
    cx, cy = np.array(center[:2]) + child_dir * limb_len
    return float(cx), float(cy)


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE-SPECIFIC SKELETON BUILDERS
# All return a dict {landmark_idx: (x, y, z, visibility)}
# ─────────────────────────────────────────────────────────────────────────────

def _add_noise(val, sigma, rng):
    return float(val) + rng.normal(0, sigma)


def build_skeleton(exercise: str, primary_angle: float, secondary_angle: float,
                   age_group: str, noise: float, rng: np.random.Generator) -> dict:
    """
    Forward-kinematics skeleton for the given exercise and angles.

    Key joints placed geometrically; all others use BASE_SKELETON + noise.
    """
    skeleton = {}
    scale = AGE_SCALE[age_group]
    ua = _L['upper_arm'] * scale   # upper-arm length
    fa = _L['forearm']   * scale   # forearm length
    th = _L['thigh']     * scale   # thigh (for chest)

    # Slightly randomise base shoulder/hip positions for realism
    s_x = _add_noise(0.600, 0.010, rng)
    s_y = _add_noise(0.280, 0.010, rng)
    h_x = _add_noise(0.570, 0.008, rng)
    h_y = _add_noise(0.620, 0.008, rng)

    shoulder = (s_x, s_y, 0.0)
    hip      = (h_x, h_y, 0.0)

    # ── BICEP / BACK ────────────────────────────────────────────────────────
    # secondary = angle(hip, shoulder, elbow)  at shoulder
    #   place elbow by rotating hip-direction at shoulder by +secondary_angle
    # primary   = angle(shoulder, elbow, wrist) at elbow
    #   place wrist by rotating shoulder-direction at elbow by +primary_angle

    if exercise in ('bicep', 'back'):
        ex, ey = _place_child_joint(shoulder, hip, secondary_angle, ua)
        elbow = (ex, ey, 0.0)

        wx, wy = _place_child_joint(elbow, shoulder, primary_angle, fa)
        wrist = (wx, wy, 0.0)

        key_joints = {12: shoulder, 14: elbow, 16: wrist, 24: hip}

    # ── CHEST ───────────────────────────────────────────────────────────────
    # secondary = angle(shoulder, hip, knee)  at hip
    #   place knee by rotating shoulder-direction at hip by +secondary_angle
    # primary   = angle(shoulder, elbow, wrist) at elbow
    #   place elbow/wrist same as above but from shoulder downward
    else:  # chest
        kx, ky = _place_child_joint(hip, shoulder, secondary_angle, th)
        knee = (kx, ky, 0.0)

        # For chest, elbow hangs roughly below the shoulder
        # We reuse the same kinematics: hip plays the role of "reference"
        # secondary_angle for elbow is a fixed ~170° (arm goes straight down)
        # then primary_angle bends the forearm
        elbow_secondary = 170.0  # elbow hangs directly below shoulder
        ex, ey = _place_child_joint(shoulder, hip, elbow_secondary, ua)
        elbow = (ex, ey, 0.0)

        wx, wy = _place_child_joint(elbow, shoulder, primary_angle, fa)
        wrist = (wx, wy, 0.0)

        key_joints = {12: shoulder, 14: elbow, 16: wrist, 24: hip, 26: knee}

    # ── ASSEMBLE FULL 33-LANDMARK SKELETON ──────────────────────────────────
    for idx in range(33):
        base = BASE_SKELETON[idx]
        if idx in key_joints:
            kj = key_joints[idx]
            x = _add_noise(kj[0], noise * 0.5, rng)
            y = _add_noise(kj[1], noise * 0.5, rng)
            z = _add_noise(kj[2] if len(kj) > 2 else 0.0, noise, rng)
        else:
            x = _add_noise(base[0], noise, rng)
            y = _add_noise(base[1], noise, rng)
            z = _add_noise(base[2], noise, rng)

        # Visibility: key joints get high confidence; others slightly lower
        vis_base = 0.95 if idx in key_joints else 0.80
        vis = float(np.clip(_add_noise(vis_base, 0.03, rng), 0.5, 1.0))
        skeleton[idx] = (x, y, z, vis)

    return skeleton


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _sample_angle(low: float, high: float, jitter: float, rng) -> float:
    """Uniform sample in [low, high] plus small Gaussian jitter."""
    val = rng.uniform(low, high)
    val += rng.normal(0, jitter)
    return float(np.clip(val, low - jitter * 3, high + jitter * 3))


def generate_samples(exercise: str, age_group: str, form_type: str,
                     angle_cfg: dict, label: int, n_samples: int,
                     rng: np.random.Generator, noise: float = 0.008) -> pd.DataFrame:
    """
    Generate n_samples rows for the given exercise/age_group/form.
    Returns a DataFrame in the exact format of video_pipeline.py output.
    """
    prim_low,  prim_high  = angle_cfg['primary']
    sec_low,   sec_high   = angle_cfg['secondary']
    jitter = 2.0  # degrees of random jitter around the uniform sample

    rows = []
    for frame_n in range(n_samples):
        p_angle = _sample_angle(prim_low,  prim_high,  jitter, rng)
        s_angle = _sample_angle(sec_low,   sec_high,   jitter, rng)

        skeleton = build_skeleton(exercise, p_angle, s_angle,
                                  age_group, noise, rng)

        row = {}
        for idx in range(33):
            x, y, z, vis = skeleton[idx]
            row[f'landmark_{idx}_x']          = round(x,   6)
            row[f'landmark_{idx}_y']          = round(y,   6)
            row[f'landmark_{idx}_z']          = round(z,   6)
            row[f'landmark_{idx}_visibility'] = round(vis, 6)

        row['frame']         = frame_n
        row['video_path']    = f'synthetic/{exercise}/{age_group}/{form_type}'
        row['video_file']    = f'synthetic_{form_type}_{frame_n:05d}.mp4'
        row['exercise_type'] = exercise
        row['age_group']     = age_group
        row['label']         = label
        row['form_type']     = form_type
        row['augmentation']  = 'synthetic'
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION — print angle statistics after generation
# ─────────────────────────────────────────────────────────────────────────────

def verify_angles(df: pd.DataFrame, exercise: str) -> None:
    """
    Recompute primary and secondary angles from generated landmarks and
    print statistics so you can confirm the distributions are correct.
    """
    primaries = []
    secondaries = []

    for _, row in df.iterrows():
        def lm(idx):
            return (row[f'landmark_{idx}_x'],
                    row[f'landmark_{idx}_y'],
                    row[f'landmark_{idx}_z'])

        s  = lm(12)   # shoulder
        e  = lm(14)   # elbow
        w  = lm(16)   # wrist
        h  = lm(24)   # hip
        kn = lm(26)   # knee

        prim = _calc_angle_2d(s, e, w)   # at elbow

        if exercise in ('bicep', 'back'):
            sec = _calc_angle_2d(h, s, e)   # at shoulder
        else:
            sec = _calc_angle_2d(s, h, kn)  # at hip

        primaries.append(prim)
        secondaries.append(sec)

    print(f"   primary_angle   -> mean={np.mean(primaries):6.1f} deg  "
          f"std={np.std(primaries):5.1f}  "
          f"[{np.min(primaries):.1f}, {np.max(primaries):.1f}]")
    print(f"   secondary_angle -> mean={np.mean(secondaries):6.1f} deg  "
          f"std={np.std(secondaries):5.1f}  "
          f"[{np.min(secondaries):.1f}, {np.max(secondaries):.1f}]")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — generate all datasets
# ─────────────────────────────────────────────────────────────────────────────

def main():
    SAMPLES_PER_CLASS = 1200   # good-form samples per age group
    SEED              = 42
    OUTPUT_DIR        = Path('data/training')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')

    exercises  = ['bicep', 'back', 'chest']
    age_groups = ['children', 'adult', 'senior']

    total_files = 0

    for exercise in exercises:
        ex_cfg = ANGLE_CONFIGS[exercise]

        for age in age_groups:
            print(f"\n{'='*60}")
            print(f"  {exercise.upper()} — {age}")
            print(f"{'='*60}")

            # ── GOOD FORM ────────────────────────────────────────────────
            good_cfg = ex_cfg['good_form'][age]
            print(f"\n  [good_form]  primary={good_cfg['primary']}  "
                  f"secondary={good_cfg['secondary']}")

            df_good = generate_samples(
                exercise, age, 'good_form', good_cfg,
                label=1, n_samples=SAMPLES_PER_CLASS, rng=rng
            )
            verify_angles(df_good, exercise)

            fname_good = OUTPUT_DIR / f'{exercise}_{age}_good_form_synthetic_{ts}.csv'
            df_good.to_csv(fname_good, index=False)
            print(f"  -> saved {len(df_good)} rows  ->  {fname_good.name}")

            # ── BAD FORM  ────────────────────────────────────────────────
            bad_frames = []
            bad_cfgs   = ex_cfg['bad_form']
            n_per_bad  = SAMPLES_PER_CLASS // len(bad_cfgs)  # balanced split

            for bad_name, bad_age_cfgs in bad_cfgs.items():
                bad_cfg = bad_age_cfgs[age]
                print(f"\n  [bad_form / {bad_name}]  "
                      f"primary={bad_cfg['primary']}  secondary={bad_cfg['secondary']}")

                df_bad_part = generate_samples(
                    exercise, age, f'bad_form_{bad_name}', bad_cfg,
                    label=0, n_samples=n_per_bad, rng=rng
                )
                verify_angles(df_bad_part, exercise)
                bad_frames.append(df_bad_part)

            df_bad = pd.concat(bad_frames, ignore_index=True)

            fname_bad = OUTPUT_DIR / f'{exercise}_{age}_bad_form_synthetic_{ts}.csv'
            df_bad.to_csv(fname_bad, index=False)
            print(f"\n  -> saved {len(df_bad)} rows  ->  {fname_bad.name}")

            total_files += 2

    print(f"\n{'='*60}")
    print(f"  DONE -- {total_files} CSV files written to {OUTPUT_DIR}/")
    print(f"  Each file ready for train_universal.py")
    print(f"")
    print(f"  Next step:")
    print(f"    python train_universal.py")
    print(f"  (It auto-discovers all CSVs in data/training/)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
