"""
Centralized configuration for AI Gym FYP.
All model paths, exercise names, and age groups live here.
Import this instead of hardcoding paths across files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
MODELS_DIR      = BASE_DIR / 'data' / 'models'
TRAINING_DIR    = BASE_DIR / 'data' / 'training'
LOGS_DIR        = BASE_DIR / 'logs'
CALIBRATION_DIR = BASE_DIR / 'user_calibration'
SESSION_DIR     = BASE_DIR / 'session_logs'

# ── Exercises & age groups ───────────────────────────────────────────────────
# Long-form keys per the Stage 5a naming reconciliation. The rule engines,
# ML aggregator, and rep counters all key on these exact strings; classifier
# body-part outputs are routed through core.exercise_mapping.BODY_PART_TO_EXERCISE.
EXERCISES  = ['bicep_curl', 'bent_over_row', 'push_up']
AGE_GROUPS = ['children', 'adult', 'senior']

# Age boundaries (used to map numeric age → age_group string)
AGE_BOUNDARIES = {'children': (0, 16), 'adult': (17, 45), 'senior': (46, 120)}

def age_to_group(age: int) -> str:
    """Convert a numeric age to the matching age_group string."""
    for group, (lo, hi) in AGE_BOUNDARIES.items():
        if lo <= age <= hi:
            return group
    return 'adult'

# ── Model path helpers ───────────────────────────────────────────────────────
def model_path(exercise: str, age_group: str) -> Path:
    """Return the Path to a trained model file."""
    return MODELS_DIR / f'{exercise}_{age_group}.pkl'

def model_exists(exercise: str, age_group: str) -> bool:
    return model_path(exercise, age_group).exists()

# ── Default model paths (for legacy code using string literals) ───────────────
DEFAULT_BICEP_CURL_MODEL    = str(model_path('bicep_curl',    'adult'))
DEFAULT_BENT_OVER_ROW_MODEL = str(model_path('bent_over_row', 'adult'))
DEFAULT_PUSH_UP_MODEL       = str(model_path('push_up',       'adult'))

# ── Calibration ──────────────────────────────────────────────────────────────
CALIBRATION_MAX_FILES = 5   # keep only the N most recent calibration files per user

def cleanup_calibration_files(user_id: str = 'default', keep: int = CALIBRATION_MAX_FILES) -> int:
    """
    Remove old calibration JSON files for a user, keeping only the `keep` most recent.
    Returns the number of files deleted.
    """
    pattern = f'user_{user_id}_*.json'
    files = sorted(CALIBRATION_DIR.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    to_delete = files[keep:]
    for f in to_delete:
        f.unlink()
    return len(to_delete)

def cleanup_all_calibration_files(keep: int = CALIBRATION_MAX_FILES) -> int:
    """Remove old calibration files for ALL users. Returns total files deleted."""
    all_files = sorted(CALIBRATION_DIR.glob('user_*.json'),
                       key=lambda f: f.stat().st_mtime, reverse=True)
    to_delete = all_files[keep:]
    for f in to_delete:
        f.unlink()
    return len(to_delete)


# ── Rep Counter FSM calibration ─────────────────────────────────────────────

@dataclass(frozen=True)
class RepCounterConfig:
    """Per-exercise thresholds that drive the rep-counter finite state machine."""
    top_threshold: float          # angle (deg) considered "top" of movement
    bottom_threshold: float       # angle (deg) considered "bottom" of movement
    hysteresis: float             # dead-band width (deg) to suppress jitter
    min_rep_duration_s: float     # reps shorter than this are rejected
    max_rep_duration_s: float     # reps longer than this are rejected
    # Degrees the (smoothed) angle must fall back from its ascent peak to count as
    # a "top turnaround" — closes a rep even when the user does NOT re-extend past
    # ``top_threshold`` between reps (the common real-world case; the absolute
    # threshold alone misses ~3/4 of real reps). Must exceed frame-to-frame jitter.
    reversal_margin: float = 35.0


REP_COUNTER_CONFIGS: dict[str, RepCounterConfig] = {
    # Bicep curl — primary angle: shoulder(12)→elbow(14)→wrist(16)
    # Extended arm ≈ 160-170°, fully curled ≈ 30-50°
    'bicep_curl': RepCounterConfig(
        top_threshold=140.0,
        bottom_threshold=60.0,
        hysteresis=10.0,
        min_rep_duration_s=0.8,
        max_rep_duration_s=8.0,
    ),
    # Bent-over row — primary angle: shoulder(12)→elbow(14)→wrist(16)
    # Arm extended forward ≈ 150-170°, pulled back ≈ 40-60°
    'bent_over_row': RepCounterConfig(
        top_threshold=140.0,
        bottom_threshold=60.0,
        hysteresis=10.0,
        min_rep_duration_s=0.8,
        max_rep_duration_s=8.0,
    ),
    # Push-up — primary angle: shoulder(12)→elbow(14)→wrist(16)
    # Arms locked out ≈ 155-170°, bottom position ≈ 60-90°
    'push_up': RepCounterConfig(
        top_threshold=140.0,
        bottom_threshold=80.0,
        hysteresis=10.0,
        min_rep_duration_s=0.8,
        max_rep_duration_s=8.0,
    ),
}


# ── Rep Segmenter (smoothed, amplitude-relative prominence) ──────────────────
# Used by core.rep_segmenter (batch find_peaks for /api/score; streaming online
# detector for the live path). A rep is one curl-bottom: a local minimum of the
# median-smoothed primary angle, confirmed by a rebound of at least the
# amplitude-relative prominence and spaced at least `refractory_s` from the prior
# rep. Amplitude-relative prominence makes it robust to the shifted/narrow real
# ROM band (~80-120 deg) measured on the bicep clips.

@dataclass(frozen=True)
class RepSegmenterConfig:
    """Per-exercise parameters for the prominence-based rep segmenter."""
    smoothing_window_frames: int = 5      # median-filter window (odd); rejects spike jitter
    prominence_frac: float = 0.35         # min dip depth as a fraction of amplitude (p95-p5)
    prominence_floor_deg: float = 12.0    # absolute min dip depth; also the streaming cold-start value
    refractory_s: float = 0.8             # min seconds between counted reps (== min_rep_duration)
    amplitude_window_s: float = 5.0       # streaming: rolling window for the p95-p5 amplitude estimate
    fps_fallback: float = 30.0            # used when frame timestamps are absent/degenerate


REP_SEGMENTER_CONFIGS: dict[str, RepSegmenterConfig] = {
    # bicep_curl is tuned against real clips in Stage C; row/push_up keep defaults
    # until they have real recordings (real-data calibration queued in TODO.md).
    'bicep_curl':    RepSegmenterConfig(),
    'bent_over_row': RepSegmenterConfig(),
    'push_up':       RepSegmenterConfig(),
}
