"""
Centralized configuration for AI Gym FYP.
All model paths, exercise names, and age groups live here.
Import this instead of hardcoding paths across files.
"""
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
MODELS_DIR      = BASE_DIR / 'data' / 'models'
TRAINING_DIR    = BASE_DIR / 'data' / 'training'
LOGS_DIR        = BASE_DIR / 'logs'
CALIBRATION_DIR = BASE_DIR / 'user_calibration'
SESSION_DIR     = BASE_DIR / 'session_logs'

# ── Exercises & age groups ───────────────────────────────────────────────────
EXERCISES  = ['bicep', 'back', 'chest']
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
DEFAULT_BICEP_MODEL  = str(model_path('bicep',  'adult'))
DEFAULT_BACK_MODEL   = str(model_path('back',   'adult'))
DEFAULT_CHEST_MODEL  = str(model_path('chest',  'adult'))

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
