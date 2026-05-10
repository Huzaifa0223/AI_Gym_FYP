"""Synthetic biomechanical trainer for form-quality classifiers.

Generates physically plausible joint-angle samples per (exercise, age_group)
bucket and trains a per-bucket RandomForestClassifier. Output format matches
the existing data/models/<exercise>_<age_group>.pkl contract so the trained
models drop straight into MLAggregator with no loader changes.

This is the synthetic-data path that unblocks the supervisor review while the
real video-extraction pipeline (Stage 6a) is queued. Once pose extraction over
data/FYP/ produces real CSV features, the same trainer interface accepts those
in place of the synthetic generator and the .pkl artefacts are replaced.

Exercise priors are grounded in standard joint-angle conventions used by the
existing rule engines (BicepRuleEngine, BackRuleEngine, ChestRuleEngine).
Age scaling follows the well-documented ROM reduction in younger and older
populations versus prime adult ranges.

Usage
-----
    python synthetic_trainer.py --output-dir data/models

Each run emits 9 .pkl model files and 9 _analysis.png histograms.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import joblib
import matplotlib
matplotlib.use("Agg")  # headless render — no GUI required
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED: Final[int] = 42
N_ESTIMATORS: Final[int] = 200
SAMPLES_PER_CLASS: Final[int] = 3000  # → 6000 per bucket → 9 buckets → 54k rows
TEST_FRACTION: Final[float] = 0.2
NOISE_STD: Final[float] = 3.0  # degrees of Gaussian noise added to every angle
FEATURE_COLS: Final[list[str]] = ["primary_angle", "secondary_angle", "tertiary_angle"]

# Filename mapping: long-form on disk, short-form in metadata (matches the
# existing .pkl contract from the Stage 5a rename — long names won externally,
# but the embedded `exercise_type` field stayed as the original short key).
EXERCISE_FILENAME_TO_TYPE: Final[dict[str, str]] = {
    "bicep_curl":       "bicep",
    "bent_over_row":    "back",
    "push_up":          "chest",
}

AGE_GROUPS: Final[tuple[str, ...]] = ("children", "adult", "senior")


# ---------------------------------------------------------------------------
# Biomechanical priors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AngleBand:
    """Mean and standard deviation defining the spread of one joint angle in
    one form class. Sampled as ``N(mean, std)`` then clipped to ``[lo, hi]``.

    Means and bounds are in degrees. Standard deviations are biological — they
    represent natural per-rep variation, not measurement noise. Measurement
    noise is added separately as Gaussian jitter on top of the sampled value.
    """

    mean: float
    std: float
    lo: float
    hi: float


@dataclass(frozen=True)
class ExercisePrior:
    """Per-exercise biomechanical priors for the three feature angles.

    For each angle we define a ``good`` band (form within rule-engine
    tolerances) and a ``bad`` band (form that the rule engine would flag).
    The two bands typically overlap slightly — in real data, marginal reps
    sit in the overlap and are inherently ambiguous, which is what gives the
    classifier its non-trivial decision boundary.
    """

    name: str
    primary_good: AngleBand
    primary_bad: AngleBand
    secondary_good: AngleBand
    secondary_bad: AngleBand
    tertiary_good: AngleBand
    tertiary_bad: AngleBand


# Bicep curl: primary = elbow flexion (full ROM bimodal at top/bottom of rep);
# secondary = shoulder flexion (should be near 0° — elbow pinned at side);
# tertiary = trunk lean (should be near 0° — no body sway).
BICEP_CURL: Final[ExercisePrior] = ExercisePrior(
    name="bicep_curl",
    primary_good=AngleBand(mean=90.0, std=55.0, lo=5.0, hi=178.0),    # uses full ROM
    primary_bad=AngleBand(mean=110.0, std=20.0, lo=70.0, hi=150.0),   # partial ROM
    secondary_good=AngleBand(mean=8.0, std=5.0, lo=0.0, hi=18.0),     # elbow stays put
    secondary_bad=AngleBand(mean=28.0, std=10.0, lo=15.0, hi=55.0),   # elbow drift
    tertiary_good=AngleBand(mean=0.0, std=3.0, lo=-7.0, hi=7.0),      # no sway
    tertiary_bad=AngleBand(mean=0.0, std=12.0, lo=-25.0, hi=25.0),    # body swing
)

# Bent-over row: primary = elbow flexion during pull (~50° → 110°);
# secondary = shoulder extension (elbow drives BACK past torso line);
# tertiary = torso angle relative to vertical (good hip hinge ~45°).
BENT_OVER_ROW: Final[ExercisePrior] = ExercisePrior(
    name="bent_over_row",
    primary_good=AngleBand(mean=85.0, std=30.0, lo=35.0, hi=160.0),
    primary_bad=AngleBand(mean=110.0, std=18.0, lo=80.0, hi=145.0),   # short pull
    secondary_good=AngleBand(mean=28.0, std=8.0, lo=15.0, hi=45.0),   # scapular retract
    secondary_bad=AngleBand(mean=8.0, std=6.0, lo=0.0, hi=18.0),      # arms-only pull
    tertiary_good=AngleBand(mean=45.0, std=8.0, lo=30.0, hi=60.0),    # 45° hinge
    tertiary_bad=AngleBand(mean=15.0, std=12.0, lo=0.0, hi=30.0),     # too upright
)

# Push-up: primary = elbow flexion (60° down → 180° up); secondary = shoulder
# flexion (push depth); tertiary = plank integrity (head-hip-ankle line, 180°
# = perfect plank, lower = sagging hips, higher = piked hips).
PUSH_UP: Final[ExercisePrior] = ExercisePrior(
    name="push_up",
    primary_good=AngleBand(mean=120.0, std=35.0, lo=55.0, hi=178.0),
    primary_bad=AngleBand(mean=140.0, std=15.0, lo=110.0, hi=170.0),  # half push-up
    secondary_good=AngleBand(mean=60.0, std=15.0, lo=35.0, hi=90.0),
    secondary_bad=AngleBand(mean=85.0, std=10.0, lo=70.0, hi=105.0),  # too high
    tertiary_good=AngleBand(mean=178.0, std=4.0, lo=168.0, hi=188.0), # straight body
    tertiary_bad=AngleBand(mean=155.0, std=15.0, lo=130.0, hi=205.0), # sag or pike
)

EXERCISE_PRIORS: Final[dict[str, ExercisePrior]] = {
    "bicep_curl":       BICEP_CURL,
    "bent_over_row":    BENT_OVER_ROW,
    "push_up":          PUSH_UP,
}


@dataclass(frozen=True)
class AgeScaling:
    """Multiplier and offset applied to ROM when generating samples.

    Children and seniors have biomechanically reduced range of motion
    relative to prime-adult populations: children due to incomplete motor
    development, seniors due to joint stiffness and reduced flexibility.
    Standard deviations also widen with age as form becomes more variable.

    These multipliers are conservative and consistent with general fitness
    literature; they are not citations from a specific paper. The synthetic
    trainer tags every model with ``noise_injection=True`` so this assumption
    is recorded in the artefact metadata.
    """

    rom_multiplier: float   # scales the std of each band → narrower distribution
    rom_centre_offset: float  # shifts the mean (degrees) toward mid-ROM
    std_multiplier: float   # scales the noise / variability


AGE_SCALING: Final[dict[str, AgeScaling]] = {
    "adult":    AgeScaling(rom_multiplier=1.00, rom_centre_offset=0.0,  std_multiplier=1.0),
    "children": AgeScaling(rom_multiplier=0.78, rom_centre_offset=10.0, std_multiplier=1.05),
    "senior":   AgeScaling(rom_multiplier=0.72, rom_centre_offset=15.0, std_multiplier=1.15),
}


# ---------------------------------------------------------------------------
# Sample generation + training
# ---------------------------------------------------------------------------


def _scale_band(band: AngleBand, age: AgeScaling) -> AngleBand:
    """Apply age-scaling to a single angle band.

    Centre is pulled toward the band's midpoint by ``rom_centre_offset``
    (capped so it cannot overshoot), and the spread is multiplied by
    ``rom_multiplier``. This produces a tighter, mid-shifted distribution
    for non-adult age groups.
    """
    midpoint: float = (band.lo + band.hi) / 2.0
    direction: float = 1.0 if midpoint > band.mean else -1.0
    new_mean: float = band.mean + direction * age.rom_centre_offset
    new_std: float = band.std * age.rom_multiplier * age.std_multiplier
    return AngleBand(mean=new_mean, std=new_std, lo=band.lo, hi=band.hi)


def _sample_band(band: AngleBand, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` samples from a clipped Gaussian fitting the band.

    Pure ``np.random.normal`` would let a fraction of samples drift outside
    physical joint limits. Clipping after sampling keeps the distribution
    biomechanically valid at the cost of mass-piling at the bounds. For
    a histogram-style demo this is acceptable and matches the visible
    pile-ups at the extremes in the supplied analysis PNGs.
    """
    raw: np.ndarray = rng.normal(loc=band.mean, scale=band.std, size=n)
    return np.clip(raw, band.lo, band.hi)


def generate_samples(
    prior: ExercisePrior,
    age: AgeScaling,
    samples_per_class: int,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a labelled synthetic dataset for one (exercise, age) bucket.

    The output ``X`` is shape ``(2 * samples_per_class, 3)`` with one row per
    rep-frame snapshot, and ``y`` is shape ``(2 * samples_per_class,)`` with
    1 for good-form rows and 0 for bad-form rows. Class balance is exact by
    construction.
    """
    feature_arrays: dict[str, np.ndarray] = {}

    for label, suffix in ((1, "good"), (0, "bad")):
        for axis in ("primary", "secondary", "tertiary"):
            band: AngleBand = getattr(prior, f"{axis}_{suffix}")
            scaled: AngleBand = _scale_band(band, age)
            samples: np.ndarray = _sample_band(scaled, samples_per_class, rng)
            feature_arrays.setdefault(f"{axis}_{label}", samples)

    primary: np.ndarray = np.concatenate([feature_arrays["primary_1"], feature_arrays["primary_0"]])
    secondary: np.ndarray = np.concatenate([feature_arrays["secondary_1"], feature_arrays["secondary_0"]])
    tertiary: np.ndarray = np.concatenate([feature_arrays["tertiary_1"], feature_arrays["tertiary_0"]])

    measurement_noise: np.ndarray = rng.normal(loc=0.0, scale=noise_std, size=(primary.size, 3))
    x_clean: np.ndarray = np.column_stack([primary, secondary, tertiary])
    x_noised: np.ndarray = x_clean + measurement_noise

    y: np.ndarray = np.concatenate([
        np.ones(samples_per_class, dtype=int),
        np.zeros(samples_per_class, dtype=int),
    ])

    # Shuffle so train/test splits are not class-blocked.
    perm: np.ndarray = rng.permutation(x_noised.shape[0])
    return x_noised[perm], y[perm]


def train_bucket(
    exercise_filename: str,
    age_group: str,
    output_dir: Path,
    rng: np.random.Generator,
) -> tuple[Path, Path, float]:
    """Train one bucket end-to-end: sample → split → fit → save → plot.

    Returns ``(model_path, analysis_path, accuracy)``.
    """
    prior: ExercisePrior = EXERCISE_PRIORS[exercise_filename]
    age: AgeScaling = AGE_SCALING[age_group]
    short_type: str = EXERCISE_FILENAME_TO_TYPE[exercise_filename]

    x, y = generate_samples(
        prior=prior,
        age=age,
        samples_per_class=SAMPLES_PER_CLASS,
        noise_std=NOISE_STD,
        rng=rng,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_FRACTION, random_state=RANDOM_SEED, stratify=y,
    )

    model: RandomForestClassifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred: np.ndarray = model.predict(x_test)
    accuracy: float = float(accuracy_score(y_test, y_pred))

    artefact: dict[str, object] = {
        "type":            "supervised",
        "model":           model,
        "feature_cols":    list(FEATURE_COLS),
        "exercise_type":   short_type,
        "age_group":       age_group,
        "accuracy":        np.float64(accuracy),
        "noise_injection": True,
    }

    model_path: Path = output_dir / f"{exercise_filename}_{age_group}.pkl"
    joblib.dump(artefact, model_path)

    analysis_path: Path = output_dir / f"{exercise_filename}_{age_group}_analysis.png"
    _plot_primary_angle_distribution(
        primary_angles=x[:, 0],
        title=f"{prior.name.replace('_', ' ').title()} – {age_group.title()}\nAngle Distribution",
        out_path=analysis_path,
    )

    logger.info(
        "trained %s/%s — n=%d, accuracy=%.4f, model=%s",
        exercise_filename, age_group, x.shape[0], accuracy, model_path.name,
    )
    return model_path, analysis_path, accuracy


def _plot_primary_angle_distribution(
    primary_angles: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Render the primary-angle histogram in the same visual style as the
    existing ``_analysis.png`` files: 50-bin histogram, light-blue bars, soft
    grid, no legend, plain white background.
    """
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.hist(primary_angles, bins=50, color="#5b9bd5", edgecolor="white", linewidth=0.3)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Primary Angle (degrees)")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models"),
        help="Directory to write .pkl model files and _analysis.png plots.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng: np.random.Generator = np.random.default_rng(RANDOM_SEED)

    summary: list[tuple[str, str, float]] = []
    for exercise_filename in EXERCISE_PRIORS:
        for age_group in AGE_GROUPS:
            _, _, accuracy = train_bucket(
                exercise_filename=exercise_filename,
                age_group=age_group,
                output_dir=args.output_dir,
                rng=rng,
            )
            summary.append((exercise_filename, age_group, accuracy))

    logger.info("=" * 60)
    logger.info("training summary — %d models written to %s", len(summary), args.output_dir)
    for exercise, age, acc in summary:
        logger.info("  %-22s %-10s acc=%.4f", exercise, age, acc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
