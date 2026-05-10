"""Contract tests for training.synthetic_trainer.

Each test trains into a pytest tmp directory; nothing is written to
``data/models``. The training itself is fast — one bucket fits a 200-tree
RandomForest on 4800 rows in ~1-2s — so we share a single trained artefact
across the four artefact-shape tests via a module-scoped fixture.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from training import synthetic_trainer as st


REQUIRED_ARTEFACT_KEYS = {
    "type",
    "model",
    "feature_cols",
    "exercise_type",
    "age_group",
    "accuracy",
    "noise_injection",
}


@pytest.fixture(scope="module")
def trained_artefact(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out_dir: Path = tmp_path_factory.mktemp("synth_models")
    rng = np.random.default_rng(st.RANDOM_SEED)
    model_path, _, _ = st.train_bucket(
        exercise_filename="bicep_curl",
        age_group="adult",
        output_dir=out_dir,
        rng=rng,
    )
    return joblib.load(model_path)


def test_artefact_has_required_keys(trained_artefact: dict) -> None:
    assert set(trained_artefact.keys()) == REQUIRED_ARTEFACT_KEYS


def test_random_forest_has_200_estimators(trained_artefact: dict) -> None:
    model = trained_artefact["model"]
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 200


def test_feature_cols_are_three_angles(trained_artefact: dict) -> None:
    assert trained_artefact["feature_cols"] == [
        "primary_angle",
        "secondary_angle",
        "tertiary_angle",
    ]


def test_accuracy_above_threshold(trained_artefact: dict) -> None:
    assert float(trained_artefact["accuracy"]) >= 0.85


def test_no_triceps_in_priors() -> None:
    assert "tricep_extension" not in st.EXERCISE_PRIORS
    assert "tricep_extension" not in st.EXERCISE_FILENAME_TO_TYPE
    assert set(st.EXERCISE_PRIORS) == {"bicep_curl", "bent_over_row", "push_up"}


def test_main_writes_nine_models(tmp_path: Path) -> None:
    out_dir = tmp_path / "models"
    rc = st.main(["--output-dir", str(out_dir), "--log-level", "WARNING"])
    assert rc == 0

    pkls = sorted(p.name for p in out_dir.glob("*.pkl"))
    pngs = sorted(p.name for p in out_dir.glob("*_analysis.png"))
    assert len(pkls) == 9, f"expected 9 .pkl, got {len(pkls)}: {pkls}"
    assert len(pngs) == 9, f"expected 9 _analysis.png, got {len(pngs)}: {pngs}"

    expected_pkls = {
        f"{ex}_{age}.pkl"
        for ex in ("bicep_curl", "bent_over_row", "push_up")
        for age in ("children", "adult", "senior")
    }
    assert set(pkls) == expected_pkls
