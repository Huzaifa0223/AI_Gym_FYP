"""
tests.test_main_loop — smoke tests for the OpenCV entry point ``main.py``.

Stage 0 backfill (per ``docs/architecture_3pillar.md`` §1.3 and §7).  These
tests exercise the import path, the no-IO constructor, ``setup_exercise``, and
``_process_frame`` on a solid-colour numpy frame.  They explicitly do **not**
call ``initialize()`` or ``run()`` — both open a camera and would block in CI.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_main_imports_cleanly() -> None:
    """Importing ``main`` must not open a camera or raise."""
    import main  # noqa: F401


def test_aigymtrainer_constructs_without_io() -> None:
    """``AIGymTrainer()`` initialises components without I/O blocking."""
    import main

    trainer = main.AIGymTrainer()
    # Documented attributes from main.py:838-867
    assert trainer.config is not None
    assert trainer.pose_analyzer is not None
    assert trainer.ml_model is not None
    assert trainer.rep_counter is not None
    # State defaults
    assert trainer.exercise_instance is None
    assert trainer.visualizer is None
    assert trainer.exercise_type == 'bicep_curl'
    assert trainer.age_group == 'adult'


@pytest.mark.parametrize("exercise,age", [
    ('bicep_curl', 25),
    ('bent_over_row', 25),
    ('push_up', 25),
])
def test_setup_exercise_assigns_instance(exercise: str, age: int) -> None:
    """``setup_exercise`` must populate ``exercise_instance`` for all 3 exercises."""
    import main

    trainer = main.AIGymTrainer()
    trainer.setup_exercise(exercise, age)
    assert trainer.exercise_instance is not None
    assert trainer.exercise_type == exercise
    assert trainer.age_group == 'adult'  # 25 → adult per get_exercise_config


def test_process_frame_on_solid_frame_returns_tuple() -> None:
    """``_process_frame`` on a solid (no-pose) frame returns ``(frame, {})``."""
    import collections
    import main

    trainer = main.AIGymTrainer()
    trainer.setup_exercise('bicep_curl', 25)
    frame = np.full((120, 160, 3), 32, dtype=np.uint8)
    angle_history: collections.deque[float] = collections.deque(maxlen=30)

    annotated, info = trainer._process_frame(frame, fps=30.0, angle_history=angle_history)

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == frame.shape
    assert isinstance(info, dict)
    # No pose detected on a solid frame → analysis_results stays empty.
    assert info == {}
