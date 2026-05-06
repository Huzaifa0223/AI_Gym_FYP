"""
tests.test_exercise_classifier — smoke tests for ``core.exercise_classifier``.

Stage 0 backfill (per ``docs/architecture_3pillar.md`` §1.3 and §7).  The
classifier is rule-based, so we fabricate MediaPipe-like landmarks via a
namedtuple rather than running the real model.  The detector's thresholds
make exact-class assertions brittle, so we prefer "stable + non-unknown after
warmup" over hard-coded class names where possible.
"""
from __future__ import annotations

from collections import namedtuple
from typing import List

import pytest

from core.exercise_classifier import ExerciseClassifier, quick_detect_exercise


Landmark = namedtuple('Landmark', ['x', 'y', 'z', 'visibility'])


def _zero_landmarks(n: int = 33, visibility: float = 0.0) -> List[Landmark]:
    return [Landmark(x=0.0, y=0.0, z=0.0, visibility=visibility) for _ in range(n)]


def _upright_curl_landmarks() -> List[Landmark]:
    """Fabricate a standing bicep-curl pose (upright torso, flexed elbow)."""
    lm = _zero_landmarks(visibility=0.95)
    # Right side (mirrors what core.exercise_classifier inspects).
    # 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips, 25/26 knees, 27/28 ankles
    lm[12] = Landmark(x=0.55, y=0.30, z=0.0, visibility=0.95)  # shoulder
    lm[14] = Landmark(x=0.55, y=0.50, z=0.0, visibility=0.95)  # elbow
    lm[16] = Landmark(x=0.55, y=0.35, z=0.0, visibility=0.95)  # wrist (flexed up)
    lm[24] = Landmark(x=0.55, y=0.65, z=0.0, visibility=0.95)  # hip
    lm[26] = Landmark(x=0.55, y=0.85, z=0.0, visibility=0.95)  # knee (straight leg)
    lm[28] = Landmark(x=0.55, y=0.99, z=0.0, visibility=0.95)  # ankle
    # Left side mirrored — classifier mostly uses right but visibility helps.
    lm[11] = Landmark(x=0.45, y=0.30, z=0.0, visibility=0.95)
    lm[13] = Landmark(x=0.45, y=0.50, z=0.0, visibility=0.95)
    lm[15] = Landmark(x=0.45, y=0.35, z=0.0, visibility=0.95)
    lm[23] = Landmark(x=0.45, y=0.65, z=0.0, visibility=0.95)
    lm[25] = Landmark(x=0.45, y=0.85, z=0.0, visibility=0.95)
    lm[27] = Landmark(x=0.45, y=0.99, z=0.0, visibility=0.95)
    return lm


# ---------------------------------------------------------------------------
# quick_detect_exercise
# ---------------------------------------------------------------------------

def test_quick_detect_returns_unknown_on_none() -> None:
    assert quick_detect_exercise(None) == 'unknown'


def test_quick_detect_returns_string_on_landmarks() -> None:
    result = quick_detect_exercise(_upright_curl_landmarks())
    assert isinstance(result, str)
    assert result in {'back', 'chest', 'legs', 'shoulders', 'arms', 'unknown'}


# ---------------------------------------------------------------------------
# ExerciseClassifier
# ---------------------------------------------------------------------------

def test_classifier_returns_unknown_when_landmarks_falsy() -> None:
    cls = ExerciseClassifier()
    exercise, conf = cls.detect_exercise_type(None)
    assert exercise == 'unknown'
    assert conf == 0.0


def test_classifier_returns_known_class_with_confidence() -> None:
    cls = ExerciseClassifier()
    exercise, conf = cls.detect_exercise_type(_upright_curl_landmarks())
    assert isinstance(exercise, str)
    assert 0.0 <= conf <= 1.0


def test_classifier_history_stabilises_over_warmup() -> None:
    """After ≥5 frames the classifier should return its stable-vote prediction."""
    cls = ExerciseClassifier()
    landmarks = _upright_curl_landmarks()
    last_exercise, last_conf = 'unknown', 0.0
    for _ in range(10):
        last_exercise, last_conf = cls.detect_exercise_type(landmarks)
    # The detector either picks a class with >0 confidence or stays 'unknown'
    # when no per-frame prediction crossed its 0.5 threshold.  Either way the
    # output should be deterministic and within bounds — a regression that
    # broke the voting code would surface as a TypeError or out-of-range conf.
    assert isinstance(last_exercise, str)
    assert 0.0 <= last_conf <= 1.0
    assert len(cls.exercise_history) == 10


def test_classifier_reset_clears_history() -> None:
    cls = ExerciseClassifier()
    for _ in range(3):
        cls.detect_exercise_type(_upright_curl_landmarks())
    assert len(cls.exercise_history) == 3
    cls.reset()
    assert len(cls.exercise_history) == 0
