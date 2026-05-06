"""
tests.test_object_detector — Detector protocol, lazy YOLO construction,
StubDetector behaviour, and EquipmentDetector allowlist filtering.

The :class:`Yolov8NanoDetector` is exercised only at construction time —
its :meth:`detect` would import ultralytics and load weights, which the
test suite avoids per the "no binary fixtures, model-free CI" project
constraint.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.object_detector import (
    Detector,
    EquipmentDetector,
    StubDetector,
    Yolov8NanoDetector,
)
from core.schemas import EquipmentDetection


def _det(label: str, conf: float = 0.9, frame_id: int = 0) -> EquipmentDetection:
    return EquipmentDetection(
        label=label,
        confidence=conf,
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        frame_id_detected=frame_id,
    )


# ---------------------------------------------------------------------------
# Yolov8NanoDetector — construction smoke
# ---------------------------------------------------------------------------

class TestYolov8NanoSmoke:
    def test_constructs_without_invoking_model(self) -> None:
        # Construction must not import ultralytics or load weights.
        detector = Yolov8NanoDetector(weights_path="not/needed/at/init.pt")
        assert detector is not None

    def test_constructs_with_default_path(self) -> None:
        detector = Yolov8NanoDetector()
        assert detector is not None

    def test_satisfies_detector_protocol(self) -> None:
        # Detector is @runtime_checkable, so isinstance works.
        assert isinstance(Yolov8NanoDetector(), Detector)


# ---------------------------------------------------------------------------
# StubDetector
# ---------------------------------------------------------------------------

class TestStubDetector:
    def test_returns_canned_tuple_unchanged(self) -> None:
        canned = (_det("dumbbell", conf=0.92, frame_id=7),)
        stub = StubDetector(canned)
        result = stub.detect(np.zeros((10, 10, 3), dtype=np.uint8), frame_id=99)
        assert result == canned

    def test_default_construction_returns_empty(self) -> None:
        stub = StubDetector()
        result = stub.detect(np.zeros((10, 10, 3), dtype=np.uint8), frame_id=0)
        assert result == ()

    def test_satisfies_detector_protocol(self) -> None:
        assert isinstance(StubDetector(), Detector)


# ---------------------------------------------------------------------------
# EquipmentDetector — wrapper + allowlist
# ---------------------------------------------------------------------------

class TestEquipmentDetector:
    def test_passthrough_without_allowlist(self) -> None:
        canned = (_det("person"), _det("dumbbell"), _det("cat"))
        wrapped = EquipmentDetector(StubDetector(canned))
        result = wrapped.detect(
            np.zeros((10, 10, 3), dtype=np.uint8), frame_id=0,
        )
        assert result == canned

    def test_allowlist_filters_unwanted_classes(self) -> None:
        canned = (_det("person"), _det("dumbbell"), _det("cat"))
        wrapped = EquipmentDetector(
            StubDetector(canned),
            allowed_classes=("dumbbell", "kettlebell"),
        )
        result = wrapped.detect(
            np.zeros((10, 10, 3), dtype=np.uint8), frame_id=0,
        )
        assert len(result) == 1
        assert result[0].label == "dumbbell"

    def test_allowlist_with_no_matches_returns_empty(self) -> None:
        canned = (_det("person"), _det("cat"))
        wrapped = EquipmentDetector(
            StubDetector(canned),
            allowed_classes=("dumbbell",),
        )
        result = wrapped.detect(
            np.zeros((10, 10, 3), dtype=np.uint8), frame_id=0,
        )
        assert result == ()
