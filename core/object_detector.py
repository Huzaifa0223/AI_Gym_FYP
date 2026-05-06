"""
core.object_detector — equipment detector with a swappable backend.

Three layers:

* :class:`Detector`               — Protocol; one method ``detect``.
* :class:`Yolov8NanoDetector`     — production. Lazy-imports
                                    ``ultralytics``; the ``YOLO`` model is
                                    not loaded until the first
                                    :meth:`detect` call so unit tests can
                                    construct without pulling weights or
                                    the framework into memory.
* :class:`StubDetector`           — deterministic test seam returning a
                                    canned :class:`EquipmentDetection`
                                    tuple regardless of input.
* :class:`EquipmentDetector`      — public-facing wrapper that holds any
                                    :class:`Detector` and applies an
                                    optional class allowlist. Swapping
                                    weights (COCO → fine-tuned gym
                                    dataset) becomes a config change at
                                    construction time.

The DI pattern mirrors :class:`training.pose_extractor.PoseExtractor`:
production class is real but offline-construct-friendly; stub is the
behavioural test seam.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import numpy as np

from core.schemas import EquipmentDetection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Detector(Protocol):
    """One frame in, a tuple of equipment detections out."""

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int,
    ) -> tuple[EquipmentDetection, ...]: ...


# ---------------------------------------------------------------------------
# Production — YOLOv8 (lazy import, lazy weights load)
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt"
)


class Yolov8NanoDetector:
    """YOLOv8-nano backed by ultralytics. Construction is cheap.

    The weights file and the ``ultralytics`` package are only touched on
    the first :meth:`detect` call — unit tests can construct freely
    without either being available.

    Args:
        weights_path:    Path to a YOLOv8 ``.pt`` file. Defaults to
                         ``models/yolov8n.pt`` at repo root.
        conf_threshold:  Minimum YOLO confidence to keep a detection.

    Notes on COCO classes:
        Stock YOLOv8-nano weights are trained on COCO. **None of**
        dumbbells, kettlebells, barbells, weight benches, or yoga mats are
        COCO classes. With COCO weights this detector will mostly emit
        ``person``, ``bottle``, ``backpack``, ``sports ball`` — the gym-
        relevant work is the model-fine-tune queued in ``TODO.md`` (viva
        blocker). Stage 4 ships the framework only.
    """

    def __init__(
        self,
        weights_path: Path | str | None = None,
        *,
        conf_threshold: float = 0.25,
    ) -> None:
        self._weights_path = (
            Path(weights_path) if weights_path is not None
            else _DEFAULT_WEIGHTS_PATH
        )
        self._conf_threshold = conf_threshold
        self._model: object | None = None

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int,
    ) -> tuple[EquipmentDetection, ...]:
        """Run YOLO inference on *frame* and return COCO detections."""
        if self._model is None:
            self._model = self._load_model()

        # ultralytics' YOLO callable returns a list of Results objects, one
        # per input image. We pass a single frame so only the first result
        # is consumed.
        results = self._model(  # type: ignore[operator]
            frame,
            verbose=False,
            conf=self._conf_threshold,
        )
        out: list[EquipmentDetection] = []
        names = getattr(self._model, "names", {})
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_idx = int(box.cls[0])
                label = str(names.get(cls_idx, f"class_{cls_idx}"))
                conf = float(box.conf[0])
                xy = box.xyxy[0].tolist()
                out.append(EquipmentDetection(
                    label=label,
                    confidence=conf,
                    bbox_xyxy=(float(xy[0]), float(xy[1]),
                               float(xy[2]), float(xy[3])),
                    frame_id_detected=frame_id,
                ))
        return tuple(out)

    def _load_model(self) -> object:
        """Lazy-load ultralytics + the weights file."""
        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"YOLOv8 weights not found at {self._weights_path}. "
                "Fetch with `python -m scripts.fetch_yolo_weights`."
            )
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Install via "
                "`pip install ultralytics==8.3.0`."
            ) from exc
        logger.info("Loading YOLOv8 weights from %s", self._weights_path)
        return YOLO(str(self._weights_path))


# ---------------------------------------------------------------------------
# Stub — deterministic test seam
# ---------------------------------------------------------------------------

class StubDetector:
    """Test stub returning a canned detection tuple unchanged.

    Args:
        canned: Detections to return on every :meth:`detect` call.
                The tuple is returned as-is — its ``frame_id_detected``
                values are *not* re-stamped, so tests have full control
                over what comes back.
    """

    def __init__(self, canned: tuple[EquipmentDetection, ...] = ()) -> None:
        self._canned = canned

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int,
    ) -> tuple[EquipmentDetection, ...]:
        return self._canned


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class EquipmentDetector:
    """Public-facing equipment detector with an optional class allowlist.

    The wrapper holds any :class:`Detector` implementation. When an
    allowlist is configured, only detections whose ``label`` appears in it
    are forwarded to the caller. Swapping COCO weights for a fine-tuned
    gym-equipment model becomes a constructor-arg change.

    Args:
        detector:        The underlying :class:`Detector`.
        allowed_classes: Iterable of acceptable label strings, or ``None``
                         to disable filtering. Comparison is case-sensitive.
    """

    def __init__(
        self,
        detector: Detector,
        *,
        allowed_classes: tuple[str, ...] | None = None,
    ) -> None:
        self._detector = detector
        self._allowed: frozenset[str] | None = (
            frozenset(allowed_classes) if allowed_classes is not None else None
        )

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int,
    ) -> tuple[EquipmentDetection, ...]:
        raw = self._detector.detect(frame, frame_id)
        if self._allowed is None:
            return raw
        return tuple(d for d in raw if d.label in self._allowed)


__all__ = [
    "Detector",
    "Yolov8NanoDetector",
    "StubDetector",
    "EquipmentDetector",
]
