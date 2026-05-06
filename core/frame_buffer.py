"""
core.frame_buffer — single-slot buffer for the latest decoded frame.

Per-frame request handlers (e.g. ``POST /api/process-frame``) write the
freshly-decoded frame here; the equipment-detection background task
reads on its 1Hz tick. Same lock-free CPython pattern as
:class:`core.equipment_pipeline.EquipmentStateHolder`: replacing a
single attribute reference is GIL-atomic, so readers always observe a
consistent snapshot without locking.

The buffer is a single slot — late writes overwrite earlier ones. The
equipment loop only cares about the most recent frame.
"""
from __future__ import annotations

import numpy as np


class LatestFrameBuffer:
    """Single-slot buffer for the latest decoded frame.

    The slot stores ``(frame, frame_id, timestamp_ms)`` or ``None`` when
    no frame has been written yet. The shape matches the
    :data:`core.equipment_pipeline.FrameSource` callable contract, so a
    bound method like ``buffer.get`` can be passed directly.
    """

    def __init__(self) -> None:
        self._slot: tuple[np.ndarray, int, int] | None = None

    def get(self) -> tuple[np.ndarray, int, int] | None:
        """Return the latest snapshot, or ``None`` when none has been set."""
        return self._slot

    def set(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
    ) -> None:
        """Replace the slot with a fresh frame snapshot."""
        self._slot = (frame, frame_id, timestamp_ms)

    def clear(self) -> None:
        """Reset the slot to empty. Idempotent."""
        self._slot = None


__all__ = ["LatestFrameBuffer"]
