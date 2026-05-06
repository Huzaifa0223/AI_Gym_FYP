"""
core.equipment_pipeline — equipment-detection state and background loop.

Consolidates two tightly coupled concerns:

* :class:`EquipmentState`        — frozen snapshot of the most-recent
                                   detections.
* :class:`EquipmentStateHolder`  — mutable wrapper around a single
                                   :class:`EquipmentState` reference.
                                   Reads (``get``) and writes (``set``)
                                   are atomic in CPython because they
                                   replace a single attribute reference
                                   under the GIL — no lock required.
                                   Readers always see a consistent
                                   snapshot.
* :func:`run_equipment_detection_loop` — async background loop. Calls
                                   ``frame_source()`` once per cycle,
                                   dispatches the detector to a thread
                                   pool (so blocking inference does not
                                   stall the event loop), and replaces
                                   the holder's reference. Throttling is
                                   monotonic-time-aware: a slow detector
                                   "catches up" by sleeping zero between
                                   cycles instead of accumulating
                                   schedule debt.

See ``docs/architecture_3pillar.md`` §4 for the threading model.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import numpy as np

from core.object_detector import EquipmentDetector
from core.schemas import EquipmentDetection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State + holder
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EquipmentState:
    """Immutable snapshot of the most-recent detections.

    Attributes:
        detections:                Tuple of detections from the latest
                                   completed cycle. Empty when none.
        last_updated_frame_id:     Source frame index. ``-1`` until the
                                   first cycle completes.
        last_updated_timestamp_ms: Source frame timestamp. ``-1`` until
                                   the first cycle completes.
    """
    detections: tuple[EquipmentDetection, ...]
    last_updated_frame_id: int
    last_updated_timestamp_ms: int


_INITIAL_STATE = EquipmentState(
    detections=(),
    last_updated_frame_id=-1,
    last_updated_timestamp_ms=-1,
)


class EquipmentStateHolder:
    """Atomic single-slot holder for :class:`EquipmentState`.

    Single-attribute assignment in CPython is GIL-serialised, so ``set``
    and ``get`` do not need a lock — readers always observe a consistent
    snapshot.
    """

    def __init__(self, initial: EquipmentState | None = None) -> None:
        self._state: EquipmentState = (
            initial if initial is not None else _INITIAL_STATE
        )

    def get(self) -> EquipmentState:
        return self._state

    def set(self, new: EquipmentState) -> None:
        self._state = new


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------

# A frame_source callable returns either ``(frame, frame_id, timestamp_ms)``
# or ``None`` to indicate "no frame available this cycle." It is injected
# by Stage 5; Stage 4 only consumes the contract.
FrameSource = Callable[[], Optional[tuple[np.ndarray, int, int]]]


async def run_equipment_detection_loop(
    holder: EquipmentStateHolder,
    detector: EquipmentDetector,
    frame_source: FrameSource,
    stop_event: asyncio.Event,
    *,
    target_hz: float = 1.0,
) -> None:
    """Async loop that keeps *holder* up-to-date at ~*target_hz* cycles/s.

    Each iteration:

    1. Pulls a frame via ``frame_source``.
    2. Runs ``detector.detect`` in a thread-pool executor (so model
       inference does not block the event loop).
    3. Replaces the state in *holder* with a fresh snapshot.
    4. Sleeps ``max(0, period - elapsed)``; if the iteration ran longer
       than ``period``, the next one fires immediately.

    Shutdown is graceful: ``stop_event.set()`` causes the loop to exit
    within at most one detector cycle. The implementation uses
    :func:`asyncio.wait_for` on the event so a pending sleep aborts as
    soon as the event is set.

    Exceptions raised by ``frame_source`` or ``detector.detect`` are
    logged and swallowed — a transient error does not kill the loop.
    Tests should avoid relying on that and assert on the holder state
    directly.
    """
    period = 1.0 / target_hz
    loop = asyncio.get_running_loop()

    while not stop_event.is_set():
        cycle_start = loop.time()

        try:
            frame_info = frame_source()
        except Exception:  # pylint: disable=broad-except
            logger.exception("frame_source raised; skipping cycle")
            frame_info = None

        if frame_info is not None:
            frame, frame_id, timestamp_ms = frame_info
            detections: tuple[EquipmentDetection, ...] = ()
            try:
                detections = await loop.run_in_executor(
                    None, detector.detect, frame, frame_id,
                )
            except Exception:  # pylint: disable=broad-except
                logger.exception("detector.detect raised; emitting empty cycle")
            holder.set(EquipmentState(
                detections=tuple(detections),
                last_updated_frame_id=frame_id,
                last_updated_timestamp_ms=timestamp_ms,
            ))

        elapsed = loop.time() - cycle_start
        sleep_for = period - elapsed
        if sleep_for > 0:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=sleep_for)
                # stop_event was set during sleep — exit promptly.
                break
            except asyncio.TimeoutError:
                # Period elapsed normally.
                pass


__all__ = [
    "EquipmentState",
    "EquipmentStateHolder",
    "FrameSource",
    "run_equipment_detection_loop",
]
