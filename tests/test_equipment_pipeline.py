"""
tests.test_equipment_pipeline — EquipmentState immutability,
EquipmentStateHolder atomic semantics, and the async background loop.

The async loop is exercised at high ``target_hz`` (sub-second periods) so
the suite does not spend real seconds waiting at the spec'd 1 Hz cadence.
Tests use :func:`asyncio.run` directly to avoid a ``pytest-asyncio``
dependency.
"""
from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import Awaitable, Callable, Optional, TypeVar

import numpy as np
import pytest

from core.equipment_pipeline import (
    EquipmentState,
    EquipmentStateHolder,
    run_equipment_detection_loop,
)
from core.object_detector import EquipmentDetector, StubDetector
from core.schemas import EquipmentDetection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T = TypeVar("T")


def _run(coro_factory: Callable[[], Awaitable[T]]) -> T:
    """Run *coro_factory* on a fresh asyncio loop. Returns its result."""
    return asyncio.run(coro_factory())


def _det(label: str = "dumbbell", frame_id: int = 0) -> EquipmentDetection:
    return EquipmentDetection(
        label=label,
        confidence=0.9,
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        frame_id_detected=frame_id,
    )


def _frame_factory(start_id: int = 0):
    """Return a frame_source callable that emits monotonically-increasing IDs."""
    counter = {"next": start_id}

    def source() -> Optional[tuple[np.ndarray, int, int]]:
        i = counter["next"]
        counter["next"] += 1
        return (np.zeros((4, 4, 3), dtype=np.uint8), i, i * 10)

    return source


class _CountingDetector:
    """Records each detect call. Optional sleep simulates slow inference."""

    def __init__(self, sleep_s: float = 0.0) -> None:
        self.calls = 0
        self._sleep_s = sleep_s

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int,
    ) -> tuple[EquipmentDetection, ...]:
        self.calls += 1
        if self._sleep_s > 0:
            time.sleep(self._sleep_s)
        return (_det(label="probe", frame_id=frame_id),)


# ---------------------------------------------------------------------------
# EquipmentState — frozen
# ---------------------------------------------------------------------------

class TestEquipmentStateImmutability:
    def test_detections_field_frozen(self) -> None:
        state = EquipmentState(
            detections=(_det(),),
            last_updated_frame_id=1,
            last_updated_timestamp_ms=10,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.detections = ()  # type: ignore[misc]

    def test_frame_id_field_frozen(self) -> None:
        state = EquipmentState(
            detections=(),
            last_updated_frame_id=-1,
            last_updated_timestamp_ms=-1,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.last_updated_frame_id = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EquipmentStateHolder — atomic single-slot
# ---------------------------------------------------------------------------

class TestEquipmentStateHolder:
    def test_default_initial_state(self) -> None:
        h = EquipmentStateHolder()
        s = h.get()
        assert s.detections == ()
        assert s.last_updated_frame_id == -1
        assert s.last_updated_timestamp_ms == -1

    def test_set_replaces_state(self) -> None:
        h = EquipmentStateHolder()
        new_state = EquipmentState(
            detections=(_det(),),
            last_updated_frame_id=42,
            last_updated_timestamp_ms=420,
        )
        h.set(new_state)
        assert h.get() is new_state
        assert h.get().last_updated_frame_id == 42

    def test_repeated_sets_keep_latest(self) -> None:
        h = EquipmentStateHolder()
        for fid in (1, 2, 3):
            h.set(EquipmentState(
                detections=(_det(frame_id=fid),),
                last_updated_frame_id=fid,
                last_updated_timestamp_ms=fid * 10,
            ))
        assert h.get().last_updated_frame_id == 3


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------

class TestRunEquipmentDetectionLoop:
    def test_invokes_detector_at_target_rate(self) -> None:
        holder = EquipmentStateHolder()
        counting = _CountingDetector()
        detector = EquipmentDetector(counting)

        async def _go() -> None:
            stop = asyncio.Event()
            task = asyncio.create_task(run_equipment_detection_loop(
                holder, detector, _frame_factory(),
                stop, target_hz=200.0,  # period = 5ms
            ))
            await asyncio.sleep(0.05)  # ~10 cycles
            stop.set()
            await asyncio.wait_for(task, timeout=0.5)

        _run(_go)

        # Generous slop for scheduling jitter.
        assert counting.calls >= 5
        assert holder.get().last_updated_frame_id >= 0

    def test_slow_detector_runs_back_to_back_no_crash(self) -> None:
        holder = EquipmentStateHolder()
        # Detector takes 20ms; period at 200 Hz is 5ms -> sleep_for == 0.
        slow = _CountingDetector(sleep_s=0.02)
        detector = EquipmentDetector(slow)

        async def _go() -> None:
            stop = asyncio.Event()
            task = asyncio.create_task(run_equipment_detection_loop(
                holder, detector, _frame_factory(),
                stop, target_hz=200.0,
            ))
            await asyncio.sleep(0.1)  # ~5 cycles when each takes 20ms
            stop.set()
            await asyncio.wait_for(task, timeout=0.5)

        _run(_go)

        # ~5 calls in 100ms when each takes 20ms; allow slop down to 3.
        assert slow.calls >= 3
        assert holder.get().last_updated_frame_id >= 0

    def test_graceful_shutdown_within_one_cycle(self) -> None:
        holder = EquipmentStateHolder()
        detector = EquipmentDetector(_CountingDetector())
        shutdown_time: dict[str, float] = {}

        async def _go() -> None:
            stop = asyncio.Event()
            # Period = 200ms (target_hz=5).
            task = asyncio.create_task(run_equipment_detection_loop(
                holder, detector, _frame_factory(),
                stop, target_hz=5.0,
            ))
            await asyncio.sleep(0.05)
            t0 = time.monotonic()
            stop.set()
            await asyncio.wait_for(task, timeout=0.5)
            shutdown_time["dt"] = time.monotonic() - t0

        _run(_go)

        assert shutdown_time["dt"] < 0.4  # well within one 200ms period

    def test_handles_none_frames_without_writing_state(self) -> None:
        holder = EquipmentStateHolder()
        counting = _CountingDetector()
        detector = EquipmentDetector(counting)

        def no_frame_source() -> Optional[tuple[np.ndarray, int, int]]:
            return None

        async def _go() -> None:
            stop = asyncio.Event()
            task = asyncio.create_task(run_equipment_detection_loop(
                holder, detector, no_frame_source,
                stop, target_hz=200.0,
            ))
            await asyncio.sleep(0.05)
            stop.set()
            await asyncio.wait_for(task, timeout=0.5)

        _run(_go)

        # Detector never called when frame_source returns None.
        assert counting.calls == 0
        # Holder state remains the initial sentinel.
        assert holder.get().last_updated_frame_id == -1

    def test_detector_exception_does_not_kill_loop(self) -> None:
        holder = EquipmentStateHolder()

        class _BlowingUpDetector:
            def __init__(self) -> None:
                self.calls = 0

            def detect(self, frame, frame_id):  # type: ignore[no-untyped-def]
                self.calls += 1
                raise RuntimeError("simulated inference failure")

        bad = _BlowingUpDetector()
        detector = EquipmentDetector(bad)

        async def _go() -> None:
            stop = asyncio.Event()
            task = asyncio.create_task(run_equipment_detection_loop(
                holder, detector, _frame_factory(),
                stop, target_hz=200.0,
            ))
            await asyncio.sleep(0.05)
            stop.set()
            await asyncio.wait_for(task, timeout=0.5)

        _run(_go)

        # Loop kept calling despite each detect() raising.
        assert bad.calls >= 2
