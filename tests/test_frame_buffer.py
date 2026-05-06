"""
tests.test_frame_buffer — LatestFrameBuffer set/get/clear semantics.
"""
from __future__ import annotations

import numpy as np

from core.frame_buffer import LatestFrameBuffer


class TestLatestFrameBuffer:
    def test_initial_state_is_none(self) -> None:
        buf = LatestFrameBuffer()
        assert buf.get() is None

    def test_set_and_get_round_trips(self) -> None:
        buf = LatestFrameBuffer()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        buf.set(frame, frame_id=42, timestamp_ms=1234)

        snapshot = buf.get()
        assert snapshot is not None
        f, fid, ts = snapshot
        assert f is frame
        assert fid == 42
        assert ts == 1234

    def test_set_overwrites_previous(self) -> None:
        buf = LatestFrameBuffer()
        f1 = np.zeros((4, 4, 3), dtype=np.uint8)
        f2 = np.ones((4, 4, 3), dtype=np.uint8)

        buf.set(f1, 1, 100)
        buf.set(f2, 2, 200)

        snapshot = buf.get()
        assert snapshot is not None
        f, fid, ts = snapshot
        assert f is f2
        assert fid == 2
        assert ts == 200

    def test_clear_resets_to_none(self) -> None:
        buf = LatestFrameBuffer()
        buf.set(np.zeros((4, 4, 3), dtype=np.uint8), 1, 100)
        buf.clear()
        assert buf.get() is None

    def test_clear_idempotent(self) -> None:
        buf = LatestFrameBuffer()
        buf.clear()
        buf.clear()
        assert buf.get() is None
