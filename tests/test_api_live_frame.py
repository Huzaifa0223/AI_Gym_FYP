"""
tests.test_api_live_frame — tests for the live per-frame 3-pillar path.

Two layers:
* :class:`TestLiveSession` drives :class:`core.live_session.LiveSession` with a
  synthetic angle sequence (no MediaPipe) — asserts the StreamingRepSegmenter is
  driven, ``counter`` increments, ``rep_quality`` is ``None`` until the first rep
  then a held float, and ``feedback`` holds last-known.
* :class:`TestLiveFrameEndpoint` exercises ``POST /api/live-frame`` over the
  TestClient: the no-pose fallback (solid frame) and the success serialization
  path (MediaPipe monkeypatched to canned landmarks). Mirrors the legacy
  ``/api/process-frame`` no-pose smoke test; the legacy endpoint is untouched.
"""
from __future__ import annotations

import base64
import os
from types import SimpleNamespace
from typing import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure the API-key middleware does not challenge us during tests.
os.environ.pop("AI_GYM_API_KEY", None)

from config import RepSegmenterConfig  # noqa: E402
from core.live_session import LiveSession, NEUTRAL_FEEDBACK  # noqa: E402
from core.rep_counter import FrameFeatures  # noqa: E402
from api_backend import app  # noqa: E402

FPS = 30.0
_CFG = RepSegmenterConfig()  # default params — independent of deployed bicep tuning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_features(n_reps: int, spr: int = 60) -> list[FrameFeatures]:
    """``n_reps`` clean curls as per-frame FrameFeatures (primary oscillates)."""
    t = np.linspace(0.0, n_reps * 2 * np.pi, n_reps * spr, endpoint=False)
    primary = 100.0 + 60.0 * np.cos(t)  # 40..160 deg
    # secondary/tertiary held constant (not what the segmenter keys on).
    return [FrameFeatures(float(p), 15.0, 25.0) for p in primary]


def _solid_frame_b64(width: int = 160, height: int = 120, shade: int = 32) -> str:
    frame = np.full((height, width, 3), shade, dtype=np.uint8)
    ok, buf = cv2.imencode('.png', frame)
    assert ok, "cv2.imencode failed for solid frame"
    return base64.b64encode(buf.tobytes()).decode('ascii')


def _fake_pose_result(n: int = 33):
    """A MediaPipe-shaped result with 33 constant landmarks (no rep, just a pose)."""
    landmark = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(n)]
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmark))


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# LiveSession unit tests (no MediaPipe)
# ---------------------------------------------------------------------------

class TestLiveSession:
    def test_segmenter_is_driven_and_counts(self) -> None:
        session = LiveSession("bicep_curl", "adult", config=_CFG)
        feats = _cosine_features(5)
        for i, f in enumerate(feats):
            session.process_features(f, i / FPS)
        assert session.rep_count == 5

    def test_rep_quality_none_until_first_rep_then_held_float(self) -> None:
        session = LiveSession("bicep_curl", "adult", config=_CFG)
        feats = _cosine_features(3)

        # Before any frame: neutral state.
        assert session.last_form_score is None
        assert session.last_feedback == NEUTRAL_FEEDBACK

        first_score_idx = None
        for i, f in enumerate(feats):
            res = session.process_features(f, i / FPS)
            if res.rep_closed and first_score_idx is None:
                first_score_idx = i
            if first_score_idx is None:
                # Pre-first-rep frames stay neutral.
                assert res.rep_quality is None
                assert res.feedback == NEUTRAL_FEEDBACK

        assert first_score_idx is not None, "expected at least one rep to close"
        # After the first rep, the score is a held float and feedback changed.
        assert isinstance(session.last_form_score, float)
        assert session.last_feedback != NEUTRAL_FEEDBACK

    def test_counter_and_primary_angle_update_every_frame(self) -> None:
        session = LiveSession("bicep_curl", "adult", config=_CFG)
        feats = _cosine_features(2)
        last_counter = 0
        for i, f in enumerate(feats):
            res = session.process_features(f, i / FPS)
            assert res.primary_angle == pytest.approx(f.primary_angle)
            assert res.counter >= last_counter  # monotonic non-decreasing
            last_counter = res.counter

    def test_live_cadence_5fps_end_to_end(self) -> None:
        # The live feed runs at ~5 fps, not the 30 fps the other tests use. Feed at
        # dt=0.2s and confirm the segmenter + scorer still count and score a rep.
        session = LiveSession("bicep_curl", "adult", config=_CFG)
        feats = _cosine_features(5, spr=12)  # 12 samples/rep
        dt = 0.2
        scored = False
        for i, f in enumerate(feats):
            res = session.process_features(f, i * dt)
            if res.rep_quality is not None:
                scored = True
        assert abs(session.rep_count - 5) <= 1
        assert scored and isinstance(session.last_form_score, float)


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------

class TestLiveFrameEndpoint:
    def test_no_pose_solid_frame_returns_neutral(self, client: TestClient) -> None:
        resp = client.post(
            "/api/live-frame",
            json={
                "age": 25,
                "exercise_type": "bicep_curl",
                "auto_detect": False,
                "frame_data": _solid_frame_b64(),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["counter"] == 0
        assert body["rep_quality"] is None        # neutral, never a misleading 0%
        assert body["feedback"] == "No pose detected"
        assert body["landmarks"] is None
        assert isinstance(body["session_id"], str) and body["session_id"]

    def test_missing_exercise_type_returns_400(self, client: TestClient) -> None:
        resp = client.post(
            "/api/live-frame",
            json={"age": 25, "frame_data": _solid_frame_b64()},
        )
        assert resp.status_code == 400
        assert "exercise_type" in resp.text

    def test_success_path_shape(self, client: TestClient, monkeypatch) -> None:
        # Canned pose so we exercise the success serialization without a webcam.
        import api_backend
        monkeypatch.setattr(
            api_backend.exercise_manager.pose, "process",
            lambda _rgb: _fake_pose_result(),
        )
        resp = client.post(
            "/api/live-frame",
            json={
                "age": 25,
                "exercise_type": "bicep_curl",
                "auto_detect": False,
                "frame_data": _solid_frame_b64(),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["counter"] == 0                 # constant pose → no rep
        assert isinstance(body["primary_angle"], (int, float))
        assert body["rep_quality"] is None          # no rep closed yet
        assert isinstance(body["landmarks"], list) and len(body["landmarks"]) == 33
        assert {"x", "y", "z", "visibility"} <= set(body["landmarks"][0])
