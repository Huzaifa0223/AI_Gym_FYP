"""
tests.test_api_5c — Stage 5c API surface.

Covers the new endpoints (``GET /api/reference/{id}``, ``GET /api/equipment``)
and the lifespan-initialised ``app.state``. The /api/score happy path is
already covered by the pre-existing ``tests/test_api_score.py``; this
suite adds the headline-population assertions on top.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure the API-key middleware does not challenge us during tests.
os.environ.pop("AI_GYM_API_KEY", None)

import api_backend  # noqa: E402
from api_backend import app  # noqa: E402
from core.equipment_pipeline import EquipmentState  # noqa: E402
from core.schemas import EquipmentDetection  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FPS = 30
_WIDTH = 160
_HEIGHT = 120


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def _write_solid_video(path: Path, *, seconds: float = 1.0) -> bytes:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, _FPS, (_WIDTH, _HEIGHT))
    if not writer.isOpened():
        pytest.skip("MJPG VideoWriter unavailable in this OpenCV build")
    try:
        for i in range(int(seconds * _FPS)):
            shade = 20 + (i % 200)
            writer.write(np.full((_HEIGHT, _WIDTH, 3), shade, dtype=np.uint8))
    finally:
        writer.release()
    return path.read_bytes()


@pytest.fixture
def synthetic_video_bytes(tmp_path: Path) -> bytes:
    return _write_solid_video(tmp_path / "synthetic.avi")


# ---------------------------------------------------------------------------
# /api/score — populated headline
# ---------------------------------------------------------------------------

class TestScoreHeadlinePopulated:
    def test_response_includes_populated_headline(
        self, client: TestClient, synthetic_video_bytes: bytes,
    ) -> None:
        resp = client.post(
            "/api/score",
            files={"video": ("clip.avi", synthetic_video_bytes, "video/x-msvideo")},
            data={"exercise": "bicep_curl", "age_group": "adult"},
        )
        assert resp.status_code == 200
        body = resp.json()
        # New field is present and non-null.
        assert body["headline"] is not None
        headline = body["headline"]
        # Schema fields populated.
        assert "rep_count" in headline
        assert "score" in headline
        assert "form_feedback" in headline
        assert "source_breakdown" in headline
        # Solid video → 0 reps → defensible "no reps" headline.
        assert headline["rep_count"] == 0
        assert headline["score"] == 0
        assert headline["form_feedback"] == "No reps detected"
        assert headline["source_breakdown"] == {
            "rules": 0, "rf": None, "cnn_lstm": None,
        }


# ---------------------------------------------------------------------------
# /api/score — graceful degradation
# ---------------------------------------------------------------------------

class TestScoreThresholdsMissing:
    def test_missing_thresholds_returns_422(
        self, client: TestClient, tmp_path: Path,
        synthetic_video_bytes: bytes,
    ) -> None:
        # Override the lifespan-installed threshold provider with one
        # pointing at an empty dir, forcing ThresholdsNotFoundError.
        from core.threshold_provider import ThresholdProvider
        original = app.state.threshold_provider
        app.state.threshold_provider = ThresholdProvider(exercises_dir=tmp_path)
        try:
            resp = client.post(
                "/api/score",
                files={"video": ("clip.avi", synthetic_video_bytes, "video/x-msvideo")},
                data={"exercise": "bicep_curl", "age_group": "adult"},
            )
            assert resp.status_code == 422
            assert "thresholds" in resp.text.lower() or "bicep_curl" in resp.text
        finally:
            app.state.threshold_provider = original


# ---------------------------------------------------------------------------
# /api/reference/{exercise_id}
# ---------------------------------------------------------------------------

class TestReferenceSkeleton:
    def test_bicep_curl_returns_placeholder(self, client: TestClient) -> None:
        resp = client.get("/api/reference/bicep_curl")
        assert resp.status_code == 200
        body = resp.json()
        assert body["exercise"] == "bicep_curl"
        assert "frames" in body
        assert isinstance(body["frames"], list)
        assert len(body["frames"]) == 4  # one per non-IDLE FSM phase
        phases = [f["phase"] for f in body["frames"]]
        assert phases == ["DESCENDING", "BOTTOM", "ASCENDING", "TOP"]
        # Each phase has 33 landmarks + 33 visibility scores.
        for f in body["frames"]:
            assert len(f["landmarks"]) == 33
            assert len(f["visibility"]) == 33

    def test_unknown_exercise_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/reference/nonexistent_exercise")
        assert resp.status_code == 404
        assert "nonexistent_exercise" in resp.text


# ---------------------------------------------------------------------------
# /api/equipment
# ---------------------------------------------------------------------------

class TestEquipmentEndpoint:
    def test_initial_state_returns_sentinel(self, client: TestClient) -> None:
        resp = client.get("/api/equipment")
        assert resp.status_code == 200
        body = resp.json()
        # Sentinel: no detections, last_updated_* < 0.
        assert body["detections"] == []
        assert body["last_updated_frame_id"] < 0
        assert body["last_updated_timestamp_ms"] < 0

    def test_returns_holder_state_after_manual_set(
        self, client: TestClient,
    ) -> None:
        # Bypass the YOLO loop and write a state directly to the holder so
        # the endpoint has something concrete to return.
        det = EquipmentDetection(
            label="dumbbell",
            confidence=0.92,
            bbox_xyxy=(10.0, 20.0, 110.0, 220.0),
            frame_id_detected=42,
        )
        original = app.state.equipment_holder.get()
        try:
            app.state.equipment_holder.set(EquipmentState(
                detections=(det,),
                last_updated_frame_id=42,
                last_updated_timestamp_ms=4200,
            ))
            resp = client.get("/api/equipment")
            assert resp.status_code == 200
            body = resp.json()
            assert body["last_updated_frame_id"] == 42
            assert body["last_updated_timestamp_ms"] == 4200
            assert len(body["detections"]) == 1
            d = body["detections"][0]
            assert d["label"] == "dumbbell"
            assert d["confidence"] == pytest.approx(0.92)
            assert d["bbox_xyxy"] == [10.0, 20.0, 110.0, 220.0]
            assert d["frame_id_detected"] == 42
        finally:
            app.state.equipment_holder.set(original)


# ---------------------------------------------------------------------------
# Lifespan — app.state initialised during request handling
# ---------------------------------------------------------------------------

class TestLifespan:
    def test_app_state_holders_present_during_request(
        self, client: TestClient,
    ) -> None:
        # If lifespan ran, these attrs are set on app.state.
        assert hasattr(app.state, "frame_buffer")
        assert hasattr(app.state, "frame_counter")
        assert hasattr(app.state, "equipment_holder")
        assert hasattr(app.state, "equipment_detector")
        assert hasattr(app.state, "equipment_task")
        assert hasattr(app.state, "cnn_lstm_scorer")
        assert hasattr(app.state, "threshold_provider")

    def test_equipment_task_running_during_request(
        self, client: TestClient,
    ) -> None:
        # Inside the TestClient context the task is alive.
        assert not app.state.equipment_task.done()


# ---------------------------------------------------------------------------
# /api/process-frame — frame_buffer wiring
# ---------------------------------------------------------------------------

class TestProcessFrameBuffersFrame:
    def test_decoded_frame_lands_in_buffer(self, client: TestClient) -> None:
        import base64
        import io

        # Build a tiny solid PNG (no pose; pipeline returns "no pose").
        frame = np.full((120, 160, 3), 50, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", frame)
        assert ok
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        # Buffer starts empty (or carries leftover from earlier tests).
        before = app.state.frame_buffer.get()

        resp = client.post(
            "/api/process-frame",
            json={"age": 25, "frame_data": b64},
        )
        assert resp.status_code == 200

        after = app.state.frame_buffer.get()
        assert after is not None
        # frame_id monotonically advanced (or simply differs from before).
        if before is not None:
            assert after[1] > before[1]
        else:
            assert after[1] >= 0
