"""
tests.test_api_process_frame — smoke tests for POST /api/process-frame.

Stage 0 backfill (per ``docs/architecture_3pillar.md`` §1.3 and §7).  The goal
is endpoint-shape coverage during the Pydantic v1 → v2 migration of
``ExerciseRequest.validate_frame_size`` — not pose-detection correctness.
A solid-colour frame is enough: MediaPipe cannot label it, so the endpoint
returns its documented "no pose detected" fallback.
"""
from __future__ import annotations

import base64
import os
from typing import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure the API-key middleware does not challenge us during tests.
os.environ.pop("AI_GYM_API_KEY", None)

from api_backend import app, MAX_FRAME_SIZE_MB  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def _solid_frame_b64(width: int = 160, height: int = 120, shade: int = 32) -> str:
    """Return base64-encoded PNG of a solid-colour frame."""
    frame = np.full((height, width, 3), shade, dtype=np.uint8)
    ok, buf = cv2.imencode('.png', frame)
    assert ok, "cv2.imencode unexpectedly failed for solid frame"
    return base64.b64encode(buf.tobytes()).decode('ascii')


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMissingFrameData:
    def test_missing_frame_data_returns_422(self, client: TestClient) -> None:
        resp = client.post("/api/process-frame", json={"age": 25})
        assert resp.status_code == 422
        body = resp.json()
        # Pydantic v2 keeps the "detail" wrapper but reshapes the error array.
        assert "detail" in body
        assert any("frame_data" in str(err) for err in body["detail"])


class TestOversizedFrame:
    def test_oversized_frame_returns_422(self, client: TestClient) -> None:
        # Generate a base64 string that would decode to > MAX_FRAME_SIZE_MB.
        # The validator estimates size as len(v) * 0.75 / 1024**2 — cross that
        # threshold by an obvious margin so we trigger the ValueError path.
        raw_size = int((MAX_FRAME_SIZE_MB + 2) * 1024 * 1024 / 0.75)
        oversized = "A" * raw_size
        resp = client.post(
            "/api/process-frame",
            json={"age": 25, "frame_data": oversized},
        )
        assert resp.status_code == 422
        body_text = resp.text.lower()
        assert "frame size" in body_text or "exceeds" in body_text


class TestAgeOutOfRange:
    @pytest.mark.parametrize("bad_age", [5, 200])
    def test_out_of_range_age_returns_422(
        self, client: TestClient, bad_age: int
    ) -> None:
        resp = client.post(
            "/api/process-frame",
            json={"age": bad_age, "frame_data": _solid_frame_b64()},
        )
        assert resp.status_code == 422
        assert "age" in resp.text.lower()


class TestSolidFrameNoPose:
    def test_valid_request_solid_frame_returns_no_pose(
        self, client: TestClient
    ) -> None:
        resp = client.post(
            "/api/process-frame",
            json={"age": 25, "frame_data": _solid_frame_b64()},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Byte-exact match with api_backend.py:362 fallback.
        assert body["success"] is False
        assert body["feedback"] == "No pose detected"
        assert body["detected_exercise"] == "unknown"
        assert body["stage"] == "unknown"
        assert body["counter"] == 0
        # Session id should be echoed back even on no-pose path.
        assert isinstance(body["session_id"], str) and body["session_id"]
