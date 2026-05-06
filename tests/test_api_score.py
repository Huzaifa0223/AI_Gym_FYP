"""
tests.test_api_score — integration tests for the POST /api/score endpoint.

Uses FastAPI's ``TestClient`` against the in-process app and synthesises
videos with ``cv2.VideoWriter`` at runtime so no binary fixtures are
committed.  The synthetic clips are solid-colour frames that MediaPipe
cannot detect a person in — this is intentional.  The suite exercises the
pipeline plumbing (request parsing, validation, error mapping, response
schema), not pose accuracy.
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FPS = 30
_WIDTH = 160
_HEIGHT = 120


def _write_solid_video(path: Path, *, seconds: float = 2.0, fps: int = _FPS) -> None:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (_WIDTH, _HEIGHT))
    if not writer.isOpened():
        pytest.skip("MJPG VideoWriter unavailable in this OpenCV build")

    try:
        for i in range(int(round(seconds * fps))):
            shade = 20 + (i % 200)
            writer.write(np.full((_HEIGHT, _WIDTH, 3), shade, dtype=np.uint8))
    finally:
        writer.release()


@pytest.fixture
def synthetic_video_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "synthetic.avi"
    _write_solid_video(path, seconds=2.0, fps=_FPS)
    return path.read_bytes()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def _post_score(
    client: TestClient,
    *,
    video_bytes: bytes | None,
    exercise: str | None = "bicep_curl",
    age_group: str | None = "adult",
    filename: str = "clip.avi",
    content_type: str = "video/x-msvideo",
    include_video: bool = True,
):
    files = {}
    if include_video:
        files["video"] = (filename, video_bytes or b"", content_type)
    data = {}
    if exercise is not None:
        data["exercise"] = exercise
    if age_group is not None:
        data["age_group"] = age_group
    return client.post("/api/score", files=files, data=data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHappyPath:
    """1 & 8. Valid request returns 200 + schema with frames_processed > 0."""

    def test_valid_bicep_adult_returns_200(
        self, client: TestClient, synthetic_video_bytes: bytes
    ) -> None:
        resp = _post_score(client, video_bytes=synthetic_video_bytes)
        assert resp.status_code == 200, resp.text

        body = resp.json()
        assert body["exercise"] == "bicep_curl"
        assert body["age_group"] == "adult"
        assert isinstance(body["reps"], list)
        # Synthetic clip has no pose, so zero reps is expected.
        assert body["total_reps"] == 0
        assert body["average_score"] is None
        assert body["frames_processed"] > 0
        assert body["frames_with_landmarks"] == 0
        assert body["processing_time_s"] >= 0.0

        # Schema keys exist
        for key in (
            "exercise", "age_group", "total_reps", "average_score",
            "frames_processed", "frames_with_landmarks",
            "processing_time_s", "reps",
        ):
            assert key in body


class TestInvalidExercise:
    """2. Unknown exercise → 400."""

    def test_invalid_exercise_returns_400(
        self, client: TestClient, synthetic_video_bytes: bytes
    ) -> None:
        resp = _post_score(
            client, video_bytes=synthetic_video_bytes, exercise="deadlift"
        )
        assert resp.status_code == 400
        assert "deadlift" in resp.json()["detail"]


class TestInvalidAgeGroup:
    """3. Unknown age_group → 400."""

    def test_invalid_age_group_returns_400(
        self, client: TestClient, synthetic_video_bytes: bytes
    ) -> None:
        resp = _post_score(
            client, video_bytes=synthetic_video_bytes, age_group="infant"
        )
        assert resp.status_code == 400
        assert "infant" in resp.json()["detail"]


class TestMissingVideoField:
    """4. Missing video form field → 422."""

    def test_missing_video_returns_422(self, client: TestClient) -> None:
        resp = _post_score(
            client, video_bytes=None, include_video=False
        )
        assert resp.status_code == 422


class TestEmptyVideo:
    """5. Empty body → 400."""

    def test_empty_video_returns_400(self, client: TestClient) -> None:
        resp = _post_score(client, video_bytes=b"")
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()


class TestUndecodableBytes:
    """6. Text-file bytes with a video content-type → 400 (decode error)."""

    def test_non_video_bytes_returns_400(self, client: TestClient) -> None:
        resp = _post_score(
            client, video_bytes=b"this is definitely not a video file",
        )
        assert resp.status_code == 400


class TestOversizedUpload:
    """7. Patched MAX_VIDEO_BYTES → 413 for anything over the cap."""

    def test_oversized_upload_returns_413(
        self,
        client: TestClient,
        synthetic_video_bytes: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(api_backend, "MAX_VIDEO_BYTES", 100)
        resp = _post_score(client, video_bytes=synthetic_video_bytes)
        assert resp.status_code == 413
        assert "exceed" in resp.json()["detail"].lower() or \
               "limit" in resp.json()["detail"].lower()


class TestUnsupportedMediaType:
    """Content-type that isn't video/* is rejected with 415."""

    def test_image_content_type_returns_415(
        self, client: TestClient, synthetic_video_bytes: bytes
    ) -> None:
        resp = _post_score(
            client,
            video_bytes=synthetic_video_bytes,
            content_type="image/png",
        )
        assert resp.status_code == 415
