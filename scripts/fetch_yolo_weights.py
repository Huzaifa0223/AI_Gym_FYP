"""
scripts.fetch_yolo_weights — download YOLOv8-nano weights to ``models/``.

Fails loud on network error rather than silently leaving the user with a
stale or missing weights file. If the target already exists, exits success
without downloading.

Usage:
    python -m scripts.fetch_yolo_weights
"""
from __future__ import annotations

import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


# Pinned to the same major as ``ultralytics==8.3.0`` in requirements.txt.
# The release tag below is the pinned weights drop, not the package version.
_DOWNLOAD_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TARGET = _REPO_ROOT / "models" / "yolov8n.pt"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    del argv  # currently unused; reserved for --force etc.

    _TARGET.parent.mkdir(parents=True, exist_ok=True)

    if _TARGET.exists():
        logger.info("YOLOv8 weights already present at %s", _TARGET)
        return 0

    logger.info("Downloading YOLOv8 weights from %s", _DOWNLOAD_URL)
    try:
        urllib.request.urlretrieve(_DOWNLOAD_URL, _TARGET)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        # Fail loud — do not leave a partial file behind.
        if _TARGET.exists():
            try:
                _TARGET.unlink()
            except OSError:
                pass
        logger.error("Download failed: %s", exc)
        return 1

    logger.info("Saved %s", _TARGET)
    return 0


if __name__ == "__main__":
    sys.exit(main())
