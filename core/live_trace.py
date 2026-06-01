"""core.live_trace — opt-in per-frame trace for the live scoring path.

The live ``POST /api/live-frame`` path is otherwise write-nothing: there is no
artefact to inspect after a real camera run, which is exactly why the deployed
streaming-segmenter config could never be validated against real live cadence
(it was tuned on 30 fps batch clips). This module adds an **opt-in,
behaviour-neutral** trace: when ``AI_GYM_LIVE_TRACE`` is set, every live frame
appends one JSON object (a JSONL line) capturing the raw + smoothed angle, the
observed fps, the primary-joint visibility, the segmenter's internal state, the
gate decision, and the latency. Disabled (the env var unset) it is a no-op, so
demo / production runs are unaffected.

Enable:
    AI_GYM_LIVE_TRACE=1                      -> writes ``logs/live_trace.jsonl``
    AI_GYM_LIVE_TRACE=path/to/trace.jsonl    -> writes that file

Each line is one frame; analyse with pandas (``pd.read_json(path, lines=True)``)
to pick the gate threshold and decide the smoothing / prominence fixes from real
numbers.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENV_VAR = "AI_GYM_LIVE_TRACE"
_DEFAULT_PATH = Path("logs") / "live_trace.jsonl"
_TRUTHY = {"1", "true", "True", "yes", "on"}


class LiveTracer:
    """Append-only JSONL writer for live per-frame telemetry.

    One instance per process (created in the FastAPI lifespan). When *path* is
    ``None`` the tracer is disabled and :meth:`record` is a no-op, so call sites
    can invoke it unconditionally without a feature check.

    Args:
        path: Destination JSONL file, or ``None`` to disable tracing.
    """

    def __init__(self, path: Path | None) -> None:
        self._path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Live trace ENABLED -> %s", path)

    @classmethod
    def from_env(cls) -> "LiveTracer":
        """Build a tracer from ``AI_GYM_LIVE_TRACE`` (see the module docstring).

        Unset/empty -> disabled. ``1``/``true``/``yes``/``on`` -> the default
        ``logs/live_trace.jsonl``. Anything else is treated as the target path.
        """
        raw = os.environ.get(_ENV_VAR, "").strip()
        if not raw:
            return cls(None)
        path = _DEFAULT_PATH if raw in _TRUTHY else Path(raw)
        return cls(path)

    @property
    def enabled(self) -> bool:
        """``True`` when frames are being written to a file."""
        return self._path is not None

    def record(self, fields: dict[str, Any]) -> None:
        """Append one frame record as a JSON line (no-op when disabled).

        A ``wall_ts`` (ISO-8601) is prepended to every record. Write/serialise
        failures are logged and swallowed — tracing must never break the live
        request path.
        """
        if self._path is None:
            return
        record = {"wall_ts": datetime.now().isoformat(), **fields}
        try:
            line = json.dumps(record, default=str)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except (OSError, TypeError, ValueError):
            logger.exception("LiveTracer failed to write a frame record")


__all__ = ["LiveTracer"]
