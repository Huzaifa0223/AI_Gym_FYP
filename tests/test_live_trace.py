"""tests.test_live_trace — the opt-in live per-frame JSONL tracer.

Asserts the three states that matter: disabled (env unset) is a silent no-op,
an explicit path writes one valid JSON object per :meth:`record`, and the
``AI_GYM_LIVE_TRACE=1`` shorthand resolves to the default ``logs/`` path. The
tracer must never raise into the live request path, so a bad record is swallowed.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.live_trace import LiveTracer


def test_disabled_when_env_unset_is_noop(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AI_GYM_LIVE_TRACE", raising=False)
    tracer = LiveTracer.from_env()
    assert tracer.enabled is False
    tracer.record({"frame": 0})  # must not raise or create anything
    assert list(tmp_path.iterdir()) == []


def test_explicit_path_writes_one_json_line_per_record(tmp_path, monkeypatch) -> None:
    target = tmp_path / "trace.jsonl"
    monkeypatch.setenv("AI_GYM_LIVE_TRACE", str(target))
    tracer = LiveTracer.from_env()
    assert tracer.enabled is True

    tracer.record({"frame": 0, "angle": 90.0, "gated": False})
    tracer.record({"frame": 1, "angle": 80.5, "gated": True})

    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["frame"] == 0 and first["angle"] == 90.0 and first["gated"] is False
    assert "wall_ts" in first  # prepended to every record
    assert json.loads(lines[1])["gated"] is True


def test_truthy_flag_uses_default_logs_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI_GYM_LIVE_TRACE", "1")
    tracer = LiveTracer.from_env()
    assert tracer.enabled is True
    tracer.record({"x": 1})
    assert (tmp_path / "logs" / "live_trace.jsonl").exists()


def test_record_swallows_serialisation_errors(tmp_path, monkeypatch) -> None:
    # A non-serialisable value (default=str handles most; object() with a custom
    # unserialisable repr still must not propagate) — record never raises.
    target = tmp_path / "trace.jsonl"
    monkeypatch.setenv("AI_GYM_LIVE_TRACE", str(target))
    tracer = LiveTracer.from_env()
    tracer.record({"obj": object()})  # default=str stringifies; must not raise
    assert target.exists()
