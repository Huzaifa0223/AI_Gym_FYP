"""
tests.test_threshold_provider — disk lookup, cache, runtime-shadows-seed,
fail-loud paths.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.schemas import HeuristicThresholds
from core.threshold_provider import (
    MalformedThresholdsError,
    ThresholdProvider,
    ThresholdsNotFoundError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _good_threshold_dict(exercise: str = "bicep_curl") -> dict:
    return {
        "exercise": exercise,
        "rom_band": [80.0, 160.0],
        "speed_band": [1.0, 5.0],
        "tempo_stability_max_cv": 0.6,
        "tempo_symmetry_min_ratio": 0.7,
        "exercise_specific": {},
        "sources": {
            "rom_band": "derived_from_video",
            "speed_band": "derived_from_video",
            "tempo_stability_max_cv": "derived_from_video",
            "tempo_symmetry_min_ratio": "derived_from_video",
        },
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadFromSeed:
    def test_load_returns_heuristic_thresholds(self, tmp_path: Path) -> None:
        seed = tmp_path / "bicep_curl" / "seed_thresholds.json"
        _write_json(seed, _good_threshold_dict())

        provider = ThresholdProvider(exercises_dir=tmp_path)
        result = provider.load("bicep_curl")

        assert isinstance(result, HeuristicThresholds)
        assert result.exercise == "bicep_curl"
        assert result.rom_band == (80.0, 160.0)


class TestCache:
    def test_second_load_does_not_reread_disk(self, tmp_path: Path) -> None:
        seed = tmp_path / "bicep_curl" / "seed_thresholds.json"
        _write_json(seed, _good_threshold_dict())

        provider = ThresholdProvider(exercises_dir=tmp_path)
        first = provider.load("bicep_curl")

        # After the first load, replace the file with garbage. If the cache
        # works, the second load should still succeed and return the same
        # object the first call cached.
        with seed.open("w", encoding="utf-8") as f:
            f.write("{ malformed json")

        second = provider.load("bicep_curl")
        assert second == first

    def test_reload_bypasses_cache(self, tmp_path: Path) -> None:
        seed = tmp_path / "bicep_curl" / "seed_thresholds.json"
        _write_json(seed, _good_threshold_dict())

        provider = ThresholdProvider(exercises_dir=tmp_path)
        first = provider.load("bicep_curl")

        # Rewrite the file with different rom_band values.
        new_data = _good_threshold_dict()
        new_data["rom_band"] = [70.0, 150.0]
        _write_json(seed, new_data)

        reloaded = provider.reload("bicep_curl")
        assert reloaded != first
        assert reloaded.rom_band == (70.0, 150.0)


class TestRuntimeShadowsSeed:
    def test_runtime_file_takes_precedence(self, tmp_path: Path) -> None:
        seed = tmp_path / "bicep_curl" / "seed_thresholds.json"
        runtime = tmp_path / "bicep_curl" / "heuristic_thresholds.json"

        seed_data = _good_threshold_dict()
        seed_data["rom_band"] = [80.0, 160.0]
        _write_json(seed, seed_data)

        runtime_data = _good_threshold_dict()
        runtime_data["rom_band"] = [90.0, 150.0]  # different
        _write_json(runtime, runtime_data)

        provider = ThresholdProvider(exercises_dir=tmp_path)
        result = provider.load("bicep_curl")

        assert result.rom_band == (90.0, 150.0)


class TestNotFound:
    def test_missing_exercise_raises(self, tmp_path: Path) -> None:
        provider = ThresholdProvider(exercises_dir=tmp_path)
        with pytest.raises(ThresholdsNotFoundError):
            provider.load("nonexistent_exercise")


class TestMalformed:
    def test_malformed_json_raises_with_path(self, tmp_path: Path) -> None:
        seed = tmp_path / "bicep_curl" / "seed_thresholds.json"
        seed.parent.mkdir(parents=True, exist_ok=True)
        seed.write_text("{ this is not valid json", encoding="utf-8")

        provider = ThresholdProvider(exercises_dir=tmp_path)
        with pytest.raises(MalformedThresholdsError) as exc_info:
            provider.load("bicep_curl")

        assert "seed_thresholds.json" in str(exc_info.value)

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        seed = tmp_path / "bicep_curl" / "seed_thresholds.json"
        bad = _good_threshold_dict()
        del bad["rom_band"]
        _write_json(seed, bad)

        provider = ThresholdProvider(exercises_dir=tmp_path)
        with pytest.raises(MalformedThresholdsError, match="rom_band"):
            provider.load("bicep_curl")

    def test_invalid_source_value_raises(self, tmp_path: Path) -> None:
        # __post_init__ validation in HeuristicThresholds rejects unknown
        # source values — the provider must surface those as
        # MalformedThresholdsError tagged with the file path.
        seed = tmp_path / "bicep_curl" / "seed_thresholds.json"
        bad = _good_threshold_dict()
        bad["sources"]["rom_band"] = "made_up_source"
        _write_json(seed, bad)

        provider = ThresholdProvider(exercises_dir=tmp_path)
        with pytest.raises(MalformedThresholdsError) as exc_info:
            provider.load("bicep_curl")

        assert "seed_thresholds.json" in str(exc_info.value)


class TestSeedNoteFieldDropped:
    def test_seed_note_does_not_break_construction(self, tmp_path: Path) -> None:
        # The committed seed_thresholds.json includes a `_seed_note`
        # commentary field; the provider must drop underscore-prefixed keys
        # before passing to HeuristicThresholds (which would otherwise raise
        # TypeError on an unexpected kwarg).
        seed = tmp_path / "bicep_curl" / "seed_thresholds.json"
        data = _good_threshold_dict()
        data["_seed_note"] = "Placeholder seed; calibrator overwrites at runtime."
        _write_json(seed, data)

        provider = ThresholdProvider(exercises_dir=tmp_path)
        result = provider.load("bicep_curl")
        assert result.exercise == "bicep_curl"
