"""
core.threshold_provider — load HeuristicThresholds from disk with caching.

Each per-exercise lookup checks two files in priority order:

1. ``exercises/<exercise>/heuristic_thresholds.json`` — runtime-calibrated
   output of :mod:`training.data_collector`. May not exist before the
   calibrator has been run for this exercise.
2. ``exercises/<exercise>/seed_thresholds.json`` — committed seed in the
   :class:`HeuristicThresholds` shape, present in every supported
   exercise's directory.

Neither file present → :class:`ThresholdsNotFoundError`. JSON parse
failure or unexpected shape → :class:`MalformedThresholdsError` with the
offending file path in the message. Other shape constraints (sources
allowlist, band ordering) are validated by
:class:`HeuristicThresholds.__post_init__` already; the provider does
not duplicate that validation.

A lazy per-exercise cache is kept so the per-frame scoring path does
not re-read disk on every score call. :meth:`reload` bypasses the cache
when a calibrator run has just rewritten the file mid-session.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.schemas import HeuristicThresholds

logger = logging.getLogger(__name__)


_RUNTIME_FILENAME = "heuristic_thresholds.json"
_SEED_FILENAME = "seed_thresholds.json"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ThresholdsNotFoundError(FileNotFoundError):
    """Neither runtime nor seed thresholds file exists for the requested exercise."""


class MalformedThresholdsError(ValueError):
    """Thresholds file exists but is unreadable or has an invalid shape."""


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class ThresholdProvider:
    """Lazy disk loader with a per-exercise cache.

    Args:
        exercises_dir: Root directory holding per-exercise subfolders.
                       Defaults to ``Path("exercises")`` (relative to the
                       caller's CWD). Tests pass a ``tmp_path``.
    """

    def __init__(self, exercises_dir: Path = Path("exercises")) -> None:
        self._exercises_dir = exercises_dir
        self._cache: dict[str, HeuristicThresholds] = {}

    # -- Public API ----------------------------------------------------------

    def load(self, exercise: str) -> HeuristicThresholds:
        """Return the thresholds for *exercise*, hitting cache on repeat calls."""
        cached = self._cache.get(exercise)
        if cached is not None:
            return cached

        thresholds = self._load_from_disk(exercise)
        self._cache[exercise] = thresholds
        return thresholds

    def reload(self, exercise: str) -> HeuristicThresholds:
        """Re-read the file from disk, bypassing the cache.

        Used when the calibrator has rewritten the file mid-session.
        """
        thresholds = self._load_from_disk(exercise)
        self._cache[exercise] = thresholds
        return thresholds

    # -- Internals -----------------------------------------------------------

    def _load_from_disk(self, exercise: str) -> HeuristicThresholds:
        runtime_path = self._exercises_dir / exercise / _RUNTIME_FILENAME
        seed_path = self._exercises_dir / exercise / _SEED_FILENAME

        if runtime_path.exists():
            return self._read_thresholds_file(runtime_path)
        if seed_path.exists():
            return self._read_thresholds_file(seed_path)

        raise ThresholdsNotFoundError(
            f"No thresholds file for {exercise!r}. Looked for "
            f"{runtime_path} and {seed_path}."
        )

    @staticmethod
    def _read_thresholds_file(path: Path) -> HeuristicThresholds:
        """Parse JSON, build a :class:`HeuristicThresholds`, surface failures."""
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except json.JSONDecodeError as exc:
            raise MalformedThresholdsError(
                f"Malformed JSON in {path}: {exc}"
            ) from exc
        except OSError as exc:
            raise MalformedThresholdsError(
                f"Failed to read {path}: {exc}"
            ) from exc

        # The seed JSON includes a `_seed_note` field for human readers; drop
        # any leading-underscore key before constructing the dataclass so we
        # do not leak commentary fields into the model.
        cleaned = {k: v for k, v in raw.items() if not k.startswith("_")}

        try:
            return HeuristicThresholds(
                exercise=cleaned["exercise"],
                rom_band=tuple(cleaned["rom_band"]),
                speed_band=tuple(cleaned["speed_band"]),
                tempo_stability_max_cv=float(cleaned["tempo_stability_max_cv"]),
                tempo_symmetry_min_ratio=float(cleaned["tempo_symmetry_min_ratio"]),
                exercise_specific=dict(cleaned.get("exercise_specific", {})),
                sources=dict(cleaned["sources"]),
            )
        except KeyError as exc:
            raise MalformedThresholdsError(
                f"Missing required field in {path}: {exc}"
            ) from exc
        except (TypeError, ValueError) as exc:
            # `__post_init__` raises ValueError on allowlist/band-ordering
            # violations; surface those with the file path attached.
            raise MalformedThresholdsError(
                f"Invalid thresholds in {path}: {exc}"
            ) from exc


__all__ = [
    "ThresholdProvider",
    "ThresholdsNotFoundError",
    "MalformedThresholdsError",
]
