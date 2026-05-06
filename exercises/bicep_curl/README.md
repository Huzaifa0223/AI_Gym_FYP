# exercises/bicep_curl/

Per-exercise calibration data directory for the bicep curl. Owned by the
Stage 3 calibrator (`training/data_collector.py`).

**Do not add an `__init__.py` here.** A regular Python package at this path
would shadow the existing `exercises/bicep_curl.py` module and break every
import that currently resolves `from exercises.bicep_curl import …`.
Without `__init__.py` the directory is a pure data dir; Python's import
system prefers the `.py` module over a namespace package.

## Files

| File | Source | Gitignored |
| --- | --- | --- |
| `seed_thresholds.json`        | committed | no  |
| `heuristic_thresholds.json`   | written by calibrator at runtime | yes |
| `calibration_log.json`        | written by calibrator at runtime | yes |

The seed mirrors the `HeuristicThresholds` dataclass shape with zeroed
values. The calibrator overwrites this file in place.
