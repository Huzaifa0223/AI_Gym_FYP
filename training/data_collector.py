"""
training.data_collector — Two-video heuristic threshold calibrator (Stage 3).

CLI::

    python -m training.data_collector \\
        --exercise bicep_curl \\
        --good <good_video_path> \\
        --bad  <bad_video_path> \\
        [--user-height-cm <float>]

For each video, iterates frames via a :class:`PoseExtractor`, drops no-pose
frames, computes per-frame elbow + shoulder angles, aggregates whole-video
stats, and emits two artefacts under ``exercises/<exercise>/``:

* ``heuristic_thresholds.json`` — the calibrated
  :class:`HeuristicThresholds` (gitignored).
* ``calibration_log.json`` — per-video stats and the health verdict
  (gitignored).

Status verdict
--------------
* ``ok``           — good video has ≥ 30 valid frames *and* the bad video
                     falls outside the good-video band on at least one
                     threshold.
* ``inconclusive`` — either good has < 30 valid frames, *or* bad falls
                     within all good-video bands. CLI exit code is
                     non-zero; both JSONs are still written.

Out of scope (deliberate, per ``docs/architecture_3pillar.md`` §1):
* FSM phase bucketing — uses whole-video aggregates only.
* Automatic threshold tightening from bad-video deviation — calibrator
  warns; humans review the log.
* Multi-person filtering — calibration video is assumed single-person.
* Seed thresholds for any exercise other than bicep_curl.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np

from core.schemas import HeuristicThresholds, LandmarkFrame
from training.pose_extractor import MediaPipePoseExtractor, PoseExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_EXERCISES: Final[tuple[str, ...]] = ("bicep_curl",)

_MIN_VALID_FRAMES: Final[int] = 30  # good video gate

# MediaPipe Pose landmark indices used by bicep-curl angle math.
_RIGHT_SHOULDER: Final[int] = 12
_RIGHT_ELBOW: Final[int] = 14
_RIGHT_WRIST: Final[int] = 16
_RIGHT_HIP: Final[int] = 24

_MIN_LANDMARK_VISIBILITY: Final[float] = 0.5

# Fallback / default values when whole-video derivation cannot proceed.
_DEFAULT_SPEED_BAND: Final[tuple[float, float]] = (1.0, 5.0)
_FPS_FALLBACK: Final[float] = 30.0

# Margin factor applied to good-video CV when emitting
# ``tempo_stability_max_cv``: 1.0 means "good's CV is the threshold."
_CV_THRESHOLD_MARGIN: Final[float] = 1.0

# Margin factor for the lower-bound symmetry threshold.  0.85 gives a 15%
# slack below good's mean ratio.
_SYMMETRY_THRESHOLD_MARGIN: Final[float] = 0.85


# ---------------------------------------------------------------------------
# Per-video stats dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoStats:
    """Aggregate stats produced for a single video."""
    frame_count: int
    valid_frame_count: int
    dropped_frame_count: int
    elbow_mean_deg: float
    elbow_std_deg: float
    shoulder_mean_deg: float
    shoulder_std_deg: float
    tempo_cv: float
    tempo_symmetry_ratio: float
    detected_peaks: int


# ---------------------------------------------------------------------------
# Angle math (kept local to avoid coupling utils.angle_utils to this stage)
# ---------------------------------------------------------------------------

def _angle_3d(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> float:
    """Angle (degrees) at *b* formed by the rays b→a and b→c."""
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    bax, bay, baz = ax - bx, ay - by, az - bz
    bcx, bcy, bcz = cx - bx, cy - by, cz - bz
    norm_ba = math.sqrt(bax * bax + bay * bay + baz * baz)
    norm_bc = math.sqrt(bcx * bcx + bcy * bcy + bcz * bcz)
    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return 0.0
    cos = (bax * bcx + bay * bcy + baz * bcz) / (norm_ba * norm_bc)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _bicep_angles(lf: LandmarkFrame) -> tuple[float, float] | None:
    """Return ``(elbow_flexion, shoulder_flexion)`` for a bicep-curl frame.

    Returns ``None`` if any of the four required landmarks
    (right shoulder/elbow/wrist/hip) has visibility < 0.5 — those frames
    are dropped before aggregation.
    """
    required = (_RIGHT_SHOULDER, _RIGHT_ELBOW, _RIGHT_WRIST, _RIGHT_HIP)
    if not all(lf.visibility[i] >= _MIN_LANDMARK_VISIBILITY for i in required):
        return None
    elbow = _angle_3d(
        lf.landmarks[_RIGHT_SHOULDER],
        lf.landmarks[_RIGHT_ELBOW],
        lf.landmarks[_RIGHT_WRIST],
    )
    shoulder = _angle_3d(
        lf.landmarks[_RIGHT_HIP],
        lf.landmarks[_RIGHT_SHOULDER],
        lf.landmarks[_RIGHT_ELBOW],
    )
    return elbow, shoulder


# ---------------------------------------------------------------------------
# Whole-video aggregation
# ---------------------------------------------------------------------------

def _detect_peaks(angles: list[float], min_distance: int = 5) -> list[int]:
    """Find local maxima above the 75th percentile, separated by ≥ *min_distance*.

    Returns a list of frame indices.  Used for both rep-count estimation
    (drives ``speed_band``) and tempo-symmetry rep splitting.
    """
    if len(angles) < 3:
        return []
    arr = np.asarray(angles, dtype=float)
    threshold = float(np.percentile(arr, 75))
    peaks: list[int] = []
    last_peak = -min_distance
    for i in range(1, len(arr) - 1):
        if (arr[i] >= threshold
                and arr[i] >= arr[i - 1]
                and arr[i] >= arr[i + 1]
                and i - last_peak >= min_distance):
            peaks.append(i)
            last_peak = i
    return peaks


def _compute_stats(extractor: PoseExtractor, video_path: Path) -> VideoStats:
    """Run *extractor* on *video_path* and aggregate the angle stats."""
    elbow_angles: list[float] = []
    shoulder_angles: list[float] = []
    frame_count = 0
    dropped = 0

    for frame in extractor.extract(video_path):
        frame_count += 1
        if frame is None:
            dropped += 1
            continue
        result = _bicep_angles(frame)
        if result is None:
            dropped += 1
            continue
        elbow_angles.append(result[0])
        shoulder_angles.append(result[1])

    valid = len(elbow_angles)
    if valid == 0:
        return VideoStats(
            frame_count=frame_count,
            valid_frame_count=0,
            dropped_frame_count=dropped,
            elbow_mean_deg=0.0,
            elbow_std_deg=0.0,
            shoulder_mean_deg=0.0,
            shoulder_std_deg=0.0,
            tempo_cv=0.0,
            tempo_symmetry_ratio=1.0,
            detected_peaks=0,
        )

    elbow_arr = np.asarray(elbow_angles, dtype=float)
    shoulder_arr = np.asarray(shoulder_angles, dtype=float)

    # Tempo stability — CV of frame-to-frame |Δangle|.
    if len(elbow_arr) >= 2:
        deltas = np.abs(np.diff(elbow_arr))
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas))
        tempo_cv = std_d / mean_d if mean_d > 1e-9 else 0.0
    else:
        tempo_cv = 0.0

    # Tempo symmetry — average descending/ascending sample ratio per detected rep.
    peaks = _detect_peaks(elbow_angles)
    ratios: list[float] = []
    for i in range(len(peaks) - 1):
        seg = elbow_arr[peaks[i]: peaks[i + 1] + 1]
        if len(seg) < 4:
            continue
        trough_idx = int(np.argmin(seg))
        descending = trough_idx + 1
        ascending = len(seg) - trough_idx
        if descending >= 2 and ascending >= 2:
            ratios.append(descending / ascending)
    tempo_symmetry_ratio = float(np.mean(ratios)) if ratios else 1.0

    return VideoStats(
        frame_count=frame_count,
        valid_frame_count=valid,
        dropped_frame_count=dropped,
        elbow_mean_deg=float(np.mean(elbow_arr)),
        elbow_std_deg=float(np.std(elbow_arr)),
        shoulder_mean_deg=float(np.mean(shoulder_arr)),
        shoulder_std_deg=float(np.std(shoulder_arr)),
        tempo_cv=tempo_cv,
        tempo_symmetry_ratio=tempo_symmetry_ratio,
        detected_peaks=len(peaks),
    )


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class ThresholdCalibrator:
    """Two-video calibrator emitting :class:`HeuristicThresholds` JSON.

    Args:
        exercise:    Exercise key — Stage 3 ships only ``"bicep_curl"``.
        extractor:   Injected :class:`PoseExtractor`. Defaults to
                     :class:`MediaPipePoseExtractor`. Tests pass a
                     :class:`StubPoseExtractor`.
        output_dir:  Override for the artefact output directory. Defaults
                     to ``<repo_root>/exercises/<exercise>/``. Tests pass
                     a ``tmp_path`` to keep the suite hermetic.
    """

    def __init__(
        self,
        exercise: str,
        *,
        extractor: PoseExtractor | None = None,
        output_dir: Path | None = None,
    ) -> None:
        if exercise not in _SUPPORTED_EXERCISES:
            raise ValueError(
                f"Unsupported exercise {exercise!r}; Stage 3 ships only "
                f"{list(_SUPPORTED_EXERCISES)}"
            )
        self._exercise = exercise
        self._extractor: PoseExtractor = (
            extractor if extractor is not None else MediaPipePoseExtractor()
        )
        if output_dir is None:
            repo_root = Path(__file__).resolve().parent.parent
            output_dir = repo_root / "exercises" / exercise
        self._output_dir = output_dir

    # -- Public API ----------------------------------------------------------

    def calibrate(
        self,
        good_video: Path,
        bad_video: Path,
        *,
        user_height_cm: float | None = None,
    ) -> tuple[HeuristicThresholds, dict, str]:
        """Run calibration end-to-end without writing files.

        Returns ``(thresholds, calibration_log_dict, status)``.
        """
        good = _compute_stats(self._extractor, good_video)
        bad = _compute_stats(self._extractor, bad_video)

        thresholds = self._build_thresholds(good)
        differentiating, warnings = self._verdict_inputs(good, bad, thresholds)

        if good.valid_frame_count < _MIN_VALID_FRAMES:
            warnings.append(
                f"good video has only {good.valid_frame_count} valid "
                f"frames (need >= {_MIN_VALID_FRAMES})"
            )

        if good.valid_frame_count < _MIN_VALID_FRAMES or not differentiating:
            status = "inconclusive"
            if not differentiating:
                warnings.append(
                    "bad video stats fall within good-video bands on every "
                    "threshold — calibration cannot differentiate good from bad"
                )
        else:
            status = "ok"

        log = {
            "exercise": self._exercise,
            "status": status,
            "user_height_cm": user_height_cm,
            "good_video": asdict(good),
            "bad_video": asdict(bad),
            "differentiating_thresholds": differentiating,
            "warnings": warnings,
        }
        return thresholds, log, status

    def write(
        self,
        thresholds: HeuristicThresholds,
        calibration_log: dict,
    ) -> tuple[Path, Path]:
        """Persist *thresholds* and *calibration_log* under :attr:`_output_dir`.

        Returns ``(thresholds_path, log_path)``.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        thresholds_path = self._output_dir / "heuristic_thresholds.json"
        with thresholds_path.open("w", encoding="utf-8") as f:
            json.dump(_thresholds_to_dict(thresholds), f, indent=2)

        log_path = self._output_dir / "calibration_log.json"
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(calibration_log, f, indent=2)

        return thresholds_path, log_path

    # -- Internals -----------------------------------------------------------

    def _build_thresholds(self, good: VideoStats) -> HeuristicThresholds:
        rom_band = (
            max(0.0, good.elbow_mean_deg - good.elbow_std_deg),
            good.elbow_mean_deg + good.elbow_std_deg,
        )

        # Speed band: derive from peak intervals when peak detection succeeded.
        if good.detected_peaks >= 2 and good.valid_frame_count > 0:
            inter_peak_frames = good.valid_frame_count / max(1, good.detected_peaks - 1)
            mean_rep_s = inter_peak_frames / _FPS_FALLBACK
            speed_band = (max(0.5, mean_rep_s * 0.5), mean_rep_s * 1.5)
        else:
            speed_band = _DEFAULT_SPEED_BAND

        return HeuristicThresholds(
            exercise=self._exercise,
            rom_band=rom_band,
            speed_band=speed_band,
            tempo_stability_max_cv=good.tempo_cv * _CV_THRESHOLD_MARGIN,
            tempo_symmetry_min_ratio=good.tempo_symmetry_ratio * _SYMMETRY_THRESHOLD_MARGIN,
            exercise_specific={},
            sources={
                "rom_band": "derived_from_video",
                "speed_band": "derived_from_video",
                "tempo_stability_max_cv": "derived_from_video",
                "tempo_symmetry_min_ratio": "derived_from_video",
            },
        )

    @staticmethod
    def _verdict_inputs(
        good: VideoStats,
        bad: VideoStats,
        thresholds: HeuristicThresholds,
    ) -> tuple[list[str], list[str]]:
        """Determine which thresholds differentiate good from bad."""
        differentiating: list[str] = []
        warnings: list[str] = []

        # rom_band: bad's mean elbow angle should fall outside good's band.
        if not (thresholds.rom_band[0] <= bad.elbow_mean_deg <= thresholds.rom_band[1]):
            differentiating.append("rom_band")
        # tempo_stability_max_cv: bad's CV should exceed the threshold.
        if bad.tempo_cv > thresholds.tempo_stability_max_cv:
            differentiating.append("tempo_stability_max_cv")
        # tempo_symmetry_min_ratio: bad's ratio should fall below the threshold.
        if bad.tempo_symmetry_ratio < thresholds.tempo_symmetry_min_ratio:
            differentiating.append("tempo_symmetry_min_ratio")

        return differentiating, warnings


def _thresholds_to_dict(t: HeuristicThresholds) -> dict:
    """Serialise :class:`HeuristicThresholds` to a JSON-safe dict."""
    return {
        "exercise": t.exercise,
        "rom_band": list(t.rom_band),
        "speed_band": list(t.speed_band),
        "tempo_stability_max_cv": t.tempo_stability_max_cv,
        "tempo_symmetry_min_ratio": t.tempo_symmetry_min_ratio,
        "exercise_specific": dict(t.exercise_specific),
        "sources": dict(t.sources),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Two-video heuristic threshold calibrator (Stage 3).",
    )
    parser.add_argument(
        "--exercise", required=True, choices=list(_SUPPORTED_EXERCISES),
    )
    parser.add_argument("--good", required=True, type=Path, help="Good-form video")
    parser.add_argument("--bad", required=True, type=Path, help="Bad-form video")
    parser.add_argument("--user-height-cm", type=float, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    calibrator = ThresholdCalibrator(args.exercise)
    thresholds, log, status = calibrator.calibrate(
        args.good, args.bad, user_height_cm=args.user_height_cm,
    )
    thresholds_path, log_path = calibrator.write(thresholds, log)

    logger.info("Wrote %s", thresholds_path)
    logger.info("Wrote %s", log_path)
    logger.info("Status: %s", status)
    if log["differentiating_thresholds"]:
        logger.info(
            "Differentiating thresholds: %s",
            ", ".join(log["differentiating_thresholds"]),
        )
    for w in log["warnings"]:
        logger.warning("Warning: %s", w)

    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["ThresholdCalibrator", "VideoStats", "main"]
