"""
core.schemas — Frozen dataclass and Pydantic data contracts for the 3-pillar
pipeline.

Five types live here:

* :class:`LandmarkFrame`       — single MediaPipe pose snapshot, body-segment
                                 normalised.
* :class:`PrimaryUserBox`      — primary-user lock state from the user-tracker.
* :class:`EquipmentDetection`  — single YOLOv8-nano detection (1 Hz cadence).
* :class:`HeuristicThresholds` — per-exercise threshold bundle loaded from
                                 ``exercises/<ex>/heuristic_thresholds.json``.
* :class:`HeadlineScore`       — Pydantic v2 model surfaced as
                                 ``ScoreResponse.headline`` for the React
                                 frontend.

See ``docs/architecture_3pillar.md`` §3 for the contract rationale and
``docs/architecture_3pillar.md`` §1.2 for why ``HeuristicThresholds`` ships
runtime validation while the other schemas do not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Allowed sources for HeuristicThresholds.sources values
# ---------------------------------------------------------------------------
#
# Fail-loud allowlist for the threshold-citation field.  Any value outside
# this set in a loaded JSON is a typo or an unsanctioned new source — both
# should hard-fail at construction time rather than silently flow into the
# rule engines.

ALLOWED_THRESHOLD_SOURCES: Final[frozenset[str]] = frozenset({
    "nsca",                # National Strength & Conditioning Association tables
    "acsm",                # American College of Sports Medicine guidelines
    "derived_from_video",  # calibrator output (training/data_collector.py)
    "synthetic",           # biomechanical prior / synthetic-trainer band, not measured
})


# ---------------------------------------------------------------------------
# Frozen dataclasses — pipeline-internal contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LandmarkFrame:
    """Single MediaPipe pose snapshot, normalised to body-segment-length units.

    Attributes:
        frame_id:        Monotonic frame index inside the current session.
        timestamp_ms:    Wall-clock timestamp at frame capture (milliseconds).
        landmarks:       33 × ``(x, y, z)`` tuples — one per MediaPipe Pose
                         landmark. Coordinates are expressed in body-segment-
                         length units when normalisation has been applied;
                         raw normalised coords (≈ ``[0, 1]``) otherwise.
        visibility:      33 visibility scores in ``[0, 1]``, one per landmark.
        user_height_cm:  Estimated user height; ``None`` when normalisation
                         is unavailable for this frame (e.g. legs out of frame).
    """
    frame_id: int
    timestamp_ms: int
    landmarks: tuple[tuple[float, float, float], ...]
    visibility: tuple[float, ...]
    user_height_cm: float | None


@dataclass(frozen=True)
class PrimaryUserBox:
    """Primary-user lock state — the bbox/track currently driving scoring.

    Attributes:
        track_id:               Tracker-internal id assigned at acquisition.
        bbox_xyxy:              ``(x1, y1, x2, y2)`` pixel coordinates of the
                                primary-user bbox.
        bbox_area_px:           Pixel area of *bbox_xyxy*.
        frames_since_acquired:  Frame count since the lock was first acquired.
                                Drives hysteresis logic (see
                                :class:`core.config.PrimaryUserConfig`).
    """
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    bbox_area_px: float
    frames_since_acquired: int


@dataclass(frozen=True)
class EquipmentDetection:
    """One YOLOv8-nano equipment detection (1 Hz cadence — see spec §4).

    Attributes:
        label:              Equipment class — ``'dumbbell'`` | ``'kettlebell'``
                            | ``'mat'`` | ``'bench'``. The label set is
                            documented in spec §3 and is not validated at
                            runtime here (the YOLO model's class table is the
                            source of truth).
        confidence:         YOLOv8 confidence score in ``[0, 1]``.
        bbox_xyxy:          ``(x1, y1, x2, y2)`` pixel coordinates.
        frame_id_detected:  Frame index in which the detection was made.
    """
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    frame_id_detected: int


@dataclass(frozen=True)
class HeuristicThresholds:
    """Per-exercise threshold bundle loaded by ``ThresholdProvider``.

    Loaded at startup from ``exercises/<ex>/heuristic_thresholds.json`` and
    consumed by the existing rule engines (``core/rule_engine.py``).

    Attributes:
        exercise:                 One of ``'bicep_curl'``, ``'bent_over_row'``,
                                  ``'push_up'``.
        rom_band:                 ``(min_acceptable_rom_deg,
                                  max_acceptable_rom_deg)``.
        speed_band:               ``(min_rep_seconds, max_rep_seconds)``.
        tempo_stability_max_cv:   Maximum acceptable coefficient-of-variation
                                  of frame-to-frame angle deltas.
        tempo_symmetry_min_ratio: Minimum descending/ascending phase-time
                                  ratio.
        exercise_specific:        Free-form per-exercise thresholds (e.g.
                                  plank mean angle, scapular retraction max
                                  angle).
        sources:                  Maps each threshold name to a citation
                                  source. Values must be one of
                                  :data:`ALLOWED_THRESHOLD_SOURCES`
                                  (``'nsca'``, ``'acsm'``,
                                  ``'derived_from_video'``, ``'synthetic'``).
                                  ``'synthetic'`` marks a hand-set
                                  biomechanical prior (e.g. a seed file or a
                                  synthetic-trainer band) that has not been
                                  measured from real captures.

    Raises:
        ValueError: If any value in *sources* is outside
            :data:`ALLOWED_THRESHOLD_SOURCES`. Fail-loud on typos in JSON.
    """
    exercise: str
    rom_band: tuple[float, float]
    speed_band: tuple[float, float]
    tempo_stability_max_cv: float
    tempo_symmetry_min_ratio: float
    exercise_specific: dict[str, float]
    sources: dict[str, str]

    def __post_init__(self) -> None:
        bad = sorted(set(self.sources.values()) - ALLOWED_THRESHOLD_SOURCES)
        if bad:
            raise ValueError(
                f"HeuristicThresholds.sources contains unknown values "
                f"{bad}; allowed: {sorted(ALLOWED_THRESHOLD_SOURCES)}"
            )
        # Band ordering — defends against manual JSON edits that swap the
        # min/max values. The calibrator is the primary gate, but a one-line
        # runtime check is consistent with the fail-loud theme of *sources*
        # and costs nothing on the construction path.
        if self.rom_band[0] > self.rom_band[1]:
            raise ValueError(
                f"HeuristicThresholds.rom_band is inverted: "
                f"{self.rom_band[0]} > {self.rom_band[1]}"
            )
        if self.speed_band[0] > self.speed_band[1]:
            raise ValueError(
                f"HeuristicThresholds.speed_band is inverted: "
                f"{self.speed_band[0]} > {self.speed_band[1]}"
            )


# ---------------------------------------------------------------------------
# Pydantic model — surfaced via /api/score
# ---------------------------------------------------------------------------

class HeadlineScore(BaseModel):
    """Frontend-facing summary of a scoring run (spec §3).

    Surfaced as ``ScoreResponse.headline`` so the React app can read a
    single object instead of traversing the per-rep structure. Defaults to
    ``None`` on the response until the score aggregator is wired in.
    """
    rep_count: int = Field(..., ge=0)
    form_feedback: str
    score: int = Field(..., ge=0, le=100)
    source_breakdown: dict[str, int | None]


__all__ = [
    "ALLOWED_THRESHOLD_SOURCES",
    "LandmarkFrame",
    "PrimaryUserBox",
    "EquipmentDetection",
    "HeuristicThresholds",
    "HeadlineScore",
]
