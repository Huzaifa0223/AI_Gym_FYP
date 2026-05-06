"""
tests.test_schemas — validate the 3-pillar data contracts in core.schemas.

Covers immutability of every frozen dataclass, the runtime sources-allowlist
on :class:`HeuristicThresholds`, and shape-level checks on the Pydantic
:class:`HeadlineScore` model. See ``docs/architecture_3pillar.md`` §3.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from pydantic import ValidationError

from core.schemas import (
    ALLOWED_THRESHOLD_SOURCES,
    EquipmentDetection,
    HeadlineScore,
    HeuristicThresholds,
    LandmarkFrame,
    PrimaryUserBox,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sane_thresholds(**overrides: Any) -> HeuristicThresholds:
    base: dict[str, Any] = dict(
        exercise="bicep_curl",
        rom_band=(80.0, 160.0),
        speed_band=(1.0, 5.0),
        tempo_stability_max_cv=0.6,
        tempo_symmetry_min_ratio=0.7,
        exercise_specific={"min_torso_angle": 30.0},
        sources={
            "rom_band": "nsca",
            "speed_band": "acsm",
            "min_torso_angle": "derived_from_video",
        },
    )
    base.update(overrides)
    return HeuristicThresholds(**base)


# ---------------------------------------------------------------------------
# Frozen-dataclass immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_landmark_frame_frozen(self) -> None:
        lf = LandmarkFrame(
            frame_id=0,
            timestamp_ms=0,
            landmarks=tuple((0.0, 0.0, 0.0) for _ in range(33)),
            visibility=tuple(1.0 for _ in range(33)),
            user_height_cm=170.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            lf.frame_id = 1  # type: ignore[misc]

    def test_primary_user_box_frozen(self) -> None:
        pub = PrimaryUserBox(
            track_id=1,
            bbox_xyxy=(0.0, 0.0, 100.0, 200.0),
            bbox_area_px=20000.0,
            frames_since_acquired=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            pub.track_id = 2  # type: ignore[misc]

    def test_equipment_detection_frozen(self) -> None:
        det = EquipmentDetection(
            label="dumbbell",
            confidence=0.92,
            bbox_xyxy=(0.0, 0.0, 50.0, 50.0),
            frame_id_detected=10,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            det.label = "bench"  # type: ignore[misc]

    def test_heuristic_thresholds_frozen(self) -> None:
        ht = _sane_thresholds()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ht.exercise = "bent_over_row"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HeuristicThresholds — runtime validation + biomechanical sanity
# ---------------------------------------------------------------------------

class TestHeuristicThresholds:
    def test_rom_band_ordered(self) -> None:
        ht = _sane_thresholds()
        assert ht.rom_band[0] < ht.rom_band[1]

    def test_speed_band_ordered(self) -> None:
        ht = _sane_thresholds()
        assert ht.speed_band[0] < ht.speed_band[1]

    def test_sources_allows_all_documented_values(self) -> None:
        ht = _sane_thresholds(sources={
            "a": "nsca",
            "b": "acsm",
            "c": "derived_from_video",
        })
        assert set(ht.sources.values()) <= ALLOWED_THRESHOLD_SOURCES

    @pytest.mark.parametrize("typo", [
        "NSCA",                # case-sensitive — uppercase is wrong
        "national_strength",   # invented full-name expansion
        "video",               # truncation of derived_from_video
        "",                    # empty string
    ])
    def test_unknown_source_raises(self, typo: str) -> None:
        with pytest.raises(ValueError, match="unknown values"):
            _sane_thresholds(sources={"rom_band": typo})

    def test_allowlist_is_frozen(self) -> None:
        # Cannot mutate the allowlist itself — it is a frozenset.
        with pytest.raises(AttributeError):
            ALLOWED_THRESHOLD_SOURCES.add("invented")  # type: ignore[attr-defined]

    def test_error_message_lists_offenders(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _sane_thresholds(sources={"a": "made_up", "b": "nsca"})
        assert "made_up" in str(exc_info.value)

    def test_inverted_rom_band_raises(self) -> None:
        with pytest.raises(ValueError, match="rom_band is inverted"):
            _sane_thresholds(rom_band=(160.0, 80.0))

    def test_inverted_speed_band_raises(self) -> None:
        with pytest.raises(ValueError, match="speed_band is inverted"):
            _sane_thresholds(speed_band=(5.0, 1.0))

    def test_equal_band_endpoints_accepted(self) -> None:
        # Degenerate but legal: equal endpoints define a zero-width band.
        ht = _sane_thresholds(rom_band=(120.0, 120.0), speed_band=(2.0, 2.0))
        assert ht.rom_band == (120.0, 120.0)
        assert ht.speed_band == (2.0, 2.0)


# ---------------------------------------------------------------------------
# HeadlineScore — Pydantic v2 model
# ---------------------------------------------------------------------------

class TestHeadlineScore:
    def test_valid_construction(self) -> None:
        hs = HeadlineScore(
            rep_count=10,
            form_feedback="Solid technique",
            score=82,
            source_breakdown={"rules": 78, "rf": 82, "cnn_lstm": None},
        )
        assert hs.score == 82
        assert hs.source_breakdown["cnn_lstm"] is None

    def test_score_lower_bound_enforced(self) -> None:
        with pytest.raises(ValidationError):
            HeadlineScore(
                rep_count=0,
                form_feedback="x",
                score=-1,
                source_breakdown={},
            )

    def test_score_upper_bound_enforced(self) -> None:
        with pytest.raises(ValidationError):
            HeadlineScore(
                rep_count=0,
                form_feedback="x",
                score=101,
                source_breakdown={},
            )

    def test_rep_count_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            HeadlineScore(
                rep_count=-1,
                form_feedback="x",
                score=50,
                source_breakdown={},
            )

    def test_round_trip_serialization_with_null_in_breakdown(self) -> None:
        # Mixed int|None values inside source_breakdown are the edge case:
        # null must survive JSON round-trip without coercing to 0 (which would
        # masquerade as a real ML score of zero).
        original = HeadlineScore(
            rep_count=10,
            form_feedback="Solid technique",
            score=82,
            source_breakdown={"rules": 78, "rf": 82, "cnn_lstm": None},
        )
        rebuilt = HeadlineScore.model_validate_json(original.model_dump_json())
        assert rebuilt == original
        assert rebuilt.source_breakdown["cnn_lstm"] is None
        assert rebuilt.source_breakdown["rules"] == 78


# ---------------------------------------------------------------------------
# ScoreResponse.headline — populated path
# ---------------------------------------------------------------------------

class TestScoreResponseHeadline:
    """The default-None path is implicitly exercised by tests/test_api_score.py;
    this class covers the populated path so a typo in the field declaration
    cannot ship silently while no caller reads it."""

    def test_construct_and_round_trip_with_populated_headline(self) -> None:
        from api_backend import ScoreResponse

        headline = HeadlineScore(
            rep_count=5,
            form_feedback="Good control",
            score=75,
            source_breakdown={"rules": 70, "rf": 80, "cnn_lstm": None},
        )
        resp = ScoreResponse(
            exercise="bicep_curl",
            age_group="adult",
            total_reps=5,
            average_score=75.0,
            frames_processed=300,
            frames_with_landmarks=290,
            processing_time_s=1.2,
            reps=[],
            headline=headline,
        )
        rebuilt = ScoreResponse.model_validate_json(resp.model_dump_json())
        assert rebuilt.headline is not None
        assert rebuilt.headline == headline
        assert rebuilt.headline.source_breakdown["cnn_lstm"] is None

    def test_default_headline_is_none(self) -> None:
        from api_backend import ScoreResponse

        resp = ScoreResponse(
            exercise="bicep_curl",
            age_group="adult",
            total_reps=0,
            average_score=None,
            frames_processed=0,
            frames_with_landmarks=0,
            processing_time_s=0.0,
            reps=[],
        )
        assert resp.headline is None
        # And it survives the round-trip as null.
        rebuilt = ScoreResponse.model_validate_json(resp.model_dump_json())
        assert rebuilt.headline is None
