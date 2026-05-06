# 3-Pillar Pipeline Architecture Specification

**Audit reference:** `docs/audit_2026-05-06.md` (commit `6802abe`)

## 1. Audit-driven adjustments to the original brief

Three findings change what lands in early stages:

**1.1 HeuristicScorer is not a new class.** `BaseRuleEngine` already implements the four universal heuristic checks (ROM, speed, tempo stability, tempo symmetry); `BackRuleEngine` and `ChestRuleEngine` extend with exercise-specific checks. Building a parallel `HeuristicScorer` would duplicate this hierarchy and break the "do not break this structure" rule. The two-video calibrator therefore produces **threshold JSON files** consumed by existing rule engines, not a new scorer class. The bicep rule engine TODO is filled by the calibrator's output for free — no manual rule authoring needed.

**1.2 Age-group split coexists with body-segment normalization.** `MLAggregator` loads `{exercise}_{age_group}.pkl`. The original plan called for fully demographic-blind processing. These conflict. Pragmatic resolution for the FYP timeline: existing RandomForest models keep the age-group split (don't retrain — no time), body-segment normalization applies only to new components (CNN+LSTM input pipeline + heuristic threshold extractor). The asymmetry gets documented in CLAUDE.md and the FYP report as a defensible incremental migration. Score aggregator weights both correctly — see §3.

**1.3 Three pre-existing hygiene items must land before pillar code.** All small, low-risk, unblock downstream work:
- `pickle.load` → `joblib.load` in `core/ml_aggregator.py` (CLAUDE.md §10 compliance)
- Pydantic v1 `@validator` → v2 in `/api/process-frame` schemas (consistency with `/api/score`)
- CLAUDE.md sync for the new stack reality

These bundle into Stage 0 as a single foundation commit. The audit also identified untested areas (`/api/process-frame`, `main.py` loop, `core/exercise_classifier.py`); minimum smoke tests for these are added in Stage 0 to prevent silent regressions during the Pydantic migration.

## 2. Pipeline diagram

```
                       ┌──────────────────────────────────────┐
                       │ MediaPipe Pose (Primary User only)   │
                       │ Body-segment normalized landmarks    │
                       └──────────────────┬───────────────────┘
                                          │
              ┌───────────────────────────┼────────────────────────────┐
              ▼                           ▼                            ▼
   ┌──────────────────┐      ┌────────────────────┐      ┌────────────────────────┐
   │ Rule Engines     │      │ MLAggregator       │      │ CNN+LSTM Form Quality  │
   │ (existing) +     │      │ (existing,         │      │ (graceful null when    │
   │ calibrated       │      │  age-grouped RF)   │      │  weights missing)      │
   │ thresholds       │      │                    │      │                        │
   └────────┬─────────┘      └─────────┬──────────┘      └───────────┬────────────┘
            │                          │                             │
            └──────────────────────────┼─────────────────────────────┘
                                       ▼
                       ┌──────────────────────────────┐
                       │ ScoreAggregator              │
                       │ Weights when CNN+LSTM ready: │
                       │   rules 0.4, RF 0.3, NN 0.3  │
                       │ Weights when CNN+LSTM null:  │
                       │   rules 0.6, RF 0.4          │
                       └──────────────┬───────────────┘
                                      ▼
                              ScoreResponse JSON

YOLOv8-nano: out-of-band asyncio task at 1 Hz, writes EquipmentState singleton.
React polls GET /api/equipment separately.
```

## 3. Data contracts

Existing schemas (`ScoreResponse`, `RepResult`, `ViolationResult`) are preserved unchanged. New types added to `core/schemas.py`:

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class LandmarkFrame:
    """Single MediaPipe pose snapshot, normalized to body-segment-length units."""
    frame_id: int
    timestamp_ms: int
    landmarks: tuple[tuple[float, float, float], ...]  # 33 × (x, y, z)
    visibility: tuple[float, ...]
    user_height_cm: float | None  # null when normalization not available

@dataclass(frozen=True)
class PrimaryUserBox:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    bbox_area_px: float
    frames_since_acquired: int

@dataclass(frozen=True)
class EquipmentDetection:
    label: str  # "dumbbell" | "kettlebell" | "mat" | "bench"
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    frame_id_detected: int

@dataclass(frozen=True)
class HeuristicThresholds:
    """Loaded from exercises/<ex>/heuristic_thresholds.json by ThresholdProvider."""
    exercise: str
    rom_band: tuple[float, float]
    speed_band: tuple[float, float]
    tempo_stability_max_cv: float
    tempo_symmetry_min_ratio: float
    exercise_specific: dict[str, float]
    sources: dict[str, str]  # threshold_name -> "nsca" | "acsm" | "derived_from_video"
```

`ScoreResponse` is **extended** (not replaced) with a non-breaking `headline` field that matches the simple frontend contract:

```python
class HeadlineScore(BaseModel):
    rep_count: int
    form_feedback: str
    score: int  # 0-100
    source_breakdown: dict[str, int | None]  # {"rules": 78, "rf": 82, "cnn_lstm": null}

class ScoreResponse(BaseModel):
    # ... existing fields preserved ...
    headline: HeadlineScore  # NEW
```

Frontend reads `response.headline`; existing consumers continue using the nested structure. Zero breakage.

## 4. Threading model

- FastAPI endpoints stay async
- Per-frame path (MediaPipe → user_tracker → rule engines → MLAggregator → score aggregator) runs synchronously inside the request handler. CPU-bound ~30 ms p95.
- CNN+LSTM inference dispatched via `asyncio.run_in_executor(thread_pool)` once weights are present. Pre-warmed thread pool sized to 1 (single GIL-held workload, no benefit from more).
- YOLOv8-nano runs as an `asyncio.create_task` started at app boot, looping at 1 Hz, writing to a module-level `EquipmentState` dataclass. Lock-free reads from the per-frame path are safe because state mutates ≤1×/sec and torn reads are functionally equivalent to a 1-frame-stale read.
- Per-frame budget: ≤90 ms p95 (10 ms slack within the 100 ms hard limit).

`ultralytics==8.3.0` pinned in `requirements.txt`. Weights file `models/yolov8n.pt` gitignored. Fetch script `scripts/fetch_yolo_weights.py` for reproducible setup.

## 5. Reference skeleton

Per-exercise reference keypoint sequences in `exercises/<ex>/reference_skeleton.json`, one keyframe per FSM phase. `GET /api/reference/{exercise_id}` returns the sequence. React renders the ghosted overlay using its existing canvas; Python stays headless. This isolation matters for eventual independent deployment of the FastAPI server.

## 6. Score aggregator contract

`core/score_aggregator.py` exposes:

```python
def aggregate(
    rules_score: int,
    rf_score: int | None,
    cnn_lstm_score: int | None,
    feedback_messages: list[str],
) -> HeadlineScore: ...
```

Rules: weights renormalize when any input is None. If both `rf_score` and `cnn_lstm_score` are None, `final = rules_score` and `source_breakdown` reflects the missing components. The aggregator never raises on missing inputs — it logs a warning at debug level and degrades.

## 7. Test strategy

Stage 0 backfills smoke tests for the three audit-identified gaps: `tests/test_api_process_frame.py`, `tests/test_main_loop.py`, `tests/test_exercise_classifier.py`. Total coverage rises to ~80 tests before pillar work begins. Per-stage targets unchanged from the original brief.

## 8. Exit criterion

This document committed to `docs/architecture_3pillar.md` with message:
`docs(arch): 3-pillar pipeline specification referencing audit 6802abe`

Phase 3 staging proceeds in the order set in the original brief, with Stage 0 expanded as described in §1.3.
