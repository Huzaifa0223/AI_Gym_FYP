# Future Work & Technical Debt

## Rep Counter Calibration
- **Task:** Calibrate `REP_COUNTER_CONFIGS` per-exercise using real skeleton recordings.
- **Context:** Current values (top=140, bottom=60, hys=10) are conservative heuristics. Chest diverges slightly (bottom=80), but all require empirical tuning based on actual user data to improve accuracy across body types.
- **Priority:** Medium (do before final FYP submission/demo).
- **Acceptance criteria:**
  - Record 20+ reps per exercise per age group using `training/auto_skeleton_recorder.py`
  - Compute per-exercise mean peak and trough angles ± 1 standard deviation
  - Update `config.py` REP_COUNTER_CONFIGS with data-driven thresholds
  - Add a one-line justification comment citing the sample size and date

## Back + Chest Rule Engine Calibration
- **Task:** Empirically tune rule thresholds using recorded good-form / bad-form samples.
- **Current heuristic values:**
  - Back: `max_retraction_angle=50°`, `max_std=15°` for torso stability
  - Chest: `min_plank=165°`, `min_leg=170°`
  - Shared: tempo symmetry ratio in `[0.7, 1.5]`
- **Priority:** Medium (do before final FYP submission).

## Bicep Curl Threshold Citation Review (Stage 3 follow-up)
- **Task:** Cross-reference the calibrated `heuristic_thresholds.json` for
  bicep_curl against NSCA / ACSM published bicep-curl form guidance.
- **Context:** Stage 3 (`training/data_collector.py`) currently writes every
  threshold with `sources: "derived_from_video"`. The framework is in place;
  no value has been retagged with `nsca` or `acsm` because doing so without
  a citation would be fabrication. NSCA/ACSM thresholds may differ from
  video-derived ones — that delta needs to be reconciled, not silently
  overwritten.
- **Priority:** High — must complete before viva so the report can defend
  the threshold provenance.
- **Acceptance criteria:**
  - For each threshold (`rom_band`, `speed_band`, `tempo_stability_max_cv`,
    `tempo_symmetry_min_ratio`), record the published reference and page.
  - Update `sources` keys in the JSON to `"nsca"` or `"acsm"` where the
    citation supports it; leave as `"derived_from_video"` only when no
    published equivalent exists.
  - The `ALLOWED_THRESHOLD_SOURCES` allowlist in `core/schemas.py` already
    accepts these three values — no schema change required.

## Equipment Detector — Fine-tune YOLOv8-nano on Gym Dataset (VIVA-BLOCKING)
- **Task:** Replace stock COCO weights with a YOLOv8-nano model fine-tuned on
  a gym-equipment dataset, OR drop in a pre-trained gym-equipment checkpoint.
- **Context:** Stage 4 (`core/object_detector.py`,
  `core/equipment_pipeline.py`) ships the framework with stock COCO weights.
  COCO does not include dumbbells, kettlebells, barbells, weight benches, or
  yoga mats — so the live detector mostly returns ``person`` / ``bottle`` /
  ``backpack`` / ``sports ball``. Equipment-driven coaching in Stage 5 will
  appear sparse until this lands.
- **Priority:** **Viva-blocking** — without this, equipment-driven feedback
  is non-functional.
- **Acceptance criteria:**
  - Fine-tune (Roboflow Universe has several gym-equipment datasets at
    ~500–2000 images each) or pull a vetted pre-trained checkpoint.
  - Drop the new weights into `models/yolov8n.pt` (or update the path in
    `Yolov8NanoDetector`'s constructor default).
  - Configure `EquipmentDetector(allowed_classes=("dumbbell", "kettlebell",
    "barbell", "bench", "mat"))` at the call site.
  - Add a one-line provenance comment naming the dataset / source / date.

## Stage 5 Prerequisite: Reconcile `exercise` Field Naming
- **Task:** Standardise on the longer descriptive form everywhere.
- **Context:** Stage 3's calibrator output writes
  `"exercise": "bicep_curl"` (matching the directory name and existing
  `exercises/bicep_curl.py` module). The rule engines in
  `core/rule_engine.py` use the short form `"bicep" / "back" / "chest"`,
  matching the trained `.pkl` filenames (`bicep_adult.pkl` etc.).
  Aliasing between the two is a code smell.
- **Resolution:** Rename rule-engine keys to the long form
  (`bicep_curl`, `bent_over_row`, `bench_press`). Do not alias.
- **Priority:** Must complete in Stage 5 before any consumer reads both
  artefacts at once.
- **Acceptance criteria:**
  - `_ENGINE_CLASSES` keys in `core/rule_engine.py` updated.
  - All call sites (`core/form_scorer.FormScorer`, `api_backend.score_video`,
    config maps in `config.py` / `core/rep_counter.py`) updated.
  - Tests still pass; pkl filename mapping handled at the storage layer
    (model_path in config.py) without leaking the short form upward.

## API Integration Follow-ups
- **WebSocket endpoint** for real-time scoring from a streaming landmark feed (not yet implemented — /api/score is one-shot video only).
- **Progress stream** for long videos: return Server-Sent Events with per-rep scores as they close, instead of waiting for the whole video to process.
- **Priority:** Low — current endpoint is viva-demoable as-is.
