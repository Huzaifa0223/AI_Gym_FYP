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

## Reference Skeleton — Hand-author Pose Keyframes (VIVA-BLOCKING)
- **Task:** Replace the zeroed placeholder content of
  `exercises/bicep_curl/reference_skeleton.json` (and add equivalents for
  `bent_over_row` and `push_up`) with hand-authored MediaPipe pose
  keyframes — one per non-IDLE FSM phase (DESCENDING / BOTTOM /
  ASCENDING / TOP).
- **Context:** Stage 5c (`api_backend.py:get_reference_skeleton` +
  `core/rep_counter.RepState`) ships the schema and the GET endpoint.
  The frontend's ghost-overlay renderer reads this JSON to draw the
  reference pose. The placeholder JSON has the right shape but every
  landmark is `(0, 0, 0)` and every visibility is `0.0`.
- **Priority:** **Viva-blocking** — overlay rendering is dead until this
  ships, even though the API layer is wired correctly.
- **Acceptance criteria:**
  - 33-landmark `(x, y, z)` arrays per keyframe in MediaPipe normalised
    coords (`[0, 1]` for x/y; z is depth-relative).
  - Visibility values reflect which body parts are reliably observable
    in that phase (e.g. occluded shoulder during a deep curl bottom →
    lower visibility).
  - Source: either trace from a high-quality demo video frame-by-frame,
    or generate from a posed model. Document the source in the JSON's
    `_note` field.

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

## API Integration Follow-ups
- **WebSocket endpoint** for real-time scoring from a streaming landmark feed (not yet implemented — /api/score is one-shot video only).
- **Progress stream** for long videos: return Server-Sent Events with per-rep scores as they close, instead of waiting for the whole video to process.
- **Priority:** Low — current endpoint is viva-demoable as-is.

## CNN+LSTM Form Scorer (supervisor-mandated)
- **Status (2026-05-30): BUILT, trained, integrated, tested — off by default.**
  `core/cnn_lstm_model.py` (`FormQualityNet`: 1-D CNN → LSTM),
  `core/cnn_lstm_scorer.CnnLstmFormScorer`, `training/train_cnn_lstm.py`. 86% on
  synthetic test data; clean reps score sensibly (good ≈85 / partial ≈1). Enable
  with `AI_GYM_ENABLE_CNN_LSTM=1`. See `docs/ML_FACTS.md` §5.
- **Remaining (prerequisite to fairly evaluating + fusing it):** fix real-clip
  **rep segmentation** — the CNN+LSTM was *not* fairly tested on real video. The
  rep counter returns **0 reps on 45 of 62 real clips** (captures only ~24% of
  ~105 estimated curls) because it only closes a rep when the arm re-extends past
  ~130°, which real curlers don't do between reps. The earlier "8–24%
  recognition" measured the **segmenter, not the model**. Fix: peak/turnaround
  segmentation (close on direction reversal, not a fixed top threshold) + signal
  smoothing; then re-score real reps and decide on fusion. Separately, real
  bad-form data is needed to validate bad-form discrimination. Until evaluated it
  stays **not fused**.
- **Original task (done):** build the mandated CNN+LSTM (was a null seam —
  `NullFormQualityScorer` returning `None`).
- **Real training data already extracted:**
  `data/training/bicep_adult_good_form_20260207_171944.csv` holds **62 real
  bicep good-form videos** (~14,006 frames) as full 33-landmark
  `(x, y, z, visibility)` sequences, grouped by the `video_file` column — i.e.
  ready CNN+LSTM input, no re-extraction needed. (gitignored; local disk only.)
- **Decision (2026-05-30):** keep the synthetic RandomForest as the deployed
  form classifier and route this real data to the CNN+LSTM instead — avoids the
  real-good-vs-synthetic-bad artifact trap in the RF. See `docs/ML_FACTS.md` §2.
- **Gaps to fill before training:**
  - Good-form only (`label=1`). Binary good/bad needs negatives: record real
    bad-form, or use a one-class / reconstruction framing, or pair with
    synthetic bad-form (and document the mix honestly).
  - Only `bicep_curl/adult` has real sequences; `bent_over_row` / `push_up`
    have none yet.
  - Train on Colab/Kaggle GPU; run inference locally (CLAUDE.md forbids cloud
    inference at runtime, not cloud *training*).
- **Priority:** High — this is the supervisor-mandated deliverable.

## Stage 6 — Synthetic Trainer & YOLO Operationalization
- `training/synthetic_trainer.py` provides reproducibility for the 9 RandomForest form-quality models; real-data trainer to follow in 6a.
- YOLOv8-nano runs with COCO classes; gym-specific fine-tuning queued for Stage 7.
