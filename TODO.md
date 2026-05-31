# Future Work & Technical Debt

## Rep Counting — Prominence Segmenter (Stage C DONE 2026-05-31)
- **Shipped as the default counting path:** `core/rep_segmenter.py`. `/api/score`
  uses `segment_reps_batch` (whole-video `find_peaks`); the live feed uses
  `StreamingRepSegmenter` via `make_rep_counter(...)`. The legacy FSM stays
  reachable via `make_rep_counter(..., counter='fsm')` (graceful fallback). Earlier
  step: FSM top-turnaround close (`RepCounterConfig.reversal_margin`).
- **Honest accuracy on real bicep (measured on the batch path /api/score runs;
  62 clips deduped of re-encoded copies, bad-form excluded, side/back separate):**
  - Held-out **FRONT exact-match 62% (n=8)**; **within ±1 100%** — but within-±1 is
    **generous** on this single-rep-dominated set (0/1/2 all pass; multi-rep ≤ 4).
  - **FRONT multi-rep exact ~75% (n=8 all-unique)**; held-out multi-rep is only
    **n=3 → indicative, not robust**.
  - One **real miss**: clip 47 (true 2, detected 1) — shallow rep below
    `prominence_floor_deg=18`. Other gaps are definitional (clip 43 prep-move
    over-count; clip 7 ambiguous half-rep).
- **Params tuned on ONLY n=13 front clips** (`REP_SEGMENTER_CONFIGS['bicep_curl']`
  = window 7 / frac 0.25 / floor 18 / refractory 1.2). **RE-TUNE if a larger or
  cleaner real set is added** — not robust at n=13.
- **Remaining:**
  - `bent_over_row` / `push_up` keep DEFAULT segmenter params — real-data
    calibration queued (no real recordings; verify they don't regress on synthetic).
  - Revisit `prominence_floor_deg` to catch clip-47-type shallow reps once more
    data exists (trade-off: re-introduces prep-move over-count like clip 43).
  - Tuning harness: `tools/tune_rep_segmenter.py` (dedup/split/grid/held-out eval).

## Live Path Convergence — `/api/live-frame` (2026-05-31)
- **Shipped:** the React live camera now drives the 3-pillar pipeline via
  `POST /api/live-frame` — `core/live_session.LiveSession` wraps the
  `StreamingRepSegmenter`, `FormScorer`, and `ScoreAggregator` (form scoring on
  rep close, counter every frame). Legacy `/api/process-frame` is untouched and
  still reachable.
- **Smoothing is now time-based** (`RepSegmenterConfig.smoothing_window_seconds`)
  so one value adapts across fps regimes (~7 frames @30fps batch, ~1 @~5fps live);
  added `streaming_fps_fallback` for cold-start. **Batch held-out numbers are
  byte-identical after the change** (front exact 62%, within ±1 100%).
- **UNVALIDATED — live-cadence rep accuracy.** The batch numbers were measured at
  ~30fps on uploaded clips; the live feed runs at ~5fps with ~10–15 samples/rep —
  a harder regime. **Do NOT present the batch 62%/100% as the live number.** The
  10-curl on-camera smoke test (count ≈ 10) has not been run yet. **If it under/
  over-counts, add a separate `REP_SEGMENTER_LIVE_CONFIGS` (live endpoint reads
  it; batch `REP_SEGMENTER_CONFIGS` stays as-is), tune bicep there, and document
  BOTH regimes here.** Until then, live accuracy is unmeasured.
- **Cadence (intentional):** `rep_quality`/`feedback` update on rep close and hold
  between reps; `rep_quality` is `null` (UI shows "—") until the first rep.

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
- **Why it's not fused — two findings, both verified:**
  1. **Rep segmentation (pipeline-wide bug):** the production FSM returns 0 reps
     on 45 of 62 real clips (~24% capture) because it only closes a rep when the
     arm re-extends past ~130°. Affects rep count, RF, and rules — not just the
     CNN+LSTM. Fix: peak/turnaround segmentation in `core/rep_counter.py` (close
     on direction reversal + smoothing). The trainer's *validation* already uses
     peak segmentation; the live FSM still needs it.
  2. **Sim-to-real transfer gap (the blocker for fusion):** even with proper
     peak-segmentation, the synthetic-trained CNN+LSTM recognises only ~6% of real
     good-form reps (real ROM ~65° vs synthetic ~113°). Synthetic tuning can't fix
     this — it needs **real labelled sequences** (record good + bad reps, extract
     landmark sequences, retrain). Until then it stays **not fused**.
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
