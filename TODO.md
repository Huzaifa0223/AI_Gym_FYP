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

## API Integration Follow-ups
- **WebSocket endpoint** for real-time scoring from a streaming landmark feed (not yet implemented — /api/score is one-shot video only).
- **Progress stream** for long videos: return Server-Sent Events with per-rep scores as they close, instead of waiting for the whole video to process.
- **Priority:** Low — current endpoint is viva-demoable as-is.
