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
