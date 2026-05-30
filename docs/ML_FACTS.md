# ML Facts — the honest, verifiable picture

**This is the single source of truth for what the ML actually is.** If any other
doc disagrees with this file, this file is right and the other doc is stale. The
numbers here were read directly off the trained `.pkl` files and the source code
on 2026-05-30. Companion: [`docs/data_audit_2026-05-09.md`](data_audit_2026-05-09.md)
(dataset provenance).

> ⚠️ History note: earlier docs in this repo claimed "95%+ accuracy, 99% rep
> counting, trained on 62 videos, K-Means + Random Forest, 2.3 MB models." **None
> of those claims are accurate** and the meeting-prep files that made them have
> been removed. Use this file instead.

---

## 1. What the models actually are

- **9 models**: `{bicep_curl, bent_over_row, push_up} × {children, adult, senior}`,
  stored at `data/models/<exercise>_<age_group>.pkl`.
- **Algorithm**: `RandomForestClassifier` (200 trees) — **Random Forest only**.
  A K-Means path exists in `training/train_universal.py` but is **not** the
  deployed model; the shipped scoring path never calls K-Means.
- **Features**: exactly **3 joint angles per frame** — `primary_angle`,
  `secondary_angle`, `tertiary_angle` (not 5, not 134).
- **Labels**: class 0 = bad form, class 1 = good form.
- **Model dict on disk** (verify with `joblib.load`):
  ```python
  {'type': 'supervised', 'model': RandomForestClassifier,
   'feature_cols': ['primary_angle', 'secondary_angle', 'tertiary_angle'],
   'exercise_type': 'bicep'|'back'|'chest', 'age_group': ...,
   'accuracy': <float>, 'noise_injection': True}
  ```
  There is **no** `kmeans_model`, `scaler`, or `134 feature names`.
- **File sizes**: 5.2–8.4 MB each (within the ≤10 MB charter limit, but not the
  "2.3 MB" some old docs claimed).

## 2. How the models were trained — **synthetic data**

The deployed models were produced by `training/synthetic_trainer.py`. It does **not**
use any video. For each (exercise, age) bucket it:

1. Draws "good-form" and "bad-form" joint angles from hand-specified Gaussian
   bands (e.g. bicep good elbow angle ≈ `N(90°, 55°)`, bad ≈ `N(110°, 20°)`),
2. adds Gaussian measurement noise,
3. trains the Random Forest on the resulting 3-angle samples.

The bands are biomechanical priors chosen by the developer, **not measured from
recorded humans**. This is a legitimate, defensible approach *when described
honestly* — it was used because no labelled real dataset was available (see §4).

## 3. The real accuracy numbers (read from the `.pkl` files)

These are **synthetic self-test accuracies** — accuracy on a held-out split of the
*same synthetic generator*. They measure how separable the assumed good/bad
distributions are, **not** real-world performance.

| Exercise        | children | adult     | senior |
|-----------------|----------|-----------|--------|
| `bicep_curl`    | 77.2%    | **96.6%** | 74.9%  |
| `bent_over_row` | 81.8%    | 82.7%     | 81.7%  |
| `push_up`       | 80.5%    | 81.6%     | 80.2%  |

Range **74.9%–96.6%**, mean ≈ **81%**. The "uniform 95%" figure from old docs is
false — only `bicep_curl/adult` is near 96%.

**What this number is NOT:** it is not validated against real reps. Real-world
accuracy is **unmeasured**, because the decision boundary was authored (the
Gaussian bands), not learned from real form, and the synthetic angle ranges have
not been checked against real MediaPipe output.

## 4. The data story

- `data/FYP/` holds ~8.15 GB of scraped social-media exercise clips. They are
  **unlabelled, of unverified licence, and are NOT used to train any deployed
  model** — see [`docs/data_audit_2026-05-09.md`](data_audit_2026-05-09.md).
- `data/training/` holds 18 synthetic CSVs (good/bad × 3 exercises × 3 ages) plus
  one older real-extraction file (`bicep_adult_good_form_20260207...csv`, ~37 MB).
- The "62 videos" claim has no traceable basis for the deployed models. Drop it.

## 5. The actual scoring pipeline (`POST /api/score`)

```
video → VideoProcessor (MediaPipe) → per-frame 3 angles
      → RepCounter (5-state FSM + hysteresis) → RepEvent per rep
      → FormScorer: RandomForest score + rule-engine penalties
      → ScoreAggregator: weighted fuse → HeadlineScore
```

- **Rep counting** is a deterministic finite-state machine, not ML. Its accuracy
  has only been tested on synthetic sine waves; it is **not empirically measured**
  on real reps. (The "99% rep counting" claim was never measured.)
- **CNN+LSTM**: currently a **null seam** — `core/cnn_lstm_scorer.py` ships a
  `NullFormQualityScorer` that always returns `None`. The aggregator then drops the
  CNN+LSTM weight. **No trained CNN+LSTM exists yet.** This is the supervisor-
  mandated deliverable still to build.
- **Thresholds** (`exercises/<ex>/seed_thresholds.json`) are tagged
  `"synthetic"` — they mirror the hard-coded rule-engine defaults; they are not
  video-derived or NSCA/ACSM-sourced.

## 6. Known limitations (state these plainly in the report)

1. Models trained on synthetic priors; **no validation on real labelled reps**.
2. CNN+LSTM not yet implemented (null seam).
3. Rep-counter and rule thresholds are uncalibrated heuristics.
4. Dataset licence/provenance for `data/FYP/` is unverified.

## 7. How to talk about this honestly (viva framing)

- ✅ "Models are trained on **synthetic biomechanical data** because no labelled
  real dataset was available; accuracy on synthetic test data is 75–97%, and
  real-world accuracy is a stated limitation we have not yet measured."
- ✅ "Rep counting is a deterministic FSM with hysteresis; we validated its logic
  on synthetic signals."
- ✅ "The CNN+LSTM is architected as a pluggable scorer (the pipeline already has
  the seam); training it on labelled landmark sequences is the next milestone."
- ❌ Do **not** say "95% accuracy," "99% rep counting," "trained on 62 videos,"
  "K-Means + Random Forest," or "we didn't use deep learning."
