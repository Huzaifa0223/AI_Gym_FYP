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

### Exception — bicep good-form data is real

62 real bicep-curl videos were pose-extracted into
`data/training/bicep_adult_good_form_20260207_171944.csv` (14,006 frames,
**good-form only**, plus mirror augmentation). An earlier `bicep_curl/adult`
model was trained on this **real good-form** data combined with **synthetic
bad-form** examples. However, the **currently-deployed** `bicep_curl_adult.pkl`
is a later (2026-05-07) `synthetic_trainer.py` batch rebuild that **overwrote**
it — all 9 `.pkl` files share that timestamp and the short-form
`exercise_type='bicep'` metadata of the synthetic batch. So the *live* bicep
model is synthetic; the real 62-video data still exists in `data/training/` and
is a strong candidate for the CNN+LSTM (real good-form sequences).

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
- The "62 videos" figure is **real**: 62 bicep good-form clips were extracted into
  the Feb-7 CSV (§2). They trained an *earlier* bicep model; the **deployed** bicep
  `.pkl` is a later synthetic rebuild. For all 9 models *as deployed today*, the
  good/bad boundary is synthetic.

## 5. The actual scoring pipeline (`POST /api/score`)

```
video → VideoProcessor (MediaPipe) → per-frame 3 angles
      → RepCounter (5-state FSM + hysteresis) → RepEvent per rep
      → FormScorer: RandomForest score + rule-engine penalties
      → ScoreAggregator: weighted fuse → HeadlineScore
```

- **Rep counting** is a deterministic finite-state machine (not ML). It closes a
  rep on full re-extension **or a top turnaround**, so reps still count when the
  lifter doesn't fully re-extend. On the 62 real clips it now captures ~36 reps
  (was ~25 before the turnaround; a prominence peak-count estimates ~105), so it
  still under-counts. Root cause: real curls have a narrow ROM (~27°, e.g.
  73°→100°) against ~15°/frame jitter — a ~2:1 SNR a streaming threshold FSM
  cannot cleanly resolve. Accurate real-rep counting needs a prominence-based
  segmenter + thresholds calibrated to real recordings. (The "99% rep counting"
  claim was never measured.)
- **CNN+LSTM**: built, trained, integrated (`core/cnn_lstm_model.py` →
  `FormQualityNet` (1-D CNN → LSTM); scorer `CnnLstmFormScorer`; trainer
  `training/train_cnn_lstm.py`). 86% on held-out **synthetic** sequences; scores
  synthetic-shaped clean reps correctly (good ≈ 85, partial-ROM ≈ 1). It is
  **OFF by default** (`AI_GYM_ENABLE_CNN_LSTM=1` to enable). Evaluation on the 62
  real bicep clips revealed **two distinct problems** (both verified):
  1. **Segmentation** — the production rep counter under-segmented real video
     (0 reps on 45 of 62 clips) because it only closed a rep on re-extension past
     ~130°. A **top-turnaround close now ships in the live FSM** (real capture
     25→36), and the trainer's validation segments by curl-bottom peaks (~105).
     It still under-counts (SNR ~2:1 — see the *Rep counting* bullet above); a
     pipeline-wide limit (rep count, RF, rules), not just the CNN+LSTM.
  2. **Sim-to-real transfer — the deeper, unfixable-by-synthetic one** — even
     with proper peak-segmentation of all 105 real reps, the synthetic-trained
     model recognises only **~6%** as good (mean 7, median 1). Real good-form
     curls have lower measured ROM (~65° vs the synthetic ~113°), ~15°/frame
     jitter, and different secondary dynamics, so they land in what the synthetic
     model calls "bad." **No amount of synthetic tuning fixes this** — the real
     good/bad boundary can only be learned from real *labelled* examples. This is
     the same synthetic→real limitation as the RF models, now empirically shown.
  Conclusion: the mandated CNN+LSTM **exists, is integrated, and works on
  synthetic data**, but is honestly **not fit to fuse** until trained on real
  labelled sequences. Hence off by default; fusing it would tank real-video
  scores.
- **Thresholds** (`exercises/<ex>/seed_thresholds.json`) are tagged
  `"synthetic"` — they mirror the hard-coded rule-engine defaults; they are not
  video-derived or NSCA/ACSM-sourced.

## 6. Known limitations (state these plainly in the report)

1. Models trained on synthetic priors; **no validation on real labelled reps**.
2. CNN+LSTM is implemented and trained on synthetic sequences (86% synthetic, off
   by default), but **does not transfer to real reps**: with proper peak
   segmentation it recognises only ~6% of real good-form curls (sim-to-real gap —
   real ROM ~65° vs synthetic ~113°; see §5). Needs real labelled training data;
   real bad-form discrimination is unvalidated.
3. Rep-counter: a **top-turnaround close now ships** (real capture 25→36), but
   real counting is still limited — real curls have a narrow ROM (~27°) vs ~15°
   jitter (~2:1 SNR), and the bottom threshold (60°) is below the real median
   curl-bottom (~73°). Needs a prominence-based segmenter + thresholds calibrated
   to real recordings. Rule thresholds are likewise uncalibrated heuristics.
4. Dataset licence/provenance for `data/FYP/` is unverified.

## 7. How to talk about this honestly (viva framing)

- ✅ "Models are trained on **synthetic biomechanical data** because no labelled
  real dataset was available; accuracy on synthetic test data is 75–97%, and
  real-world accuracy is a stated limitation we have not yet measured."
- ✅ "Rep counting is a deterministic FSM with hysteresis; we validated its logic
  on synthetic signals."
- ✅ "The CNN+LSTM is built, trained, and integrated (86% synthetic). We evaluated
  it on 62 real clips with proper rep segmentation: it recognises only ~6% of real
  good-form reps — a sim-to-real gap synthetic tuning can't close (real curls have
  lower ROM than our synthetic 'good'). So it's off by default; training on real
  labelled sequences is the honest next step — the same lesson as the RF models."
- ✅ "We collected 62 real bicep good-form videos (~14k frames); that data is
  available, though the *currently deployed* bicep classifier is a synthetic
  rebuild (§2)."
- ❌ Do **not** say "95% accuracy," "99% rep counting," "the **deployed** model is
  trained on 62 videos" (it is a synthetic rebuild — see §2),
  "K-Means + Random Forest," or "we didn't use deep learning."
