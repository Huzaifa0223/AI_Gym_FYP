# Training Dataset Inventory — `data/FYP/` (2026-05-09)

Read-only baseline. **No data files were modified, no models were trained, no
loader code was written** while producing this document. Branch: `master`.
Repo: `E:\FYP 1\AI_Gym_FYP`. Tooling: PowerShell `Get-ChildItem`, OpenCV
4.11.0 (`.venv\Scripts\python.exe`), `ffprobe.exe` (Gyan FFmpeg 8.0.1).

This audit is the prerequisite for any pillar-2 (CNN+LSTM form scorer) work
described in `docs/architecture_3pillar.md`. It follows the convention of
`docs/audit_2026-05-06.md` and `docs/audit_2026-05-07_5c.md`.

---

## 1. Dataset overview

| Property | Value |
| --- | --- |
| Root | `E:\FYP 1\AI_Gym_FYP\data\FYP` |
| Top-level buckets | **12** (4 muscles × 3 age groups) |
| Total files | **4124** |
| Total size on disk | **8341.38 MB (8.15 GB)** |
| Distinct extensions | `.mp4` = 4118, `.mkv` = 6 |
| Image / tabular / array files | **0** (no `.png`, `.jpg`, `.csv`, `.json`, `.npy`, `.pkl`, `.parquet`, `.txt`) |
| README / LICENSE / metadata sidecars | **0** found anywhere under the root |

The dataset is **video-only**. There is no companion CSV, JSON, NPY, PKL,
or text file at any depth — confirmed by `Get-ChildItem -Recurse -File |
Group-Object Extension`.

---

## 2. Top-level folder structure

```
data/FYP/
├── BACK/
│   ├── AGE 10–16 (Youth)/
│   ├── AGE 17–45 (Adults)/
│   └── AGE 46–85 (Seniors)/
├── Bicep/
│   ├── Age 10–16 (Youth, default goal healthstrength gain)/
│   ├── Age 17–45 (Adults)/
│   └── Age 46–85 (Seniors)/
├── Chest/
│   ├── AGE 17–45 (Adults)/
│   ├── AGE 46–85 (Seniors)/
│   └── for age 10 - 16/
└── TRICEPS/
    ├── AGE 10–16 (Youth)/
    ├── AGE 17–45 (Adults)/
    └── AGE 46–85 (Seniors)/
```

**Folder-name inconsistency** (will need normalisation in any loader):

* Casing: `BACK` / `TRICEPS` (uppercase) vs `Bicep` / `Chest` (title case).
* Age separator: en-dash `–` in most folders, hyphen `-` in `Chest/for age 10 - 16`.
* Age folder phrasing: `AGE 10–16 (Youth)` (BACK, TRICEPS) vs
  `Age 10–16 (Youth, default goal healthstrength gain)` (Bicep) vs
  `for age 10 - 16` (Chest).

Below each age folder is a goal/level layer (`GOAL MAINTAIN WEIGHT`,
`Level 1 (Beginner)`, etc.) and below that an exercise folder
(`Dumbbell Row`, `Bench Dips`, …). Clip files live at the bottom.

The 4 muscle groups (`BACK`, `Bicep`, `Chest`, `TRICEPS`) **do not align
1-to-1** with the project's existing exercise keys (`bicep_curl`,
`bent_over_row`, `push_up`) defined in `CLAUDE.md` §11. The dataset
introduces a **fourth muscle group, `TRICEPS`, that is not currently
modelled** by the existing rule engines or RandomForest models.

---

## 3. Inventory table — per (muscle × age) bucket

| Muscle | Age folder | Files | Size (MB) | Extensions | Goal/Level subfolders | Exercise folders |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| BACK    | AGE 10–16 (Youth)                                       |  51 |   93.46 | mp4=51         | 1 | 5 |
| BACK    | AGE 17–45 (Adults)                                      | 628 | 1206.15 | mp4=627, mkv=1 | 3 | 57 |
| BACK    | AGE 46–85 (Seniors)                                     | 645 | 1197.06 | mp4=644, mkv=1 | 3 | 57 |
| Bicep   | Age 10–16 (Youth, default goal healthstrength gain)     |  49 |   57.21 | mp4=49         | 1 | 5 |
| Bicep   | Age 17–45 (Adults)                                      | 192 |  303.27 | mp4=190, mkv=2 | 3 | 19 |
| Bicep   | Age 46–85 (Seniors)                                     | 188 |  297.52 | mp4=186, mkv=2 | 3 | 19 |
| Chest   | AGE 17–45 (Adults)                                      | 526 | 2263.83 | mp4=526        | 3 | 57 |
| Chest   | AGE 46–85 (Seniors)                                     | 552 | 1355.01 | mp4=552        | 3 | 57 |
| Chest   | for age 10 - 16                                         |  50 |  391.57 | mp4=50         | 5 | 5 |
| TRICEPS | AGE 10–16 (Youth)                                       |  58 |   58.11 | mp4=58         | 1 | 5 |
| TRICEPS | AGE 17–45 (Adults)                                      | 606 |  629.89 | mp4=606        | 3 | 57 |
| TRICEPS | AGE 46–85 (Seniors)                                     | 579 |  488.31 | mp4=579        | 3 | 57 |
| **Total** |                                                       | **4124** | **8341.38** | **mp4=4118, mkv=6** | — | — |

`Goal/Level subfolders` counts the immediate children of the age folder.
`Exercise folders` counts directories that contain at least one video
(across the goal × level grid).

The Youth (10–16) buckets are markedly smaller (49–58 clips each,
~5 exercises) and use a single "Default goal Health & Strength Gain"
sub-tree, while Adults and Seniors fan out across three goals × three
levels × ~5–6 exercises = ~57 exercise folders.

---

## 4. Sub-hierarchy snapshot

**BACK / AGE 17–45 (Adults)** — representative full layout:

```
GOAL MAINTAIN WEIGHT/
  LEVEL 1 — Beginner (5 exercises, 12 reps, 3 sets)/
    Dumbbell Row, Face Pull, Hyperextension, Lat Pulldown, Seated Cable Row
  LEVEL 2 — Intermediate (6 exercises, 15 reps, 3 sets)/
    Cable Row, Dumbbell Row, Hyperextension, Lat Pulldown, …
  LEVEL 3 — …
GOAL WEIGHT GAIN (Muscle Building)/
  LEVEL 1, 2, 3 — same shape
GOAL WEIGHT LOSS/
  LEVEL 1, 2, 3 — same shape
```

**Chest / for age 10 - 16** — flat (no goal/level layer):

```
Chest Dips (bench-assisted, feet on floor)/
PUSH UPS/
Resistance Band Chest Press/
incline wall push ups/
knee push ups/
```

**Bicep / Age 10–16 (Youth)** — single goal layer:

```
Level 1 (Beginner) – 5 exercises, 12 reps, 3 sets/
  Bodyweight Chin-Ups (assisted if needed), …
```

The other 9 buckets follow the three-goal × three-level grid identical
to BACK/Adults above. Goal-folder names vary slightly between Adults
and Seniors (`GOAL WEIGHT GAIN (Muscle Building)` vs
`GOAL WEIGHT GAIN (Seniors)` vs
`GOAL WEIGHT GAIN (Muscle Building for Seniors)`).

---

## 5. Sample video metadata

One alphabetically-first clip was probed per bucket via
`cv2.VideoCapture` + `ffprobe`. All 12 opened cleanly.

| Muscle | Age | Sample file | W×H | FPS | Frames | Duration | Size (MB) | Audio |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| BACK    | Youth   | `Advanced Bird Dog.mp4`                         | 480×480 | 30.00 |  622 |  20.7 s | 0.87 | aac |
| BACK    | Adults  | `Dumbbell Row …try this 🤟🏽.mp4`               | 270×480 | 30.00 |  570 |  19.0 s | 1.11 | aac |
| BACK    | Seniors | `Back Extension _ Machine ….mp4`                | 640×360 | 25.00 |  919 |  36.8 s | 1.59 | aac |
| Bicep   | Youth   | `Assisted Chin Ups.mp4`                         | 270×480 | 29.95 |  575 |  19.2 s | 0.83 | aac |
| Bicep   | Adults  | `_The PERFECT Barbell Bicep Curl_.mp4`          | 480×854 | 30.00 |  262 |   8.7 s | 0.64 | aac |
| Bicep   | Seniors | `Don't Do Biceps Curls Like This ❌.mp4`         | 480×854 | 30.00 |  674 |  22.5 s | 1.67 | aac |
| Chest   | Adults  | `Cable Chest Press Variations….mp4`             | 480×854 | 29.97 |  663 |  22.1 s | 1.66 | aac |
| Chest   | Seniors | `CHEST EXERCISE_ Technogym ….mp4`               | 854×480 | 25.00 | 2661 | 106.4 s | 15.50 | aac |
| Chest   | Youth   | `Bench Dips (feet elevated).mp4`                | 480×854 | 23.98 |  228 |   9.5 s | 0.52 | aac |
| TRICEPS | Youth   | `Bench Dip Mistakes (FIX THESE!).mp4`           | 480×854 | 29.97 |  308 |  10.3 s | 0.74 | aac |
| TRICEPS | Adults  | `Bench Dip Mistakes (FIX THESE!).mp4`           | 480×854 | 29.97 |  308 |  10.3 s | 0.74 | aac |
| TRICEPS | Seniors | `Cable Overhead Tricep Extension….mp4`          | 360×640 | 30.00 |  194 |   6.5 s | 0.32 | aac |

**Codec / format observations:**

* Every probed clip carries an **AAC audio track** — this is consistent
  with social-media (YouTube/Instagram/TikTok) downloads.
* Resolutions are **all mobile-portrait or mobile-landscape**: 480×480,
  480×854 (vertical 9:16), 270×480, 360×640, 640×360, 854×480. None of
  the typical webcam resolutions (640×480, 1280×720) appear.
* Frame rates cluster at 30 / 29.97 / 25 fps — consistent with re-encoded
  social-media uploads, not original captures.
* Two identical-name clips (`Bench Dip Mistakes (FIX THESE!).mp4`) appear
  in both `TRICEPS\AGE 10–16` and `TRICEPS\AGE 17–45` with **identical
  size, duration, and frame stats** — suggesting cross-bucket file
  duplication; deduplication will likely be needed before training.

---

## 6. First 10 frames per sample

Decoded via `cap.read()`; numpy uint8 arrays in BGR order
(`cv2`'s native channel order). For each sample we record `shape`,
`dtype`, `mean`, `min`, `max` — which fully characterises frame format
without reproducing pixel data.

| Bucket | Frame `shape` (H, W, C) | `dtype` | mean range (f0 → f9) | min | max |
| --- | --- | --- | --- | ---: | ---: |
| BACK / Youth        | (480, 480, 3) | uint8 | 139.56 → 139.59 |  2 | 255 |
| BACK / Adults       | (480, 270, 3) | uint8 |  83.58 →  82.55 |  0 | 255 |
| BACK / Seniors      | (360, 640, 3) | uint8 | 126.40 → 126.53 |  0 | 255 |
| Bicep / Youth       | (480, 270, 3) | uint8 |  98.76 →  99.58 |  0 | 254 |
| Bicep / Adults      | (854, 480, 3) | uint8 |  96.33 →  96.60 |  0 | 249 |
| Bicep / Seniors     | (854, 480, 3) | uint8 | 115.85 → 116.25 |  0 | 255 |
| Chest / Adults      | (854, 480, 3) | uint8 |  85.97 →  85.71 |  0 | 255 |
| Chest / Seniors     | (480, 854, 3) | uint8 |   0.00 →  14.08 |  0 |  61 |
| Chest / Youth       | (854, 480, 3) | uint8 |  46.44 →  46.89 |  0 | 255 |
| TRICEPS / Youth     | (854, 480, 3) | uint8 |  74.77 →  79.46 |  0 | 255 |
| TRICEPS / Adults    | (854, 480, 3) | uint8 |  74.77 →  79.46 |  0 | 255 |
| TRICEPS / Seniors   | (640, 360, 3) | uint8 | 100.51 → 101.16 |  0 | 255 |

`Chest / Seniors` opens with 6 black frames (mean = 0, max ≤ 1) before
fading in — typical YouTube-style title card. This will need to be
trimmed for any frame-zero feature extraction.

---

## 7. Image / tabular / array files

**None present.** The extension scan over all 4124 files returned only
`.mp4` (4118) and `.mkv` (6); zero image, CSV, JSON, NPY, PKL, parquet,
or plain-text files. Therefore no rows-to-dump, no schema, no shape /
dtype to document.

---

## 8. Labels — good vs bad form

**No machine-readable labelling scheme exists in this dataset.** The
following label channels were checked and are all absent:

| Channel | Status |
| --- | --- |
| Subfolders named `good/` or `bad/` | absent |
| Subfolders named `correct/incorrect`, `right/wrong`, `pos/neg` | absent |
| `labels.csv`, `manifest.csv`, `annotations.csv` | absent (no `.csv` files anywhere) |
| Sidecar JSON next to each clip (`<name>.json`) | absent (no `.json` files anywhere) |
| Per-clip TXT manifest | absent (no `.txt` files anywhere) |
| README / LICENSE | absent |

Filenames carry **only implicit, non-machine-readable signals** (English
phrases and emojis indicating tutorial vs anti-pattern intent):

```
The PERFECT Dumbbell Row 🔥 BEST Guide Ever Made.mp4
STOP doing dumbbell rows like this.mp4
Don't Do Biceps Curls Like This ❌.mp4
Bench Dip Mistakes (FIX THESE!).mp4
Dumbbell Hammer Curl! Do's and Don'ts ❌✅ #dumbbell #hammercurls ….mp4
Cable Fly Mistake!  STOP DOING THIS!.mp4
✅ The PERFECT Cable Bicep Curl.mp4
🛑 STOP Doing Cable Rows LIKE THIS!.mp4
Back Extension Machine the RIGHT way✅….mp4
4 Facepull Mistakes You Need to FIX!.mp4
```

These cues are heuristic ("PERFECT", "STOP", "Don't", "MISTAKE", "FIX",
"WRONG", ❌, ✅, 🛑) and are **not reliable training labels** — many
"PERFECT" clips include side-by-side bad demonstrations, and many "STOP"
or "MISTAKE" clips show the corrected form afterwards. **Manual labelling
will be required** before any supervised training (for pillar-2's
CNN+LSTM form scorer or for new RandomForest models) is possible.

---

## 9. Granularity — rep / set / session?

Sampled clip durations range from **6.5 s to 106.4 s**, with most
falling in the **9–22 s** range. File-size distribution across the full
4124-file set ranges from ~0.3 MB to ~16 MB.

Combined with the social-media-style filenames ("How To Do …", "Tips
for …", "BEST Guide Ever Made"), this strongly indicates that each
clip is a **third-party tutorial / demo clip**, not a controlled-capture
single-rep or single-set recording. A single clip typically contains:

* a talking-head intro / outro,
* multiple repetitions (often **mixed good and bad** demonstrations
  intercut in the same clip),
* graphical overlays, captions, transitions,
* sometimes side-by-side or split-screen comparisons.

**Implication for the pipeline:** the existing rep counter
(`core/rep_counter.py`, `BaseRepCounter` FSM) cannot be applied to
these clips as-is to extract per-rep training samples — the FSM assumes
a single subject performing clean reps. Any future use of this dataset
for supervised training will require:

1. A rep-segmentation / shot-detection pre-pass to split each clip into
   per-rep windows.
2. A manual labelling pass (or a weak-supervision pass) to assign
   good / bad form to each rep window.

---

## 10. Source, license, provenance

**No source, no license, no attribution metadata was shipped with this
dataset.** There is no `README.md`, `LICENSE`, `SOURCES.txt`,
`attributions.csv`, or sidecar JSON anywhere under `data/FYP/`.

Filename evidence strongly suggests **third-party origin** (likely
scraped from YouTube / Instagram / TikTok):

* presence of `#shorts`, `#fitness`, `#gym`, `#bodybuilding` hashtags
  embedded in filenames,
* heavy use of emojis and clickbait phrasing ("KILLING GAINS!",
  "BEST Guide Ever Made"),
* AAC audio tracks at consumer encoder bitrates,
* mobile-portrait (480×854 9:16) and mobile-landscape resolutions
  characteristic of social-media content,
* duplicate clips appearing across age folders (e.g. `Bench Dip
  Mistakes (FIX THESE!).mp4` is byte-identical between
  `TRICEPS\AGE 10–16` and `TRICEPS\AGE 17–45`).

**Risk for the FYP report:** redistribution and academic-submission
rights for these clips are **unverified**. This must be resolved before
any FYP submission, demo, or CI fixture relies on these files. The
`.gitignore` already excludes `data/` (per `CLAUDE.md` §10), so the
clips will not enter the public git history; this audit doc is the only
record that they were inspected.

---

## 11. Implications for the 3-pillar pipeline

Cross-referencing `docs/architecture_3pillar.md` and `CLAUDE.md` §13:

* **Pillar 1 — RandomForest, age-group keyed** (`data/models/{exercise}_{age}.pkl`).
  Per `CLAUDE.md` §13 these models are explicitly **not retrained for this
  FYP**. This dataset therefore does **not** affect pillar 1.
* **Pillar 2 — CNN+LSTM form scorer** (`core/form_quality_scorer.py`,
  currently a null seam per the architecture doc). This pillar **cannot
  be trained from this dataset as-shipped** because:
  1. there are no labels (§8 above),
  2. clips are tutorial demos, not per-rep captures (§9 above),
  3. licensing is unverified (§10 above).
* **Pillar 3 — heuristic threshold extractor**. Body-segment-normalised,
  demographic-blind by construction; does not rely on supervised labels.
  Could in principle consume **landmark sequences** extracted from these
  clips for unsupervised threshold calibration, but the rep-segmentation
  blocker from §9 still applies, and the source-licensing blocker from
  §10 still applies.

The dataset also introduces `TRICEPS` as a fourth muscle group, which
is **not represented** in the current exercise mapping
(`core/exercise_mapping.BODY_PART_TO_EXERCISE` covers `arms`, `back`,
`chest` only — see `CLAUDE.md` §11). Adding tricep coverage would
require a new exercise key, a new rule engine, and entries in
`AGE_CONFIGS` — none of which is in scope for this audit.

---

## 12. Recommendations (out of scope for this commit, recorded for follow-up)

These items are **not** acted on in this audit. They are recorded for the
project memory so the next planning round can pick them up:

1. Clarify dataset source and license; obtain written permission from
   rights-holders or replace with self-captured / openly licensed clips
   before FYP submission.
2. If the dataset is to be used at all: add a manual or weak-supervision
   labelling pass (good / bad per rep) plus a rep-segmentation pre-pass.
3. Decide whether `TRICEPS` is in scope for the FYP. If yes, extend
   `BODY_PART_TO_EXERCISE`, add a `TricepRuleEngine`, add age configs.
4. Normalise folder names (lowercase, ASCII hyphen, single canonical age
   phrasing) **only if** a loader is later written — do not rename now,
   to avoid breaking any in-progress external work on the dataset.
5. Deduplicate cross-bucket identical clips (confirmed at least one pair
   in TRICEPS Youth ↔ TRICEPS Adults).
