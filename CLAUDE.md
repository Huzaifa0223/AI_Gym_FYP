# AI Gym Trainer FYP — Engineering Charter

This file is the authoritative project constitution for Claude Code and all human
contributors. Read it before touching any file.

---

## 1. Project Purpose

Real-time edge AI fitness coach that detects exercise form and counts reps using a
laptop webcam. No cloud inference. No GPU. All processing runs on the local machine
within a strict latency budget.

---

## 2. Tech Stack (canonical — do not deviate)

| Layer            | Technology                            | Version  |
|------------------|---------------------------------------|----------|
| Language         | Python                                | 3.11     |
| Pose estimation  | MediaPipe Pose (33-landmark model)    | 0.10.x   |
| Computer vision  | OpenCV                                | 4.8.x    |
| Numerical        | NumPy, Pandas                         | see reqs |
| ML classifier    | scikit-learn RandomForestClassifier   | 1.3.x    |
| Model IO         | joblib                                | 1.3.x    |
| API backend      | FastAPI + uvicorn[standard]           | 0.104.x  |
| Data validation  | Pydantic v2                           | 2.5.x    |
| CV loop (desktop)| OpenCV window via `main.py`           | —        |
| Dev toolchain    | Claude Code native binary             | latest   |
| Object detection | YOLOv8-nano (ultralytics)             | 8.3.x    |

> Object detection is added by Stage 4 of the 3-pillar pipeline. Weights live at
> `models/yolov8n.pt` and are gitignored — fetch with
> `python -m scripts.fetch_yolo_weights`. The fetch script requires network
> access; runtime loading from the local file is offline.

### Product UI & app-data service (front of the stack)

The product UI is a **React + Vite** single-page app, served by a local
**Node/Express + SQLite** app-data service (user accounts, session/workout
history). This is **application-data plumbing, not ML**: all pose estimation,
rep counting, and form scoring run in the Python edge process with **no cloud
inference**. The original FYP proposal specified a React / Node / MySQL stack;
SQLite is the local-dev substitute for MySQL. The UI talks to the Python ML
service over HTTP (`POST /api/live-frame` for the live camera path,
`POST /api/score` for uploaded video).

### FORBIDDEN technologies (never introduce)

- Streamlit / Flask / Django (no second Python web framework — FastAPI is canonical)
- Docker / Kubernetes (out of scope for FYP hardware target)
- **Any cloud inference endpoint (OpenAI, AWS, GCP, Azure AI).** This is the
  hard line: ML inference stays on-device. Cloud *training* (Colab/Kaggle GPU)
  is allowed; cloud *runtime inference* is not.

> Removed from this list (2026-05-31): the React/Node/npm and MySQL/external-DB
> bans. The shipped product uses them for the UI + app-data tier (above); the
> ban predated that decision and conflicted with the product as built.

---

## 3. Hardware & Performance Constraints

- **Target machine:** 11th Gen Intel Core i5, 8 GB RAM, integrated graphics (no dGPU)
- **OS:** Windows 11 primary; must also pass CI on ubuntu-latest
- **Inference budget:** ≤ 100 ms end-to-end per frame (MediaPipe + RF predict combined)
- **Model size:** each `.pkl` file must stay ≤ 10 MB; enforce during training
- **Memory ceiling:** the FastAPI process must stay ≤ 512 MB RSS under load

---

## 4. Repository File Layout

```
AI_Gym_FYP/
├── core/               # ExerciseClassifier and shared business logic
├── exercises/          # Per-exercise classes (BaseExercise ABC + bicep/back/chest)
├── pipeline/           # Video pipeline, skeleton recorder
├── utils/              # angle_utils, pose_utils, visual_utils, expert_coach
├── training/           # Offline scripts: collect_data, train_universal, etc.
├── tools/              # Dev helpers, architecture docs, example clients
├── tests/              # pytest unit & integration tests
├── scripts/            # setup_dev.ps1, setup_dev.sh, CI helpers
├── data/
│   ├── models/         # Trained .pkl files (gitignored)
│   └── training/       # CSV datasets (gitignored)
├── .github/workflows/  # GitHub Actions CI
├── main.py             # Entry point: OpenCV real-time loop
├── api_backend.py      # Entry point: FastAPI REST API
├── config.py           # Centralised paths, constants, age-group helpers
└── requirements.txt    # Pinned third-party deps (Python 3.11, Win + Linux)
```

---

## 5. Entry Points

```bash
# Real-time CV loop (OpenCV window)
python main.py

# REST API server (hot-reload for development)
uvicorn api_backend:app --reload --host 0.0.0.0 --port 8000

# Train all 9 ML models (3 exercises × 3 age groups)
python training/train_universal.py

# Install dependencies
pip install -r requirements.txt
```

---

## 6. Coding Standards (non-negotiable)

### Type hints
Every public function signature must carry full PEP 484 type hints including return
type. Use `from __future__ import annotations` at the top of each file so forward
references resolve lazily.

### Docstrings
Google-style docstrings on every public function and class. Mathematical helper
functions **must** include a geometric/algorithmic explanation in the docstring
(why this formula, what coordinate system, what edge cases).

### Logging
Use `stdlib logging` exclusively. No `print()` for debug output. Initialise a
module-level logger: `logger = logging.getLogger(__name__)`. No bare `except:`
clauses — always catch a specific exception type.

### PEP 8
Lines ≤ 100 characters. Verified by pylint using the project `.pylintrc`.

### Error handling
Validate at system boundaries (HTTP requests, file I/O, external API calls).
Trust internal invariants; do not add redundant guards inside well-typed functions.

---

## 7. ML Model Conventions

- 9 models total: `{exercise}_{age_group}.pkl` → `bicep_adult.pkl`, etc.
- Stored as dicts: `{'type': 'supervised', 'model': RandomForest, 'feature_cols': [...]}`
- 3 features per model: `primary_angle`, `secondary_angle`, `tertiary_angle`
- Class 0 = bad form, Class 1 = good form
- Load via `joblib.load()`; never use `pickle.load()` directly in production code

### Angle definitions

| Exercise        | primary              | secondary             | tertiary              |
|-----------------|----------------------|-----------------------|-----------------------|
| `bicep_curl`    | shoulder→elbow→wrist | hip→shoulder→elbow    | hip→shoulder→wrist    |
| `bent_over_row` | shoulder→elbow→wrist | hip→shoulder→elbow    | hip→shoulder→wrist    |
| `push_up`       | shoulder→elbow→wrist | shoulder→hip→knee     | hip→knee→ankle        |

---

## 8. Dev Toolchain

Claude Code native binary is the standard development assistant for this project.
Install via the official installer (see `scripts/setup_dev.ps1` / `setup_dev.sh`).
Do **not** install Claude Code via npm.

```bash
# Verify installation
claude --version
```

---

## 9. Testing

- All tests live in `tests/`
- Run with: `pytest tests/ -v`
- CI runs pytest on every push/PR to `master` (see `.github/workflows/ci.yml`)
- Do not mock the RandomForest models in unit tests — use fixture CSVs and
  train a tiny model in the test setup to catch real sklearn API changes

---

## 10. What Claude Code Must Never Do

1. Introduce any package from the FORBIDDEN list above.
2. Add `print()` statements for debugging.
3. Write bare `except:` or `except Exception:` without re-raising or structured logging.
4. Create Streamlit, Flask, or any UI framework outside of the approved stack.
5. Hard-code absolute file paths — always use `pathlib.Path` relative to `config.BASE_DIR`.
6. Commit trained `.pkl` files or raw video data to git.
7. Bypass `.pylintrc` rules with inline `# pylint: disable` without a comment explaining why.

---

## 11. Exercise naming convention

The system has three layers that talk about exercises in different vocabularies.
**Do not collapse them.**

1. **Body-part classifier** (`core/exercise_classifier.py`) — emits coarse body
   parts (`arms`, `back`, `chest`, `legs`, `shoulders`, `unknown`). It does
   *not* emit specific exercise keys. Treating its output as a rule-engine
   key is a bug.
2. **Body-part → exercise map** (`core/exercise_mapping.BODY_PART_TO_EXERCISE`)
   — single source of truth. Maps each body part to exactly one specific
   exercise:
   - `arms`  → `bicep_curl`
   - `back`  → `bent_over_row`
   - `chest` → `push_up`

   Body parts intentionally absent from the map (`legs`, `shoulders`,
   `unknown`) flow through to `NullRuleEngine` / caller default — preserving
   graceful degradation.
3. **Rule engines, ML aggregator, rep counters, configs** — keyed on the
   long-form exercise key (`bicep_curl`, `bent_over_row`, `push_up`).
   Trained model filenames follow the same scheme:
   `data/models/{exercise}_{age_group}.pkl` →
   `bicep_curl_adult.pkl`, `bent_over_row_senior.pkl`, etc.

When adding a new exercise, add it to the layer-3 keys *and* to
`BODY_PART_TO_EXERCISE` in one commit.

## 12. Pipeline composition (3-pillar wiring, post-Stage 5c)

The video-scoring path composes per-stage components in `core/`:

```
POST /api/score
    └── ScoringPipeline (core/pipeline.py)
        ├── ThresholdProvider.load(exercise)         # 422 if missing
        ├── VideoProcessor.process_bytes()           # frames → features
        ├── make_rep_counter()                       # FSM rep detection
        ├── FormScorer.score(rep)                    # per-rep ML + rules
        ├── FormQualityScorer.score(rep)             # CNN+LSTM (null seam)
        └── score_aggregator.aggregate(...)          # → HeadlineScore
```

The pipeline runs synchronously inside an executor thread
(`loop.run_in_executor`) so the FastAPI event loop stays free for the
1 Hz YOLOv8 equipment task. `LatestFrameBuffer` is teed from
`POST /api/process-frame` so the equipment task has live frames to
process. `app.state` carries the long-lived holders / detector / task /
provider / scorer.

Out-of-band components:

* **YOLOv8 equipment detector** — async background task at 1 Hz, reads
  the latest frame from `LatestFrameBuffer`, writes `EquipmentState` to
  the holder. `GET /api/equipment` reads the holder snapshot.
* **Reference skeleton** — static JSON at
  `exercises/<exercise>/reference_skeleton.json` served by
  `GET /api/reference/{exercise_id}` for the React ghost-overlay.

## 13. Known asymmetries (3-pillar migration)

The 3-pillar pipeline (`docs/architecture_3pillar.md`) introduces a deliberate
inconsistency that contributors must understand before changing the scoring path:

- **Existing RandomForest models stay age-group-keyed.** They are loaded as
  `data/models/{exercise}_{age_group}.pkl` and the API maps numeric age →
  `children` / `adult` / `senior` before lookup. These models are **not**
  retrained for this FYP — the cost/benefit is wrong for the timeline.
- **New components use body-segment normalization instead.** The CNN+LSTM form
  scorer and the heuristic threshold extractor consume landmarks already
  rescaled to body-segment-length units, so they are demographic-blind by
  construction.
- **The `ScoreAggregator` weights both correctly.** When the CNN+LSTM is
  available the weights are `rules 0.4 / RF 0.3 / NN 0.3`; otherwise
  `rules 0.6 / RF 0.4`. See `docs/architecture_3pillar.md` §3 and §6.

This asymmetry is documented in the FYP report as a defensible incremental
migration. Do not "harmonize" by retraining the RF models or by removing the
age-group split unless the project explicitly schedules it.
