# AI Gym — Status & Handoff (Quick Note)

## Current Status
- Framework, API, and tooling: **Ready**
- Trained models: **0 / 9** (bicep/back/chest × children/adult/senior)
- Readiness: `/ready` returns `ready: false` until models exist

## What works now
- Health & readiness endpoints
- Model inventory endpoint: `/api/models/status`
- Session isolation (UUIDs)
- Graceful degradation (`model_missing`, `degraded` flags)
- Structured logging with latency

## How to check
```bash
# With server running
curl http://localhost:8000/api/models/status
curl http://localhost:8000/ready
```

## How to train the missing models
```powershell
# Process videos + train all missing models (expects videos/ populated)
./scripts/train_all.ps1

# If CSVs already exist
./scripts/train_all.ps1 -SkipVideo
```
Models are expected at `data/models/{exercise}_{age}.pkl`.

## How to test quickly
```powershell
# Basic endpoint tests
./test_server.ps1

# Advanced feature tests
./test_advanced_features.ps1
```

## Data needed to finish
Add good-form videos to:
```
videos/{exercise}/{age_group}/good_form/
  exercise: bicep | back | chest
  age_group: children | adult | senior
```
Then rerun training.

---

## Developer Environment

### Requirements

- **Python 3.11** (exactly) — the project uses type-hint syntax and library
  versions pinned to 3.11. Check with `python --version`.
- **Git** for cloning and version control.
- **Claude Code native binary** — the standard dev-toolchain for this project
  (see installation below).

---

### One-command setup

**Windows (PowerShell)**

```powershell
# From repo root — creates .venv, installs deps, checks for Claude Code
powershell -ExecutionPolicy Bypass -File scripts\setup_dev.ps1
```

**Linux / macOS / WSL (bash)**

```bash
# Make executable once, then run
chmod +x scripts/setup_dev.sh
./scripts/setup_dev.sh
```

Both scripts are **idempotent** — safe to re-run after pulling new changes.

---

### Start the REST API

```bash
# Activate venv first
# Windows:  .\.venv\Scripts\Activate.ps1
# Linux:    source .venv/bin/activate

uvicorn api_backend:app --reload --host 0.0.0.0 --port 8000
```

- API docs (Swagger UI): http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Model status: http://localhost:8000/api/models/status

---

### Score a workout video

`POST /api/score` accepts a full workout clip and returns a rep-by-rep
breakdown in a single call.

```
POST /api/score
Content-Type: multipart/form-data

Form fields:
  video      — UploadFile (mp4/mov/avi/webm, max 50 MB)
  exercise   — one of: bicep, back, chest
  age_group  — one of: children, adult, senior
```

**curl example**

```bash
curl -X POST http://localhost:8000/api/score \
     -F "video=@/path/to/workout.mp4;type=video/mp4" \
     -F "exercise=bicep" \
     -F "age_group=adult"
```

**Sample response**

```json
{
  "exercise": "bicep",
  "age_group": "adult",
  "total_reps": 2,
  "average_score": 82.5,
  "frames_processed": 180,
  "frames_with_landmarks": 176,
  "processing_time_s": 4.21,
  "reps": [
    {
      "rep_number": 1,
      "start_ts": 0.40,
      "end_ts": 2.10,
      "duration_s": 1.70,
      "peak_angle": 162.4,
      "trough_angle": 42.1,
      "ml_score": 88.0,
      "ml_confidence": 0.91,
      "rule_penalty": 5.0,
      "final_score": 83.0,
      "violations": [
        {
          "name": "too_fast",
          "severity": "minor",
          "message": "Rep too fast (0.7s < 1.0s) — control the negative",
          "penalty": 5.0
        }
      ]
    }
  ]
}
```

`ml_score`, `ml_confidence`, and `final_score` are `null` when the ML
model is unavailable for the requested `{exercise, age_group}` pair.
Error responses: **400** (bad input / undecodable video), **413** (>50 MB),
**415** (non-`video/*` content type), **422** (missing form field).

---

### Start the real-time CV loop

```bash
python main.py
```

Opens an OpenCV window. Press `q` to quit. Requires a webcam.

---

### Run tests

```bash
pytest tests/ -v
```

---

### Train all 9 ML models

```bash
python training/train_universal.py
```

Models are written to `data/models/` as `{exercise}_{age_group}.pkl`.

---

### Claude Code — standard dev toolchain

Claude Code is the official AI assistant integrated into this project's workflow.

**Install (Linux / macOS / WSL)**

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**Install (Windows PowerShell)**

```powershell
irm https://claude.ai/install.ps1 | iex
```

Do **not** use `npm install -g @anthropic-ai/claude-code`.

**Verify**

```bash
claude --version
```

**Usage**

```bash
# Interactive session from repo root
claude

# Single command
claude "explain the angle calculation in utils/angle_utils.py"
```

Claude Code reads `CLAUDE.md` at the repo root for project-specific instructions.
All architectural decisions and coding standards are documented there.
