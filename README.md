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
