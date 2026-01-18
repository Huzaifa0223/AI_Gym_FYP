# AI Gym - Copilot Instructions

## Project Overview
**AI Gym** is a real-time exercise form detection system using MediaPipe pose estimation and ML models. It automatically detects exercises (bicep/back/chest), counts reps, provides form feedback, and adapts to age groups (children/adult/senior).

**Current Status**: 60% complete. Framework & API ready, need ML models trained for 8/9 exercise-age combinations.

---

## Architecture At a Glance

### Three Core Layers
1. **Pose Detection** → MediaPipe (33 landmarks, real-time)
2. **Exercise Logic** → Exercise classes calculate angles, check form, count reps
3. **Backend API** → FastAPI REST endpoints for frontend integration

### Data Flow
```
Camera Frame → MediaPipe Landmarks → Exercise.process_frame() 
→ ML Model Classify → Rep Detection → Feedback/Counter → JSON Response
```

### Directory Structure
- `exercises/` - Exercise classes (base, back, chest, bicep)
- `data/models/` - Trained ML models (`.pkl` files)
- `data/training/` - Training CSV files from video processing
- `videos/` - Source videos organized by exercise/age/quality
- `api_backend.py` - FastAPI server
- `main.py` - Desktop app with visual feedback

---

## Key Files & Their Roles

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `exercises/base_exercise.py` | Abstract base for all exercises | `BaseExercise`, `ExerciseConfig`, `ExerciseResult`, `AGE_CONFIGS` |
| `exercises/back_exercise.py` | Back row detection | `BackExercise` - calculate angles, form checking |
| `exercises/chest_exercise.py` | Push-up detection | `ChestExercise` |
| `bicep_curl.py` | Bicep curl detection | `BicepCurlExercise` |
| `pose_utils.py` | Landmark & angle utilities | `calculate_angle_3d()`, `PoseLandmark`, `AngleResult` |
| `exercise_classifier.py` | Auto-detect exercise type | `ExerciseClassifier.detect_exercise_type()` |
| `video_pipeline.py` | Extract pose data from videos | `VideoProcessor.extract_landmarks_from_video()` |
| `train_universal.py` | Train all ML models | `ExerciseModelTrainer` |
| `api_backend.py` | FastAPI REST server | `/api/process-frame`, `/api/start-session` |
| `main.py` | Desktop real-time app | ML initialization, frame loop, visual feedback |

---

## Critical Patterns & Conventions

### 1. **Age Groups Matter**
Every exercise has **age-specific configurations** in `AGE_CONFIGS` dict:
```python
AGE_CONFIGS = {
    'children': {
        'bicep': ExerciseConfig(min_angle=60, max_angle=160, flexibility_tolerance=15.0, ...),
        'back': ExerciseConfig(...),
        ...
    },
    'adult': {...},
    'senior': {...}
}
```
- **Children**: More flexibility (±15°), slower reps allowed (5s)
- **Adult**: Standard ranges, stricter timing
- **Senior**: More flexibility (±20°), longer reps allowed

**When adding features**, always check `AGE_CONFIGS` and test across age groups.

### 2. **Exercise Class Pattern**
All exercises inherit from `BaseExercise` and implement:
```python
class MyExercise(BaseExercise):
    def get_required_landmarks(self) -> List[int]:
        return [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
    
    def calculate_angles(self, landmarks) -> Dict[str, float]:
        # Return dict with 'primary', 'secondary', 'confidence'
        return {'primary': 145.2, 'secondary': 85.3, 'confidence': 0.98}
    
    def check_form(self, landmarks, angles) -> Tuple[str, Tuple, float]:
        # Return (feedback_text, RGB_color, quality_score)
        return ("✓ GOOD FORM", (0, 255, 0), 95.0)
    
    def detect_rep(self, angle: float) -> bool:
        # Returns True when rep is completed
```

### 3. **Angle Calculation**
Use `pose_utils.calculate_angle_3d()` for robustness:
```python
from pose_utils import calculate_angle_3d, AngleResult

result: AngleResult = calculate_angle_3d(landmark_a, landmark_b, landmark_c)
if result.is_valid:
    angle = result.angle
    confidence = result.confidence
```
Returns `AngleResult` with error handling, not raw float.

### 4. **Model Paths Convention**
Models stored as: `data/models/{exercise}_{age_group}.pkl`
```python
'data/models/bicep_adult.pkl'
'data/models/back_children.pkl'
'data/models/chest_senior.pkl'
```
**Always check model existence** before loading in production code.

### 5. **Feedback Format**
Exercise form checking returns: `(message: str, color_rgb: Tuple, quality_score: float)`
```python
# Good form
("✓ GOOD FORM", (0, 255, 0), 95.0)

# Issues
("⚠ ELBOW POSITION", (0, 165, 255), 60.0)
("❌ BACK TOO ROUNDED", (0, 0, 255), 40.0)
```

### 6. **Training Data Pipeline**
```
Videos (videos/exercise/age/quality/) 
→ video_pipeline.py (extract landmarks)
→ CSV files (data/training/exercise_age.csv)
→ train_universal.py (train models)
→ data/models/exercise_age.pkl
```
Each CSV has columns: `landmark_0_x`, `landmark_0_y`, `landmark_0_z`, ..., `landmark_32_visibility`

### 7. **API Response Format**
All `/api/process-frame` responses follow:
```json
{
  "success": true,
  "counter": 5,
  "feedback": "✓ GOOD FORM",
  "primary_angle": 145.2,
  "secondary_angle": 85.3,
  "rep_quality": 95.0,
  "confidence": 0.98,
  "detected_exercise": "bicep",
  "detection_confidence": 0.92,
  "is_valid_rep": true,
  "timestamp": "2024-01-18T..."
}
```

---

## Critical Developer Workflows

### Training a New Model
```bash
# 1. Add videos to videos/exercise/age/good_form/
# 2. Process videos
python video_pipeline.py

# 3. Train (generates data/models/exercise_age.pkl)
python train_universal.py --exercise bicep --age adult

# 4. Test immediately
python api_backend.py  # In terminal 1
python example_client.py test  # In terminal 2
```

### Testing Exercise Detection
```bash
# Desktop app with visual feedback
python main.py --exercise bicep --age adult

# API testing
python example_client.py demo bicep 25

# See real-time feedback on stdout + visual overlay
```

### Adding a New Exercise Type
1. Create `exercises/my_exercise.py` inheriting `BaseExercise`
2. Add to `AGE_CONFIGS` dict in `base_exercise.py`
3. Implement required methods (angles, form check, rep detection)
4. Import in `api_backend.py` and `main.py`
5. Register in `exercise_classifier.py` signatures
6. Collect videos → train model

### Debugging Form Detection
- Check `landmarks` validity (MediaPipe confidence > 0.5)
- Verify angle calculation in `calculate_angles()`
- Check form thresholds in `check_form()` method
- Look at `pose_utils.AngleResult.is_valid` flag
- Compare actual vs expected angle ranges in `AGE_CONFIGS`

---

## Important Integration Points

### MediaPipe Landmarks
33-point body model used throughout:
- **11, 12**: Left/Right Shoulders
- **13, 14**: Left/Right Elbows  
- **15, 16**: Left/Right Wrists
- **23, 24**: Left/Right Hips
- **25, 26**: Left/Right Knees
- **27, 28**: Left/Right Ankles

Only use these indices; MediaPipe updates may shift indices.

### ML Models Used
- **K-Means**: Clustering for angle ranges (identifies "up" vs "down" states)
- **Random Forest**: Classification for form quality assessment
- Models stored as pickled scikit-learn objects

### External Dependencies
- `mediapipe==0.10.9` - Pose detection (non-negotiable version)
- `opencv-python==4.8.1.78` - Video/image I/O
- `fastapi==0.104.1` - REST API
- `scikit-learn==1.3.0` - ML training/inference

**Version mismatches cause landmark index shifts** — respect pinned versions.

---

## Common Pitfalls & Solutions

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| "Model not found" errors | Path mismatch `data/models/` vs `models/` | Use absolute paths via `os.path.join()` |
| Angle calculations return 0 | Landmark visibility < 0.5 | Check `result.is_valid` before using angle |
| Rep counter stuck | State machine never transitions | Verify angle crosses both min/max thresholds |
| Different results per age group | Config not loaded for age | Always pass age_group to `BaseExercise.__init__()` |
| Slow API responses (>100ms) | Model inference bottleneck | Profile with `timeit`, consider quantization |
| False rep counts | Form check too permissive | Increase `min_rep_time`, lower quality threshold |

---

## Before Committing Changes

1. **Test across all age groups** — Use `example_client.py` with different ages
2. **Check model loading** — Verify `data/models/exercise_age.pkl` exists before deploy
3. **Validate API response** — Ensure JSON includes all required fields
4. **Profile latency** — Real-time needs < 100ms per frame
5. **Update `ARCHITECTURE.py`** if major flow changes

---

## Getting Started Tasks

- [ ] Read `COMPLETE_GUIDE.md` for current state
- [ ] Review `ARCHITECTURE.py` (run it: `python ARCHITECTURE.py`)
- [ ] Check `exercises/base_exercise.py` for exercise pattern
- [ ] Run `python api_backend.py` locally
- [ ] Test with `python example_client.py test`
- [ ] Review video folder structure in `QUICK_START.md`
