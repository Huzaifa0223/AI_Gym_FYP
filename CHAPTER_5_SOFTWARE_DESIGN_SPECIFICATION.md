# Chapter 5: Software Design Specification

**AI Gym — Real-Time Exercise Form Detection System**

**Date:** February 8, 2026

---

## 5.1 Design Methodology and Software Process Model

### 5.1.1 Design Methodology: Object-Oriented Programming (OOP)

**Justification:**

AI Gym employs **Object-Oriented Programming (OOP)** as its core design methodology for the following reasons:

1. **Modularity and Reusability**
   - Each exercise type (bicep curl, back row, chest press) is encapsulated as a separate class inheriting from `BaseExercise`
   - Common functionality (angle calculations, form checking, rep detection) is reused across all exercises without code duplication
   - Age-specific variations are managed through configuration objects, not duplicated code

2. **Encapsulation and Maintainability**
   - Internal state (rep counter, current stage, confidence scores) is encapsulated within exercise objects
   - Clear separation of concerns: Pose detection → Exercise Logic → Feedback
   - Changes to one exercise type don't affect others due to inheritance hierarchy

3. **Extensibility**
   - New exercises can be added by creating a new class inheriting from `BaseExercise`
   - New features (age adaptation, form strictness levels) are added through configuration inheritance
   - The system can easily scale from 3 exercises (current) to 9+ exercises without architectural changes

4. **Abstraction**
   - Complex operations (3D angle calculation, confidence scoring, state machine logic) are hidden behind simple method interfaces
   - Frontend developers interact with a clean API (`process_frame()`, `detect_rep()`) without understanding internal complexity
   - ML model integration is abstracted, allowing model changes without code rewrites

5. **Polymorphism**
   - Different exercise classes implement the same interface differently (e.g., `calculate_angles()` calculates different joint angles for each exercise)
   - The system treats all exercises uniformly through the `BaseExercise` interface
   - Auto-detection system leverages polymorphism to automatically select appropriate exercise handler

**OOP Patterns Used:**

- **Template Method Pattern:** `BaseExercise.process_frame()` defines the overall algorithm; subclasses implement specific steps
- **Strategy Pattern:** Different form-checking strategies for different age groups via `AGE_CONFIGS`
- **Factory Pattern:** `ExerciseClassifier` creates appropriate exercise instances based on detected movement
- **Singleton Pattern:** ML model handler loads models once and reuses them

---

### 5.1.2 Software Process Model: Agile Incremental Development

**Chosen Model:** Iterative/Agile with sprint-based development

**Rationale:**

1. **Flexibility with Requirements**
   - Exercise types, age groups, and form parameters evolve based on testing results
   - User feedback (from coaches, athletes) drives feature prioritization
   - Agile allows rapid pivots without major rework

2. **Continuous Integration & Deployment**
   - Training pipeline runs automatically when new training data arrives (`train_universal.py`)
   - API backend can be updated without retraining all models
   - Desktop application (`main.py`) can be enhanced while backend remains stable

3. **Risk Mitigation**
   - Early prototype (bicep adult model exists) validates core architecture
   - Each age group trained independently reduces failure scope
   - Regular testing of inference accuracy (<100ms latency requirement)

4. **Stakeholder Visibility**
   - Weekly demos show progress (current: 1/9 models trained)
   - Documentation updated in parallel with development
   - Clear roadmap: Phase 1 (framework ✓) → Phase 2 (training) → Phase 3 (deployment)

**Development Phases:**

| Phase | Status | Duration | Deliverables |
|-------|--------|----------|--------------|
| **Phase 1: Architecture & Framework** | ✅ Complete | 8 weeks | FastAPI backend, Exercise classes, Model pipeline, MediaPipe integration |
| **Phase 2: Model Training** | 🔄 In Progress | 4-6 weeks | Train remaining 8 models (3 exercises × 3 age groups) |
| **Phase 3: Integration & Testing** | ⏳ Planned | 2 weeks | End-to-end testing, performance optimization, API documentation |
| **Phase 4: Deployment & Monitoring** | ⏳ Planned | 2 weeks | Cloud deployment, error tracking, user feedback loop |

---

## 5.2 System Overview

### 5.2.1 Purpose and Context

**AI Gym** is a real-time exercise form detection and rep-counting system designed to provide automated coaching feedback to users performing strength training exercises. It combines computer vision (MediaPipe pose estimation) with machine learning to:

1. **Automatically detect** which exercise the user is performing
2. **Count repetitions** accurately with rep quality assessment
3. **Analyze form** and provide corrective feedback in real-time
4. **Adapt to user age groups** (children, adults, seniors) with age-appropriate form standards
5. **Maintain persistent data** across workout sessions for progress tracking

**System Context:**

```
┌─────────────────────┐
│   End User (Coach/  │
│    Athlete)         │
└──────────┬──────────┘
           │ Webcam Video
           ↓
┌──────────────────────────────────┐
│      AI Gym System                │
│  ┌────────────────────────────┐  │
│  │ Pose Detection (MediaPipe) │  │
│  └────────────┬───────────────┘  │
│               ↓                   │
│  ┌────────────────────────────┐  │
│  │ Exercise Recognition       │  │
│  │ (Auto-detection classifier)│  │
│  └────────────┬───────────────┘  │
│               ↓                   │
│  ┌────────────────────────────┐  │
│  │ Exercise Processing        │  │
│  │ (Rep counting, Form check) │  │
│  └────────────┬───────────────┘  │
│               ↓                   │
│  ┌────────────────────────────┐  │
│  │ Real-Time Feedback         │  │
│  │ (Visual + text)            │  │
│  └────────────────────────────┘  │
└─────────────────────────────────┘
           ↓
┌──────────────────────┐
│  Feedback Display    │
│  • Rep count         │
│  • Form quality      │
│  • Target angle      │
│  • Corrective tips   │
└──────────────────────┘
```

### 5.2.2 Functional Overview

**Core Functions:**

1. **Real-Time Pose Estimation**
   - Captures full-body skeleton (33 landmarks) from webcam frames
   - Runs at 30 FPS to ensure smooth feedback
   - Filters out low-confidence detections (<0.5 visibility threshold)

2. **Exercise Auto-Detection**
   - Analyzes pose landmarks to determine which exercise is being performed
   - Uses movement patterns: torso angle, arm plane, active joints, key angles
   - Returns confidence score for user awareness

3. **Angle Calculation**
   - Calculates 3D angles at key joints (elbows, knees, shoulders, hips)
   - Primary angle: main movement axis (e.g., elbow angle for bicep curl)
   - Secondary angle: stability check (e.g., torso angle for back row)
   - Returns confidence based on landmark visibility

4. **Form Assessment**
   - Checks if current pose matches "good form" criteria for the exercise
   - Provides specific feedback: "✓ GOOD FORM", "⚠ ELBOW POSITION", "❌ BACK TOO ROUNDED"
   - Assigns quality score (0-100) representing form correctness

5. **Rep Detection and Counting**
   - Uses state machine: "up" ↔ "down"
   - Rep completes when angle transitions from full range (e.g., 50° to 170°) back down
   - Validates rep timing (min/max rep duration)
   - Counts only "valid reps" (good form + proper timing)

6. **Age-Specific Adaptation**
   - Children: Flexible standards (±15° tolerance, 5s max rep time) for developing bodies
   - Adults: Standard form requirements (±10° tolerance, 4s max rep time)
   - Seniors: Extra flexibility (±20° tolerance, 5s+ rep time) for joint health

7. **Session Management**
   - Starts sessions with UUID for state continuity
   - Maintains rep counter across frames
   - Optionally saves workout summaries and session history

---

## 5.2.1 Architectural Design

### 5.2.1.1 System Architecture Overview

AI Gym follows a **Three-Tier Layered Architecture** with clear separation of concerns:

```
┌────────────────────────────────────────────────────────────┐
│              PRESENTATION LAYER                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Frontend Interfaces                                │  │
│  │  • Desktop App (Python + OpenCV)                    │  │
│  │  • Web Dashboard (REST API Consumer)                │  │
│  │  • Mobile App (REST API Consumer)                   │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────┬─────────────────────────────────────────────┘
             │ HTTP Requests (JSON)
             ↓
┌────────────────────────────────────────────────────────────┐
│              BUSINESS LOGIC LAYER                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  FastAPI Backend (api_backend.py)                   │  │
│  │  • Request validation & deserialization             │  │
│  │  • Session management & routing                     │  │
│  │  • Exercise instantiation & delegation              │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Exercise Classes (exercises/ folder)               │  │
│  │  • BaseExercise (abstract)                          │  │
│  │  • BicepCurlExercise, BackExercise, ChestExercise  │  │
│  │  • Shared logic: angle calculation, form checking   │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Auto-Classifier (exercise_classifier.py)           │  │
│  │  • Detects exercise from pose landmarks             │  │
│  │  • Returns exercise type + confidence               │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────┬─────────────────────────────────────────────┘
             │ Low-Level Operations
             ↓
┌────────────────────────────────────────────────────────────┐
│              DATA & INFRASTRUCTURE LAYER                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Pose Detection (MediaPipe)                         │  │
│  │  • 33-point skeleton extraction from frames         │  │
│  │  • Landmark visibility & confidence scoring         │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ML Models (data/models/)                           │  │
│  │  • K-Means: Identify up/down states                 │  │
│  │  • Random Forest: Form quality classification       │  │
│  │  • 9 models total: 3 exercises × 3 age groups       │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Utility Functions (pose_utils.py)                  │  │
│  │  • 3D angle calculations with confidence            │  │
│  │  • Landmark visibility filtering                    │  │
│  │  • Vector math operations                           │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Data Storage                                       │  │
│  │  • Training CSVs: data/training/                    │  │
│  │  • Session logs: session_logs/                      │  │
│  │  • User calibration: user_calibration/              │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### 5.2.1.2 Module Decomposition and Relationships

**Module Hierarchy:**

```
AI_Gym_FYP (Root)
│
├── api_backend.py [FastAPI Server]
│   └── Depends on: exercises/*, exercise_classifier.py, pose_utils.py
│
├── main.py [Desktop Application]
│   └── Depends on: exercises/*, visual_utils.py, pose_utils.py, expert_coach.py
│
├── exercises/ [Exercise Logic Package]
│   ├── base_exercise.py [Abstract Base Class]
│   │   └── Defines: BaseExercise, ExerciseConfig, ExerciseResult, AGE_CONFIGS
│   ├── bicep_curl.py [Bicep Implementation]
│   ├── back_exercise.py [Back Implementation]
│   └── chest_exercise.py [Chest Implementation]
│
├── exercise_classifier.py [Auto-Detection]
│   └── Implements: ExerciseClassifier.detect_exercise_type()
│
├── pose_utils.py [Pose Utilities]
│   └── Provides: calculate_angle_3d(), PoseLandmark, AngleResult
│
├── video_pipeline.py [Training Data Extraction]
│   └── Extracts landmarks from videos → CSVs → data/training/
│
├── train_universal.py [ML Model Training]
│   └── Converts CSVs → Trained models → data/models/
│
├── visual_utils.py [Rendering Engine]
│   └── Provides: VisualFeedback, DisplayMetrics for visual feedback
│
├── expert_coach.py [Advanced Feedback]
│   └── Provides: ExpertCoach for personalized coaching tips
│
├── data/ [Data Directory]
│   ├── models/ [Trained ML Models] (.pkl files)
│   └── training/ [Training Datasets] (.csv files)
│
├── videos/ [Source Videos]
│   ├── bicep/
│   │   ├── children/good_form/
│   │   ├── adult/good_form/
│   │   └── senior/good_form/
│   ├── back/, chest/, etc.
│   └── [Used by video_pipeline.py to generate training data]
│
└── scripts/ [Automation Scripts]
    ├── train_all.ps1 [PowerShell training pipeline]
    └── train_all.py [Python training pipeline]
```

**Key Dependencies:**

```
Frontend (Web/Mobile/Desktop)
    ↓
api_backend.py / main.py
    ↓ (instantiates)
Exercise Classes (bicep_curl.py, back_exercise.py, etc.)
    ├→ base_exercise.py (inherits from, uses AGE_CONFIGS)
    ├→ pose_utils.py (imports calculate_angle_3d, AngleResult)
    └→ ML Models in data/models/ (loaded via config)
    
exercise_classifier.py
    ├→ pose_utils.py (uses landmark analysis)
    └→ All exercise classes (identifies which to use)

video_pipeline.py
    ├→ videos/ (reads source videos)
    └→ generates data/training/ .csv files
    
train_universal.py
    ├→ data/training/ .csv (reads training data)
    └→ generates data/models/ .pkl files
```

### 5.2.1.3 Architecture Style: Model-View-Controller (MVC) Adapted

**Adapted MVC Pattern for API-Centric System:**

```
┌──────────────────────────────────────────────────────────┐
│                    M O D E L LAYER                       │
│           (Business Logic & Data Management)             │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Exercise Models                                  │ │
│  │  • BaseExercise (ABC)                            │ │
│  │  • BicepCurlExercise, BackExercise, etc.        │ │
│  │  Responsibilities:                               │ │
│  │  - Calculate angles from landmarks              │ │
│  │  - Check form (joint positions, posture)        │ │
│  │  - Detect rep completion                        │ │
│  │  - Maintain rep counter & state machine         │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Auto-Classifier                                  │ │
│  │  • ExerciseClassifier                            │ │
│  │  Responsibilities:                               │ │
│  │  - Analyze pose landmarks                       │ │
│  │  - Determine exercise type                      │ │
│  │  - Return confidence score                      │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  ML Model Layer                                   │ │
│  │  • K-Means clusters (up/down state detection)    │ │
│  │  • Random Forest (form quality classification)   │ │
│  │  • Stored: data/models/{exercise}_{age}.pkl      │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
         ↑                              ↑
         │                              │ Model instances
         │                              │ created per request
         │                              │
┌──────────┴──────────────────────────┴─────────────────┐
│               C O N T R O L L E R                      │
│           (Request Handling & Routing)                │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  FastAPI Application (api_backend.py)           │ │
│  │  Endpoints:                                      │ │
│  │  • POST /api/process-frame                     │ │
│  │  • POST /api/start-session                     │ │
│  │  • POST /api/end-session                       │ │
│  │  • GET  /api/models/status                     │ │
│  │  • GET  /ready                                 │ │
│  │                                                 │ │
│  │  Responsibilities:                              │ │
│  │  - Validate incoming requests                  │ │
│  │  - Decode base64 frame data                    │ │
│  │  - Route to appropriate exercise model         │ │
│  │  - Coordinate multi-step processing            │ │
│  │  - Format responses as JSON                    │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
         ↑                              ↓
         │ JSON Response              JSON Request
         │ (counter, feedback, etc)    (frame, exercise, age)
         │                              │
┌────────┴──────────────────────────────┴─────────────┐
│               V I E W LAYER                         │
│        (Presentation & User Feedback)               │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │  Web Dashboard                               │  │
│  │  • Displays rep counter                      │  │
│  │  • Shows live feedback text                  │  │
│  │  • Charts progress over sessions             │  │
│  └──────────────────────────────────────────────┘  │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │  Desktop Application (main.py)               │  │
│  │  • Real-time visual overlay on webcam        │  │
│  │  • Joint skeleton drawing (MediaPipe)        │  │
│  │  • Color-coded feedback (Red/Yellow/Green)   │  │
│  │  • Rep counter  + angle displays             │  │
│  └──────────────────────────────────────────────┘  │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │  Mobile App (Future)                         │  │
│  │  • Simplified UI for on-the-go tracking      │  │
│  │  • Push notifications for form issues        │  │
│  └──────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

**Why MVC for an API-Centric System?**

- **Separation of Concerns:** Each layer has single responsibility
- **Testability:** Models can be unit-tested independently from controller logic
- **Reusability:** Models used by both FastAPI and desktop application
- **Maintainability:** Changes to UI don't affect business logic
- **Scalability:** Easy to add new view (web dashboard, mobile) without changing models

---

## 5.4 Design Models

### 5.4.1 Class Diagram (UML 2.5)

**Exercise Class Hierarchy:**

```
┌─────────────────────────────────────────────────────────────┐
│ <<abstract>> BaseExercise                                   │
│─────────────────────────────────────────────────────────────│
│ Attributes:                                                 │
│  - config: ExerciseConfig                                   │
│  - rep_counter: int                                         │
│  - current_stage: str                                       │
│  - angle_history: deque                                     │
│  - state_machine: StateMachine                              │
│  - ml_model: sklearn.base.BaseEstimator                    │
│─────────────────────────────────────────────────────────────│
│ Methods:                                                    │
│  + __init__(config: ExerciseConfig)                         │
│  + process_frame(frame: np.ndarray, landmarks): dict        │
│  + calculate_angles(landmarks) -> dict {abstract}           │
│  + check_form(landmarks, angles) -> tuple {abstract}        │
│  + detect_rep(angle: float) -> bool {abstract}              │
│  + get_feedback() -> str                                    │
│  + load_model(path: str): bool                              │
│  + reset_session()                                          │
│  + get_valid_reps() -> int                                  │
└─────────────────────────────────────────────────────────────┘
         △                   △                   △
         │                   │                   │
    ┌────┴─────┐      ┌──────┴──────┐      ┌────┴──────────┐
    │           │      │             │      │               │
┌───┴──────┐  ┌┴──────┴──┐  ┌──────┴─┐  ┌┴────────────┐
│BicepCurl │  │Back Row  │  │ Chest  │  │ [Future]   │
│Exercise  │  │Exercise  │  │Exercise│  │ LegExercise│
├──────────┤  ├──────────┤  ├────────┤  └────────────┘
│calculate │  │calculate │  │calc...  │
│_angles() │  │_angles()│  │_angles() │
│          │  │          │  │          │
│check_form│  │check_form│  │check_form│
│()        │  │()        │  │()        │
│          │  │          │  │          │
│detect_rep│  │detect_rep│  │detect_rep│
│()        │  │()        │  │()        │
└──────────┘  └──────────┘  └─────────┘
```

**Configuration and Data Classes:**

```
┌──────────────────────────────────────────┐        ┌──────────────────────┐
│ ExerciseConfig                           │        │ ExerciseResult       │
├──────────────────────────────────────────┤        ├──────────────────────┤
│ name: str                                │        │ counter: int         │
│ age_group: str ('children'|'adult'|...)  │        │ feedback: str        │
│ min_angle: float                         │        │ primary_angle: float │
│ max_angle: float                         │        │ secondary_angle: flt │
│ min_rep_time: float                      │        │ stage: str (up/down) │
│ max_rep_time: float                      │        │ confidence: float    │
│ flexibility_tolerance: float             │        │ is_valid_rep: bool   │
│ speed_tolerance: float                   │        │ rep_quality: float   │
│ form_strictness: float                   │        │ exercise_type: str   │
│ model_path: str                          │        │ age_group: str       │
└──────────────────────────────────────────┘        │ timestamp: datetime  │
         ▲                                           └──────────────────────┘
         │
         │ contains multiple
         │
┌──────────────────────────────────────────┐
│ AGE_CONFIGS: Dict                        │
├──────────────────────────────────────────┤
│ Key: 'children' | 'adult' | 'senior'   │
│ Value: Dict[str, ExerciseConfig]        │
│   Subkeys: 'bicep', 'back', 'chest'    │
└──────────────────────────────────────────┘
```

**ML and Utility Classes:**

```
┌─────────────────────────────────────────┐  ┌──────────────────────────┐
│ PoseLandmark                           │  │ AngleResult              │
├─────────────────────────────────────────┤  ├──────────────────────────┤
│ x: float                                │  │ angle: float             │
│ y: float                                │  │ is_valid: bool           │
│ z: float                                │  │ confidence: float        │
│ visibility: float (0.0-1.0)             │  │ additional_info: Dict    │
│ presence: float                         │  └──────────────────────────┘
│ name: str (landmark name)               │
├─────────────────────────────────────────┤
│ + to_array() -> np.ndarray              │
│ + to_array_2d() -> np.ndarray           │
└─────────────────────────────────────────┘


┌─────────────────────────────────────────┐
│ ExerciseClassifier                      │
├─────────────────────────────────────────┤
│ - model: Any                            │
│ - scaler: StandardScaler                │
├─────────────────────────────────────────┤
│ + detect_exercise_type(landmarks,       │
│                       confidence=0.6)   │
│        -> (exercise: str,               │
│            confidence_score: float)     │
└─────────────────────────────────────────┘
```

### 5.4.2 Sequence Diagram: Real-Time Frame Processing

```
User/Client        FastAPI Backend      Exercise Class       ML Model    Database
     │                   │                      │                │            │
     │  1. POST /api/    │                      │                │            │
     │  process-frame    │                      │                │            │
     │  (frame, exercise)│                      │                │            │
     │──────────────────>│                      │                │            │
     │                   │                      │                │            │
     │                   │  2. Validate Request │                │            │
     │                   │  Decode base64      │                │            │
     │                   │──────┬──────         │                │            │
     │                   │      │ (5ms)        │                │            │
     │                   │<─────┘              │                │            │
     │                   │                      │                │            │
     │                   │  3. Get Exercise    │                │            │
     │                   │  from Session Cache │                │            │
     │                   │──────┬──────        │                │            │
     │                   │      │ (2ms)        │                │            │
     │                   │<─────┘              │                │            │
     │                   │                      │                │            │
     │                   │  4. process_frame() │                │            │
     │                   │ (raw frame)         │                │            │
     │                   │─────────────────────>│                │            │
     │                   │                      │                │            │
     │                   │                      │  5. MediaPipe │            │
     │                   │                      │  Pose detect │            │
     │                   │                      │  (30ms)       │            │
     │                   │                      │──────┬────────┘            │
     │                   │                      │      │                     │
     │                   │                      │      [Extract 33 landmarks]│
     │                   │                      │<─────┘                     │
     │                   │                      │                            │
     │                   │                      │  6. calculate_angles()     │
     │                   │                      │ (10ms)                    │
     │                   │                      │──────┐                    │
     │                   │                      │      │ (3D math)          │
     │                   │                      │<─────┘                    │
     │                   │                      │                            │
     │                   │                      │  7. check_form()           │
     │                   │                      │ (5ms)                     │
     │                   │                      │──────┬──────              │
     │                   │                      │      │ (visibility check)  │
     │                   │                      │<─────┘                    │
     │                   │                      │                            │
     │                   │                      │  8. ML predict()           │
     │                   │                      │ (form quality)             │
     │                   │                      │─────────────────────────>│
     │                   │                      │                [Load model &│
     │                   │                      │                featurize]   │
     │                   │                      │<─────────────────────────│
     │                   │                      │ (15ms)                    │
     │                   │                      │                            │
     │                   │                      │  9. detect_rep()           │
     │                   │                      │ (state machine)            │
     │                   │                      │ (8ms)                     │
     │                   │                      │──────┬──────             │
     │                   │                      │      │ (update counter) │
     │                   │                      │<─────┘                  │
     │                   │                      │                          │
     │                   │  10. Return Result  │                          │
     │                   │<─────────────────────│                          │
     │                   │ (ExerciseResult)    │                          │
     │                   │  counter: 5         │                          │
     │                   │  feedback: "✓"      │                          │
     │                   │  angles: {..}       │                          │
     │                   │                      │                          │
     │                   │  11. Format JSON    │                          │
     │                   │  Response           │                          │
     │                   │ (3ms)               │                          │
     │                   │──────┬──────        │                          │
     │                   │      │              │                          │
     │                   │<─────┘              │                          │
     │                   │                      │                          │
     │  12. JSON Response│                     │                          │
     │  (success, counter│                     │                          │
     │   feedback, ...)  │                     │                          │
     │<──────────────────│                     │                          │
     │ (Total ~80ms)    │                     │                          │
     │                   │                      │                          │

Total Latency Budget: 100ms (exceeding causes poor UX)
- MediaPipe: 30ms (largest component)
- ML Inference: 15ms
- Angle Calculation: 10ms
- Other: 25ms (validation, serialization, network round-trip)
```

### 5.4.3 State Transition Diagram: Rep Detection State Machine

```
                          angle < min_angle
                          ────────────────>
                                     ┌──────────┐
                                     │   DOWN   │<──────────┐
                                     │  State   │           │
                                     └──────────┘           │
                                            △               │
                                            │               │ angle > max_angle
                                            │               │
                        ┌───────────────────┴───────────────┴─────────────────┐
                        │                                                       │
                        │  [Rep Conditions Met]        [Rep Conditions Failed] │
                        │  • angle transitioned        • timing invalid        │
                        │  • min_rep_time passed       • form was bad         │
                        │                              • user moved too fast   │
                        │                                                       │
                        ↓                                                       ↓
                    ┌────────────────────────┐                      ┌──────────────────┐
                    │   VALID REP COUNTED    │                      │   REP REJECTED   │
                    │   counter += 1         │                      │   (Feedback:     │
                    │   (Feedback: "✓")      │                      │   "⚠ BAD FORM")  │
                    │   rep_quality = 95%    │                      │   rep_quality=0% │
                    └────────────┬───────────┘                      └────────┬─────────┘
                                 │                                           │
                      angle < max_angle (descending)   angle < max_angle (ascending)
                                 │                                           │
                                 ↓                                           ↓
                       ┌──────────────────┐
                       │      UP STATE    │
                       │  Ascending phase │
                       └────────┬─────────┘
                                │
                                │ angle continues to rise
                                │
                    When angle enters [min_angle, max_angle]:
                                │
                                ├─> If form_score > threshold
                                │   └─> Transition to DOWN (wait for descent)
                                │
                                └─> If form_score < threshold
                                    └─> Return to resting phase
                                    └─> Reset counter


                    ┌─────────────────────────────────────┐
                    │   STATE TIMING CONSTRAINTS          │
                    │                                     │
                    │ Rep Duration: min_rep_time to       │
                    │              max_rep_time           │
                    │                                     │
                    │ Children:  1.5s - 5.0s              │
                    │ Adult:     1.0s - 4.0s              │
                    │ Senior:    1.5s - 5.0s              │
                    │                                     │
                    │ If outside range: REP REJECTED      │
                    └─────────────────────────────────────┘
```

**Key States:**

| State | Condition | Next State | Action |
|-------|-----------|-----------|--------|
| **RESTING** | angle < min_angle | Waiting | No action |
| **WAITING** | min_angle < angle < max_angle | UP | Start rep timing |
| **UP** | angle increasing & in range | DOWN | Monitor form |
| **DOWN** | angle decreasing & in range | Valid/Invalid Rep | Evaluate rep quality |
| **VALID REP** | All conditions met | UP | Increment counter, provide positive feedback |
| **INVALID REP** | Timing/form failed | RESTING | Reject rep, provide corrective feedback |

### 5.4.4 Activity Diagram: Exercise Training Pipeline

```
Start
  │
  ├─> [Collect Videos]
  │   └─ User places videos in:
  │      videos/{exercise}/{age}/good_form/
  │
  ├─> [Run video_pipeline.py]
  │   │
  │   ├─ For each video file:
  │   │  │
  │   │  ├─ Open video with OpenCV
  │   │  │
  │   │  ├─ Read frames at 30 FPS
  │   │  │
  │   │  ├─ [For each frame]
  │   │  │  │
  │   │  │  ├─ Run MediaPipe Pose detection
  │   │  │  │
  │   │  │  ├─ Extract 33 landmarks
  │   │  │  │
  │   │  │  ├─ Check visibility > 0.5?
  │   │  │  │  ├─ Yes: Keep landmark
  │   │  │  │  └─ No: Discard frame
  │   │  │  │
  │   │  │  └─ Store landmarks in memory
  │   │  │
  │   │  └─ Append all landmarks to training CSV
  │   │
  │   └─ Output: data/training/{exercise}_{age}.csv
  │
  ├─> [Run train_universal.py]
  │   │
  │   ├─ Load {exercise}_{age}.csv
  │   │
  │   ├─ [Data Preprocessing]
  │   │  ├─ Remove rows with NaN values
  │   │  ├─ Normalize landmark coordinates (0-1 scale)
  │   │  ├─ Calculate angles for primary/secondary joints
  │   │  └─ Extract features (33 landmarks × 4 = 132 features)
  │   │
  │   ├─ [Split Data]
  │   │  ├─ Training set: 80%
  │   │  └─ Test set: 20%
  │   │
  │   ├─ [Train K-Means Clustering]
  │   │  ├─ Cluster on primary angle
  │   │  ├─ Identify 2 clusters: (up, down)
  │   │  └─ Learn centroids for up/down states
  │   │
  │   ├─ [Train Random Forest Classifier]
  │   │  ├─ Input: landmark features
  │   │  ├─ Output: form quality (binary: good/bad)
  │   │  ├─ Train on labeled good-form data
  │   │  └─ Evaluate on test set
  │   │
  │   ├─ [Evaluate Models]
  │   │  ├─ Compute accuracy, precision, recall
  │   │  ├─ If accuracy < 80%: Log warning
  │   │  └─ Print metrics to console
  │   │
  │   └─ [Serialize Models]
  │      ├─ Pickle K-Means model
  │      ├─ Pickle Random Forest model
  │      └─ Save to: data/models/{exercise}_{age}.pkl
  │
  ├─> [Ready for Deployment]
  │   └─ Models loaded automatically at API startup
  │
  └─> End

  [Decision Points]:
  • Video availability: If no videos, training fails
  • Landmark visibility: Frames with visibility < 0.5 rejected
  • Model performance: Models with <80% accuracy raise warnings
  • Deployment: API checks model existence at startup
```

**Data Flow for Training:**

```
videos/bicep/children/good_form/
├── video1.mp4
├── video2.mp4
└── video3.mp4
             │
             ↓ [video_pipeline.py]
             │ Extract landmarks from each frame
             ↓
data/training/bicep_children.csv
     (rows = frames, cols = 33 landmarks × 4 values)
             │
             ↓ [train_universal.py]
             │ Preprocess → Train K-Means
             │          → Train Random Forest
             ↓
data/models/bicep_children.pkl
     (Contains: K-Means model, Random Forest, scaler)
             │
             ↓ [api_backend.py at startup]
             │ Load model from disk
             ↓
Exercise instance ready for inference
```

### 5.4.5 Data Flow Diagram (DFD) - Level 0 (System Context)

```
                          ┌─────────────────────┐
                          │    End User         │
                          │ (Athlete/Coach)     │
                          └──────────┬──────────┘
                                     │
                   ┌─────────────────┼─────────────────┐
                   │ Webcam Video    │ Feedback (Text) │
                   │ (30 FPS @       │ + Visual Cues   │
                   │  640x480)       │ + Rep Counter   │
                   ↓                 ↑                 │
          ┌────────────────────────────────────┐      │
          │      AI GYM System                 │      │
          │  (Exercise Detection +             │──────┘
          │   Rep Counting +                   │
          │   Form Feedback)                   │
          └────────┬────────────────────┬──────┘
                   │                    │
                   │ Session Data       │ Optional:
                   │ (Rep count,        │ Workout logs
                   ↓  Quality scores)   ↓
         ┌──────────────────────┐  ┌──────────────┐
         │  Cloud Backend       │  │  File System │
         │ (Optional storage)   │  │ (Session logs│
         │                      │   │ & calibration)
         └──────────────────────┘  └──────────────┘
```

### 5.4.6 Data Flow Diagram (DFD) - Level 1 (Detailed Processes)

```
┌───────────────────────────────────────────────────────────────────┐
│                     AI GYM DFD - LEVEL 1                          │
└───────────────────────────────────────────────────────────────────┘

Real-time Processing:

  Webcam Stream        Exercise      Landmark     Form Check
  (RGB 640x480)       Database      Database      Database
       │                  │              │              │
       │                  │              │              │
       ↓                  ↓              ↓              ↓
    ┌──────────┐    ┌──────────┐   ┌──────────┐  ┌──────────┐
    │ D1: Raw  │    │ D2:      │   │ D3:      │  │ D4:      │
    │ Frames   │    │ Exercise │   │ Pose     │  │ Form     │
    │ Store    │    │ Configs  │   │ Reference│  │ Specs    │
    └──────────┘    └──────────┘   └──────────┘  └──────────┘
                           │              │              │
                           ↓              ↓              ↓
    ┌────────────────────────────────────────────────────────┐
    │ P1: Decode Frame                                       │
    │ Input: RGB frame (base64 from API)                     │
    │ Output: cv2.Mat frame (640x480)                        │
    └────────────────────┬─────────────────────────────────┘
                         ↓
    ┌────────────────────────────────────────────────────────┐
    │ P2: Pose Detection (MediaPipe)                         │
    │ Input: Frame                                           │
    │ Process: Run MediaPipe Pose model                      │
    │ Output: 33 landmarks + visibility scores               │
    └────────────────────┬─────────────────────────────────┘
                         ↓
    ┌────────────────────────────────────────────────────────┐
    │ P3: Exercise Classification                            │
    │ Input: 33 landmarks                                    │
    │ Process: Analyze movement patterns                     │
    │          Compare against exercise signatures           │
    │ Output: Exercise type (bicep/back/chest) + confidence  │
    └────────────────────┬─────────────────────────────────┘
                         ↓
    ┌────────────────────────────────────────────────────────┐
    │ P4: Angle Calculation                                  │
    │ Input: Landmarks (3D points)                           │
    │ Process: Calculate primary/secondary angles            │
    │          Verify landmark visibility > 0.5              │
    │ Output: Angle values + confidence scores               │
    └────────────────────┬─────────────────────────────────┘
                         ↓
    ┌────────────────────────────────────────────────────────┐
    │ P5: Form Assessment                                    │
    │ Input: Angles, landmarks, exercise config              │
    │ Process: Compare against form templates               │
    │          Check joint positions, posture                │
    │ Output: Form quality (0-100)                           │
    └────────────────────┬─────────────────────────────────┘
                         ↓
    ┌────────────────────────────────────────────────────────┐
    │ P6: ML Model Inference                                 │
    │ Input: Pose features, angles                           │
    │ Process: Load trained model from data/models/          │
    │          Run K-Means + Random Forest predictions       │
    │ Output: State prediction (up/down) + quality score     │
    └────────────────────┬─────────────────────────────────┘
                         ↓
    ┌────────────────────────────────────────────────────────┐
    │ P7: Rep Detection                                      │
    │ Input: Current angle, state, history                   │
    │ Process: Execute state machine                         │
    │          Check timing constraints (min/max rep time)   │
    │ Output: Rep counted? (bool), feedback text            │
    └────────────────────┬─────────────────────────────────┘
                         ↓
    ┌────────────────────────────────────────────────────────┐
    │ P8: Feedback Generation                                │
    │ Input: Rep status, form quality, age group             │
    │ Process: Select feedback message                       │
    │          Format response JSON                          │
    │ Output: ExerciseResult object                          │
    └────────────────────┬─────────────────────────────────┘
                         ↓
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │ D5: Session      │  │ D6: Feedback     │  │ D7: API          │
    │ State Store      │  │ History          │  │ Response Format  │
    │ (counter, stage) │  │ (for debugging)   │  │ (JSON)           │
    └──────────────────┘  └──────────────────┘  └──────────────────┘
         ↑                      │                      ↑
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                         User / Frontend
                         (Web/Mobile/Desktop)
```

**Data Stores (D1-D7):**

| ID | Name | Type | Contents |
|----|------|------|----------|
| D1 | Raw Frames Buffer | In-Memory Deque | Last 30 frames (for temporal smoothing) |
| D2 | Exercise Configs | File (Python dict) | AGE_CONFIGS in base_exercise.py  |
| D3 | Pose Reference | File + Memory | MediaPipe landmark indices (0-32) |
| D4 | Form Specifications | File (JSON/YAML) | Form angle ranges, posture rules |
| D5 | Session State | In-Memory Dictionary | Rep counter, current angle, stage |
| D6 | Feedback History | File (session logs) | All feedback messages (for progress tracking) |
| D7 | API Response Template | In-Memory | ExerciseResponse schema |

**Processes (P1-P8):** See descriptions above

---

## 5.5 Data Design

### 5.5.1 Data Transformation and Storage Architecture

**Information Domain → Data Structures:**

```
Raw Physical Movement (Camera Input)
    ↓
        Pixel coordinates (640x480 RGB)
        ↓
        [MediaPipe Pose]
        ↓
        33 Body Landmarks with visibility scores
        │ - (x, y, z) normalized coordinates (0-1 range)
        │ - visibility: confidence of detection (0-1)
        │ - presence: likelihood of limb being visible (0-1)
        ↓
        PoseLandmark objects (pose_utils.py)
        ├─ Store in OrderedDict for frame
        └─ Keep history (last 30 frames) for temporal smoothing
        ↓
        [Angle Calculation]
        ↓
        AngleResult objects
        │ - angle: float (degrees)
        │ - is_valid: bool (visibility check passed)
        │ - confidence: float (based on landmark visibility)
        ├─ primary_angle (main movement axis)
        └─ secondary_angle (stability verification)
        ↓
        [ML Feature Extraction]
        ↓
        Feature vector for model:
        │ - 33 landmarks × 3 coords = 99 features
        │ - 33 landmarks × visibility = 33 features
        │ - 2 angle values = 2 features
        │ - Total: 134 features
        └→ Stored in CSV for training (data/training/)
        ↓
        [Processing]
        ↓
        ExerciseResult object
        │ - counter: int (rep count)
        │ - feedback: str (form assessment)
        │ - angles: dict
        │ - quality_score: float (0-100)
        │ - stage: str ('up'/'down')
        └→ Serialized to JSON for API response
        ↓
        [Persistence]
        ↓
        Session logs (JSON files)
        └─ data/training/ CSV files for model retraining
```

### 5.5.2 Data Representation: Entity-Relationship Diagram (ERD)

**Logical Data Model:**

```
┌──────────────────────────────┐
│       User (Entity)          │
├──────────────────────────────┤
│ PK: user_id (UUID)           │
│ user_age: int                │
│ age_group: str ('children'.. │
│ preferred_exercises: List    │
└──────────────────────────────┘
           │
           │ (1:N) Conducts
           ↓
┌──────────────────────────────┐
│    WorkoutSession (Entity)   │
├──────────────────────────────┤
│ PK: session_id (UUID)        │
│ FK: user_id                  │
│ start_time: timestamp        │
│ end_time: timestamp          │
│ exercise_type: str           │
│ total_reps: int              │
│ valid_reps: int              │
│ avg_form_quality: float      │
└──────────────────────────────┘
           │
           │ (1:N) Contains
           ↓
┌──────────────────────────────────────┐
│      ExerciseFrame (Entity)          │
├──────────────────────────────────────┤
│ PK: frame_id (int, auto-increment)   │
│ FK: session_id                       │
│ timestamp: float (seconds in session) │
│ primary_angle: float                 │
│ secondary_angle: float               │
│ form_quality: float (0-100)          │
│ rep_stage: str ('up'/'down'/'rest')  │
│ feedback: str                        │
│ is_valid_rep: bool                   │
└──────────────────────────────────────┘
           │
           │ References (N:1)
           ↓
┌──────────────────────────────┐
│     ExerciseTemplate         │
├──────────────────────────────┤
│ PK: exercise_type ('bicep'..) │
│ exercise_name: str           │
│ primary_joints: List         │
│ angle_range: [min, max]      │
└──────────────────────────────┘
           
┌──────────────────────────────┐
│     AgeConfiguration         │
├──────────────────────────────┤
│ PK: age_group ('children'..) │
│ min_rep_time: float          │
│ max_rep_time: float          │
│ angle_tolerance: float       │
│ form_strictness: float       │
└──────────────────────────────┘
```

**ER Relationships:**

- **User → WorkoutSession:** One-to-Many (one user can conduct many sessions)
- **WorkoutSession → ExerciseFrame:** One-to-Many (one session contains many frames)
- **ExerciseFrame → ExerciseTemplate:** Many-to-One (many frames for one exercise type)
- **WorkoutSession → AgeConfiguration:** Many-to-One (uses age configuration based on user age)

### 5.5.3 Data Storage Implementation

**File-Based Storage (Current Implementation):**

```
data/
├── models/
│   ├── bicep_children.pkl       [Trained ML model: K-Means + RF]
│   ├── bicep_adult.pkl          [Trained ML model]
│   ├── bicep_senior.pkl         [Trained ML model]
│   ├── back_children.pkl        [Status: Missing ❌]
│   ├── back_adult.pkl           [Status: Missing ❌]
│   ├── back_senior.pkl          [Status: Missing ❌]
│   ├── chest_children.pkl       [Status: Missing ❌]
│   ├── chest_adult.pkl          [Status: Missing ❌]
│   └── chest_senior.pkl         [Status: Missing ❌]
│
└── training/
    ├── bicep_adult_good_form_20260207_171944.csv
    │   Columns: landmark_0_x, landmark_0_y, landmark_0_z, 
    │           landmark_0_visibility, ..., landmark_32_visibility
    │   Rows: ~5000 (one per processed video frame)
    │
    ├── back_all_ages_YYYYMMDD.csv [Generated when videos processed]
    └── chest_all_ages_YYYYMMDD.csv [Generated when videos processed]

session_logs/
├── workout_20251218_143215.json
│   └─ Structure:
│      {
│        "session_id": "uuid-...",
│        "user_id": "user-1",
│        "exercise": "bicep",
│        "age_group": "adult",
│        "start_time": "2025-12-18T14:32:15Z",
│        "end_time": "2025-12-18T14:35:42Z",
│        "total_reps": 12,
│        "valid_reps": 10,
│        "frames": [
│          {
│            "frame_num": 0,
│            "timestamp": 0.0,
│            "primary_angle": 145.2,
│            "form_quality": 95.0,
│            "feedback": "✓ GOOD FORM"
│          },
│          ...
│        ]
│      }
└── workout_20251218_150320.json
    └─ ...

user_calibration/
└── user_default_20260207_165503.json
    └─ Structure:
       {
         "user_id": "user-1",
         "age": 25,
         "age_group": "adult",
         "height_cm": 180,
         "limb_proportions": {..},
         "personal_ranges": {
           "bicep": {"min_angle": 48, "max_angle": 172},
           ...
         }
       }
```

**Data Format Specifications:**

**CSV Format (Training Data):**
```csv
landmark_0_x,landmark_0_y,landmark_0_z,landmark_0_visibility,landmark_1_x,...
0.45,0.32,0.12,0.98,0.46,0.33,0.11,0.97,...
0.44,0.33,0.11,0.97,0.47,0.32,0.12,0.96,...
...
```

**JSON Format (Session Logs):**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user-default",
  "exercise_type": "bicep",
  "age_group": "adult",
  "start_time": "2026-02-07T17:19:44.123456Z",
  "end_time": "2026-02-07T17:22:18.987654Z",
  "total_frames": 5400,
  "total_reps": 12,
  "valid_reps": 10,
  "average_quality": 87.5,
  "frames": [
    {
      "frame_num": 0,
      "timestamp_ms": 0,
      "primary_angle": 145.2,
      "secondary_angle": 15.3,
      "form_quality": 95.0,
      "stage": "up",
      "feedback": "✓ GOOD FORM",
      "rep_count": 0
    }
  ]
}
```

**Pickle Format (ML Models):**
```python
# File: data/models/bicep_adult.pkl
# Contains:
{
  'kmeans_model': sklearn.cluster.KMeans,  # Trained on primary angles
  'random_forest_model': sklearn.ensemble.RandomForestClassifier,  # Form quality
  'scaler': sklearn.preprocessing.StandardScaler,  # Feature normalization
  'metadata': {
    'exercise': 'bicep',
    'age_group': 'adult',
    'training_samples': 4132,
    'model_accuracy': 0.87,
    'feature_names': [list of 134 feature names]
  }
}
```

---

## 5.5.1 Data Dictionary

**Alphabetical listing of all system entities and attributes:**

### Core Data Entities

| Entity | Type | Description |
|--------|------|-------------|
| **AGE_CONFIGS** | Dict[str, Dict[str, ExerciseConfig]] | Master configuration for all age groups and exercises. Keys: 'children', 'adult', 'senior'. Values: exercise-specific configs. |
| **age_group** | str | User's age classification: 'children' (10-16), 'adult' (17-45), 'senior' (46+). Determines form standards and feedback tone. |
| **AngleResult** | Dataclass | Structured result of angle calculation containing: angle (float), is_valid (bool), confidence (float), additional_info (dict). |

### Exercise Configuration Attributes

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| **config.name** | str | Human-readable exercise name | "bicep_curl" |
| **config.age_group** | str | Age group this config applies to | "adult" |
| **config.min_angle** | float | Minimum valid angle for exercise (degrees) | 50 |
| **config.max_angle** | float | Maximum valid angle for exercise (degrees) | 170 |
| **config.min_rep_time** | float | Minimum duration for valid rep (seconds) | 1.0 |
| **config.max_rep_time** | float | Maximum duration for valid rep (seconds) | 4.0 |
| **config.flexibility_tolerance** | float | Extra degrees allowed for age group | 10.0 |
| **config.speed_tolerance** | float | Extra seconds allowed for age group | 0.5 |
| **config.form_strictness** | float | Multiplier for form checking (0.7 = lenient, 1.0 = standard) | 1.0 |
| **config.model_path** | str | File path to trained ML model | "data/models/bicep_adult.pkl" |
| **config.strict_mode_threshold** | int | Form quality threshold (0-100) | 25 |

### Exercise Result Attributes

| Attribute | Type | Description | Range |
|-----------|------|-------------|-------|
| **result.counter** | int | Total repetitions counted in session | 0-∞ |
| **result.feedback** | str | Human-readable form assessment | "✓ GOOD FORM", "⚠ ELBOW POSITION", "❌ BACK TOO ROUNDED" |
| **result.primary_angle** | float | Main movement angle (degrees) | Depends on exercise |
| **result.secondary_angle** | float | Stability/posture angle (degrees) | Depends on exercise |
| **result.stage** | str | Current rep stage | "up", "down", "rest" |
| **result.confidence** | float | Confidence in angle measurement (0-1) | 0.0 - 1.0 |
| **result.is_valid_rep** | bool | Whether last rep was counted as valid | true/false |
| **result.rep_quality** | float | Form quality score of last rep (0-100) | 0 - 100 |
| **result.exercise_type** | str | Detected/specified exercise | "bicep", "back", "chest" |
| **result.age_group** | str | Age group used for standards | "children", "adult", "senior" |
| **result.timestamp** | datetime | Frame processing timestamp | ISO 8601 format |

### Pose Landmark Attributes

| Attribute | Type | Description | Range |
|-----------|------|-------------|-------|
| **landmark.x** | float | Normalized X coordinate (left-right) | 0.0 - 1.0 |
| **landmark.y** | float | Normalized Y coordinate (top-bottom) | 0.0 - 1.0 |
| **landmark.z** | float | Normalized Z coordinate (depth) | -1.0 - 1.0 |
| **landmark.visibility** | float | Detection confidence (MediaPipe) | 0.0 - 1.0 |
| **landmark.presence** | float | Likelihood of body part being visible | 0.0 - 1.0 |
| **landmark.name** | str | Human-readable landmark name | "left_shoulder", "right_elbow", etc. |

### MediaPipe Body Landmarks (33 total)

| Index | Name | Body Part | Usage |
|-------|------|-----------|-------|
| 0 | NOSE | Head | Posture check |
| 11 | LEFT_SHOULDER | Upper body | Arm exercises, posture |
| 12 | RIGHT_SHOULDER | Upper body | Arm exercises, posture |
| 13 | LEFT_ELBOW | Arm | Primary angle (bicep/back) |
| 14 | RIGHT_ELBOW | Arm | Primary angle (bicep/back) |
| 15 | LEFT_WRIST | Arm | Form validation |
| 16 | RIGHT_WRIST | Arm | Form validation |
| 23 | LEFT_HIP | Torso | Stability check |
| 24 | RIGHT_HIP | Torso | Stability check |
| [Others: knees, ankles, etc.] | ... | ... | Posture validation |

### API Request/Response Attributes

**ExerciseRequest (Input):**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| exercise_type | str | No | "bicep", "back", "chest" (auto-detected if omitted) |
| age | int | Yes | User age (8-100) |
| frame_data | str | Yes | Base64-encoded image frame (max 5MB) |
| auto_detect | bool | No | Enable exercise auto-detection (default: true) |
| session_id | str | No | UUID for session continuity |

**ExerciseResponse (Output):**
| Field | Type | Description |
|-------|------|-------------|
| success | bool | Request processed successfully |
| counter | int | Total reps counted this session |
| feedback | str | Form feedback text |
| stage | str | Current rep phase ("up"/"down"/"rest") |
| primary_angle | float | Main exercise angle (degrees) |
| secondary_angle | float | Secondary angle for stability |
| rep_quality | float | Quality of current rep (0-100) |
| confidence | float | Confidence in angle measurement (0-1) |
| detected_exercise | str | Exercise type (auto-detected or user-specified) |
| detection_confidence | float | Confidence in exercise detection (0-1) |
| is_valid_rep | bool | Was the last rep counted as valid? |
| model_missing | bool | True if required ML model not found |
| degraded | bool | True if running in fallback mode |
| session_id | str | Session UUID for state tracking |
| api_version | str | API version number |
| latency_ms | float | Processing time in milliseconds |
| timestamp | str | Frame processing timestamp (ISO 8601) |

### Training Data Attributes

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| landmark_i_x | float | MediaPipe | X coordinate of landmark i |
| landmark_i_y | float | MediaPipe | Y coordinate of landmark i |
| landmark_i_z | float | MediaPipe | Z (depth) coordinate of landmark i |
| landmark_i_visibility | float | MediaPipe | Detection confidence for landmark i |

(Repeated for i = 0 to 32, totaling 132 columns per row)

### Session Log Attributes

| Field | Type | Description |
|-------|------|-------------|
| session_id | str | Unique session identifier (UUID v4) |
| user_id | str | User identifier |
| exercise_type | str | Exercise performed |
| age_group | str | Age classification used |
| start_time | str | Session start (ISO 8601) |
| end_time | str | Session end (ISO 8601) |
| total_frames | int | Frames processed in session |
| total_reps | int | Total repetitions counted |
| valid_reps | int | Valid repetitions (good form) |
| average_quality | float | Mean form quality (0-100) |
| frames | List[dict] | Array of frame-level data |

### Calibration Data Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| user_id | str | User identifier |
| age | int | User age |
| age_group | str | Calculated age group |
| height_cm | int | User height (for normalization) |
| limb_proportions | dict | Ratio of limb lengths |
| personal_ranges | dict | User-specific angle ranges |

---

## 5.6 User Interface Design

### 5.6.1 System Perspective and User Interaction

**User Interaction Models:**

The AI Gym system provides interfaces for two types of users:

1. **End Users (Athletes/Coaches)**
   - Interact through desktop application or web dashboard
   - See real-time rep counter and form feedback
   - Receive visual cues (color overlays on skeleton)
   - Get corrective tips (e.g., "Lower your elbow")

2. **System Administrators/ML Engineers**
   - Use CLI tools for data collection and model training
   - Monitor API endpoints for system health
   - Test models before deployment

### 5.6.2 Interface Specification: Desktop Application (main.py)

**Window Layout and Components:**

```
┌─────────────────────────────────────────────────────────┐
│                    AI GYM TRAINER                       │
│              Real-time Exercise Coach                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────┐  ┌──────────────┐ │
│  │                                │  │  Rep Counter │ │
│  │  Live Webcam Feed              │  │              │ │
│  │  + Skeleton Overlay            │  │  Reps: 5     │ │
│  │  (33 joint points)             │  │  Quality: 95%│ │
│  │  + Joint Circle Colors:        │  │  Stage: UP   │ │
│  │    - Green: Good form          │  │              │ │
│  │    - Yellow: Minor issues      │  │ Angles:      │ │
│  │    - Red: Bad form             │  │ Primary: 145°│ │
│  │                                │  │ Secondary: 15│ │
│  │  + Angle Arc Visualization     │  │              │ │
│  │  (shows current vs target)     │  │ Confidence:  │ │
│  │                                │  │ 98%          │ │
│  ├────────────────────────────────┤  └──────────────┘ │
│  │  Feedback Ticker               │                   │
│  │  > "✓ GOOD FORM"               │  [START] [STOP]   │
│  │  > "Lower your elbows"         │                   │
│  │  > "⚠ ELBOW TOO HIGH"          │ [CALIBRATE]       │
│  │                                │                   │
│  └────────────────────────────────┘                    │
│                                                          │
├─────────────────────────────────────────────────────────┤
│  Settings:  Exercise: Bicep  │  Age: 25  │  FPS: 29.8  │
│  Session ID: hash-123-abc-def ...                      │
└─────────────────────────────────────────────────────────┘
```

**Screen Objects and Interactions:**

| Component | Type | Purpose | User Action |
|-----------|------|---------|-------------|
| **Live Feed** | Video Display | Shows user movement with skeleton overlay | User moves in front of camera |
| **Skeleton Drawing** | Visual Cue | 33 joints connected as skeleton | Real-time update (30 FPS) |
| **Joint Circles** | Color Coded Feedback | Green = good, Yellow = warning, Red = bad form | Visual form assessment |
| **Angle Arc** | Angle Visualization | Shows current angle vs target range | User monitors primary movement angle |
| **Rep Counter** | Text Label | Displays current rep count | Auto-increments on valid reps |
| **Quality Score** | Progress Bar | Visual representation of form quality (0-100%) | Updates each frame |
| **Feedback Ticker** | Text Area | Scrolling feedback messages | Lists recent form corrections |
| **Stage Indicator** | Text Label | Shows current rep phase ("UP", "DOWN", "REST") | Updates every frame |
| **Angle Display** | Numeric Labels | Shows primary and secondary angles in degrees | Updates 30x per second |
| **Confidence Gauge** | Progress Bar | Shows detection confidence (0-100%) | Visual quality indicator |
| **START Button** | Interactive Button | Begins rep counting for new session | Click to initialize |
| **STOP Button** | Interactive Button | Ends session and saves logs | Click to finalize |
| **CALIBRATE Button** | Interactive Button | Starts user calibration process | Click for personalization |
| **Settings Panel** | Configuration Area | Shows exercise type, age, FPS | Read-only display |

### 5.6.3 Interface Specification: REST API (api_backend.py)

**Endpoint Hierarchy:**

```
GET /                              [Root - Returns API info]
GET /docs                          [Swagger UI auto-documentation]
GET /ready                         [Readiness check]

POST /api/process-frame            [Main inference endpoint]
POST /api/start-session            [Initialize new session]
POST /api/end-session              [Finalize session]
GET  /api/models/status            [Check available models]

GET  /api/health                   [Health check]
GET  /api/version                  [API version]
```

**Primary Endpoint: POST /api/process-frame**

**Request Format:**
```json
{
  "exercise_type": "bicep",        // Optional: auto-detect if omitted
  "age": 25,                        // Required: user age
  "frame_data": "iVBORw0KG...",    // Required: base64 encoded RGB image
  "auto_detect": true,              // Optional: enable auto-detection
  "session_id": "550e8400-..."      // Optional: for session continuity
}
```

**Response Format:**
```json
{
  "success": true,
  "counter": 5,
  "feedback": "✓ GOOD FORM",
  "stage": "up",
  "primary_angle": 145.2,
  "secondary_angle": 15.3,
  "rep_quality": 95.0,
  "confidence": 0.98,
  "detected_exercise": "bicep",
  "detection_confidence": 0.92,
  "is_valid_rep": true,
  "model_missing": false,
  "degraded": false,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "api_version": "2.1.0",
  "latency_ms": 87.34,
  "timestamp": "2026-02-08T14:32:15.123456Z"
}
```

**Other Endpoints:**

**POST /api/start-session**
```json
Request:
{
  "user_id": "user-1",
  "age": 25,
  "exercise_type": "bicep"
}

Response:
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "started",
  "timestamp": "2026-02-08T14:30:00Z"
}
```

**POST /api/end-session**
```json
Request:
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}

Response:
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_reps": 12,
  "valid_reps": 10,
  "average_quality": 87.5,
  "duration_seconds": 178,
  "log_saved": true,
  "log_path": "session_logs/workout_20260208_143215.json"
}
```

**GET /api/models/status**
```json
Response:
{
  "system_ready": false,
  "total_models": 9,
  "trained_models": 1,
  "missing_models": 8,
  "models": {
    "bicep_adult": {
      "available": true,
      "path": "data/models/bicep_adult.pkl",
      "accuracy": 0.87,
      "training_samples": 4132
    },
    "bicep_children": {
      "available": false,
      "reason": "File not found"
    },
    ...
  }
}
```

### 5.6.4 Screen Images and Visual Feedback Examples

**Desktop App - Good Form State:**

```
┌─────────────────────────────────────────────────┐
│ AI GYM - Good Form Example (Bicep Curl)         │
├─────────────────────────────────────────────────┤
│                                    Rep: 3        │
│  ⭕ Head (visible, good)           Quality: 98%  │
│   │                                Angle: 145°  │
│  ⭕─⭕ Shoulders (GREEN)           Confidence:99%│
│   └─┘                                           │
│   │ ⭕─⭕  Elbows (GREEN)          [START][STOP] │
│   │ │ │ ⭕─⭕ Wrists (GREEN)                     │
│   │ │ │                                         │
│   │ ⭕─⭕  Ribs/Core (GREEN)     "✓ GREAT FORM" │
│   │   │                         "Keep elbows"  │
│   │  ⭕─⭕  Hips (GREEN)         "tucked tight"│
│   │  │   │                                     │
│   └─⭕─⭕  Knees (GRAY)                         │
│        ⭕─⭕  Ankles (GRAY)                    │
│                                                │
│ ◀─────[50°]─[145°] SUCCESS ─────▶            │
│        Min        Current        Max         │
└─────────────────────────────────────────────────┘
```

**Desktop App - Bad Form State:**

```
┌─────────────────────────────────────────────────┐
│ AI GYM - Form Issue Detected (Elbow Position)  │
├─────────────────────────────────────────────────┤
│                                    Rep: 3        │
│  ⭕ Head (visible, good)           Quality: 32%  │
│   │                                Angle: 95°   │
│  ⭕─⭕ Shoulders (YELLOW)          Confidence:92%│
│   └─┘                                           │
│   │ ⭕─⚠️⭕ Elbows (RED!!!)         [START][STOP] │
│   │ │ │ ⭕─⭕ Wrists (GREEN)                     │
│   │ │ │                                         │
│   │ ⭕─⭕  Ribs/Core (YELLOW)    "❌ BAD FORM"   │
│   │   │                         "⚠ Elbows too │
│   │  ⭕─⭕  Hips (YELLOW)        high! Keep at |
│   │  │   │                      your sides!"  │
│   └─⭕─⭕  Knees (GRAY)                         │
│        ⭕─⭕  Ankles (GRAY)                    │
│                                                │
│ ◀─────[50°]──[95°]─ TOO LOW ─────▶            │
│        Min    Current        Max              │
└─────────────────────────────────────────────────┘
```

**Color Coding Convention:**

| Color | Signal | Meaning | Action |
|-------|--------|---------|--------|
| 🟢 **Green** | ✓ Good | All parameters within range | Continue current movement |
| 🟡 **Yellow** | ⚠ Warning | Minor deviation from ideal form | Adjust before rep completes |
| 🔴 **Red** | ❌ Critical | Form significantly incorrect | Stop, reset, try again |
| ⚪ **Gray** | Info | Body part not involved in exercise | Normal, no action needed |

### 5.6.5 Form Feedback Messages

**Dynamic Feedback Generation:**

Feedback is generated based on current form state and age group:

**Bicep Curl - Good Form (All Age Groups):**
- "✓ PERFECT FORM"
- "✓ NICE CONTROL"
- "✓ STEADY MOVEMENT"

**Bicep Curl - Form Issues:**

For Elbows:
- "⚠ Elbows slightly high" (95-105°) → "Bring elbows closer to sides"
- "❌ Elbows too high" (>105°) → "Keep elbows against your ribs!"
- "❌ Elbows flaring out" → "Keep elbows tucked in!"

For Back/Torso:
- "⚠ Using momentum" → "Control the weight, slower rep"
- "❌ Swinging back" → "NO swinging! Engage core!"

For Wrist:
- "⚠ Wrist bent" → "Keep wrist straight"

**Age-Specific Feedback Tone:**

- **Children:** Encouraging, shorter messages ("Great job!", "Nice rep!")
- **Adult:** Direct, professional ("Good form", "Adjust elbows")
- **Senior:** Careful, health-focused ("Excellent control", "Protect your joints")

---

## Summary

This Chapter 5 Software Design Specification document provides:

✅ **Design Methodology Justification** - OOP patterns and Agile process model explained

✅ **System Overview** - Three-tier architecture with clear layer separation

✅ **Architectural Design** - MVC adapted for API-centric system with module decomposition

✅ **Design Models:**
  - Class Diagram (Exercise hierarchy)
  - Sequence Diagram (Frame processing pipeline)
  - State Transition Diagram (Rep detection state machine)
  - Activity Diagram (Training pipeline)
  - Data Flow Diagrams (multi-level DFD)

✅ **Data Design** - Transformation from physical movement to data structures, ERD, storage specification

✅ **Data Dictionary** - Comprehensive listing of all entities, attributes, types, and ranges

✅ **User Interface Design:**
  - Desktop Application layout and components
  - REST API endpoint specifications
  - Visual feedback mockups and color conventions
  - Form feedback message library

**Current Implementation Status:**
- Framework & Architecture: **100% Complete** ✅
- Training Pipeline: **Ready** ✅
- ML Models: **1/9 Trained** (bicep_adult) - 8 remaining
- Deployment: **Pending** (awaiting model completion)

---

**Document Version:** 1.0  
**Last Updated:** February 8, 2026  
**Author:** AI Gym Development Team
