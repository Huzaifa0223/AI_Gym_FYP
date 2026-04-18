"""
FastAPI Backend for AI Gym - Production Ready
Provides REST API endpoints for exercise detection and tracking
Supports multiple exercises and age groups
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, status
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Literal
import cv2
import math
import mediapipe as mp
import numpy as np
import base64
import io
from datetime import datetime
import json
import os
import uuid
import logging
from pathlib import Path
import time

# Import exercise classes
from exercises.base_exercise import get_exercise_config, ExerciseResult
from exercises.back_exercise import BackExercise
from exercises.chest_exercise import ChestExercise
from exercises.bicep_curl import BicepCurlExercise

# Import auto-detection
from core.exercise_classifier import ExerciseClassifier, quick_detect_exercise

# Video-based scoring pipeline
import config as app_config
from core.form_scorer import FormScore, FormScorer
from core.rep_counter import RepEvent, make_rep_counter
from core.rule_engine import FormViolation
from core.video_processor import (
    NoFramesExtractedError,
    UnsupportedExerciseError,
    VideoDecodeError,
    VideoProcessor,
)

# ===========================================
# LOGGING CONFIGURATION
# ===========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===========================================
# CONSTANTS
# ===========================================
MAX_FRAME_SIZE_MB = 5
MAX_IMAGE_DIMENSION = 1920
MAX_VIDEO_BYTES = 50 * 1024 * 1024  # 50 MB cap for /api/score uploads
MODELS_DIR = Path("data/models")
EXERCISES = ["bicep", "back", "chest"]
AGE_GROUPS = ["children", "adult", "senior"]

# ===========================================
# PYDANTIC MODELS
# ===========================================
class ExerciseRequest(BaseModel):
    exercise_type: Optional[str] = Field(None, description="Exercise type: bicep, back, chest, legs, shoulders, arms (auto-detect if not provided)")
    age: int = Field(..., description="User's age in years", ge=8, le=100)
    frame_data: str = Field(..., description="Base64 encoded image frame")
    auto_detect: bool = Field(True, description="Automatically detect exercise type")
    session_id: Optional[str] = Field(None, description="Optional session ID for state continuity")
    
    @validator('frame_data')
    def validate_frame_size(cls, v):
        # Rough size check (base64 is ~4/3 original size)
        size_mb = len(v) * 0.75 / (1024 * 1024)
        if size_mb > MAX_FRAME_SIZE_MB:
            raise ValueError(f"Frame size {size_mb:.2f}MB exceeds {MAX_FRAME_SIZE_MB}MB limit")
        return v
    
class ExerciseResponse(BaseModel):
    success: bool
    counter: int
    feedback: str
    stage: str
    primary_angle: float
    secondary_angle: float
    rep_quality: float
    confidence: float
    detected_exercise: str = Field(..., description="Auto-detected or specified exercise type")
    detection_confidence: float = Field(..., description="Confidence in exercise detection")
    is_valid_rep: bool
    model_missing: bool = Field(..., description="True when the expected ML model was not loaded")
    degraded: bool = Field(..., description="True when running in degraded mode (heuristics only)")
    session_id: str = Field(..., description="Session ID for state tracking")
    api_version: str = Field(default="2.1.0", description="API version")
    latency_ms: Optional[float] = Field(None, description="Processing latency in milliseconds")
    timestamp: str
    landmarks: Optional[List[Dict[str, float]]] = Field(
        None,
        description="MediaPipe pose landmarks — list of 33 points, each with x, y, z (normalized 0-1) and visibility"
    )

class UserProfile(BaseModel):
    user_id: str
    age: int = Field(..., ge=8, le=100)
    preferred_exercises: List[str] = []
    
class WorkoutSession(BaseModel):
    session_id: str
    user_id: str
    exercise_type: str
    age_group: str
    start_time: str
    total_reps: int = 0
    valid_reps: int = 0
    average_quality: float = 0.0

# ---- /api/score response schema ---------------------------------------------
# These models describe the flat JSON returned by the video-based scoring
# endpoint.  NaN values from FormScorer (unavailable ML / rules) are mapped to
# JSON ``null`` because the JSON spec does not define NaN.

class ViolationResult(BaseModel):
    name: str
    severity: Literal['minor', 'major']
    message: str
    penalty: float

class RepResult(BaseModel):
    rep_number: int
    start_ts: float
    end_ts: float
    duration_s: float
    peak_angle: float
    trough_angle: float
    ml_score: Optional[float] = None
    ml_confidence: Optional[float] = None
    rule_penalty: float
    final_score: Optional[float] = None
    violations: List[ViolationResult]

class ScoreResponse(BaseModel):
    exercise: str
    age_group: str
    total_reps: int
    average_score: Optional[float] = None
    frames_processed: int
    frames_with_landmarks: int
    processing_time_s: float
    reps: List[RepResult]


def _nan_to_none(value: float) -> Optional[float]:
    """Return ``None`` for NaN, else the value — JSON has no NaN literal."""
    return None if math.isnan(value) else value


def _to_violation_result(v: FormViolation) -> ViolationResult:
    return ViolationResult(
        name=v.name,
        severity=v.severity,
        message=v.message,
        penalty=v.penalty,
    )


def _to_rep_result(event: RepEvent, score: FormScore) -> RepResult:
    """Convert a (RepEvent, FormScore) pair into a JSON-safe RepResult."""
    return RepResult(
        rep_number=event.rep_number,
        start_ts=event.start_ts,
        end_ts=event.end_ts,
        duration_s=event.duration_s,
        peak_angle=event.peak_angle,
        trough_angle=event.trough_angle,
        ml_score=_nan_to_none(score.ml_score),
        ml_confidence=_nan_to_none(score.ml_confidence),
        rule_penalty=score.rule_penalty,
        final_score=_nan_to_none(score.final_score),
        violations=[_to_violation_result(v) for v in score.violations],
    )

# ===========================================
# FASTAPI APP
# ===========================================
app = FastAPI(
    title="AI Gym - Exercise Detection API",
    description="Real-time exercise form detection and rep counting",
    version="2.0.0"
)

# CORS configuration for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Key authentication ────────────────────────────────────────────────────
# Set the AI_GYM_API_KEY environment variable to enable auth.
# If the variable is not set the API runs open (local dev / demo mode).
_API_KEY = os.environ.get("AI_GYM_API_KEY", "")

@app.middleware("http")
async def _api_key_middleware(request: Request, call_next):
    if _API_KEY:  # only enforce when a key has been configured
        # Allow health/readiness endpoints without a key
        if request.url.path not in ("/ready", "/docs", "/openapi.json", "/redoc"):
            provided = request.headers.get("X-API-Key", "")
            if provided != _API_KEY:
                return Response(
                    content='{"detail":"Invalid or missing API key"}',
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    media_type="application/json"
                )
    return await call_next(request)

# ===========================================
# GLOBAL STATE
# ===========================================
class ExerciseManager:
    """Manages exercise instances for multiple users/sessions"""
    
    def __init__(self):
        self.sessions = {}  # session_id -> exercise instance
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.classifier = ExerciseClassifier()  # Auto-detection classifier
        self.model_availability = {}  # Cache model availability at startup
        self._check_model_availability()
    
    def _check_model_availability(self):
        """Check which models are available at startup"""
        logger.info("Checking model availability...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        for exercise in EXERCISES:
            for age_group in AGE_GROUPS:
                model_path = MODELS_DIR / f"{exercise}_{age_group}.pkl"
                is_available = model_path.exists()
                self.model_availability[f"{exercise}_{age_group}"] = is_available
                
                status = "✅" if is_available else "❌"
                logger.info(f"  {status} {model_path.name}")
        
        available_count = sum(self.model_availability.values())
        total_count = len(self.model_availability)
        logger.info(f"Model availability: {available_count}/{total_count} models loaded")
    
    def is_model_available(self, exercise_type: str, age_group: str) -> bool:
        """Check if a model is available without filesystem access"""
        return self.model_availability.get(f"{exercise_type}_{age_group}", False)
    
    def get_exercise_instance(self, session_id: str, exercise_type: str, age: int, user_id: Optional[str] = None):
        """Get or create exercise instance for a session"""
        if session_id not in self.sessions:
            logger.info(f"Creating new session: {session_id} (exercise={exercise_type}, age={age})")
            config = get_exercise_config(exercise_type, age)
            
            # Create appropriate exercise instance
            if exercise_type == 'bicep' or exercise_type == 'arms':
                # Use existing bicep class with config
                exercise = BicepCurlExercise(config)
            elif exercise_type == 'back':
                exercise = BackExercise(config)
            elif exercise_type == 'chest':
                exercise = ChestExercise(config)
            elif exercise_type == 'legs':
                exercise = BackExercise(config)  # Placeholder - reuse back logic
            elif exercise_type == 'shoulders':
                exercise = BackExercise(config)  # Placeholder - reuse back logic
            else:
                raise ValueError(f"Unknown exercise type: {exercise_type}")
            
            self.sessions[session_id] = {
                'exercise': exercise,
                'exercise_type': exercise_type,
                'start_time': datetime.now(),
                'frame_count': 0,
                'user_id': user_id,
                'age': age
            }
        
        return self.sessions[session_id]
    
    def remove_session(self, session_id: str):
        """Remove session to free memory"""
        if session_id in self.sessions:
            del self.sessions[session_id]

# Initialize manager
exercise_manager = ExerciseManager()

# ===========================================
# API ENDPOINTS
# ===========================================
@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "service": "AI Gym Exercise Detection API",
        "version": "2.0.0",
        "supported_exercises": ["bicep", "back", "chest"],
        "age_groups": ["children (8-15)", "adult (16-59)", "senior (60+)"]
    }

@app.post("/api/process-frame", response_model=ExerciseResponse)
async def process_frame(request: ExerciseRequest):
    """
    Process a single frame for exercise detection with auto-detection
    
    Args:
        request: ExerciseRequest containing optional exercise type, age, and frame data
        
    Returns:
        ExerciseResponse with rep count, feedback, form analysis, and detected exercise
    """
    start_time = time.time()
    
    try:
        # Generate or use provided session_id
        session_id = request.session_id or str(uuid.uuid4())
        
        # Decode base64 image
        image_data = base64.b64decode(request.frame_data)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Validate image dimensions
        height, width = frame.shape[:2]
        if height > MAX_IMAGE_DIMENSION or width > MAX_IMAGE_DIMENSION:
            raise HTTPException(
                status_code=400, 
                detail=f"Image dimensions {width}x{height} exceed maximum {MAX_IMAGE_DIMENSION}px"
            )
        
        # Process with MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = exercise_manager.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            latency_ms = (time.time() - start_time) * 1000
            logger.warning(f"No pose detected (session={session_id}, latency={latency_ms:.2f}ms)")
            return ExerciseResponse(
                success=False,
                counter=0,
                feedback="No pose detected",
                stage="unknown",
                primary_angle=0.0,
                secondary_angle=0.0,
                rep_quality=0.0,
                confidence=0.0,
                is_valid_rep=False,
                detected_exercise="unknown",
                detection_confidence=0.0,
                model_missing=False,
                degraded=False,
                session_id=session_id,
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat(),
                landmarks=None
            )

        # Extract all 33 MediaPipe pose landmarks to send to the frontend
        landmarks_data = [
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            }
            for lm in results.pose_landmarks.landmark
        ]
        
        # Auto-detect exercise if not provided or auto_detect is True
        exercise_type = request.exercise_type
        detection_confidence = 1.0
        
        if request.auto_detect or not exercise_type:
            detected_exercise, detection_confidence = exercise_manager.classifier.detect_exercise_type(
                results.pose_landmarks.landmark
            )
            
            # Warn if detection confidence is low
            if detection_confidence < 0.6:
                logger.warning(f"Low detection confidence: {detection_confidence:.2f} for {detected_exercise}")
            
            exercise_type = detected_exercise if detection_confidence > 0.6 else (exercise_type or 'back')
        
        # Determine age group for model checking
        if request.age < 16:
            age_group = 'children'
        elif request.age < 60:
            age_group = 'adult'
        else:
            age_group = 'senior'
        
        # Check model availability and set degraded flag
        model_available = exercise_manager.is_model_available(exercise_type, age_group)
        degraded = not model_available
        
        # Create or get session
        session = exercise_manager.get_exercise_instance(
            session_id, exercise_type, request.age, user_id=None
        )
        
        # Process frame with exercise detector
        exercise_result = session['exercise'].process_frame(
            results.pose_landmarks.landmark
        )
        session['frame_count'] += 1
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log request
        logger.info(
            f"Processed frame: session={session_id}, exercise={exercise_type}, "
            f"age_group={age_group}, counter={exercise_result.counter}, "
            f"model_missing={exercise_result.model_missing}, degraded={degraded}, "
            f"detection_conf={detection_confidence:.2f}, latency={latency_ms:.2f}ms"
        )
        
        return ExerciseResponse(
            success=True,
            counter=exercise_result.counter,
            feedback=exercise_result.feedback,
            stage=exercise_result.stage,
            primary_angle=exercise_result.primary_angle,
            secondary_angle=exercise_result.secondary_angle,
            rep_quality=exercise_result.rep_quality,
            confidence=exercise_result.confidence,
            is_valid_rep=exercise_result.is_valid_rep,
            detected_exercise=exercise_type,
            detection_confidence=detection_confidence,
            model_missing=exercise_result.model_missing,
            degraded=degraded,
            session_id=session_id,
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat(),
            landmarks=landmarks_data
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/api/start-session")
async def start_session(
    user_id: str = Form(...),
    exercise_type: str = Form(...),
    age: int = Form(...)
):
    """Start a new workout session"""
    session_id = f"{user_id}_{exercise_type}_{datetime.now().timestamp()}"
    
    # Validate inputs
    if exercise_type not in ['bicep', 'back', 'chest', 'legs', 'shoulders', 'arms']:
        raise HTTPException(status_code=400, detail="Invalid exercise type")
    
    if age < 8 or age > 100:
        raise HTTPException(status_code=400, detail="Age must be between 8 and 100")
    
    # Create session
    session = exercise_manager.get_exercise_instance(session_id, exercise_type, age)
    
    # Determine age group
    if age < 16:
        age_group = 'children'
    elif age < 60:
        age_group = 'adult'
    else:
        age_group = 'senior'
    
    return {
        "success": True,
        "session_id": session_id,
        "exercise_type": exercise_type,
        "age_group": age_group,
        "start_time": session['start_time'].isoformat()
    }

@app.post("/api/end-session")
async def end_session(session_id: str = Form(...)):
    """End a workout session and get summary"""
    if session_id not in exercise_manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = exercise_manager.sessions[session_id]
    exercise = session['exercise']
    
    summary = {
        "session_id": session_id,
        "start_time": session['start_time'].isoformat(),
        "end_time": datetime.now().isoformat(),
        "total_reps": exercise.counter,
        "frames_processed": session['frame_count'],
        "exercise_type": exercise.config.name,
        "age_group": exercise.config.age_group
    }
    
    # Clean up session
    exercise_manager.remove_session(session_id)
    
    return summary

@app.post("/api/score", response_model=ScoreResponse)
async def score_video(
    video: UploadFile = File(..., description="Workout video (mp4/mov/avi/webm)"),
    exercise: str = Form(..., description="One of: bicep, back, chest"),
    age_group: str = Form(..., description="One of: children, adult, senior"),
) -> ScoreResponse:
    """Score a full workout video, rep by rep.

    Runs the VideoProcessor → RepCounter → FormScorer pipeline end-to-end
    and returns per-rep scores, detected violations, and pipeline stats.
    """
    # -- 1. Validate enums up-front (fast fail, no upload read) ------------
    if exercise not in app_config.EXERCISES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid exercise {exercise!r}. Must be one of {app_config.EXERCISES}",
        )
    if age_group not in app_config.AGE_GROUPS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid age_group {age_group!r}. Must be one of {app_config.AGE_GROUPS}",
        )

    # -- 2. Validate content-type (reject non-video uploads early) ---------
    content_type = (video.content_type or "").lower()
    if content_type and not content_type.startswith("video/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type {video.content_type!r} — expected video/*",
        )

    # -- 3. Read body & enforce size cap -----------------------------------
    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Empty video upload")
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Video exceeds {MAX_VIDEO_BYTES // (1024 * 1024)} MB limit "
                f"({len(video_bytes) / (1024 * 1024):.1f} MB received)"
            ),
        )

    # -- 4. Run the pipeline ----------------------------------------------
    start_monotonic = time.monotonic()
    rep_results: List[RepResult] = []
    frames_processed = 0
    frames_with_landmarks = 0

    try:
        with VideoProcessor(exercise) as vp:
            counter = make_rep_counter(exercise, age_group)
            scorer = FormScorer(exercise, age_group)

            for pf in vp.process_bytes(video_bytes):
                frames_processed += 1
                if not pf.landmarks_detected:
                    continue
                frames_with_landmarks += 1
                event = counter.update_angle(
                    pf.primary_angle,
                    pf.timestamp,
                    features=pf.features,
                )
                if event is not None:
                    score = scorer.score(event)
                    rep_results.append(_to_rep_result(event, score))
    except (VideoDecodeError, NoFramesExtractedError) as exc:
        # Bad input, not a server fault — surface the reason.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnsupportedExerciseError as exc:
        # Defensive: enum check above should prevent this, but double-guard.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "Unhandled error in /api/score (exercise=%s, age_group=%s)",
            exercise, age_group,
        )
        raise HTTPException(status_code=500, detail="Internal processing error")

    processing_time_s = time.monotonic() - start_monotonic

    # -- 5. Aggregate --------------------------------------------------------
    valid_scores = [r.final_score for r in rep_results if r.final_score is not None]
    average_score = (sum(valid_scores) / len(valid_scores)) if valid_scores else None

    logger.info(
        "Scored video: exercise=%s, age_group=%s, reps=%d, frames=%d/%d, time=%.2fs",
        exercise, age_group, len(rep_results),
        frames_with_landmarks, frames_processed, processing_time_s,
    )

    return ScoreResponse(
        exercise=exercise,
        age_group=age_group,
        total_reps=len(rep_results),
        average_score=average_score,
        frames_processed=frames_processed,
        frames_with_landmarks=frames_with_landmarks,
        processing_time_s=processing_time_s,
        reps=rep_results,
    )

@app.get("/api/exercises")
async def get_exercises():
    """Get list of supported exercises"""
    return {
        "exercises": [
            {
                "id": "bicep",
                "name": "Bicep Curl",
                "description": "Arm curl exercise for biceps",
                "muscles": ["biceps", "forearms"]
            },
            {
                "id": "back",
                "name": "Back Row",
                "description": "Rowing exercise for back muscles",
                "muscles": ["lats", "rhomboids", "traps"]
            },
            {
                "id": "chest",
                "name": "Push-up",
                "description": "Bodyweight chest exercise",
                "muscles": ["chest", "triceps", "shoulders"]
            }
        ]
    }

@app.get("/api/age-groups")
async def get_age_groups():
    """Get information about age-specific configurations"""
    return {
        "age_groups": [
            {
                "name": "children",
                "age_range": "8-15 years",
                "description": "Modified exercises with more flexibility and slower pace",
                "adjustments": "Less strict form requirements, longer rep time allowed"
            },
            {
                "name": "adult",
                "age_range": "16-59 years",
                "description": "Standard exercise parameters",
                "adjustments": "Standard form requirements and timing"
            },
            {
                "name": "senior",
                "age_range": "60+ years",
                "description": "Gentler exercises with safety focus",
                "adjustments": "Reduced range of motion, slower pace, less strict form"
            }
        ]
    }

@app.get("/api/models/status")
async def get_models_status():
    """Get status of all ML models (which are loaded/missing)"""
    models = []
    for exercise in EXERCISES:
        for age_group in AGE_GROUPS:
            key = f"{exercise}_{age_group}"
            is_available = exercise_manager.model_availability.get(key, False)
            model_path = MODELS_DIR / f"{key}.pkl"
            
            models.append({
                "exercise": exercise,
                "age_group": age_group,
                "model_name": f"{key}.pkl",
                "available": is_available,
                "path": str(model_path)
            })
    
    available_count = sum(m["available"] for m in models)
    total_count = len(models)
    
    return {
        "summary": {
            "total_models": total_count,
            "available_models": available_count,
            "missing_models": total_count - available_count,
            "coverage_percent": round((available_count / total_count) * 100, 1)
        },
        "models": models,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Basic health check for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(exercise_manager.sessions),
        "mediapipe_loaded": exercise_manager.pose is not None
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check for k8s/load balancers"""
    available_models = sum(exercise_manager.model_availability.values())
    total_models = len(exercise_manager.model_availability)
    
    is_ready = (
        exercise_manager.pose is not None and 
        available_models > 0  # At least one model available
    )
    
    return {
        "ready": is_ready,
        "mediapipe_initialized": exercise_manager.pose is not None,
        "models_available": available_models,
        "models_total": total_models,
        "active_sessions": len(exercise_manager.sessions),
        "timestamp": datetime.now().isoformat()
    }

# ===========================================
# RUN SERVER
# ===========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
