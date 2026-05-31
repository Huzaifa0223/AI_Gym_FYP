"""
FastAPI Backend for AI Gym - Production Ready
Provides REST API endpoints for exercise detection and tracking
Supports multiple exercises and age groups
"""
import asyncio
import itertools
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, status
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import AsyncIterator, Optional, List, Dict, Literal
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
from core.exercise_mapping import BODY_PART_TO_EXERCISE

# Video-based scoring pipeline
import config as app_config
from core.cnn_lstm_scorer import (
    CnnLstmFormScorer,
    CnnLstmModelNotAvailableError,
    NullFormQualityScorer,
)
from core.equipment_pipeline import (
    EquipmentStateHolder,
    run_equipment_detection_loop,
)
from core.form_scorer import FormScore, FormScorer
from core.frame_buffer import LatestFrameBuffer
from core.live_session import LiveSession
from core.object_detector import (
    EquipmentDetector,
    StubDetector,
    Yolov8NanoDetector,
)
from core.pipeline import ScoringPipeline
from core.rep_counter import RepEvent, make_rep_counter
from core.rule_engine import FormViolation
from core.schemas import HeadlineScore
from core.threshold_provider import (
    MalformedThresholdsError,
    ThresholdProvider,
    ThresholdsNotFoundError,
)
from core.video_processor import (
    NoFramesExtractedError,
    UnsupportedExerciseError,
    VideoDecodeError,
    VideoProcessor,
    extract_features,
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
EXERCISES = ["bicep_curl", "bent_over_row", "push_up"]
AGE_GROUPS = ["children", "adult", "senior"]

# ===========================================
# PYDANTIC MODELS
# ===========================================
class ExerciseRequest(BaseModel):
    exercise_type: Optional[str] = Field(None, description="Exercise key: bicep_curl, bent_over_row, push_up (auto-detect from body-part classifier if not provided)")
    age: int = Field(..., description="User's age in years", ge=8, le=100)
    frame_data: str = Field(..., description="Base64 encoded image frame")
    auto_detect: bool = Field(True, description="Automatically detect exercise type")
    session_id: Optional[str] = Field(None, description="Optional session ID for state continuity")
    
    @field_validator('frame_data')
    @classmethod
    def validate_frame_size(cls, v: str) -> str:
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

class LiveFrameResponse(BaseModel):
    """Response for the live per-frame 3-pillar path (POST /api/live-frame).

    Field names mirror what the React CameraWorkout already reads, so repointing
    the frontend is a URL change. Cadence: ``counter`` / ``primary_angle`` /
    ``landmarks`` update every frame; ``rep_quality`` / ``feedback`` update only
    on rep close and hold last-known between reps. ``rep_quality`` is ``null``
    until the first rep so the UI shows a neutral "—", never a misleading 0%.
    """
    success: bool = Field(..., description="False on no-pose / pre-detection frames")
    counter: int = Field(..., description="Running rep count (StreamingRepSegmenter)")
    primary_angle: float = Field(..., description="This frame's primary joint angle (deg)")
    rep_quality: Optional[float] = Field(
        None,
        description="Fused 0-100 form score; null until the first rep closes, then held",
    )
    feedback: str = Field(..., description="Coaching text; neutral pre-first-rep, then held")
    landmarks: Optional[List[Dict[str, float]]] = Field(
        None, description="MediaPipe pose landmarks (33 points) or null on no-pose"
    )
    session_id: str
    api_version: str = Field(default="2.1.0")
    latency_ms: Optional[float] = None
    timestamp: str

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
    headline: Optional[HeadlineScore] = None


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
# LIFESPAN — startup / shutdown wiring (Stage 5c)
# ===========================================
#
# Initialises long-lived 3-pillar pipeline state on app.state:
#   * frame_buffer       — LatestFrameBuffer; populated by /api/process-frame,
#                          consumed by the equipment background task.
#   * frame_counter      — itertools.count() yielding monotonic frame ids.
#   * equipment_holder   — EquipmentStateHolder; written by the loop, read by
#                          GET /api/equipment.
#   * equipment_detector — EquipmentDetector wrapping Yolov8NanoDetector when
#                          weights are present, or StubDetector(()) as a safe
#                          fallback so the endpoint never 500s.
#   * equipment_task     — asyncio task running run_equipment_detection_loop
#                          at 1 Hz.
#   * cnn_lstm_scorer    — NullFormQualityScorer (returns None until a trained
#                          CNN+LSTM model lands; that work is post-5c).
#   * threshold_provider — Stage 5b ThresholdProvider with the production
#                          exercises/ directory.
#
# Shutdown sets the stop event and awaits the task with a timeout so the
# TestClient context manager exits promptly during pytest runs.

YOLO_WEIGHTS_PATH = Path("models/yolov8n.pt")
CNN_LSTM_WEIGHTS_PATH = Path("data/models/cnn_lstm_form_scorer.pt")
EQUIPMENT_TASK_SHUTDOWN_TIMEOUT_S = 2.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise app.state on startup; tear it down on shutdown."""
    # -- Buffers + monotonic frame counter ----------------------------------
    app.state.frame_buffer = LatestFrameBuffer()
    app.state.frame_counter = itertools.count()

    # -- Live per-frame 3-pillar sessions (StreamingRepSegmenter + FormScorer +
    # ScoreAggregator), one LiveSession per session_id, used by /api/live-frame.
    # Separate from ExerciseManager.sessions (the legacy /api/process-frame path)
    # so both paths stay reachable.
    app.state.live_sessions = {}

    # -- Equipment detector (real if weights present, stub otherwise) -------
    if YOLO_WEIGHTS_PATH.exists():
        app.state.equipment_detector = EquipmentDetector(
            Yolov8NanoDetector(weights_path=YOLO_WEIGHTS_PATH),
        )
    else:
        logger.warning(
            "YOLOv8 weights not found at %s — running with StubDetector "
            "(no detections). Run `python -m scripts.fetch_yolo_weights`.",
            YOLO_WEIGHTS_PATH,
        )
        app.state.equipment_detector = EquipmentDetector(StubDetector())

    # -- Equipment background task ------------------------------------------
    app.state.equipment_holder = EquipmentStateHolder()
    app.state.equipment_stop = asyncio.Event()

    def _frame_source():
        return app.state.frame_buffer.get()

    app.state.equipment_task = asyncio.create_task(
        run_equipment_detection_loop(
            app.state.equipment_holder,
            app.state.equipment_detector,
            _frame_source,
            app.state.equipment_stop,
            target_hz=1.0,
        )
    )

    # -- Pipeline-side singletons -------------------------------------------
    # CNN+LSTM form scorer: OFF by default. It is trained on synthetic temporal
    # sequences and has a *measured sim-to-real gap* (see docs/ML_FACTS.md) — it
    # recognises only a minority of real good-form reps, so fusing it into the
    # headline score (weight 0.3) would degrade real-video results. Set
    # AI_GYM_ENABLE_CNN_LSTM=1 to enable the experimental scorer (e.g. for a
    # synthetic-data demo). On any load failure (missing weights, torch DLL
    # issues) it falls back to the null scorer so the API never 500s.
    app.state.cnn_lstm_scorer = NullFormQualityScorer()
    if os.environ.get("AI_GYM_ENABLE_CNN_LSTM") and CNN_LSTM_WEIGHTS_PATH.exists():
        try:
            app.state.cnn_lstm_scorer = CnnLstmFormScorer(CNN_LSTM_WEIGHTS_PATH)
            logger.info("CNN+LSTM form scorer ENABLED (experimental, synthetic-trained)")
        except CnnLstmModelNotAvailableError as exc:
            logger.warning(
                "AI_GYM_ENABLE_CNN_LSTM set but scorer failed to load (%s) — "
                "using NullFormQualityScorer", exc,
            )
    app.state.threshold_provider = ThresholdProvider()

    try:
        yield
    finally:
        # Shutdown: signal the equipment loop and wait for clean exit.
        app.state.equipment_stop.set()
        try:
            await asyncio.wait_for(
                app.state.equipment_task,
                timeout=EQUIPMENT_TASK_SHUTDOWN_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning("Equipment task did not stop within %.1fs; cancelling",
                           EQUIPMENT_TASK_SHUTDOWN_TIMEOUT_S)
            app.state.equipment_task.cancel()


# ===========================================
# FASTAPI APP
# ===========================================
app = FastAPI(
    title="AI Gym - Exercise Detection API",
    description="Real-time exercise form detection and rep counting",
    version="2.0.0",
    lifespan=lifespan,
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
        """Get or create exercise instance for a session.

        ``exercise_type`` must be one of the long-form keys in :data:`EXERCISES`
        (``bicep_curl``, ``bent_over_row``, ``push_up``). Body-part classifier
        outputs are translated to long-form keys via
        :data:`BODY_PART_TO_EXERCISE` *before* this method is called — see
        :func:`process_frame` for the routing logic.
        """
        if session_id not in self.sessions:
            logger.info(f"Creating new session: {session_id} (exercise={exercise_type}, age={age})")
            config = get_exercise_config(exercise_type, age)

            if exercise_type == 'bicep_curl':
                exercise = BicepCurlExercise(config)
            elif exercise_type == 'bent_over_row':
                exercise = BackExercise(config)
            elif exercise_type == 'push_up':
                exercise = ChestExercise(config)
            else:
                raise ValueError(
                    f"Unknown exercise type: {exercise_type!r}. "
                    f"Must be one of {EXERCISES}."
                )

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
        "supported_exercises": EXERCISES,
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

        # Tee the decoded frame into the equipment task's input buffer so
        # the background YOLOv8 loop has fresh frames to detect on. Cheap
        # reference write; defensive `hasattr` guard handles edge cases
        # where lifespan setup was bypassed (e.g. mid-reload).
        if hasattr(app.state, "frame_buffer"):
            try:
                _frame_id = next(app.state.frame_counter)
                app.state.frame_buffer.set(
                    frame, _frame_id, int(time.time() * 1000),
                )
            except Exception:  # pylint: disable=broad-except
                logger.exception("frame_buffer.set raised; continuing")
        
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
        
        # Auto-detect body part, then route through BODY_PART_TO_EXERCISE.
        # The classifier emits body-part categories (arms / back / chest / ...);
        # specific-exercise selection happens here via the mapping. Body parts
        # with no mapped exercise (legs / shoulders / unknown) fall through to
        # the user-provided value or the default.
        exercise_type = request.exercise_type
        detection_confidence = 1.0
        detected_body_part = "unknown"

        if request.auto_detect or not exercise_type:
            detected_body_part, detection_confidence = (
                exercise_manager.classifier.detect_exercise_type(
                    results.pose_landmarks.landmark
                )
            )
            if detection_confidence < 0.6:
                logger.warning(
                    f"Low detection confidence: {detection_confidence:.2f} "
                    f"for body part {detected_body_part!r}"
                )
            mapped = (
                BODY_PART_TO_EXERCISE.get(detected_body_part)
                if detection_confidence > 0.6 else None
            )
            exercise_type = mapped or exercise_type or 'bent_over_row'

        if exercise_type not in EXERCISES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported exercise type {exercise_type!r}. "
                    f"Must be one of {EXERCISES}."
                ),
            )
        
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


def _age_to_group(age: int) -> str:
    """Map a numeric age to the model age-group key.

    Mirrors the boundaries used by ``/api/process-frame`` (children < 16,
    adult < 60, senior otherwise) so the live path loads the same RF model.
    """
    if age < 16:
        return 'children'
    if age < 60:
        return 'adult'
    return 'senior'


def _get_live_session(session_id: str, exercise_type: str, age_group: str) -> LiveSession:
    """Get or create the :class:`LiveSession` for *session_id*.

    Created lazily on the first live frame (so ``/api/start-session`` stays
    unchanged). If the same session_id is reused for a *different* exercise
    (the user switched exercises), the session is reset so the count restarts
    with the correct config.
    """
    store = app.state.live_sessions
    session = store.get(session_id)
    if session is None or session.exercise != exercise_type:
        session = LiveSession(
            exercise_type, age_group,
            cnn_lstm_scorer=app.state.cnn_lstm_scorer,
        )
        store[session_id] = session
    return session


@app.post("/api/live-frame", response_model=LiveFrameResponse)
async def live_frame(request: ExerciseRequest):
    """Score a single live-camera frame with the 3-pillar streaming pipeline.

    Unlike legacy ``/api/process-frame`` (per-exercise heuristics + RF on every
    frame), this drives a session-scoped :class:`~core.live_session.LiveSession`:
    StreamingRepSegmenter counts reps, and FormScorer + ScoreAggregator run only
    when a rep closes. ``exercise_type`` is required (no auto-detect on the live
    path). Per-frame work stays within the existing process-frame budget; form
    scoring is amortised to rep close.
    """
    start_time = time.time()

    try:
        session_id = request.session_id or str(uuid.uuid4())

        exercise_type = request.exercise_type
        if not exercise_type or exercise_type not in EXERCISES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"/api/live-frame requires exercise_type in {EXERCISES} "
                    f"(got {exercise_type!r})"
                ),
            )

        # Decode base64 image
        image_data = base64.b64decode(request.frame_data)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        height, width = frame.shape[:2]
        if height > MAX_IMAGE_DIMENSION or width > MAX_IMAGE_DIMENSION:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions {width}x{height} exceed maximum {MAX_IMAGE_DIMENSION}px",
            )

        # Tee the frame into the equipment task's buffer (same as legacy path).
        if hasattr(app.state, "frame_buffer"):
            try:
                _frame_id = next(app.state.frame_counter)
                app.state.frame_buffer.set(frame, _frame_id, int(time.time() * 1000))
            except Exception:  # pylint: disable=broad-except
                logger.exception("frame_buffer.set raised; continuing")

        age_group = _age_to_group(request.age)
        session = _get_live_session(session_id, exercise_type, age_group)

        # Pose estimation (reuse the shared MediaPipe Pose instance).
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = exercise_manager.pose.process(image_rgb)

        if not results.pose_landmarks:
            latency_ms = (time.time() - start_time) * 1000
            return LiveFrameResponse(
                success=False,
                counter=session.rep_count,         # hold the count across no-pose frames
                primary_angle=0.0,
                rep_quality=session.last_form_score,   # held; None pre-first-rep
                feedback="No pose detected",
                landmarks=None,
                session_id=session_id,
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat(),
            )

        landmarks_data = [
            {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
            for lm in results.pose_landmarks.landmark
        ]

        features = extract_features(exercise_type, results.pose_landmarks.landmark)
        frame_result = session.process_features(features, time.monotonic())

        latency_ms = (time.time() - start_time) * 1000
        return LiveFrameResponse(
            success=True,
            counter=frame_result.counter,
            primary_angle=frame_result.primary_angle,
            rep_quality=frame_result.rep_quality,
            feedback=frame_result.feedback,
            landmarks=landmarks_data,
            session_id=session_id,
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Live-frame error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/api/start-session")
async def start_session(
    user_id: str = Form(...),
    exercise_type: str = Form(...),
    age: int = Form(...)
):
    """Start a new workout session"""
    session_id = f"{user_id}_{exercise_type}_{datetime.now().timestamp()}"
    
    # Validate inputs (long-form keys per Stage 5a naming reconciliation).
    if exercise_type not in EXERCISES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid exercise type {exercise_type!r}. Must be one of {EXERCISES}.",
        )
    
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
    # Free any live (3-pillar) session state for this id, regardless of whether a
    # legacy exercise session also exists. Additive — does not affect the summary.
    if hasattr(app.state, "live_sessions"):
        app.state.live_sessions.pop(session_id, None)

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
    exercise: str = Form(..., description="One of: bicep_curl, bent_over_row, push_up"),
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

    # -- 4. Run the pipeline (Stage 5c: extracted to core.pipeline) --------
    pipeline = ScoringPipeline(
        threshold_provider=app.state.threshold_provider,
        cnn_lstm_scorer=app.state.cnn_lstm_scorer,
    )

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            pipeline.score,
            video_bytes,
            exercise,
            age_group,
        )
    except ThresholdsNotFoundError as exc:
        # Schema-level: the requested exercise has no thresholds shipped.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MalformedThresholdsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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

    rep_results: List[RepResult] = [
        _to_rep_result(r.event, r.score) for r in result.per_rep
    ]

    logger.info(
        "Scored video: exercise=%s, age_group=%s, reps=%d, frames=%d/%d, "
        "time=%.2fs, headline.score=%d",
        exercise, age_group, len(rep_results),
        result.frames_with_landmarks, result.frames_processed,
        result.processing_time_s, result.headline.score,
    )

    return ScoreResponse(
        exercise=exercise,
        age_group=age_group,
        total_reps=len(rep_results),
        average_score=result.average_score,
        frames_processed=result.frames_processed,
        frames_with_landmarks=result.frames_with_landmarks,
        processing_time_s=result.processing_time_s,
        reps=rep_results,
        headline=result.headline,
    )

@app.get("/api/exercises")
async def get_exercises():
    """Get list of supported exercises"""
    return {
        "exercises": [
            {
                "id": "bicep_curl",
                "name": "Bicep Curl",
                "description": "Arm curl exercise for biceps",
                "muscles": ["biceps", "forearms"]
            },
            {
                "id": "bent_over_row",
                "name": "Bent-over Row",
                "description": "Rowing exercise for back muscles",
                "muscles": ["lats", "rhomboids", "traps"]
            },
            {
                "id": "push_up",
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

# ===========================================
# Stage 5c — reference skeleton + equipment endpoints
# ===========================================

REPO_ROOT = Path(__file__).resolve().parent
EXERCISES_DIR = REPO_ROOT / "exercises"


@app.get("/api/reference/{exercise_id}")
async def get_reference_skeleton(exercise_id: str):
    """Return the reference skeleton sequence for *exercise_id*.

    Reads ``exercises/<exercise_id>/reference_skeleton.json`` — a hand-
    authored (or placeholder) JSON containing one keyframe per non-IDLE
    FSM phase: DESCENDING / BOTTOM / ASCENDING / TOP. The frontend uses
    this for ghost-overlay rendering during live sessions.

    Returns 404 when the file is missing — Stage 5c only ships
    bicep_curl. Returns 500 only on disk read errors.
    """
    path = EXERCISES_DIR / exercise_id / "reference_skeleton.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No reference skeleton for exercise {exercise_id!r}",
        )
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.exception("Failed to read reference skeleton %s", path)
        raise HTTPException(
            status_code=500,
            detail=f"Malformed reference skeleton for {exercise_id!r}: {exc}",
        ) from exc


@app.get("/api/equipment")
async def get_equipment_state(request: Request):
    """Return the latest :class:`EquipmentState` snapshot.

    Reads ``app.state.equipment_holder``, which the background YOLOv8
    loop updates at 1 Hz. Before the first detection completes (or
    when no frames have been written to the buffer), the holder
    returns its sentinel state (``detections=()``,
    ``last_updated_frame_id=-1``, ``last_updated_timestamp_ms=-1``).
    """
    state = request.app.state.equipment_holder.get()
    return {
        "detections": [
            {
                "label": d.label,
                "confidence": d.confidence,
                "bbox_xyxy": list(d.bbox_xyxy),
                "frame_id_detected": d.frame_id_detected,
            }
            for d in state.detections
        ],
        "last_updated_frame_id": state.last_updated_frame_id,
        "last_updated_timestamp_ms": state.last_updated_timestamp_ms,
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
