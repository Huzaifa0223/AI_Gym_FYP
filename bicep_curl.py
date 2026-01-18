import pickle
import os
import numpy as np
import time
import logging
from collections import deque
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, asdict

# Try to import from utils, handle error if run standalone
try:
    from pose_utils import calculate_angle_3d, check_form_torso_relative  # type: ignore
except ImportError:
    # Fallback if running script directly for testing
    from dataclasses import dataclass
    from typing import Dict
    
    @dataclass
    class FallbackAngleResult:
        angle: float
        is_valid: bool
        confidence: float
        additional_info: Optional[Dict] = None
    
    def calculate_angle_3d(a, b, c): 
        return FallbackAngleResult(angle=0.0, is_valid=False, confidence=0.0, additional_info={"error": "pose_utils not available"})
    
    def check_form_torso_relative(a, b, c): 
        return FallbackAngleResult(angle=0.0, is_valid=False, confidence=0.0, additional_info={"error": "pose_utils not available"})

try:
    from train_advanced_model import AdvancedExerciseModel
except ImportError:
    AdvancedExerciseModel = None

# ===========================================
# DATA STRUCTURES
# ===========================================
@dataclass
class ExerciseResult:
    counter: int
    feedback: str
    color: Tuple[int, int, int]
    elbow_angle: float
    torso_angle: float
    stage: str
    confidence: float
    is_valid_rep: bool
    rep_quality: float  # 0-100 score

    # Helper to make this work with code expecting a Dictionary
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ===========================================
# MAIN EXERCISE CLASS
# ===========================================
class BicepCurlExercise:
    def __init__(self, model_path: str, strict_thresh: float = 35):
        self.strict_thresh = strict_thresh
        self.counter = 0
        self.stage = "down"
        self.feedback = "Ready"
        self.rep_history = []
        self.angle_history = deque(maxlen=5)
        self.rep_start_time = None
        self.rep_times = []
        
        # --- NEW MODEL LOADING LOGIC ---
        self.model = None
        try:
            # 1. Try Advanced Model first
            from train_advanced_model import AdvancedExerciseModel
            adv_model = AdvancedExerciseModel()
            if adv_model.load_model('advanced_bicep_model.pkl'):
                self.model = adv_model.best_model
                print("[INFO] Loaded ADVANCED Model")
        except Exception as e:
            print(f"[WARN] Advanced model skipped: {e}")

        # 2. Fallback to Basic Model if Advanced failed
        if self.model is None and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("[INFO] Loaded BASIC Model")
        # -------------------------------

        self.calibration_data = {'min_angle': 60, 'max_angle': 160}

    def _setup_logging(self):
        logger = logging.getLogger('BicepCurlExercise')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            self.logger.error(f"Model not found at {model_path}")
            # Return None to allow running without AI (fallback mode)
            return None 
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.logger.info("Model loaded successfully")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return None

    def validate_angle(self, angle: float) -> bool:
        return 20 <= angle <= 200

    def calculate_rep_quality(self, elbow_angle: float, torso_angle: float, rep_time: Optional[float] = None) -> float:
        score = 0
        # 1. Range of Motion (40%)
        min_a = self.calibration_data['min_angle']
        max_a = self.calibration_data['max_angle']
        
        if elbow_angle < min_a: rom_score = (elbow_angle / min_a) * 100
        elif elbow_angle > max_a: rom_score = 100 - ((elbow_angle - max_a) / (200 - max_a)) * 100
        else: rom_score = 100
        score += rom_score * 0.4

        # 2. Form/Stability (40%)
        form_score = max(0, 100 - (torso_angle / self.strict_thresh * 100))
        score += min(100, form_score) * 0.4

        # 3. Tempo (20%)
        if rep_time:
            ideal = 2.0
            time_diff = abs(rep_time - ideal)
            tempo_score = max(0, 100 - (time_diff / ideal * 100))
            score += tempo_score * 0.2
        else:
            score = score * 1.25 # Scale up if no time data

        return min(100, score)

    def process(self, shoulder: List[float], elbow: List[float], wrist: List[float], hip: Optional[List[float]] = None) -> Dict:
        """
        Process frame landmarks. 
        RETURNS: Dictionary (for compatibility with main.py)
        """
        
        # 1. Angle Calculation
        try:
            # CALL THE NEW UTILITY
            angle_result = calculate_angle_3d(shoulder, elbow, wrist)
            
            # EXTRACT THE DATA (This is the fix!)
            if angle_result.is_valid:
                raw_angle = angle_result.angle
            else:
                raw_angle = 0.0
                self.logger.warning(f"Invalid angle detected: {angle_result.additional_info}")

        except Exception as e:
            raw_angle = 0.0

        # Smoothing
        self.angle_history.append(raw_angle)
        smoothed_angle = float(np.median(self.angle_history))

        # 2. Form Analysis (Torso)
        torso_angle = 0.0
        if hip is not None:
            # CALL NEW UTILITY
            torso_result = check_form_torso_relative(shoulder, elbow, hip)
            # EXTRACT DATA
            torso_angle = float(torso_result.angle)

        # 3. AI Prediction
        is_valid_angle = self.validate_angle(smoothed_angle)
        current_state = "DOWN"
        confidence = 0.0

        if is_valid_angle and self.model:
            try:
                prediction = self.model.predict([[smoothed_angle]])[0]
                current_state = "UP" if prediction == 1 else "DOWN"
                
                if hasattr(self.model, 'predict_proba'):
                    confidence = self.model.predict_proba([[smoothed_angle]])[0][prediction] * 100
            except:
                pass # Fail silently to geometric logic

        # 4. Feedback Logic
        feedback = "GOOD FORM"
        color = (0, 255, 0)
        feedback_type = "good"

        if not is_valid_angle:
            feedback = "ADJUST CAMERA"
            color = (0, 0, 255)
            feedback_type = "invalid"
        elif torso_angle > self.strict_thresh:
            feedback = "ELBOW SWING!"
            color = (0, 0, 255)
            feedback_type = "bad"
        elif smoothed_angle < self.calibration_data['min_angle']:
            feedback = "FULL EXTENSION"
            color = (0, 165, 255)
            feedback_type = "partial"
        
        # 5. Rep Counting State Machine
        is_valid_rep = False
        rep_quality = 0.0

        if feedback_type == "good":
            if current_state == "UP":
                if self.stage != "up":
                    self.stage = "up"
                    self.rep_start_time = time.time()
            
            elif current_state == "DOWN":
                if self.stage == "up":
                    self.stage = "down"
                    self.counter += 1
                    is_valid_rep = True
                    
                    # Calculate Stats
                    duration = 0
                    if self.rep_start_time:
                        duration = time.time() - self.rep_start_time
                        self.rep_times.append(duration)
                    
                    rep_quality = self.calculate_rep_quality(smoothed_angle, torso_angle, duration)
                    self.rep_history.append({
                        "rep": self.counter, 
                        "quality": rep_quality,
                        "duration": duration
                    })
                    self.logger.info(f"Rep {self.counter} | Quality: {rep_quality:.1f}%")

        # 6. Return Result (As Dictionary for Compatibility)
        result = ExerciseResult(
            counter=self.counter,
            feedback=feedback,
            color=color,
            elbow_angle=smoothed_angle,
            torso_angle=torso_angle,
            stage=self.stage,
            confidence=confidence,
            is_valid_rep=is_valid_rep,
            rep_quality=rep_quality
        )
        
        # IMPORTANT: Return as dict so main.py doesn't crash
        return result.to_dict()