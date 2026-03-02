"""
AI GYM TRAINER - ENHANCED EDGE CASE PROTECTION
Main Application File with Professional ML Integration
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
import json
import datetime
import sys
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# MediaPipe solutions - import after mediapipe module
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose_module

# Import your visual engine
try:
    from utils.visual_utils import VisualFeedback, DisplayMetrics
    VISUAL_AVAILABLE = True
except ImportError:
    print("[WARNING] Visual utils not found. Using basic rendering.")
    VISUAL_AVAILABLE = False

try:
    from utils.expert_coach import ExpertCoach
    COACH_AVAILABLE = True
except ImportError:
    print("[WARNING] Expert coach not found. Using basic feedback.")
    COACH_AVAILABLE = False

# ===========================================
# CONFIGURATION DATACLASSES
# ===========================================
@dataclass
class AppConfig:
    """Centralized configuration management"""
    # Model settings
    model_path: str = 'data/models/bicep_adult.pkl'
    advanced_model_path: str = 'data/models/advanced_bicep_model.pkl'
    
    # Camera settings
    target_fps: int = 30
    resolution: Tuple[int, int] = (640, 480)
    camera_backend: str = 'DSHOW' if sys.platform == 'win32' else 'ANY'
    
    # Exercise parameters
    strict_mode_threshold: int = 25
    calibration_time: int = 5
    min_rep_angle: int = 50
    max_rep_angle: int = 170
    min_rep_time: float = 1.0
    max_rep_time: float = 4.0
    
    # Performance — frame-skip activates when FPS drops below this value.
    # 20 gives headroom before the feed becomes visibly laggy.
    enable_optimizations: bool = True
    low_perf_threshold: int = 20
    buffer_size: int = 30
    
    # ML Settings
    confidence_threshold: float = 0.7
    smoothing_window: int = 5
    adaptive_thresholds: bool = True
    
    # File paths
    session_logs: str = 'session_logs'
    calibration_data: str = 'user_calibration'

@dataclass
class UserProfile:
    """Store user-specific data"""
    user_id: str = "default"
    personal_min_angle: float = 0
    personal_max_angle: float = 0
    flexibility_factor: float = 0.2
    previous_sessions: List[Dict] = field(default_factory=list)
    calibration_data: Dict = field(default_factory=dict)

# ===========================================
# ENHANCED ML MODEL HANDLER
# ===========================================
class MLModelHandler:
    """Professional ML model management with fallback strategies"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = None
        self.model_type = None
        self.feature_scaler = None
        self.is_advanced = False
        
    def load_model(self) -> bool:
        """Load model with intelligent fallback system"""
        # TEMPORARILY DISABLED: ML model needs state-labeled training data
        # to distinguish between "up" vs "down" bicep positions
        print("[INFO] ML model training requires state labels - using heuristic for demo")
        print("[INFO] Using heuristic fallback model")
        self.model = HeuristicModel()
        self.model_type = "heuristic"
        return True
        
        # Future: Enable when training data includes up/down state labels
        models_to_try = [
            (self.config.advanced_model_path, self._load_advanced_model),
            (self.config.model_path, self._load_basic_model),
        ]
        
        for model_path, loader in models_to_try:
            if os.path.exists(model_path):
                try:
                    if loader(model_path):
                        print(f"[SUCCESS] Loaded model: {model_path}")
                        return True
                except Exception as e:
                    print(f"[WARNING] Failed to load {model_path}: {e}")
        
        # Fallback to heuristic model
        print("[INFO] Using heuristic fallback model")
        self.model = HeuristicModel()
        self.model_type = "heuristic"
        return True
    
    def _load_advanced_model(self, path: str) -> bool:
        """Load advanced ML model with feature engineering"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
                
            if isinstance(model_data, dict):
                # Advanced model with metadata
                self.model = model_data.get('model')
                self.feature_scaler = model_data.get('scaler')
                self.model_type = model_data.get('type', 'advanced')
                self.is_advanced = True
                
                # Validate model structure
                if self.model is None:
                    return False
                sample_input = np.zeros((1, model_data.get('n_features', 2)))
                prediction = self.model.predict(sample_input)
                print(f"[DEBUG] Advanced model test prediction: {prediction}")
                return True
        except Exception as e:
            print(f"[ERROR] Advanced model load failed: {e}")
        return False
    
    def _load_basic_model(self, path: str) -> bool:
        """Load basic scikit-learn model"""
        try:
            with open(path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # Handle dict model format (from train_universal.py)
            if isinstance(loaded_data, dict):
                self.model = loaded_data.get('model')
                self.model_type = loaded_data.get('type', 'supervised')
                if self.model is None:
                    print("[ERROR] Dict model missing 'model' key")
                    return False
            else:
                self.model = loaded_data
                self.model_type = "basic"
            
            # Test model compatibility
            try:
                # Try 2-feature input
                test_input = [[0, 0]]
                self.model.predict(test_input)
                if not isinstance(loaded_data, dict):
                    self.model_type = "basic_2feature"
            except ValueError:
                # Try 1-feature input
                test_input = [[0]]
                self.model.predict(test_input)
                if not isinstance(loaded_data, dict):
                    self.model_type = "basic_1feature"
            
            return True
        except Exception as e:
            print(f"[ERROR] Basic model load failed: {e}")
            return False
    
    def predict(self, features: List[float], context: Optional[Dict] = None) -> Tuple[int, float]:
        """
        Enhanced prediction with context awareness
        
        Returns:
            tuple: (prediction_class, confidence_score)
        """
        try:
            # Feature engineering based on model type
            processed_features = self._engineer_features(features, context)
            
            # Scale features if scaler available
            if self.feature_scaler:
                processed_features = self.feature_scaler.transform([processed_features])[0]
            
            # Get prediction
            if self.model is None:
                return self._heuristic_fallback(features), 0.5
            prediction = self.model.predict([processed_features])[0]
            
            # Get confidence if available
            confidence = 1.0
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba([processed_features])[0]
                confidence = max(proba)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            # Fallback to heuristic
            return self._heuristic_fallback(features), 0.5
    
    def _engineer_features(self, features: List[float], context: Optional[Dict] = None) -> List[float]:
        """Create enhanced feature set for ML model"""
        base_features = list(features)
        
        if context and self.is_advanced:
            # Add temporal features
            if 'angle_history' in context:
                angles = context['angle_history']
                if len(angles) > 1:
                    base_features.append(float(np.std(angles)))  # Angle variance
                    base_features.append(float(np.mean(np.diff(angles))))  # Angle velocity
            
            # Add form features
            if 'upper_arm_angle' in context:
                base_features.append(context['upper_arm_angle'])
        
        return base_features
    
    def _heuristic_fallback(self, features: List[float]) -> int:
        """Fallback prediction when ML model fails"""
        elbow_angle = features[0] if len(features) > 0 else 90
        swing_dist = features[1] if len(features) > 1 else 0
        
        # Simple heuristic rules
        if swing_dist > 0.15:  # High elbow swing
            return 3  # Bad form
        elif elbow_angle > 160:
            return 0  # Down position
        elif elbow_angle < 60:
            return 1  # Up position
        else:
            return 2  # Middle position

class HeuristicModel:
    """Pure heuristic model for fallback"""
    def predict(self, X):
        # Expects [elbow_angle, swing_dist]
        predictions = []
        for features in X:
            elbow_angle, swing_dist = features[0], features[1] if len(features) > 1 else 0
            
            if swing_dist > 0.15:
                predictions.append(3)  # Bad form
            elif elbow_angle > 160:
                predictions.append(0)  # Down
            elif elbow_angle < 60:
                predictions.append(1)  # Up
            else:
                predictions.append(2)  # Middle
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Return probability estimates (dummy implementation)"""
        predictions = self.predict(X)
        # Return dummy probabilities with high confidence for heuristic predictions
        probas = []
        for pred in predictions:
            proba = [0.1, 0.1, 0.1, 0.1]
            proba[int(pred)] = 0.9
            probas.append(proba)
        return np.array(probas)

# ===========================================
# ENHANCED POSE ANALYZER
# ===========================================
class PoseAnalyzer:
    """Advanced pose analysis with biomechanical insights"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.mp_pose = mp_pose_module  # Reference the imported pose module
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1,
            enable_segmentation=False
        )
        
    def calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Robust angle calculation with error correction"""
        try:
            a, b, c = map(np.array, [a, b, c])
            
            ba = a - b
            bc = c - b
            
            dot_product = np.dot(ba, bc)
            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)
            
            if norm_ba < 1e-6 or norm_bc < 1e-6:
                return 0.0
            
            cosine_angle = np.clip(dot_product / (norm_ba * norm_bc), -1.0, 1.0)
            angle = np.degrees(np.arccos(cosine_angle))
            
            return float(angle)
        except Exception:
            return 0.0
    
    def analyze_pose(self, landmarks, side: str = "right") -> Dict[str, Any]:
        """Comprehensive pose analysis"""
        if not landmarks:
            return {}
        
        if side.lower() == "right":
            joints = {
                "shoulder": (11, 12),  # Actually left shoulder for right arm
                "elbow": (13, 14),
                "wrist": (15, 16)
            }
        else:
            joints = {
                "shoulder": (12, 11),  # Right shoulder for left arm
                "elbow": (14, 13),
                "wrist": (16, 15)
            }
        
        # Extract joint positions
        shoulder = self._get_joint_position(landmarks, joints["shoulder"][0])
        elbow = self._get_joint_position(landmarks, joints["elbow"][0])
        wrist = self._get_joint_position(landmarks, joints["wrist"][0])
        
        # Calculate angles
        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        
        # Calculate upper arm deviation from vertical
        vertical_vector = np.array([0, 1])
        upper_arm_vector = elbow - shoulder
        upper_arm_angle = self._angle_between_vectors(upper_arm_vector, vertical_vector)
        
        # Calculate swing (horizontal movement)
        swing_distance = abs(shoulder[0] - elbow[0])
        
        # Calculate joint velocities if history available
        velocity = self._calculate_joint_velocity(elbow, getattr(self, '_prev_elbow', None))
        self._prev_elbow = elbow
        
        return {
            "elbow_angle": elbow_angle,
            "upper_arm_angle": upper_arm_angle,
            "swing_distance": swing_distance,
            "velocity": velocity,
            "joints": {
                "shoulder": shoulder,
                "elbow": elbow,
                "wrist": wrist
            }
        }
    
    def _get_joint_position(self, landmarks, index: int) -> np.ndarray:
        """Get normalized joint position"""
        if hasattr(landmarks[index], 'x'):
            return np.array([landmarks[index].x, landmarks[index].y])
        return np.array([0, 0])
    
    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors"""
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 0.0
        cos_angle = np.clip(dot / norm, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def _calculate_joint_velocity(self, current_pos: np.ndarray, prev_pos: Optional[np.ndarray]) -> float:
        """Calculate joint movement velocity"""
        if prev_pos is None:
            return 0.0
        return float(np.linalg.norm(current_pos - prev_pos))
    
    def release(self):
        """Release resources"""
        if self.pose:
            self.pose.close()

# ===========================================
# ENHANCED REPETITION COUNTER
# ===========================================
class RepetitionCounter:
    """Smart repetition counting with form validation"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.count = 0
        self.state = None  # 'up', 'down', or 'bad'
        self.state_history = deque(maxlen=10)
        self.angle_history = deque(maxlen=config.buffer_size)
        self.rep_start_time = None
        self.rep_times = []
        self.form_scores = []
        
        # State machine parameters
        self.min_up_angle = config.min_rep_angle
        self.max_down_angle = config.max_rep_angle
        self.min_rep_time = config.min_rep_time
        self.max_rep_time = config.max_rep_time
        
    def update(self, elbow_angle: float, prediction: int, confidence: float, 
               upper_arm_angle: float, swing_distance: float) -> Dict[str, Any]:
        """
        Update rep counter with new frame data
        
        Returns:
            dict: Rep analysis results
        """
        self.angle_history.append(elbow_angle)
        
        # Map prediction to state
        if prediction == 3:  # Bad form
            new_state = 'bad'
        elif prediction == 1:  # Up
            new_state = 'up'
        elif prediction == 0:  # Down
            new_state = 'down'
        else:
            new_state = None
        
        # Check for state transition
        rep_completed = False
        form_quality = self._calculate_form_quality(elbow_angle, upper_arm_angle, swing_distance)
        
        if new_state == 'up' and self.state != 'up':
            # Starting a new rep
            self.state = 'up'
            self.rep_start_time = time.time()
            
        elif new_state == 'down' and self.state == 'up':
            # Completed a rep
            rep_completed = True
            self.state = 'down'
            
            if self.rep_start_time:
                rep_duration = time.time() - self.rep_start_time
                self.rep_times.append(rep_duration)
                
                # Validate rep quality
                is_valid = self._validate_rep(rep_duration, form_quality)
                if is_valid:
                    self.count += 1
                    self.form_scores.append(form_quality)
                    print(f"[REP {self.count}] Duration: {rep_duration:.2f}s, "
                          f"Form: {form_quality:.1f}%, Angle: {elbow_angle:.1f}°")
        
        elif new_state == 'bad':
            # Bad form detected
            self.state = 'bad'
            if self.rep_start_time:
                self.rep_start_time = None  # Reset if bad form during rep
        
        self.state_history.append((new_state, time.time()))
        
        return {
            'count': self.count,
            'state': self.state,
            'rep_completed': rep_completed,
            'form_quality': form_quality,
            'angle': elbow_angle
        }
    
    def _calculate_form_quality(self, elbow_angle: float, upper_arm_angle: float, 
                                swing_distance: float) -> float:
        """Calculate form quality score (0-100)"""
        score = 100.0
        
        # Penalize elbow swing
        swing_penalty = min(swing_distance * 200, 40)
        score -= swing_penalty
        
        # Penalize incorrect range of motion
        if elbow_angle < self.min_up_angle * 0.9:
            score -= 20  # Too shallow
        if elbow_angle > self.max_down_angle * 1.1:
            score -= 15  # Too extended
        
        # Penalize upper arm movement
        if upper_arm_angle > self.config.strict_mode_threshold:
            penalty = (upper_arm_angle - self.config.strict_mode_threshold) * 2
            score -= min(penalty, 30)
        
        return max(0, score)
    
    def _validate_rep(self, duration: float, form_quality: float) -> bool:
        """Validate if rep meets quality standards"""
        # Check duration
        if duration < self.min_rep_time or duration > self.max_rep_time:
            print(f"[WARNING] Rep duration {duration:.2f}s outside optimal range")
            return False
        
        # Check form quality
        if form_quality < 70:
            print(f"[WARNING] Low form quality: {form_quality:.1f}%")
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive rep statistics"""
        if not self.rep_times:
            return {}
        
        return {
            'total_reps': self.count,
            'avg_duration': np.mean(self.rep_times) if self.rep_times else 0,
            'std_duration': np.std(self.rep_times) if len(self.rep_times) > 1 else 0,
            'avg_form_score': np.mean(self.form_scores) if self.form_scores else 0,
            'best_form_score': max(self.form_scores) if self.form_scores else 0,
            'worst_form_score': min(self.form_scores) if self.form_scores else 0,
            'consistency': 100 - (np.std(self.rep_times) / np.mean(self.rep_times) * 100) 
                          if self.rep_times and np.mean(self.rep_times) > 0 else 0
        }
    
    def reset(self):
        """Reset counter"""
        self.count = 0
        self.state = None
        self.state_history.clear()
        self.angle_history.clear()
        self.rep_times.clear()
        self.form_scores.clear()
        self.rep_start_time = None

# ===========================================
# ENHANCED CAMERA HANDLER
# ===========================================
class CameraHandler:
    """Professional camera management with auto-recovery"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.cap = None
        self.camera_index = 0
        self.is_open = False
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        
    def open(self, index: Optional[int] = None) -> bool:
        """Open camera with auto-detection"""
        if index is None:
            index = self._find_best_camera()
        
        if index is None:
            print("[ERROR] No cameras found")
            return False
        
        # Set backend based on OS
        if sys.platform == 'win32':
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(index)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Failed to open camera {index}")
            return False
        
        # Configure camera
        self._configure_camera()
        self.camera_index = index
        self.is_open = True
        self.frame_count = 0
        
        print(f"[SUCCESS] Camera {index} opened at {self.config.resolution}")
        return True
    
    def _find_best_camera(self) -> Optional[int]:
        """Find working camera with best specs"""
        available_cameras = []
        
        for i in range(5):  # Check first 5 indices
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if sys.platform == 'win32' else 0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Score camera based on resolution and FPS
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    score = (width * height) + (fps * 100)
                    available_cameras.append((i, score, width, height, fps))
                
                cap.release()
        
        if not available_cameras:
            return None
        
        # Select best camera
        available_cameras.sort(key=lambda x: x[1], reverse=True)
        return available_cameras[0][0]
    
    def _configure_camera(self):
        """Configure camera settings"""
        if not self.cap:
            return
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
        
        # Try to set FPS
        self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        
        # Optional: Set camera properties
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read frame with error handling"""
        if not self.cap or not self.is_open:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            # Try to recover
            print("[WARNING] Frame read failed, attempting recovery...")
            self.release()
            time.sleep(0.1)
            if self.open(self.camera_index):
                ret, frame = self.cap.read()
            
            if not ret:
                return None
        
        self.frame_count += 1
        
        # Resize to configured resolution
        if frame.shape[1] != self.config.resolution[0] or \
           frame.shape[0] != self.config.resolution[1]:
            frame = cv2.resize(frame, self.config.resolution)
        
        return frame
    
    def get_fps(self) -> float:
        """Calculate current FPS"""
        if len(self.fps_history) < 2:
            return 0.0
        
        return float(np.mean(list(self.fps_history)))
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.is_open = False

# ===========================================
# ENHANCED FEEDBACK SYSTEM
# ===========================================
class FeedbackSystem:
    """Comprehensive feedback generation"""
    
    FEEDBACK_TYPES = {
        "good_form": {"text": "✓ PERFECT FORM", "color": (0, 255, 0)},
        "elbow_swing": {"text": "✗ ELBOW SWING DETECTED", "color": (0, 0, 255)},
        "partial_rep": {"text": "⚠ PARTIAL RANGE OF MOTION", "color": (0, 165, 255)},
        "too_fast": {"text": "⚡ TOO FAST - SLOW DOWN", "color": (0, 255, 255)},
        "too_slow": {"text": "🐌 TOO SLOW - INCREASE PACE", "color": (255, 165, 0)},
        "full_lockout": {"text": "⚠ AVOID FULL LOCKOUT", "color": (255, 255, 0)},
        "bad_posture": {"text": "⚠ IMPROVE POSTURE", "color": (255, 100, 0)},
        "good_lighting": {"text": "✓ GOOD LIGHTING", "color": (0, 255, 0)},
        "bad_lighting": {"text": "⚠ POOR LIGHTING", "color": (255, 100, 0)},
        "optimal_range": {"text": "✓ OPTIMAL RANGE", "color": (0, 255, 0)}
    }
    
    def __init__(self):
        self.feedback_history = deque(maxlen=50)
        self.current_feedback = self.FEEDBACK_TYPES["good_form"]
        
    def analyze(self, pose_data: Dict, rep_data: Dict, 
                lighting_score: float, occlusion_score: float) -> Dict:
        """Generate comprehensive feedback"""
        feedbacks = []
        
        # Pose-based feedback
        if pose_data.get('upper_arm_angle', 0) > 25:
            feedbacks.append(self.FEEDBACK_TYPES["elbow_swing"])
        
        if pose_data.get('elbow_angle', 90) < 60:
            feedbacks.append(self.FEEDBACK_TYPES["partial_rep"])
        
        if pose_data.get('elbow_angle', 90) > 170:
            feedbacks.append(self.FEEDBACK_TYPES["full_lockout"])
        
        # Rep timing feedback
        if 'rep_duration' in rep_data:
            dur = rep_data['rep_duration']
            if dur < 1.5:
                feedbacks.append(self.FEEDBACK_TYPES["too_fast"])
            elif dur > 3.5:
                feedbacks.append(self.FEEDBACK_TYPES["too_slow"])
        
        # Environmental feedback
        if lighting_score < 50:
            feedbacks.append(self.FEEDBACK_TYPES["bad_lighting"])
        elif lighting_score > 150:
            feedbacks.append(self.FEEDBACK_TYPES["good_lighting"])
        
        # Select primary feedback
        priority_order = [
            "elbow_swing", "partial_rep", "full_lockout", 
            "too_fast", "too_slow", "bad_lighting", 
            "good_lighting", "good_form"
        ]
        
        for fb_type in priority_order:
            if any(fb["text"] == self.FEEDBACK_TYPES[fb_type]["text"] for fb in feedbacks):
                self.current_feedback = self.FEEDBACK_TYPES[fb_type]
                break
        else:
            self.current_feedback = self.FEEDBACK_TYPES["good_form"]
        
        self.feedback_history.append({
            "timestamp": time.time(),
            "feedback": self.current_feedback["text"],
            "pose_data": pose_data,
            "rep_data": rep_data
        })
        
        return self.current_feedback
    
    def get_historical_analysis(self) -> Dict:
        """Get historical feedback analysis"""
        if not self.feedback_history:
            return {}
        
        # Count feedback types
        feedback_counts = defaultdict(int)
        for entry in self.feedback_history:
            feedback_counts[entry["feedback"]] += 1
        
        return {
            "total_feedbacks": len(self.feedback_history),
            "feedback_distribution": dict(feedback_counts),
            "most_common_feedback": max(feedback_counts.items(), key=lambda x: x[1])[0] 
            if feedback_counts else "None"
        }

# ===========================================
# PERFORMANCE MONITOR
# ===========================================
class PerformanceMonitor:
    """Monitor and optimize application performance"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.frame_times = deque(maxlen=30)
        self.last_time = time.time()
        self.current_fps = config.target_fps
        self.low_performance_mode = False
        self.frame_skip_counter = 0
        
    def update(self) -> float:
        """Update performance metrics"""
        current_time = time.time()
        frame_time = current_time - self.last_time
        
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) > 0:
            self.current_fps = 1.0 / float(np.mean(list(self.frame_times)))
        
        # Adaptive performance mode
        if self.current_fps < self.config.low_perf_threshold:
            self.low_performance_mode = True
        elif self.current_fps > self.config.target_fps * 0.8:
            self.low_performance_mode = False
        
        self.last_time = current_time
        return self.current_fps
    
    def should_skip_processing(self) -> bool:
        """Determine if we should skip heavy processing"""
        if not self.low_performance_mode:
            return False
        
        # Skip every other frame in low performance mode
        self.frame_skip_counter += 1
        return self.frame_skip_counter % 2 == 0
    
    def get_diagnostics(self) -> Dict:
        """Get performance diagnostics"""
        return {
            "current_fps": self.current_fps,
            "avg_frame_time": float(np.mean(list(self.frame_times))) if self.frame_times else 0,
            "low_performance_mode": self.low_performance_mode,
            "frame_skip_rate": (self.frame_skip_counter % 2) * 100,
            "recommendations": self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """Get performance improvement recommendations"""
        recs = []
        
        if self.current_fps < 20:
            recs.append("Reduce resolution")
            recs.append("Close other applications")
            recs.append("Improve lighting conditions")
        
        if self.current_fps < 15:
            recs.append("Disable visual effects")
            recs.append("Use lower camera resolution")
        
        return recs

# ===========================================
# MAIN AI GYM TRAINER CLASS
# ===========================================
class AIGymTrainer:
    """Enhanced AI Gym Trainer with professional ML integration"""
    
    def __init__(self):
        # Configuration
        self.config = AppConfig()
        self.user = UserProfile()
        
        # Core components
        self.camera = CameraHandler(self.config)
        self.pose_analyzer = PoseAnalyzer(self.config)
        self.ml_model = MLModelHandler(self.config)
        self.rep_counter = RepetitionCounter(self.config)
        self.feedback_system = FeedbackSystem()
        self.performance = PerformanceMonitor(self.config)
        
        # State
        self.running = False
        self.paused = False
        self.calibrated = False
        self.session_start_time = None
        
        # Visualization
        self.visualizer = None
        self.coach = None

        # Exercise-class integration (set via setup_exercise before run())
        self.exercise_instance = None
        self.exercise_type = 'bicep'
        self.age_group = 'adult'

        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/gym_trainer_{datetime.datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_exercise(self, exercise_type: str, age: int):
        """
        Instantiate the correct exercise class for the selected exercise and age.
        Must be called before run().  Replaces the disabled MLModelHandler path
        for all exercises.

        exercise_type: 'bicep' | 'back' | 'chest'
        age:           user's age in years (determines age-group & model file)
        """
        try:
            from exercises.base_exercise import get_exercise_config
            from exercises.back_exercise import BackExercise
            from exercises.chest_exercise import ChestExercise
            from exercises.bicep_curl import BicepCurlExercise

            config = get_exercise_config(exercise_type, age)
            self.exercise_type = exercise_type
            self.age_group = config.age_group
            self.config.model_path = config.model_path  # for logging

            if exercise_type == 'bicep':
                self.exercise_instance = BicepCurlExercise(config)
            elif exercise_type == 'back':
                self.exercise_instance = BackExercise(config)
            elif exercise_type == 'chest':
                self.exercise_instance = ChestExercise(config)
            else:
                print(f"[WARNING] Unknown exercise '{exercise_type}', defaulting to bicep")
                self.exercise_instance = BicepCurlExercise(config)

            print(f"[INFO] Exercise ready: {exercise_type.upper()} | {self.age_group.upper()}")
        except Exception as e:
            print(f"[WARNING] Could not set up exercise class: {e} — using legacy mode")
            self.exercise_instance = None

    def initialize(self) -> bool:
        """Initialize all components"""
        self.logger.info("=" * 60)
        self.logger.info("AI GYM TRAINER - PROFESSIONAL EDITION")
        self.logger.info("=" * 60)
        
        try:
            # 1. Load ML Model
            self.logger.info("Loading ML model...")
            if not self.ml_model.load_model():
                self.logger.error("Failed to load ML model")
                return False
            
            # 2. Initialize camera
            self.logger.info("Initializing camera...")
            if not self.camera.open():
                self.logger.error("Failed to initialize camera")
                return False
            
            # 3. Initialize visualizer
            if VISUAL_AVAILABLE:
                self.visualizer = VisualFeedback(  # type: ignore[possibly-undefined]
                    resolution=self.config.resolution
                )
                self.logger.info("Visual feedback system initialized")

            # 4. Initialize expert coach
            if COACH_AVAILABLE:
                self.coach = ExpertCoach()  # type: ignore[possibly-undefined]
                self.logger.info("Expert coach initialized")
            
            # 5. Create necessary directories
            os.makedirs(self.config.session_logs, exist_ok=True)
            os.makedirs(self.config.calibration_data, exist_ok=True)
            
            self.logger.info("Initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def calibrate_user(self) -> bool:
        """Enhanced user calibration"""
        self.logger.info("Starting user calibration...")
        
        # Display calibration instructions
        print("\n" + "=" * 50)
        print("PERSONALIZED CALIBRATION")
        print("=" * 50)
        print("1. Stand in SIDE VIEW of the camera")
        print("2. Keep your arm relaxed at your side")
        print("3. Ensure good lighting conditions")
        print("=" * 50)
        
        calibration_window = "Calibration"
        cv2.namedWindow(calibration_window, cv2.WINDOW_NORMAL)
        
        calibration_data = {
            "angles": [],
            "lighting": [],
            "timestamps": [],
            "joint_positions": []
        }
        
        frames_collected = 0
        target_frames = 90  # 3 seconds at 30 FPS
        
        while frames_collected < target_frames:
            frame = self.camera.read_frame()
            if frame is None:
                continue
            
            # Analyze pose
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_analyzer.pose.process(frame_rgb)
            
            display_frame = frame.copy()
            
            if results.pose_landmarks:  # type: ignore
                landmarks = results.pose_landmarks.landmark  # type: ignore
                pose_analysis = self.pose_analyzer.analyze_pose(landmarks, "right")
                
                if pose_analysis:
                    angle = pose_analysis.get('elbow_angle', 0)
                    calibration_data["angles"].append(angle)
                    calibration_data["lighting"].append(np.mean(frame))
                    calibration_data["timestamps"].append(time.time())
                    calibration_data["joint_positions"].append(pose_analysis.get('joints', {}))
                    
                    frames_collected += 1
                    
                    # Draw progress
                    progress = (frames_collected / target_frames) * 100
                    cv2.rectangle(display_frame, (50, 300), (50 + int(progress * 3), 330), 
                                 (0, 255, 0), -1)
                    cv2.putText(display_frame, f"Calibrating: {progress:.1f}%", 
                               (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Draw skeleton
                    if self.visualizer:
                        self.visualizer.draw_pose_skeleton(
                            display_frame, landmarks, self.pose_analyzer.mp_pose.POSE_CONNECTIONS
                        )
            
            # Display instructions
            cv2.putText(display_frame, "CALIBRATION IN PROGRESS", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, "Keep your arm relaxed at your side", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(calibration_window, display_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
                cv2.destroyWindow(calibration_window)
                return False
        
        cv2.destroyWindow(calibration_window)
        
        # Process calibration data
        if calibration_data["angles"]:
            self.user.personal_min_angle = float(np.percentile(calibration_data["angles"], 5))
            self.user.personal_max_angle = float(np.percentile(calibration_data["angles"], 95))
            self.user.calibration_data = calibration_data
            
            self.logger.info(f"Calibration complete. Min angle: {self.user.personal_min_angle:.1f}°, "
                           f"Max angle: {self.user.personal_max_angle:.1f}°")
            
            # Save calibration data
            self._save_calibration_data()
            self.calibrated = True
            
            return True
        
        self.logger.warning("Calibration failed - insufficient data")
        return False
    
    def _save_calibration_data(self):
        """Save user calibration data"""
        filename = f"{self.config.calibration_data}/user_{self.user.user_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "user_id": self.user.user_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "personal_min_angle": self.user.personal_min_angle,
            "personal_max_angle": self.user.personal_max_angle,
            "calibration_data": {
                "angles": self.user.calibration_data.get("angles", []),
                "lighting": self.user.calibration_data.get("lighting", []),
                "timestamps": self.user.calibration_data.get("timestamps", [])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Calibration data saved to {filename}")
    
    def run(self):
        """Main application loop"""
        try:
            if not self.initialize():
                self.logger.error("Failed to initialize application")
                return
            
            # Calibrate user
            if not self.calibrate_user():
                self.logger.warning("Using default calibration values")
            
            self.logger.info("Starting main loop...")
            print("\n[CONTROLS] Q: Quit | R: Reset | S: Save | C: Calibrate | P: Pause")
            print("[TIPS] Stand in side view, wear fitted clothing, ensure good lighting")
            
            # Main loop variables
            angle_history = deque(maxlen=self.config.smoothing_window)
            self.running = True
            self.session_start_time = time.time()
            
            # Create main window
            window_name = 'AI Gym Trainer - Professional Edition'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
            
            while self.running:
                # Handle pause
                if self.paused:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('p'):
                        self.paused = False
                        self.logger.info("Resumed")
                    continue
                
                # Performance monitoring
                fps = self.performance.update()
                
                # Skip processing if performance is low
                if self.performance.should_skip_processing():
                    continue
                
                # Read frame
                frame = self.camera.read_frame()
                if frame is None:
                    self.logger.warning("Failed to read frame")
                    time.sleep(0.01)
                    continue
                
                # Process frame
                processed_frame, analysis_results = self._process_frame(frame, fps, angle_history)
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                self._handle_keyboard(key)
                
                # Auto-save every 50 reps
                if self.rep_counter.count > 0 and self.rep_counter.count % 50 == 0:
                    self._save_session()
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def _process_frame(self, frame: np.ndarray, fps: float, angle_history: deque) -> Tuple[np.ndarray, Dict]:
        """Process a single frame"""
        # Enhance image quality
        enhanced_frame = self._enhance_image(frame)
        
        # Pose detection
        frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        results = self.pose_analyzer.pose.process(frame_rgb)
        
        display_frame = enhanced_frame.copy()
        analysis_results = {}
        
        if results.pose_landmarks:  # type: ignore
            landmarks = results.pose_landmarks.landmark  # type: ignore

            # ── Exercise-class path (back / chest / bicep via class) ──────────
            if self.exercise_instance is not None:
                result = self.exercise_instance.process_frame(landmarks)

                # Draw MediaPipe skeleton
                try:
                    mp_drawing.draw_landmarks(
                        display_frame,
                        results.pose_landmarks,  # type: ignore
                        list(mp_pose_module.POSE_CONNECTIONS),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                except Exception as sk_err:
                    self.logger.warning(f"Skeleton render error: {sk_err}")

                # Exercise label (top-right corner)
                h, w = display_frame.shape[:2]
                label = f"{self.exercise_type.upper()} | {self.age_group.upper()}"
                cv2.putText(display_frame, label, (w - 310, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Primary angle annotation
                cv2.putText(
                    display_frame,
                    f"Angle: {result.primary_angle:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )

                # UI overlays
                if self.visualizer:
                    display_frame = self.visualizer.draw_status_box(
                        display_frame, result.counter,
                        result.feedback, result.color
                    )
                    display_frame = self.visualizer.draw_progress_bar(
                        display_frame, result.rep_quality, 100, "Form Quality"
                    )
                    display_frame = self.visualizer.draw_fps(display_frame, fps)
                else:
                    # Minimal fallback when VisualFeedback is unavailable
                    cv2.rectangle(display_frame, (0, h - 100), (w, h), (0, 0, 0), -1)
                    cv2.putText(display_frame, f"Reps: {result.counter}",
                                (10, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    cv2.putText(display_frame, result.feedback,
                                (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, result.color, 2)
                    q_color = (0, 255, 0) if result.rep_quality >= 70 else (0, 165, 255)
                    cv2.putText(display_frame, f"Quality: {result.rep_quality:.0f}%",
                                (w - 220, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, q_color, 2)

                analysis_results = {
                    'counter':         result.counter,
                    'feedback':        result.feedback,
                    'primary_angle':   result.primary_angle,
                    'secondary_angle': result.secondary_angle,
                    'rep_quality':     result.rep_quality,
                    'stage':           result.stage,
                    'model_missing':   result.model_missing,
                }
                return display_frame, analysis_results
            # ── end exercise-class path ───────────────────────────────────────

            # Legacy heuristic path (unchanged below)
            pose_analysis = self.pose_analyzer.analyze_pose(landmarks, "right")
            
            if pose_analysis:
                # Prepare features for ML model
                features = [
                    pose_analysis['elbow_angle'],
                    pose_analysis['swing_distance']
                ]
                
                # Get ML prediction
                prediction, confidence = self.ml_model.predict(
                    features, 
                    context={'angle_history': list(angle_history)}
                )
                
                angle_history.append(pose_analysis['elbow_angle'])
                
                # Update rep counter
                rep_data = self.rep_counter.update(
                    elbow_angle=pose_analysis['elbow_angle'],
                    prediction=prediction,
                    confidence=confidence,
                    upper_arm_angle=pose_analysis['upper_arm_angle'],
                    swing_distance=pose_analysis['swing_distance']
                )
                
                # Generate feedback
                lighting_score = float(np.mean(frame))
                occlusion_score = self._check_occlusion(landmarks)
                
                feedback = self.feedback_system.analyze(
                    pose_analysis, 
                    rep_data,
                    lighting_score,
                    occlusion_score
                )
                
                # Get coaching tips
                coaching_tip = ""
                if self.coach:
                    coaching_tip = self.coach.get_coaching_tip(
                        int(pose_analysis['elbow_angle']),
                        int(pose_analysis['upper_arm_angle'])
                    )
                
                # Draw visualization
                if self.visualizer:
                    # ==================================================
                    # Draw MediaPipe Skeleton (Pose Landmarks)
                    # ==================================================
                    # Renders 33-point body skeleton with connections
                    # Green circles: joints | Red lines: connections
                    try:
                        mp_drawing.draw_landmarks(
                            display_frame,
                            results.pose_landmarks,  # type: ignore
                            list(mp_pose_module.POSE_CONNECTIONS),
                            mp_drawing.DrawingSpec(
                                color=(0, 255, 0),      # Green for joints
                                thickness=2,
                                circle_radius=2
                            ),
                            mp_drawing.DrawingSpec(
                                color=(255, 0, 0),      # Red for connections
                                thickness=2
                            )
                        )
                    except Exception as e:
                        self.logger.warning(f"Skeleton rendering error: {e}")
                    
                    # ==================================================
                    # Draw Custom Pose Skeleton (Existing Logic)
                    # ==================================================
                    self.visualizer.draw_pose_skeleton(
                        display_frame, landmarks, self.pose_analyzer.mp_pose.POSE_CONNECTIONS
                    )
                    
                    # Prepare metrics
                    metrics = DisplayMetrics(  # type: ignore[possibly-undefined]
                        elbow_angle=pose_analysis['elbow_angle'],
                        torso_angle=pose_analysis['upper_arm_angle'],
                        torso_threshold=self.config.strict_mode_threshold,
                        fps=fps,
                        form_quality=rep_data.get('form_quality', 0),
                        rep_count=self.rep_counter.count,
                        stage=rep_data.get('state', ''),
                        arm_side="right"
                    )
                    
                    # Draw status box
                    display_frame = self.visualizer.draw_status_box(
                        display_frame, 
                        self.rep_counter.count, 
                        feedback['text'], 
                        feedback['color']
                    )
                    
                    # Draw debug panel
                    if fps > 15:
                        display_frame = self.visualizer.draw_debug_panel(
                            display_frame, metrics
                        )
                    
                    # Draw progress bar
                    display_frame = self.visualizer.draw_progress_bar(
                        display_frame, 
                        rep_data.get('form_quality', 0), 
                        100, 
                        "Form Quality"
                    )
                    
                    # Draw FPS
                    display_frame = self.visualizer.draw_fps(display_frame, fps)
                
                # Draw coaching tip
                if coaching_tip and fps > 15:
                    self._draw_coaching_tip(display_frame, coaching_tip)
                
                analysis_results = {
                    'pose_analysis': pose_analysis,
                    'prediction': prediction,
                    'confidence': confidence,
                    'rep_data': rep_data,
                    'feedback': feedback
                }
        
        else:
            # No pose detected
            cv2.putText(display_frame, "NO POSE DETECTED", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(display_frame, "Check: Lighting | Camera View | Distance", 
                       (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw FPS
        fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        return display_frame, analysis_results
    
    def _enhance_image(self, frame: np.ndarray) -> np.ndarray:
        """Apply image enhancements"""
        enhanced = frame.copy()
        
        # Auto-brightness adjustment
        brightness = np.mean(frame)
        if brightness < 80:
            # Apply gamma correction
            gamma = 1.5
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                             for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)
        
        # Apply CLAHE for contrast
        if len(frame.shape) == 3:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _check_occlusion(self, landmarks) -> float:
        """Check if joints are occluded"""
        if not landmarks:
            return 1.0
        
        # Check key joints
        key_joints = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
        visibilities = []
        
        for joint in key_joints:
            if hasattr(landmarks[joint], 'visibility'):
                visibilities.append(landmarks[joint].visibility)
            else:
                visibilities.append(0.5)
        
        return float(np.mean(visibilities))
    
    def _draw_coaching_tip(self, frame: np.ndarray, tip: str):
        """Draw coaching tip on frame"""
        # Create background for tip
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (50, h - 120), (w - 50, h - 30), (0, 0, 0), -1)
        cv2.rectangle(frame, (50, h - 120), (w - 50, h - 30), (0, 255, 255), 2)
        
        # Split long tips
        if len(tip) > 60:
            parts = []
            words = tip.split()
            current = ""
            for word in words:
                if len(current) + len(word) + 1 < 60:
                    current += word + " "
                else:
                    parts.append(current)
                    current = word + " "
            if current:
                parts.append(current)
            
            # Draw each part
            for i, part in enumerate(parts[:2]):  # Max 2 lines
                y_pos = h - 80 + (i * 30)
                cv2.putText(frame, part, (70, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, tip, (70, h - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _handle_keyboard(self, key: int):
        """Handle keyboard input"""
        if key == ord('q'):
            self.running = False
            self.logger.info("Quit requested")
        elif key == ord('r'):
            self.rep_counter.reset()
            self.logger.info("Counter reset")
        elif key == ord('s'):
            self._save_session()
        elif key == ord('c'):
            self.calibrate_user()
        elif key == ord('p'):
            self.paused = not self.paused
            self.logger.info(f"{'Paused' if self.paused else 'Resumed'}")
        elif key == ord('d'):
            # Debug mode toggle
            self._toggle_debug_mode()
    
    def _save_session(self):
        """Save current session data"""
        if self.rep_counter.count == 0:
            self.logger.info("No reps to save")
            return
        
        session_data = {
            "session_id": f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "user_id": self.user.user_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "duration": time.time() - self.session_start_time if self.session_start_time else 0,
            "rep_stats": self.rep_counter.get_stats(),
            "performance_stats": self.performance.get_diagnostics(),
            "feedback_analysis": self.feedback_system.get_historical_analysis(),
            "config": {
                "personal_min_angle": self.user.personal_min_angle,
                "personal_max_angle": self.user.personal_max_angle,
                "model_type": self.ml_model.model_type
            }
        }
        
        filename = f"{self.config.session_logs}/{session_data['session_id']}.json"
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        self.logger.info(f"Session saved to {filename}")
        return filename
    
    def _toggle_debug_mode(self):
        """Toggle debug information display"""
        # Implement debug mode toggle if needed
        pass
    
    def cleanup(self):
        """Clean up all resources"""
        self.logger.info("Cleaning up resources...")
        
        # Release camera
        self.camera.release()
        
        # Release pose analyzer
        self.pose_analyzer.release()
        
        # Close all windows
        cv2.destroyAllWindows()
        
        # Log session summary
        self._log_session_summary()
        
        self.logger.info("Cleanup complete")
    
    def _log_session_summary(self):
        """Log session summary"""
        if self.session_start_time:
            duration = time.time() - self.session_start_time
            stats = self.rep_counter.get_stats()
            
            summary = f"""
            ========================================
            SESSION SUMMARY
            ========================================
            Total Reps: {stats.get('total_reps', 0)}
            Average Form Score: {stats.get('avg_form_score', 0):.1f}%
            Best Form Score: {stats.get('best_form_score', 0):.1f}%
            Average Rep Duration: {stats.get('avg_duration', 0):.2f}s
            Consistency: {stats.get('consistency', 0):.1f}%
            Session Duration: {duration:.1f}s
            Reps per Minute: {(stats.get('total_reps', 0) / duration * 60):.1f}
            Model Type: {self.ml_model.model_type}
            ========================================
            """
            
            self.logger.info(summary)

# ===========================================
# MAIN ENTRY POINT
# ===========================================
def main():
    """Main entry point"""
    # Windows-specific camera backend
    if sys.platform == "win32":
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
        os.environ["OPENCV_VIDEOIO_DEBUG"] = "1"
    
    print("\n" + "=" * 60)
    print("AI GYM TRAINER - PROFESSIONAL EDITION")
    print("Version 2.0 - Enhanced ML Integration")
    print("=" * 60)

    # ── Exercise selection ────────────────────────────────────────────────
    print("\nSelect exercise:")
    print("  1. Bicep Curl")
    print("  2. Back Row")
    print("  3. Chest Push-up")
    ex_input = input("Enter choice [1-3, default=1]: ").strip() or "1"
    exercise_map = {"1": "bicep", "2": "back", "3": "chest"}
    exercise_type = exercise_map.get(ex_input, "bicep")

    age_input = input("Enter your age [8-100, default=25]: ").strip() or "25"
    try:
        age = max(8, min(100, int(age_input)))
    except ValueError:
        age = 25

    print(f"\n[INFO] Starting: {exercise_type.upper()}, age {age}")
    # ─────────────────────────────────────────────────────────────────────

    try:
        trainer = AIGymTrainer()
        trainer.setup_exercise(exercise_type, age)
        trainer.run()
        
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n[INFO] Application terminated")
        sys.exit(0)

if __name__ == "__main__":
    main()
