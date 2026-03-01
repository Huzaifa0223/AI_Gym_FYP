"""
Back Exercise Detection (Bent Over Rows, Lat Pulldowns)
Tracks back, shoulder, and arm movement
"""
import pickle
import os
import numpy as np
from typing import Dict, Tuple, List
from exercises.base_exercise import BaseExercise, ExerciseConfig
from utils import calculate_angle_3d

class BackExercise(BaseExercise):
    """
    Detects back exercises like bent-over rows
    Tracks: Back angle, shoulder retraction, elbow position
    """
    
    def __init__(self, config: ExerciseConfig):
        super().__init__(config)
        self.prev_angle = None
        self.rep_start_time = None
    
    def load_model(self):
        """Load trained model for back exercises"""
        if os.path.exists(self.config.model_path):
            try:
                with open(self.config.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"[INFO] Loaded back exercise model: {self.config.model_path}")
            except Exception as e:
                print(f"[WARNING] Could not load model: {e}")
                self.model = None
        else:
            print(f"[WARNING] Model not found: {self.config.model_path}")
            self.model = None
    
    def get_required_landmarks(self) -> List[int]:
        """
        Required landmarks for back exercises:
        11, 12: Shoulders
        13, 14: Elbows  
        15, 16: Wrists
        23, 24: Hips
        """
        return [11, 12, 13, 14, 15, 16, 23, 24]
    
    def calculate_angles(self, landmarks) -> Dict[str, float]:
        """
        Calculate angles for back exercises:
        - Primary  (elbow angle):   Shoulder -> Elbow -> Wrist
        - Secondary (back lean):    Hip -> Shoulder -> Elbow
        - Tertiary (full reach):    Hip -> Shoulder -> Wrist  (trained feature)
        """
        try:
            shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
            elbow    = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
            wrist    = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
            hip      = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])

            # Primary: elbow bend during the row pull
            elbow_angle = calculate_angle_3d(shoulder, elbow, wrist)

            # Secondary: torso / back lean angle
            back_angle = calculate_angle_3d(hip, shoulder, elbow)

            # Tertiary: full arm deviation from body line — matches training feature
            tertiary_angle = calculate_angle_3d(hip, shoulder, wrist)

            # Shoulder width — useful for retraction cue
            left_shoulder  = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
            shoulder_width = float(np.linalg.norm(right_shoulder - left_shoulder))

            # Landmark visibility confidence
            avg_vis = (
                landmarks[12].visibility + landmarks[14].visibility +
                landmarks[16].visibility + landmarks[24].visibility
            ) / 4

            return {
                'primary':   elbow_angle,
                'secondary': back_angle,
                'tertiary':  tertiary_angle,
                'shoulder_width': shoulder_width,
                'confidence': float(avg_vis)
            }
        except Exception as e:
            print(f"[ERROR] Back angle calculation failed: {e}")
            return {
                'primary': 0.0, 'secondary': 0.0, 'tertiary': 0.0,
                'shoulder_width': 0.0, 'confidence': 0.0
            }
    
    def check_form(self, landmarks, angles: Dict[str, float]) -> Tuple[str, Tuple[int, int, int], float]:
        """
        Check back exercise form using ML model as primary signal.
        Rule-based checks supply specific corrective cues when ML flags bad form.

        ML classes: 0 = bad form, 1 = good form.
        Falls back to pure rule-based when no model is loaded.
        """
        primary   = angles['primary']    # elbow angle
        secondary = angles['secondary']  # back lean
        tertiary  = angles.get('tertiary', 0.0)

        # ── ML prediction ────────────────────────────────────────────────
        is_good, quality_score = self._ml_predict_form([primary, secondary, tertiary])

        if is_good is not None:
            # Model is available — use its verdict as the primary gate
            if not is_good:
                # Bad form detected: identify the most likely cause
                if secondary < 30:
                    return "BACK TOO ROUNDED", (0, 165, 255), max(30.0, quality_score)
                if primary < self.config.min_angle:
                    return "PARTIAL REP", (0, 165, 255), max(40.0, quality_score)
                if primary > self.config.max_angle:
                    return "OVEREXTENDING", (255, 165, 0), max(45.0, quality_score)
                return "CHECK YOUR FORM", (0, 165, 255), max(30.0, quality_score)
            else:
                return "GOOD FORM", (0, 255, 0), quality_score

        # ── Rule-based fallback (no model loaded) ────────────────────────
        quality_score = 100.0
        if secondary < 30:
            return "BACK TOO ROUNDED", (0, 165, 255), 60.0
        if primary < self.config.min_angle:
            quality_score -= 20
            return "PARTIAL REP", (0, 165, 255), quality_score
        if primary > self.config.max_angle:
            quality_score -= 15
            return "OVEREXTENDING", (255, 165, 0), quality_score
        return "GOOD FORM", (0, 255, 0), quality_score
    
    def detect_rep(self, landmarks, angles: Dict[str, float]) -> bool:
        """
        Detect complete back exercise rep:
        - Start: Arms extended (high angle)
        - End: Arms pulled back (low angle)
        """
        elbow_angle = angles['primary']
        
        # State machine for rep detection
        if self.stage == "extended" and elbow_angle < self.config.min_angle + 20:
            # Transitioned from extended to pulled
            self.stage = "pulled"
            return True  # Rep completed
        
        if self.stage == "pulled" and elbow_angle > self.config.max_angle - 20:
            # Transitioned from pulled to extended
            self.stage = "extended"
        
        # Initialize stage
        if self.stage is None:
            self.stage = "extended" if elbow_angle > 120 else "pulled"
        
        return False
    
