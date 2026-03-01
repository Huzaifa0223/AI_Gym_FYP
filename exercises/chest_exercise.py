"""
Chest Exercise Detection (Push-ups, Bench Press)
Tracks chest, shoulder, and arm movement
"""
import pickle
import os
import numpy as np
from typing import Dict, Tuple, List
from exercises.base_exercise import BaseExercise, ExerciseConfig
from utils import calculate_angle_3d

class ChestExercise(BaseExercise):
    """
    Detects chest exercises like push-ups and bench press
    Tracks: Elbow angle, body alignment, depth
    """
    
    def __init__(self, config: ExerciseConfig):
        super().__init__(config)
        self.prev_angle = None
        self.rep_start_time = None
        self.lowest_angle = 180  # Track depth
    
    def load_model(self):
        """Load trained model for chest exercises"""
        if os.path.exists(self.config.model_path):
            try:
                with open(self.config.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"[INFO] Loaded chest exercise model: {self.config.model_path}")
            except Exception as e:
                print(f"[WARNING] Could not load model: {e}")
                self.model = None
        else:
            print(f"[WARNING] Model not found: {self.config.model_path}")
            self.model = None
    
    def get_required_landmarks(self) -> List[int]:
        """
        Required landmarks for chest exercises:
        11, 12: Shoulders
        13, 14: Elbows
        15, 16: Wrists
        23, 24: Hips
        25, 26: Knees
        27, 28: Ankles (needed for tertiary angle: hip->knee->ankle)
        """
        return [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def calculate_angles(self, landmarks) -> Dict[str, float]:
        """
        Calculate angles for chest exercises:
        - Primary  (elbow):          Shoulder -> Elbow -> Wrist
        - Secondary (body alignment): Shoulder -> Hip   -> Knee
        - Tertiary (leg straightness): Hip     -> Knee  -> Ankle  (trained feature)
        """
        try:
            shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
            elbow    = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
            wrist    = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
            hip      = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])
            knee     = np.array([landmarks[26].x, landmarks[26].y, landmarks[26].z])
            ankle    = np.array([landmarks[28].x, landmarks[28].y, landmarks[28].z])

            # Primary: elbow bend depth
            elbow_angle = float(calculate_angle_3d(shoulder, elbow, wrist))

            # Secondary: body/plank alignment
            body_angle = float(calculate_angle_3d(shoulder, hip, knee))

            # Tertiary: leg straightness — matches trained feature hip->knee->ankle
            tertiary_angle = float(calculate_angle_3d(hip, knee, ankle))

            # Track lowest elbow position reached this rep
            if elbow_angle < self.lowest_angle:
                self.lowest_angle = elbow_angle

            # Shoulder and elbow widths for flare check
            left_shoulder  = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
            left_elbow     = np.array([landmarks[13].x, landmarks[13].y, landmarks[13].z])
            right_elbow    = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])

            shoulder_width = float(np.linalg.norm(right_shoulder - left_shoulder))
            elbow_width    = float(np.linalg.norm(right_elbow - left_elbow))

            avg_vis = (
                landmarks[12].visibility + landmarks[14].visibility +
                landmarks[16].visibility + landmarks[24].visibility
            ) / 4

            return {
                'primary':   elbow_angle,
                'secondary': body_angle,
                'tertiary':  tertiary_angle,
                'shoulder_width': shoulder_width,
                'elbow_width': elbow_width,
                'lowest': float(self.lowest_angle),
                'confidence': float(avg_vis)
            }
        except Exception as e:
            print(f"[ERROR] Chest angle calculation failed: {e}")
            return {
                'primary': 0.0, 'secondary': 0.0, 'tertiary': 0.0,
                'shoulder_width': 0.0, 'elbow_width': 0.0,
                'lowest': 180.0, 'confidence': 0.0
            }
    
    def check_form(self, landmarks, angles: Dict[str, float]) -> Tuple[str, Tuple[int, int, int], float]:
        """
        Check chest exercise form using ML model as primary signal.
        Rule-based checks supply specific corrective cues when ML flags bad form.

        ML classes: 0 = bad form, 1 = good form.
        Falls back to pure rule-based when no model is loaded.

        Note: body_angle max is ~180° — the old check of >200 was impossible.
        """
        primary   = angles['primary']    # elbow angle
        secondary = angles['secondary']  # body alignment
        tertiary  = angles.get('tertiary', 0.0)
        shoulder_width = angles.get('shoulder_width', 0.0)
        elbow_width    = angles.get('elbow_width', 0.0)

        # ── ML prediction ────────────────────────────────────────────────
        is_good, quality_score = self._ml_predict_form([primary, secondary, tertiary])

        if is_good is not None:
            if not is_good:
                # Identify the most likely cause of bad form
                if secondary < 160:
                    return "BODY SAGGING", (0, 165, 255), max(30.0, quality_score)
                if secondary > 185:
                    return "HIPS TOO HIGH", (0, 165, 255), max(40.0, quality_score)
                if elbow_width > shoulder_width * 1.5 and shoulder_width > 0:
                    return "ELBOWS TOO WIDE", (255, 200, 0), max(45.0, quality_score)
                if primary > self.config.min_angle + 30:
                    return "GO DEEPER", (255, 165, 0), max(40.0, quality_score)
                return "CHECK YOUR FORM", (0, 165, 255), max(30.0, quality_score)
            else:
                return "GOOD FORM", (0, 255, 0), quality_score

        # ── Rule-based fallback (no model loaded) ────────────────────────
        quality_score = 100.0
        if secondary < 160:
            quality_score -= 30
            return "BODY SAGGING", (0, 165, 255), quality_score
        if secondary > 185:
            quality_score -= 20
            return "HIPS TOO HIGH", (0, 165, 255), quality_score
        if primary > self.config.min_angle + 30:
            quality_score -= 25
            return "GO DEEPER", (255, 165, 0), quality_score
        if elbow_width > shoulder_width * 1.5 and shoulder_width > 0:
            quality_score -= 15
            return "ELBOWS TOO WIDE", (255, 200, 0), quality_score
        return "GOOD FORM", (0, 255, 0), quality_score
    
    def detect_rep(self, landmarks, angles: Dict[str, float]) -> bool:
        """
        Detect complete push-up rep:
        - Start: Arms extended (high angle ~170°)
        - Bottom: Arms bent (low angle ~90° or less)
        - End: Back to extended
        """
        elbow_angle = angles['primary']
        
        # State machine for rep detection
        if self.stage == "up" and elbow_angle < self.config.min_angle + 20:
            # Transitioned from up to down
            self.stage = "down"
            self.lowest_angle = elbow_angle  # Record depth
        
        elif self.stage == "down" and elbow_angle > self.config.max_angle - 20:
            # Transitioned from down to up - rep complete
            self.stage = "up"
            
            # Check if depth was sufficient
            if self.lowest_angle < self.config.min_angle + 30:
                self.lowest_angle = 180  # Reset for next rep
                return True  # Valid rep
            else:
                self.lowest_angle = 180
                return False  # Partial rep
        
        # Initialize stage
        if self.stage is None:
            self.stage = "up" if elbow_angle > 140 else "down"
        
        return False
    
