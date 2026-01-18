"""
Chest Exercise Detection (Push-ups, Bench Press)
Tracks chest, shoulder, and arm movement
"""
import pickle
import os
import numpy as np
from typing import Dict, Tuple, List
from exercises.base_exercise import BaseExercise, ExerciseConfig, ExerciseResult

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
        25, 26: Knees (for push-up form)
        """
        return [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
    
    def calculate_angles(self, landmarks) -> Dict[str, float]:
        """
        Calculate angles for chest exercises:
        - Elbow angle (primary): Shoulder -> Elbow -> Wrist
        - Body alignment (secondary): Shoulder -> Hip -> Knee
        - Chest depth: How low the body goes
        """
        try:
            # Use right side
            shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
            elbow = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
            wrist = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
            hip = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])
            knee = np.array([landmarks[26].x, landmarks[26].y, landmarks[26].z])
            
            # Primary angle: Elbow angle (push-up depth)
            elbow_angle = self._calculate_angle_3d(shoulder, elbow, wrist)
            
            # Secondary angle: Body alignment (should be straight)
            body_angle = self._calculate_angle_3d(shoulder, hip, knee)
            
            # Track lowest point reached
            if elbow_angle < self.lowest_angle:
                self.lowest_angle = elbow_angle
            
            # Shoulder width for form checking
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
            left_elbow = np.array([landmarks[13].x, landmarks[13].y, landmarks[13].z])
            right_elbow = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
            
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
            elbow_width = np.linalg.norm(right_elbow - left_elbow)
            
            return {
                'primary': elbow_angle,
                'secondary': body_angle,
                'shoulder_width': shoulder_width,
                'elbow_width': elbow_width,
                'lowest': self.lowest_angle,
                'confidence': 0.95
            }
        except Exception as e:
            print(f"[ERROR] Angle calculation failed: {e}")
            return {
                'primary': 0.0,
                'secondary': 0.0,
                'shoulder_width': 0.0,
                'elbow_width': 0.0,
                'lowest': 180,
                'confidence': 0.0
            }
    
    def check_form(self, landmarks, angles: Dict[str, float]) -> Tuple[str, Tuple[int, int, int], float]:
        """
        Check chest exercise form:
        - Body should be straight (plank position for push-ups)
        - Elbows should not flare too wide
        - Proper depth achieved
        """
        elbow_angle = angles['primary']
        body_angle = angles['secondary']
        quality_score = 100.0
        
        # Check body alignment (should be close to 180° for straight body)
        if body_angle < 160:
            quality_score -= 30
            return "⚠ BODY SAGGING", (0, 165, 255), quality_score
        
        if body_angle > 200:
            quality_score -= 20
            return "⚠ HIPS TOO HIGH", (0, 165, 255), quality_score
        
        # Check depth
        if elbow_angle > self.config.min_angle + 30:
            quality_score -= 25
            return "⚠ GO DEEPER", (255, 165, 0), quality_score
        
        # Check elbow flare
        shoulder_width = angles.get('shoulder_width', 0)
        elbow_width = angles.get('elbow_width', 0)
        
        if elbow_width > shoulder_width * 1.5:
            quality_score -= 15
            return "⚠ ELBOWS TOO WIDE", (255, 200, 0), quality_score
        
        # Good form
        return "✓ GOOD FORM", (0, 255, 0), quality_score
    
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
    
    def _calculate_angle_3d(self, a, b, c):
        """Calculate angle between three 3D points"""
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
