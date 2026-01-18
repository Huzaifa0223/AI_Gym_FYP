"""
Back Exercise Detection (Bent Over Rows, Lat Pulldowns)
Tracks back, shoulder, and arm movement
"""
import pickle
import os
import numpy as np
from typing import Dict, Tuple, List
from exercises.base_exercise import BaseExercise, ExerciseConfig, ExerciseResult

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
        - Elbow angle (primary): Shoulder -> Elbow -> Wrist
        - Back angle (secondary): Hip -> Shoulder -> Elbow
        - Torso angle: For posture check
        """
        try:
            # Use right side (can be adjusted)
            shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
            elbow = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
            wrist = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
            hip = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])
            
            # Primary angle: Elbow angle (arm pull)
            elbow_angle = self._calculate_angle_3d(shoulder, elbow, wrist)
            
            # Secondary angle: Back/torso angle
            back_angle = self._calculate_angle_3d(hip, shoulder, elbow)
            
            # Shoulder retraction (check if shoulders are pulled back)
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
            
            return {
                'primary': elbow_angle,
                'secondary': back_angle,
                'shoulder_width': shoulder_width,
                'confidence': 0.95  # Based on landmark visibility
            }
        except Exception as e:
            print(f"[ERROR] Angle calculation failed: {e}")
            return {
                'primary': 0.0,
                'secondary': 0.0,
                'shoulder_width': 0.0,
                'confidence': 0.0
            }
    
    def check_form(self, landmarks, angles: Dict[str, float]) -> Tuple[str, Tuple[int, int, int], float]:
        """
        Check back exercise form:
        - Back should be relatively straight (not rounded)
        - Elbows should pull back, not flare out
        - Controlled movement
        """
        elbow_angle = angles['primary']
        back_angle = angles['secondary']
        quality_score = 100.0
        
        # Check back position
        if back_angle < 30:
            return "⚠ BACK TOO ROUNDED", (0, 165, 255), 60.0
        
        # Check elbow angle range
        if elbow_angle < self.config.min_angle:
            quality_score -= 20
            return "⚠ PARTIAL REP", (0, 165, 255), quality_score
        
        # Check for proper form
        if elbow_angle > self.config.max_angle:
            quality_score -= 15
            return "⚠ EXTEND LESS", (255, 165, 0), quality_score
        
        # Good form
        return "✓ GOOD FORM", (0, 255, 0), quality_score
    
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
    
    def _calculate_angle_3d(self, a, b, c):
        """Calculate angle between three 3D points"""
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
