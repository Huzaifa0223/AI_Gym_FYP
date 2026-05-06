"""
Automatic Body-Part Classification

Outputs **body-part categories** — ``arms``, ``back``, ``chest``,
``legs``, ``shoulders``, ``unknown`` — based on coarse landmark patterns.
These are NOT specific exercise keys. Specific-exercise selection happens
one layer up via :data:`core.exercise_mapping.BODY_PART_TO_EXERCISE`,
which maps each body part to a single rule-engine-keyed exercise
(``arms`` → ``bicep_curl``, ``back`` → ``bent_over_row``,
``chest`` → ``push_up``). Body parts with no mapped exercise (``legs``,
``shoulders``, ``unknown``) flow to the project's ``NullRuleEngine``
fallback, preserving graceful degradation.
"""
import numpy as np
from typing import Dict, Tuple, Optional
import pickle
import os

class ExerciseClassifier:
    """
    Automatically detects which exercise is being performed
    Based on body keypoint positions and movement patterns
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.exercise_history = []  # Track last N frames
        self.confidence_threshold = 0.7
        
        # Exercise signatures - unique movement patterns
        self.exercise_signatures = {
            'back': {
                'primary_joints': [11, 12, 13, 14, 15, 16],  # Shoulders, elbows, wrists
                'movement_plane': 'horizontal',  # Pull/push movements
                'key_angle': 'elbow_shoulder',
                'torso_position': 'bent_forward'
            },
            'chest': {
                'primary_joints': [11, 12, 13, 14, 15, 16, 23, 24],
                'movement_plane': 'vertical',  # Up/down movements
                'key_angle': 'elbow_body',
                'torso_position': 'prone_or_upright'
            },
            'legs': {
                'primary_joints': [23, 24, 25, 26, 27, 28],  # Hips, knees, ankles
                'movement_plane': 'vertical',
                'key_angle': 'knee_hip',
                'torso_position': 'upright'
            },
            'shoulders': {
                'primary_joints': [11, 12, 13, 14, 15, 16],
                'movement_plane': 'vertical',  # Overhead movements
                'key_angle': 'shoulder_arm',
                'torso_position': 'upright'
            },
            'arms': {
                'primary_joints': [11, 12, 13, 14, 15, 16],
                'movement_plane': 'sagittal',  # Curl movements
                'key_angle': 'elbow',
                'torso_position': 'upright'
            }
        }
    
    def detect_exercise_type(self, landmarks, frame_count: int = 1) -> Tuple[str, float]:
        """
        Detect which exercise is being performed
        
        Args:
            landmarks: MediaPipe pose landmarks
            frame_count: Current frame number
            
        Returns:
            (exercise_type, confidence)
        """
        if not landmarks:
            return 'unknown', 0.0
        
        # Extract movement features
        features = self._extract_movement_features(landmarks)
        
        # Calculate scores for each exercise
        scores = {}
        for exercise, signature in self.exercise_signatures.items():
            score = self._calculate_exercise_score(features, signature)
            scores[exercise] = score
        
        # Get best match
        best_exercise = max(scores, key=lambda x: scores[x])
        confidence = scores[best_exercise]
        
        # Add to history for stability
        self.exercise_history.append((best_exercise, confidence))
        if len(self.exercise_history) > 10:
            self.exercise_history.pop(0)
        
        # Use voting from recent frames for stability
        if len(self.exercise_history) >= 5:
            exercise, conf = self._get_stable_prediction()
            return exercise, conf
        
        return best_exercise, confidence
    
    def _extract_movement_features(self, landmarks) -> Dict:
        """Extract key features from pose landmarks"""
        try:
            # Get key joint positions
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
            left_elbow = np.array([landmarks[13].x, landmarks[13].y, landmarks[13].z])
            right_elbow = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
            left_wrist = np.array([landmarks[15].x, landmarks[15].y, landmarks[15].z])
            right_wrist = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
            left_hip = np.array([landmarks[23].x, landmarks[23].y, landmarks[23].z])
            right_hip = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])
            left_knee = np.array([landmarks[25].x, landmarks[25].y, landmarks[25].z])
            right_knee = np.array([landmarks[26].x, landmarks[26].y, landmarks[26].z])
            
            # Calculate key angles
            elbow_angle_r = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
            knee_angle_r = self._calculate_angle(right_hip, right_knee, np.array([landmarks[28].x, landmarks[28].y, landmarks[28].z]))
            
            # Torso angle (hip to shoulder)
            torso_angle = self._calculate_angle(right_hip, right_shoulder, right_elbow)
            
            # Arm elevation (how high arms are)
            arm_elevation = right_wrist[1] - right_shoulder[1]  # Y-axis difference
            
            # Hip to shoulder ratio (bent vs upright)
            torso_vertical = abs(right_hip[1] - right_shoulder[1])
            
            # Movement plane detection
            arm_horizontal_movement = abs(right_wrist[0] - right_shoulder[0])  # X-axis
            arm_vertical_movement = abs(right_wrist[1] - right_shoulder[1])    # Y-axis
            
            features = {
                'elbow_angle': elbow_angle_r,
                'knee_angle': knee_angle_r,
                'torso_angle': torso_angle,
                'arm_elevation': arm_elevation,
                'torso_vertical': torso_vertical,
                'arm_horizontal_movement': arm_horizontal_movement,
                'arm_vertical_movement': arm_vertical_movement,
                'wrist_height': right_wrist[1],
                'shoulder_height': right_shoulder[1],
                'hip_height': right_hip[1]
            }
            
            return features
            
        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {e}")
            return {}
    
    def _calculate_exercise_score(self, features: Dict, signature: Dict) -> float:
        """Calculate how well features match exercise signature"""
        if not features:
            return 0.0
        
        score = 0.0
        
        # Check torso position
        torso_vertical = features.get('torso_vertical', 0)
        torso_angle = features.get('torso_angle', 0)
        
        if signature['torso_position'] == 'bent_forward':
            # Back exercises: torso should be bent forward
            if torso_angle < 140:  # Bent forward
                score += 30
        elif signature['torso_position'] == 'upright':
            # Legs, shoulders, arms: upright torso
            if torso_angle > 160:  # Standing straight
                score += 30
        
        # Check movement plane
        h_movement = features.get('arm_horizontal_movement', 0)
        v_movement = features.get('arm_vertical_movement', 0)
        
        if signature['movement_plane'] == 'horizontal':
            # Back: arms move horizontally (pulling)
            if h_movement > v_movement:
                score += 25
        elif signature['movement_plane'] == 'vertical':
            # Chest, shoulders: arms move vertically
            if v_movement > h_movement:
                score += 25
        
        # Check key angles
        elbow_angle = features.get('elbow_angle', 0)
        knee_angle = features.get('knee_angle', 0)
        
        if signature['key_angle'] == 'elbow':
            # Arms/biceps: focus on elbow
            if 40 < elbow_angle < 170:  # Active range
                score += 25
        elif signature['key_angle'] == 'knee_hip':
            # Legs: focus on knee
            if 60 < knee_angle < 170:
                score += 25
        elif signature['key_angle'] in ['elbow_shoulder', 'shoulder_arm']:
            # Back/shoulders: combined movement
            if 60 < elbow_angle < 170:
                score += 20
        
        # Arm elevation check
        arm_elevation = features.get('arm_elevation', 0)
        
        if signature['torso_position'] == 'upright' and signature['movement_plane'] == 'vertical':
            # Shoulders: arms should go high
            if arm_elevation < -0.2:  # Arms above shoulders
                score += 20
        
        return min(score, 100.0) / 100.0  # Normalize to 0-1
    
    def _get_stable_prediction(self) -> Tuple[str, float]:
        """Get stable prediction from recent history"""
        if not self.exercise_history:
            return 'unknown', 0.0
        
        # Count votes for each exercise
        votes = {}
        for exercise, conf in self.exercise_history[-10:]:
            if conf > 0.5:  # Only count confident predictions
                votes[exercise] = votes.get(exercise, 0) + 1
        
        if not votes:
            return 'unknown', 0.0
        
        # Get majority vote
        best_exercise = max(votes, key=lambda x: votes[x])
        confidence = votes[best_exercise] / len(self.exercise_history[-10:])
        
        return best_exercise, confidence
    
    def _calculate_angle(self, a, b, c):
        """Calculate angle between three 3D points"""
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
    
    def reset(self):
        """Reset classifier history"""
        self.exercise_history.clear()


# Quick rule-based detection for immediate use
def quick_detect_exercise(landmarks) -> str:
    """
    Quick rule-based exercise detection
    Returns: exercise type (back/chest/legs/shoulders/arms)
    """
    if not landmarks:
        return 'unknown'
    
    try:
        # Get key positions
        right_wrist = landmarks[16]
        right_elbow = landmarks[14]
        right_shoulder = landmarks[12]
        right_hip = landmarks[24]
        right_knee = landmarks[26]
        
        # Calculate torso angle (hip to shoulder vertical)
        torso_vertical = abs(right_hip.y - right_shoulder.y)
        
        # Calculate if arms are active (wrist movement)
        arm_range = abs(right_wrist.y - right_shoulder.y)
        
        # Calculate if legs are active (knee bending)
        leg_range = abs(right_knee.y - right_hip.y)
        
        # Decision tree
        
        # LEGS: Knee is actively bending
        if leg_range < 0.3 and right_knee.y < right_hip.y:
            return 'legs'
        
        # BACK: Torso bent forward + arms moving horizontally
        if torso_vertical < 0.4 and arm_range > 0.1:
            return 'back'
        
        # SHOULDERS: Arms elevated high above shoulders
        if right_wrist.y < right_shoulder.y - 0.2:
            return 'shoulders'
        
        # CHEST: Prone position or arms at chest level
        if torso_vertical < 0.3 or (right_elbow.y > right_shoulder.y):
            return 'chest'
        
        # ARMS/BICEPS: Upright with elbow flexion
        if torso_vertical > 0.4 and arm_range > 0.1:
            return 'arms'
        
        return 'unknown'
        
    except Exception as e:
        print(f"[ERROR] Quick detection failed: {e}")
        return 'unknown'
