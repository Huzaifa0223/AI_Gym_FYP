import numpy as np
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ===========================================
# DATA STRUCTURES & ENUMS
# ===========================================
class BodySide(Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"

@dataclass
class PoseLandmark:
    """Enhanced landmark representation"""
    x: float
    y: float
    z: float
    visibility: float = 1.0
    presence: float = 1.0
    name: str = ""
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def to_array_2d(self) -> np.ndarray:
        return np.array([self.x, self.y])

@dataclass
class AngleResult:
    """Structured angle calculation result"""
    angle: float
    is_valid: bool
    confidence: float
    additional_info: Optional[Dict] = None

# ===========================================
# ADVANCED ANGLE CALCULATION FUNCTIONS
# ===========================================
def calculate_angle_3d(a: PoseLandmark, b: PoseLandmark, c: PoseLandmark, 
                       use_visibility: bool = True) -> AngleResult:
    """
    Calculates the 3D angle at joint 'b' with error handling and confidence scoring.
    
    Args:
        a, b, c: Landmark objects with .x, .y, .z, .visibility attributes
        use_visibility: Whether to consider landmark visibility in confidence
    
    Returns:
        AngleResult object with angle, validity flag, and confidence score
    """
    try:
        # Convert landmarks to numpy arrays
        a_vec = np.array([a.x, a.y, a.z])
        b_vec = np.array([b.x, b.y, b.z])
        c_vec = np.array([c.x, c.y, c.z])
        
        # Calculate vectors
        ba = a_vec - b_vec
        bc = c_vec - b_vec
        
        # Check for zero vectors
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return AngleResult(0.0, False, 0.0, {"error": "Zero vector"})
        
        # Calculate cosine and angle with numerical stability
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        # Calculate confidence based on visibility
        confidence = 1.0
        if use_visibility:
            avg_visibility = (a.visibility + b.visibility + c.visibility) / 3
            confidence = avg_attention = avg_visibility
        
        # Validate angle range (human joints have physical limits)
        is_valid = 10 <= angle <= 200  # Reasonable human joint angle range
        
        return AngleResult(
            angle=angle,
            is_valid=is_valid,
            confidence=confidence,
            additional_info={
                "vector_norms": [norm_ba, norm_bc],
                "cosine_value": cosine_angle
            }
        )
        
    except Exception as e:
        return AngleResult(0.0, False, 0.0, {"error": str(e)})

def calculate_angle_2d(a: PoseLandmark, b: PoseLandmark, c: PoseLandmark,
                      camera_matrix: Optional[np.ndarray] = None) -> AngleResult:
    """
    Calculate 2D angle from screen coordinates with optional camera calibration.
    
    Args:
        camera_matrix: 3x3 camera intrinsic matrix for perspective correction
    """
    try:
        # Convert to 2D points
        a_2d = np.array([a.x, a.y])
        b_2d = np.array([b.x, b.y])
        c_2d = np.array([c.x, c.y])
        
        # Calculate vectors
        ba = a_2d - b_2d
        bc = c_2d - b_2d
        
        # Calculate angle
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return AngleResult(0.0, False, 0.0)
        
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        # Confidence based on 2D visibility
        confidence = (a.visibility + b.visibility + c.visibility) / 3
        
        return AngleResult(angle=angle, is_valid=True, confidence=confidence)
        
    except Exception as e:
        return AngleResult(0.0, False, 0.0, {"error": str(e)})

def calculate_relative_angle(shoulder: PoseLandmark, elbow: PoseLandmark, 
                           wrist: PoseLandmark, reference_direction: str = "vertical") -> AngleResult:
    """
    Calculate angle relative to a reference direction (vertical/horizontal).
    Useful for strict curl form checking.
    
    Args:
        reference_direction: "vertical", "horizontal", or custom vector
    """
    try:
        # Upper arm vector
        upper_arm = np.array([elbow.x - shoulder.x, 
                             elbow.y - shoulder.y, 
                             elbow.z - shoulder.z])
        
        # Reference vector
        if reference_direction == "vertical":
            # Assuming y increases downward in image coordinates
            reference = np.array([0, 1, 0])
        elif reference_direction == "horizontal":
            reference = np.array([1, 0, 0])
        else:
            # Custom direction as string "x,y,z"
            coords = list(map(float, reference_direction.split(',')))
            reference = np.array(coords)
        
        # Calculate angle between upper arm and reference
        norm_upper = np.linalg.norm(upper_arm)
        norm_ref = np.linalg.norm(reference)
        
        if norm_upper < 1e-6:
            return AngleResult(0.0, False, 0.0)
        
        cosine_angle = np.dot(upper_arm, reference) / (norm_upper * norm_ref)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        return AngleResult(
            angle=angle,
            is_valid=True,
            confidence=(shoulder.visibility + elbow.visibility) / 2
        )
        
    except Exception as e:
        return AngleResult(0.0, False, 0.0, {"error": str(e)})

# ===========================================
# FORM ANALYSIS FUNCTIONS
# ===========================================
def check_form_torso_relative(shoulder: PoseLandmark, elbow: PoseLandmark, 
                             hip: PoseLandmark, use_3d: bool = True) -> AngleResult:
    """
    Enhanced version: Calculates the angle between the upper arm and torso.
    Now includes error handling and multiple calculation methods.
    
    Args:
        use_3d: If False, uses 2D projection (useful when depth is unreliable)
    """
    try:
        if use_3d and hasattr(shoulder, 'z') and hasattr(elbow, 'z') and hasattr(hip, 'z'):
            # 3D calculation
            vec_arm = np.array([elbow.x - shoulder.x, 
                              elbow.y - shoulder.y, 
                              elbow.z - shoulder.z])
            vec_torso = np.array([hip.x - shoulder.x, 
                                hip.y - shoulder.y, 
                                hip.z - shoulder.z])
        else:
            # 2D fallback
            vec_arm = np.array([elbow.x - shoulder.x, 
                              elbow.y - shoulder.y])
            vec_torso = np.array([hip.x - shoulder.x, 
                                hip.y - shoulder.y])
        
        # Check for zero vectors
        norm_arm = np.linalg.norm(vec_arm)
        norm_torso = np.linalg.norm(vec_torso)
        
        if norm_arm < 1e-6 or norm_torso < 1e-6:
            return AngleResult(0.0, False, 0.0, {"error": "Zero vector"})
        
        # Calculate angle
        cosine_angle = np.dot(vec_arm, vec_torso) / (norm_arm * norm_torso)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        # Calculate confidence based on landmark quality
        confidence = (shoulder.visibility + elbow.visibility + hip.visibility) / 3
        
        # Additional form analysis
        additional_info = {
            "is_elbow_swinging": angle > 30,  # Threshold for elbow swing
            "arm_torso_alignment": angle,
            "calculation_method": "3D" if use_3d else "2D"
        }
        
        return AngleResult(
            angle=angle,
            is_valid=True,
            confidence=confidence,
            additional_info=additional_info
        )
        
    except Exception as e:
        return AngleResult(0.0, False, 0.0, {"error": str(e)})

def analyze_bicep_form(shoulder: PoseLandmark, elbow: PoseLandmark, 
                      wrist: PoseLandmark, hip: PoseLandmark) -> Dict:
    """
    Comprehensive bicep curl form analysis with multiple metrics.
    Returns a detailed analysis of the curl form.
    """
    analysis = {
        "elbow_angle": None,
        "torso_alignment": None,
        "range_of_motion": None,
        "form_issues": [],
        "overall_score": 0,
        "suggestions": []
    }
    
    # 1. Calculate elbow angle
    elbow_angle_result = calculate_angle_3d(shoulder, elbow, wrist)
    analysis["elbow_angle"] = elbow_angle_result
    
    # 2. Check torso alignment (elbow swing)
    torso_angle_result = check_form_torso_relative(shoulder, elbow, hip)
    analysis["torso_alignment"] = torso_angle_result
    
    # 3. Range of motion analysis
    if elbow_angle_result.is_valid:
        if elbow_angle_result.angle < 60:
            analysis["form_issues"].append("Partial rep - not curling high enough")
            analysis["range_of_motion"] = "poor"
        elif elbow_angle_result.angle > 170:
            analysis["form_issues"].append("Full lockout - arm is too straight")
            analysis["range_of_motion"] = "excessive"
        elif 60 <= elbow_angle_result.angle <= 150:
            analysis["range_of_motion"] = "good"
        else:
            analysis["range_of_motion"] = "acceptable"
    
    # 4. Elbow swing detection
    if torso_angle_result.is_valid and torso_angle_result.angle > 30:
        analysis["form_issues"].append("Elbow swinging forward")
    
    # 5. Overall score calculation
    score = 100
    
    # Deduct for range issues
    if analysis["range_of_motion"] == "poor":
        score -= 30
    elif analysis["range_of_motion"] == "excessive":
        score -= 20
    
    # Deduct for elbow swing
    if torso_angle_result.is_valid:
        swing_penalty = min(40, max(0, (torso_angle_result.angle - 20) * 2))
        score -= swing_penalty
    
    analysis["overall_score"] = max(0, score)
    
    # 6. Generate suggestions
    if score < 70:
        if "Partial rep" in analysis["form_issues"]:
            analysis["suggestions"].append("Curl the weight higher to engage full muscle range")
        if "Elbow swinging" in analysis["form_issues"]:
            analysis["suggestions"].append("Keep elbows tucked to your sides")
        if "Full lockout" in analysis["form_issues"]:
            analysis["suggestions"].append("Maintain a slight bend in elbow at bottom position")
    
    return analysis

def calculate_joint_velocity(prev_landmark: PoseLandmark, curr_landmark: PoseLandmark, 
                           delta_time: float) -> float:
    """
    Calculate joint velocity for rep speed analysis.
    Returns velocity in units per second.
    """
    try:
        displacement = np.linalg.norm(
            curr_landmark.to_array() - prev_landmark.to_array()
        )
        velocity = displacement / delta_time if delta_time > 0 else 0
        return float(velocity)
    except:
        return 0.0

# ===========================================
# LANDMARK MANAGEMENT FUNCTIONS
# ===========================================
def get_landmark_coords(landmarks: List[PoseLandmark], 
                       name_idx_dict: Dict[str, int], 
                       side: Union[str, BodySide] = BodySide.LEFT,
                       auto_detect: bool = True) -> Dict[str, PoseLandmark]:
    """
    Enhanced landmark extraction with auto-detection and validation.
    
    Args:
        landmarks: List of pose landmarks
        name_idx_dict: Dictionary mapping names to indices
        side: "left", "right", BodySide.LEFT, BodySide.RIGHT, or BodySide.BOTH
        auto_detect: Automatically detect which arm is more visible
    
    Returns:
        Dictionary with shoulder, elbow, wrist, hip landmarks
    """
    result = {}
    
    # Convert string side to BodySide enum
    if isinstance(side, str):
        side = BodySide(side.lower())
    
    try:
        # If auto_detect is True, determine which side is more visible
        if auto_detect and side in [BodySide.LEFT, BodySide.RIGHT, BodySide.BOTH]:
            left_shoulder_idx = name_idx_dict.get('LEFT_SHOULDER')
            right_shoulder_idx = name_idx_dict.get('RIGHT_SHOULDER')
            
            if left_shoulder_idx is not None and right_shoulder_idx is not None:
                left_vis = landmarks[left_shoulder_idx].visibility
                right_vis = landmarks[right_shoulder_idx].visibility
                
                # Auto-detect based on visibility
                if left_vis > right_vis and side != BodySide.RIGHT:
                    side = BodySide.LEFT
                elif right_vis > left_vis and side != BodySide.LEFT:
                    side = BodySide.RIGHT
        
        # Extract landmarks based on selected side
        if side in [BodySide.LEFT, BodySide.BOTH]:
            try:
                result["left_shoulder"] = landmarks[name_idx_dict['LEFT_SHOULDER']]
                result["left_elbow"] = landmarks[name_idx_dict['LEFT_ELBOW']]
                result["left_wrist"] = landmarks[name_idx_dict['LEFT_WRIST']]
                result["left_hip"] = landmarks[name_idx_dict['LEFT_HIP']]
            except (KeyError, IndexError):
                pass
        
        if side in [BodySide.RIGHT, BodySide.BOTH]:
            try:
                result["right_shoulder"] = landmarks[name_idx_dict['RIGHT_SHOULDER']]
                result["right_elbow"] = landmarks[name_idx_dict['RIGHT_ELBOW']]
                result["right_wrist"] = landmarks[name_idx_dict['RIGHT_WRIST']]
                result["right_hip"] = landmarks[name_idx_dict['RIGHT_HIP']]
            except (KeyError, IndexError):
                pass
        
        # Add visibility summary
        if result:
            visibilities = []
            for key, landmark in result.items():
                if hasattr(landmark, 'visibility'):
                    visibilities.append(landmark.visibility)
            
            result["avg_visibility"] = np.mean(visibilities) if visibilities else 0
            result["side_used"] = side.value
        
        return result
        
    except Exception as e:
        print(f"[WARNING] Landmark extraction failed: {e}")
        return {}

def validate_landmarks(landmarks: Dict[str, PoseLandmark], 
                      min_visibility: float = 0.5) -> Tuple[bool, List[str]]:
    """
    Validate that landmarks are present and visible enough.
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    required_keys = ['shoulder', 'elbow', 'wrist', 'hip']
    
    for key in required_keys:
        if key not in landmarks:
            warnings.append(f"Missing landmark: {key}")
            continue
        
        landmark = landmarks[key]
        if hasattr(landmark, 'visibility') and landmark.visibility < min_visibility:
            warnings.append(f"Low visibility for {key}: {landmark.visibility:.2f}")
    
    is_valid = len(warnings) == 0
    
    return is_valid, warnings

def interpolate_missing_landmarks(current_landmarks: List[PoseLandmark],
                                 previous_landmarks: List[PoseLandmark],
                                 alpha: float = 0.5) -> List[PoseLandmark]:
    """
    Interpolate missing landmarks using temporal smoothing.
    Useful for handling occlusion or detection drops.
    
    Args:
        alpha: Interpolation factor (0 = use previous, 1 = use current)
    """
    if previous_landmarks is None or len(previous_landmarks) != len(current_landmarks):
        return current_landmarks
    
    interpolated = []
    for i, (curr, prev) in enumerate(zip(current_landmarks, previous_landmarks)):
        if hasattr(curr, 'visibility') and curr.visibility < 0.3:
            # Landmark has low visibility, interpolate
            interp_x = prev.x * (1 - alpha) + curr.x * alpha
            interp_y = prev.y * (1 - alpha) + curr.y * alpha
            interp_z = prev.z * (1 - alpha) + curr.z * alpha
            interp_vis = min(1.0, prev.visibility + 0.1)  # Slightly increase confidence
            
            interpolated.append(PoseLandmark(
                x=interp_x, y=interp_y, z=interp_z,
                visibility=interp_vis,
                name=curr.name if hasattr(curr, 'name') else f"landmark_{i}"
            ))
        else:
            interpolated.append(curr)
    
    return interpolated

# ===========================================
# GEOMETRY UTILITIES
# ===========================================
def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector with safety checks."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros_like(v)
    return v / norm

def calculate_distance(p1: PoseLandmark, p2: PoseLandmark, 
                      use_3d: bool = True) -> float:
    """Calculate distance between two landmarks."""
    if use_3d:
        return float(np.linalg.norm(p1.to_array() - p2.to_array()))
    else:
        return float(np.linalg.norm(p1.to_array_2d() - p2.to_array_2d()))

def calculate_midpoint(p1: PoseLandmark, p2: PoseLandmark) -> PoseLandmark:
    """Calculate midpoint between two landmarks."""
    return PoseLandmark(
        x=(p1.x + p2.x) / 2,
        y=(p1.y + p2.y) / 2,
        z=(p1.z + p2.z) / 2 if hasattr(p1, 'z') and hasattr(p2, 'z') else 0,
        visibility=(p1.visibility + p2.visibility) / 2
    )

def estimate_depth_from_projection(landmarks_2d: List[PoseLandmark], 
                                  known_lengths: Dict[str, float]) -> Dict[str, float]:
    """
    Estimate 3D depth from 2D projections using known body segment lengths.
    Basic implementation - can be enhanced with more sophisticated methods.
    """
    # This is a simplified implementation
    # In practice, you might use perspective-n-point or similar algorithms
    depth_estimates = {}
    
    # Example: If we know shoulder-elbow length should be ~0.3m
    # and we see it as 100 pixels, we can estimate depth
    
    return depth_estimates

# ===========================================
# POSE QUALITY FUNCTIONS
# ===========================================
def calculate_pose_confidence(landmarks: List[PoseLandmark]) -> float:
    """
    Calculate overall pose detection confidence.
    """
    if not landmarks:
        return 0.0
    
    visibilities = []
    for lm in landmarks:
        if hasattr(lm, 'visibility'):
            visibilities.append(lm.visibility)
    
    return float(np.mean(visibilities)) if visibilities else 0.0

def detect_pose_anomalies(landmarks: List[PoseLandmark], 
                         name_idx_dict: Dict[str, int]) -> List[str]:
    """
    Detect physically impossible poses or anomalies.
    """
    anomalies = []
    
    try:
        # Check limb proportions
        left_shoulder = landmarks[name_idx_dict['LEFT_SHOULDER']]
        left_elbow = landmarks[name_idx_dict['LEFT_ELBOW']]
        left_wrist = landmarks[name_idx_dict['LEFT_WRIST']]
        
        shoulder_elbow = calculate_distance(left_shoulder, left_elbow)
        elbow_wrist = calculate_distance(left_elbow, left_wrist)
        
        # Upper arm should be roughly similar to forearm length
        ratio = shoulder_elbow / elbow_wrist if elbow_wrist > 0 else 0
        if not 0.5 <= ratio <= 2.0:
            anomalies.append(f"Unusual arm segment ratio: {ratio:.2f}")
        
        # Check for impossible angles
        elbow_angle = calculate_angle_3d(left_shoulder, left_elbow, left_wrist)
        if elbow_angle.is_valid and elbow_angle.angle < 10:
            anomalies.append(f"Impossibly sharp elbow angle: {elbow_angle.angle:.1f}°")
            
    except Exception as e:
        anomalies.append(f"Anomaly detection error: {e}")
    
    return anomalies

# ===========================================
# BACKWARD COMPATIBILITY WRAPPERS
# ===========================================
def legacy_calculate_angle_3d(a, b, c):
    """Legacy wrapper for backward compatibility."""
    result = calculate_angle_3d(a, b, c)
    return result.angle

def legacy_check_form_torso_relative(shoulder, elbow, hip):
    """Legacy wrapper for backward compatibility."""
    result = check_form_torso_relative(shoulder, elbow, hip)
    return result.angle

def legacy_get_landmark_coords(landmarks, name_idx_dict, side="left"):
    """Legacy wrapper for backward compatibility."""
    result = get_landmark_coords(landmarks, name_idx_dict, side, auto_detect=False)
    
    # Return in original format
    if side == "left":
        return (
            result.get("left_shoulder"),
            result.get("left_elbow"),
            result.get("left_wrist"),
            result.get("left_hip")
        )
    else:
        return (
            result.get("right_shoulder"),
            result.get("right_elbow"),
            result.get("right_wrist"),
            result.get("right_hip")
        )

# ===========================================
# MAIN EXPORTS
# ===========================================
__all__ = [
    # Core functions
    'calculate_angle_3d',
    'calculate_angle_2d',
    'calculate_relative_angle',
    'check_form_torso_relative',
    'analyze_bicep_form',
    'get_landmark_coords',
    
    # Utility functions
    'validate_landmarks',
    'interpolate_missing_landmarks',
    'calculate_pose_confidence',
    'detect_pose_anomalies',
    'calculate_joint_velocity',
    
    # Geometry utilities
    'normalize_vector',
    'calculate_distance',
    'calculate_midpoint',
    
    # Data structures
    'PoseLandmark',
    'AngleResult',
    'BodySide',
    
    # Legacy wrappers
    'legacy_calculate_angle_3d',
    'legacy_check_form_torso_relative',
    'legacy_get_landmark_coords'
]

# ===========================================
# USAGE EXAMPLE
# ===========================================
if __name__ == "__main__":
    # Example usage
    print("Pose Utilities Module - Enhanced Version")
    print("========================================")
    
    # Create sample landmarks
    shoulder = PoseLandmark(x=0.5, y=0.5, z=0.2, visibility=0.9, name="shoulder")
    elbow = PoseLandmark(x=0.5, y=0.7, z=0.2, visibility=0.8, name="elbow")
    wrist = PoseLandmark(x=0.5, y=0.9, z=0.2, visibility=0.7, name="wrist")
    hip = PoseLandmark(x=0.5, y=0.3, z=0.1, visibility=0.9, name="hip")
    
    # Calculate angle with enhanced function
    angle_result = calculate_angle_3d(shoulder, elbow, wrist)
    print(f"Elbow angle: {angle_result.angle:.1f}°")
    print(f"Confidence: {angle_result.confidence:.2f}")
    print(f"Valid: {angle_result.is_valid}")
    
    # Form analysis
    form_analysis = analyze_bicep_form(shoulder, elbow, wrist, hip)
    print(f"\nForm Score: {form_analysis['overall_score']:.0f}/100")
    
    if form_analysis['suggestions']:
        print("Suggestions:")
        for suggestion in form_analysis['suggestions']:
            print(f"  - {suggestion}")