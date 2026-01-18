"""
Base Exercise Class - Foundation for all exercise types
Supports age-specific variations and form detection
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np

@dataclass
class ExerciseConfig:
    """Configuration for age-specific exercise parameters"""
    name: str
    age_group: str  # 'children', 'adult', 'senior'
    min_angle: float
    max_angle: float
    min_rep_time: float
    max_rep_time: float
    strict_mode_threshold: float
    model_path: str
    
    # Age-specific adjustments
    flexibility_tolerance: float = 10.0  # Extra degrees allowed
    speed_tolerance: float = 0.5  # Extra seconds allowed
    form_strictness: float = 1.0  # Multiplier for form checking

# Age-specific configurations
AGE_CONFIGS = {
    'children': {
        'bicep': ExerciseConfig(
            name='bicep_curl',
            age_group='children',
            min_angle=60,  # More flexible for children
            max_angle=160,
            min_rep_time=1.5,
            max_rep_time=5.0,
            strict_mode_threshold=30,
            model_path='data/models/bicep_children.pkl',
            flexibility_tolerance=15.0,
            speed_tolerance=1.0,
            form_strictness=0.7
        ),
        'back': ExerciseConfig(
            name='back_row',
            age_group='children',
            min_angle=40,
            max_angle=160,
            min_rep_time=1.5,
            max_rep_time=5.0,
            strict_mode_threshold=25,
            model_path='data/models/back_children.pkl',
            flexibility_tolerance=15.0,
            speed_tolerance=1.0,
            form_strictness=0.7
        ),
        'chest': ExerciseConfig(
            name='push_up',
            age_group='children',
            min_angle=70,  # Modified push-ups allowed
            max_angle=170,
            min_rep_time=1.5,
            max_rep_time=5.0,
            strict_mode_threshold=25,
            model_path='data/models/chest_children.pkl',
            flexibility_tolerance=15.0,
            speed_tolerance=1.0,
            form_strictness=0.7
        )
    },
    'adult': {
        'bicep': ExerciseConfig(
            name='bicep_curl',
            age_group='adult',
            min_angle=50,
            max_angle=170,
            min_rep_time=1.0,
            max_rep_time=4.0,
            strict_mode_threshold=25,
            model_path='data/models/bicep_adult.pkl',
            flexibility_tolerance=10.0,
            speed_tolerance=0.5,
            form_strictness=1.0
        ),
        'back': ExerciseConfig(
            name='back_row',
            age_group='adult',
            min_angle=30,
            max_angle=170,
            min_rep_time=1.0,
            max_rep_time=4.0,
            strict_mode_threshold=20,
            model_path='data/models/back_adult.pkl',
            flexibility_tolerance=10.0,
            speed_tolerance=0.5,
            form_strictness=1.0
        ),
        'chest': ExerciseConfig(
            name='push_up',
            age_group='adult',
            min_angle=50,
            max_angle=180,
            min_rep_time=1.0,
            max_rep_time=4.0,
            strict_mode_threshold=20,
            model_path='data/models/chest_adult.pkl',
            flexibility_tolerance=10.0,
            speed_tolerance=0.5,
            form_strictness=1.0
        )
    },
    'senior': {
        'bicep': ExerciseConfig(
            name='bicep_curl',
            age_group='senior',
            min_angle=70,  # More conservative range
            max_angle=150,
            min_rep_time=2.0,
            max_rep_time=6.0,
            strict_mode_threshold=35,
            model_path='data/models/bicep_senior.pkl',
            flexibility_tolerance=20.0,
            speed_tolerance=2.0,
            form_strictness=0.6
        ),
        'back': ExerciseConfig(
            name='back_row',
            age_group='senior',
            min_angle=50,
            max_angle=150,
            min_rep_time=2.0,
            max_rep_time=6.0,
            strict_mode_threshold=30,
            model_path='data/models/back_senior.pkl',
            flexibility_tolerance=20.0,
            speed_tolerance=2.0,
            form_strictness=0.6
        ),
        'chest': ExerciseConfig(
            name='push_up',
            age_group='senior',
            min_angle=80,  # Wall push-ups or modified
            max_angle=160,
            min_rep_time=2.0,
            max_rep_time=6.0,
            strict_mode_threshold=30,
            model_path='data/models/chest_senior.pkl',
            flexibility_tolerance=20.0,
            speed_tolerance=2.0,
            form_strictness=0.6
        )
    }
}

@dataclass
class ExerciseResult:
    """Standardized result format for all exercises"""
    counter: int
    feedback: str
    color: Tuple[int, int, int]
    primary_angle: float  # Main angle being tracked
    secondary_angle: float  # Supporting angle
    stage: str
    confidence: float
    is_valid_rep: bool
    rep_quality: float  # 0-100 score
    exercise_type: str
    age_group: str
    model_missing: bool = False

class BaseExercise(ABC):
    """Abstract base class for all exercise detection"""
    
    def __init__(self, config: ExerciseConfig):
        self.config = config
        self.counter = 0
        self.stage = "down"
        self.model = None
        self.load_model()
    
    @abstractmethod
    def load_model(self):
        """Load the ML model for this exercise"""
        pass
    
    @abstractmethod
    def get_required_landmarks(self) -> List[int]:
        """Return list of MediaPipe landmark indices needed"""
        pass
    
    @abstractmethod
    def calculate_angles(self, landmarks) -> Dict[str, float]:
        """Calculate relevant angles from landmarks"""
        pass
    
    @abstractmethod
    def check_form(self, landmarks, angles: Dict[str, float]) -> Tuple[str, Tuple[int, int, int], float]:
        """Check exercise form and return (feedback, color, quality_score)"""
        pass
    
    @abstractmethod
    def detect_rep(self, landmarks, angles: Dict[str, float]) -> bool:
        """Detect if a valid rep has been completed"""
        pass
    
    def process_frame(self, landmarks) -> ExerciseResult:
        """
        Main processing method - called for each frame
        Returns standardized ExerciseResult
        """
        if not landmarks:
            return ExerciseResult(
                counter=self.counter,
                feedback="NO POSE DETECTED",
                color=(128, 128, 128),
                primary_angle=0.0,
                secondary_angle=0.0,
                stage=self.stage,
                confidence=0.0,
                is_valid_rep=False,
                rep_quality=0.0,
                exercise_type=self.config.name,
                age_group=self.config.age_group,
                model_missing=self.model is None
            )
        
        # Calculate angles specific to this exercise
        angles = self.calculate_angles(landmarks)
        
        # Check form
        feedback, color, quality = self.check_form(landmarks, angles)
        
        # Detect rep completion
        is_valid_rep = self.detect_rep(landmarks, angles)
        if is_valid_rep:
            self.counter += 1
        
        return ExerciseResult(
            counter=self.counter,
            feedback=feedback,
            color=color,
            primary_angle=angles.get('primary', 0.0),
            secondary_angle=angles.get('secondary', 0.0),
            stage=self.stage,
            confidence=angles.get('confidence', 0.0),
            is_valid_rep=is_valid_rep,
            rep_quality=quality,
            exercise_type=self.config.name,
            age_group=self.config.age_group,
            model_missing=self.model is None
        )
    
    def reset(self):
        """Reset exercise counter and state"""
        self.counter = 0
        self.stage = "down"

def get_exercise_config(exercise_type: str, age: int) -> ExerciseConfig:
    """
    Get appropriate exercise configuration based on age
    
    Args:
        exercise_type: 'bicep', 'back', or 'chest'
        age: User's age in years
    
    Returns:
        ExerciseConfig for the specified exercise and age group
    """
    if age < 16:
        age_group = 'children'
    elif age < 60:
        age_group = 'adult'
    else:
        age_group = 'senior'
    
    return AGE_CONFIGS[age_group][exercise_type]
