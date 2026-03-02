"""
utils package — shared math and visual helpers for AI Gym Trainer.

Re-exports the core angle functions from angle_utils so that existing
imports like ``from utils import calculate_angle_3d`` continue to work
after the refactor.
"""
from utils.angle_utils import calculate_angle, calculate_angle_3d

__all__ = ["calculate_angle", "calculate_angle_3d"]
