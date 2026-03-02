import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math

# ===========================================
# COLOR PALETTES
# ===========================================
class ColorPalette(Enum):
    FITNESS = "fitness"
    MEDICAL = "medical"
    GAMING = "gaming"
    MINIMAL = "minimal"

THEME_COLORS = {
    ColorPalette.FITNESS: {
        "primary": (0, 200, 0), "secondary": (255, 150, 0), "accent": (0, 100, 255),
        "background": (40, 40, 40), "text": (255, 255, 255),
        "warning": (0, 100, 255), "error": (0, 0, 255), "success": (0, 255, 150)
    },
    ColorPalette.MEDICAL: {
        "primary": (255, 255, 255), "secondary": (200, 200, 255), "accent": (255, 200, 200),
        "background": (30, 30, 50), "text": (240, 240, 240),
        "warning": (0, 200, 255), "error": (0, 0, 200), "success": (150, 255, 150)
    },
    ColorPalette.GAMING: {
        "primary": (255, 0, 255), "secondary": (0, 255, 255), "accent": (255, 255, 0),
        "background": (20, 20, 40), "text": (255, 220, 100),
        "warning": (255, 100, 0), "error": (255, 0, 100), "success": (100, 255, 100)
    },
    ColorPalette.MINIMAL: {
        "primary": (150, 150, 150), "secondary": (100, 100, 100), "accent": (200, 200, 200),
        "background": (10, 10, 10), "text": (220, 220, 220),
        "warning": (200, 150, 0), "error": (150, 0, 0), "success": (0, 150, 0)
    }
}

@dataclass
class UIPosition:
    x: int
    y: int
    width: int = 0
    height: int = 0

@dataclass
class DisplayMetrics:
    elbow_angle: float
    torso_angle: float
    torso_threshold: float
    fps: float = 0.0
    form_quality: float = 0.0
    rep_count: int = 0
    stage: str = ""
    arm_side: str = ""
    confidence: float = 0.0
    additional_metrics: Dict = None

# ===========================================
# SELF-CORRECTING VISUALIZER
# ===========================================
class VisualFeedback:
    def __init__(self, theme: ColorPalette = ColorPalette.FITNESS, 
                 resolution: Tuple[int, int] = (1280, 720)):
        self.theme = theme
        self.colors = THEME_COLORS[theme]
        
        # --- CRITICAL FIX: Initialize Scale Factor ---
        self.resolution = resolution
        self.scale_factor = resolution[0] / 1280.0
        
        # Initialize with defaults
        self._recalculate_layout(resolution[0], resolution[1])
        
    def _recalculate_layout(self, w, h):
        """Re-generates fonts and boxes based on CURRENT image size"""
        self.resolution = (w, h)
        # Calculate scale: 1280px is "1.0", 640px is "0.5"
        self.scale_factor = w / 1280.0
        sf = self.scale_factor
        
        # Define Fonts
        self.fonts = {
            "large": {"face": cv2.FONT_HERSHEY_SIMPLEX, "scale": 2.0 * sf, "thickness": max(2, int(4 * sf))},
            "medium": {"face": cv2.FONT_HERSHEY_SIMPLEX, "scale": 1.0 * sf, "thickness": max(1, int(2 * sf))},
            "small": {"face": cv2.FONT_HERSHEY_PLAIN, "scale": 1.5 * sf, "thickness": 1},
            "bold": {"face": cv2.FONT_HERSHEY_DUPLEX, "scale": 1.2 * sf, "thickness": max(1, int(3 * sf))}
        }
        
        # Define Boxes
        margin = int(20 * sf)
        self.layouts = {
            "default": {
                "status_box": UIPosition(margin, margin, int(350 * sf), int(120 * sf)),
                "debug_panel": UIPosition(margin, h - int(150 * sf) - margin, int(400 * sf), int(130 * sf)),
                "stats_panel": UIPosition(w - int(400 * sf) - margin, int(380 * sf), int(200 * sf)),
                "progress_bar": UIPosition(w // 2 - int(150 * sf), int(50 * sf), int(300 * sf), int(25 * sf)),
                "fps_display": UIPosition(w - int(120 * sf), int(40 * sf), int(100 * sf), int(30 * sf))
            }
        }

    def _check_resize(self, img):
        """Self-Correction: If image size changes, update layout immediately"""
        h, w = img.shape[:2]
        if w != self.resolution[0] or h != self.resolution[1]:
            self._recalculate_layout(w, h)

    def draw_status_box(self, img: np.ndarray, counter: int, feedback: str, 
                        feedback_color: Tuple) -> np.ndarray:
        self._check_resize(img)
        layout = self.layouts["default"]
        box = layout["status_box"]
        s = self.scale_factor
        
        # Draw Box
        cv2.rectangle(img, (box.x, box.y), (box.x + box.width, box.y + box.height), feedback_color, cv2.FILLED)
        cv2.rectangle(img, (box.x, box.y), (box.x + box.width, box.y + box.height), self.colors["text"], 2)
        
        # Draw Counter
        text_y = box.y + int(50 * s)
        cv2.putText(img, f"REPS: {counter}", (box.x + int(20 * s), text_y), 
                   self.fonts["large"]["face"], self.fonts["large"]["scale"], 
                   self.colors["text"], self.fonts["large"]["thickness"])
        
        # Draw Feedback
        text_y += int(50 * s)
        cv2.putText(img, str(feedback), (box.x + int(20 * s), text_y), 
                   self.fonts["medium"]["face"], self.fonts["medium"]["scale"], 
                   self.colors["text"], self.fonts["medium"]["thickness"])
        return img

    def draw_progress_bar(self, img: np.ndarray, value: float, max_value: float = 100,
                          label: str = "Progress") -> np.ndarray:
        self._check_resize(img)
        layout = self.layouts["default"]
        bar = layout["progress_bar"]
        s = self.scale_factor
        
        percentage = min(1.0, max(0.0, value / max_value))
        fill_width = int(bar.width * percentage)
        
        cv2.rectangle(img, (bar.x, bar.y), (bar.x + bar.width, bar.y + bar.height), self.colors["background"], cv2.FILLED)
        cv2.rectangle(img, (bar.x, bar.y), (bar.x + fill_width, bar.y + bar.height), self.colors["success"], cv2.FILLED)
        cv2.rectangle(img, (bar.x, bar.y), (bar.x + bar.width, bar.y + bar.height), self.colors["text"], 2)
        cv2.putText(img, f"{label}: {int(value)}%", (bar.x, bar.y - int(10 * s)), 
                   self.fonts["small"]["face"], self.fonts["small"]["scale"], self.colors["text"], 1)
        return img

    def draw_fps(self, img: np.ndarray, fps: float) -> np.ndarray:
        self._check_resize(img)
        layout = self.layouts["default"]
        pos = layout["fps_display"]
        s = self.scale_factor
        cv2.putText(img, f"FPS: {int(fps)}", (pos.x, pos.y + int(20 * s)), 
                   self.fonts["small"]["face"], self.fonts["small"]["scale"], self.colors["warning"], 1)
        return img

    def draw_debug_panel(self, img: np.ndarray, metrics: DisplayMetrics) -> np.ndarray:
        self._check_resize(img)
        layout = self.layouts["default"]
        panel = layout["debug_panel"]
        s = self.scale_factor
        
        cv2.rectangle(img, (panel.x, panel.y), (panel.x + panel.width, panel.y + panel.height), self.colors["background"], cv2.FILLED)
        y_offset = int(30 * s)
        lines = [f"Elbow Angle: {int(metrics.elbow_angle)}",
                 f"Torso Angle: {int(metrics.torso_angle)} (Max {metrics.torso_threshold})",
                 f"Stage: {metrics.stage}"]
        for line in lines:
            cv2.putText(img, line, (panel.x + int(10 * s), panel.y + y_offset), 
                       self.fonts["small"]["face"], self.fonts["small"]["scale"], self.colors["text"], 1)
            y_offset += int(25 * s)
        return img

    def draw_pose_skeleton(self, img: np.ndarray, landmarks, connections) -> np.ndarray:
        h, w = img.shape[:2]
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start, end = landmarks[start_idx], landmarks[end_idx]
                if start.visibility > 0.5 and end.visibility > 0.5:
                    cv2.line(img, (int(start.x * w), int(start.y * h)), (int(end.x * w), int(end.y * h)), self.colors["accent"], 2)
        for lm in landmarks:
            if lm.visibility > 0.5:
                cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 4, self.colors["primary"], -1)
        return img

# Dummy function to prevent import errors in main.py
def draw_minimal_ui(*args, **kwargs):
    pass