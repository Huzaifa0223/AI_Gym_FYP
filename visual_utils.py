import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math

# ===========================================
# COLOR PALETTES & CONSTANTS
# ===========================================
class ColorPalette(Enum):
    """Predefined color palettes for different visualization themes"""
    FITNESS = "fitness"
    MEDICAL = "medical"
    GAMING = "gaming"
    MINIMAL = "minimal"

# Base Colors (B, G, R)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_PURPLE = (255, 0, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_GRAY = (128, 128, 128)
COLOR_DARK_GRAY = (50, 50, 50)
COLOR_LIGHT_GRAY = (200, 200, 200)

# Gradient Colors for Progress/Quality
QUALITY_GRADIENT = [
    (0, 0, 255),    # Red (poor)
    (0, 100, 255),  # Orange-red
    (0, 165, 255),  # Orange
    (0, 200, 255),  # Yellow-orange
    (0, 255, 255),  # Yellow
    (0, 255, 200),  # Yellow-green
    (0, 255, 100),  # Green-yellow
    (0, 255, 0)     # Green (excellent)
]

# Theme-based color schemes
THEME_COLORS = {
    ColorPalette.FITNESS: {
        "primary": (0, 200, 0),      # Vibrant green
        "secondary": (255, 150, 0),  # Orange
        "accent": (0, 100, 255),     # Blue
        "background": (40, 40, 40),  # Dark gray
        "text": (255, 255, 255),     # White
        "warning": (0, 100, 255),    # Blue for warnings
        "error": (0, 0, 255),        # Red for errors
        "success": (0, 255, 150)     # Teal for success
    },
    ColorPalette.MEDICAL: {
        "primary": (255, 255, 255),  # White
        "secondary": (200, 200, 255),# Light blue
        "accent": (255, 200, 200),   # Light red
        "background": (30, 30, 50),  # Dark blue-gray
        "text": (240, 240, 240),     # Off-white
        "warning": (0, 200, 255),    # Cyan
        "error": (0, 0, 200),        # Dark red
        "success": (150, 255, 150)   # Light green
    },
    ColorPalette.GAMING: {
        "primary": (255, 0, 255),    # Purple
        "secondary": (0, 255, 255),  # Cyan
        "accent": (255, 255, 0),     # Yellow
        "background": (20, 20, 40),  # Very dark blue
        "text": (255, 220, 100),     # Gold text
        "warning": (255, 100, 0),    # Orange
        "error": (255, 0, 100),      # Pink
        "success": (100, 255, 100)   # Neon green
    },
    ColorPalette.MINIMAL: {
        "primary": (150, 150, 150),  # Gray
        "secondary": (100, 100, 100),# Dark gray
        "accent": (200, 200, 200),   # Light gray
        "background": (10, 10, 10),  # Near black
        "text": (220, 220, 220),     # Light gray
        "warning": (200, 150, 0),    # Dark yellow
        "error": (150, 0, 0),        # Dark red
        "success": (0, 150, 0)       # Dark green
    }
}

# ===========================================
# DATA STRUCTURES
# ===========================================
@dataclass
class UIPosition:
    """Position and size for UI elements"""
    x: int
    y: int
    width: int = 0
    height: int = 0

@dataclass
class DisplayMetrics:
    """Container for display metrics"""
    elbow_angle: float
    torso_angle: float
    torso_threshold: float
    fps: float = 0.0
    form_quality: float = 0.0
    rep_count: int = 0
    stage: str = ""
    arm_side: str = ""
    confidence: float = 0.0
    additional_metrics: Optional[Dict] = None

# ===========================================
# CORE VISUALIZATION CLASS
# ===========================================
class VisualFeedback:
    """
    Advanced visual feedback system with theme support and multiple layout options
    """
    def __init__(self, theme: ColorPalette = ColorPalette.FITNESS, 
                 resolution: Tuple[int, int] = (1280, 720)):
        """
        Initialize visual feedback system
        
        Args:
            theme: Color palette theme
            resolution: Screen resolution for layout calculations
        """
        self.theme = theme
        self.colors = THEME_COLORS[theme]
        self.resolution = resolution
        self.fonts = self._initialize_fonts()
        self.layouts = self._initialize_layouts()
        
        # Animation state
        self.pulse_phase = 0
        self.last_pulse_time = 0
        
    def _initialize_fonts(self) -> Dict:
        """Initialize font configurations"""
        return {
            "large": {
                "face": cv2.FONT_HERSHEY_SIMPLEX,
                "scale": 2.5,
                "thickness": 4
            },
            "medium": {
                "face": cv2.FONT_HERSHEY_SIMPLEX,
                "scale": 1.2,
                "thickness": 2
            },
            "small": {
                "face": cv2.FONT_HERSHEY_PLAIN,
                "scale": 1.0,
                "thickness": 1
            },
            "bold": {
                "face": cv2.FONT_HERSHEY_DUPLEX,
                "scale": 1.5,
                "thickness": 3
            }
        }
    
    def _initialize_layouts(self) -> Dict:
        """Initialize different layout configurations"""
        w, h = self.resolution
        
        return {
            "default": {
                "status_box": UIPosition(20, 20, 350, 120),
                "debug_panel": UIPosition(20, h - 150, 400, 130),
                "stats_panel": UIPosition(w - 400, 20, 380, 200),
                "progress_bar": UIPosition(w // 2 - 150, 50, 300, 25),
                "fps_display": UIPosition(w - 120, 40, 100, 30)
            },
            "compact": {
                "status_box": UIPosition(10, 10, 280, 80),
                "debug_panel": UIPosition(10, h - 100, 350, 90),
                "stats_panel": UIPosition(w - 300, 10, 290, 120),
                "progress_bar": UIPosition(w // 2 - 100, 30, 200, 20),
                "fps_display": UIPosition(w - 90, 25, 80, 25)
            },
            "minimal": {
                "status_box": UIPosition(5, 5, 200, 60),
                "debug_panel": UIPosition(5, h - 80, 250, 75),
                "progress_bar": UIPosition(w // 2 - 75, 20, 150, 15),
                "fps_display": UIPosition(w - 70, 15, 65, 20)
            }
        }
    
    def get_color_for_value(self, value: float, min_val: float = 0, 
                           max_val: float = 100, reverse: bool = False) -> Tuple:
        """
        Get gradient color based on value
        
        Args:
            value: Current value
            min_val: Minimum value
            max_val: Maximum value
            reverse: Reverse the gradient (for error states)
            
        Returns:
            BGR color tuple
        """
        if max_val <= min_val:
            return self.colors["error"]
        
        # Normalize value to 0-1
        normalized = max(0, min(1, (value - min_val) / (max_val - min_val)))
        
        if reverse:
            normalized = 1 - normalized
        
        # Map to gradient
        gradient_index = int(normalized * (len(QUALITY_GRADIENT) - 1))
        gradient_index = max(0, min(len(QUALITY_GRADIENT) - 1, gradient_index))
        
        return QUALITY_GRADIENT[gradient_index]
    
    def draw_status_box(self, img: np.ndarray, counter: int, feedback: str, 
                       feedback_color: Tuple, layout: str = "default",
                       pulse_on_good: bool = True) -> np.ndarray:
        """
        Enhanced status box with animations and better layout
        
        Args:
            img: Input image
            counter: Repetition counter
            feedback: Feedback text
            feedback_color: Box color
            layout: Layout configuration name
            pulse_on_good: Add pulsing animation for good form
        """
        layout_config = self.layouts.get(layout, self.layouts["default"])
        box = layout_config["status_box"]
        
        # Calculate pulsing effect for good form
        if pulse_on_good and feedback in ["GOOD FORM", "✓ GOOD FORM"]:
            pulse_intensity = abs(math.sin(self.pulse_phase)) * 30
            feedback_color = tuple(
                min(255, c + pulse_intensity) for c in feedback_color
            )
        
        # Draw main box with shadow effect
        shadow_offset = 5
        cv2.rectangle(img, 
                     (box.x + shadow_offset, box.y + shadow_offset),
                     (box.x + box.width + shadow_offset, box.y + box.height + shadow_offset),
                     COLOR_BLACK, cv2.FILLED)
        
        # Draw main box
        cv2.rectangle(img, 
                     (box.x, box.y),
                     (box.x + box.width, box.y + box.height),
                     feedback_color, cv2.FILLED)
        
        # Add border
        cv2.rectangle(img,
                     (box.x, box.y),
                     (box.x + box.width, box.y + box.height),
                     self.colors["text"], 2)
        
        # Draw rep counter (large)
        counter_text = f"REPS: {counter}"
        counter_size = cv2.getTextSize(counter_text, 
                                      self.fonts["large"]["face"],
                                      self.fonts["large"]["scale"],
                                      self.fonts["large"]["thickness"])[0]
        
        counter_x = box.x + 20
        counter_y = box.y + 50
        
        # Add text shadow
        cv2.putText(img, counter_text,
                   (counter_x + 2, counter_y + 2),
                   self.fonts["large"]["face"],
                   self.fonts["large"]["scale"],
                   COLOR_BLACK,
                   self.fonts["large"]["thickness"],
                   cv2.LINE_AA)
        
        # Draw main text
        cv2.putText(img, counter_text,
                   (counter_x, counter_y),
                   self.fonts["large"]["face"],
                   self.fonts["large"]["scale"],
                   self.colors["text"],
                   self.fonts["large"]["thickness"],
                   cv2.LINE_AA)
        
        # Draw feedback text
        feedback_size = cv2.getTextSize(feedback,
                                       self.fonts["medium"]["face"],
                                       self.fonts["medium"]["scale"],
                                       self.fonts["medium"]["thickness"])[0]
        
        feedback_x = box.x + (box.width - feedback_size[0]) // 2
        feedback_y = box.y + box.height - 20
        
        # Icon for feedback
        icon = ""
        if feedback.startswith("✓"):
            icon = "✓"
            text_color = self.colors["success"]
        elif feedback.startswith("⚠"):
            icon = "⚠"
            text_color = self.colors["warning"]
        elif feedback.startswith("✗"):
            icon = "✗"
            text_color = self.colors["error"]
        else:
            text_color = self.colors["text"]
        
        # Draw feedback with icon
        feedback_display = f"{icon} {feedback.replace(icon, '').strip()}" if icon else feedback
        cv2.putText(img, feedback_display,
                   (feedback_x, feedback_y),
                   self.fonts["medium"]["face"],
                   self.fonts["medium"]["scale"],
                   text_color,
                   self.fonts["medium"]["thickness"],
                   cv2.LINE_AA)
        
        return img
    
    def draw_debug_panel(self, img: np.ndarray, metrics: DisplayMetrics,
                        layout: str = "default", show_all: bool = True) -> np.ndarray:
        """
        Enhanced debug panel with multiple metrics and visual indicators
        
        Args:
            img: Input image
            metrics: DisplayMetrics object with all metrics
            layout: Layout configuration
            show_all: Show all metrics or just essential ones
        """
        layout_config = self.layouts.get(layout, self.layouts["default"])
        panel = layout_config["debug_panel"]
        
        # Draw panel background
        cv2.rectangle(img,
                     (panel.x, panel.y),
                     (panel.x + panel.width, panel.y + panel.height),
                     self.colors["background"],
                     cv2.FILLED)
        
        # Add border
        cv2.rectangle(img,
                     (panel.x, panel.y),
                     (panel.x + panel.width, panel.y + panel.height),
                     self.colors["primary"],
                     2)
        
        # Title
        cv2.putText(img, "LIVE METRICS",
                   (panel.x + 10, panel.y + 25),
                   self.fonts["bold"]["face"],
                   self.fonts["bold"]["scale"] * 0.8,
                   self.colors["primary"],
                   self.fonts["bold"]["thickness"] - 1,
                   cv2.LINE_AA)
        
        y_offset = 50
        line_height = 30
        
        # 1. Elbow Angle
        elbow_color = self.get_color_for_value(
            metrics.elbow_angle, 60, 150, reverse=False
        )
        elbow_text = f"Elbow: {int(metrics.elbow_angle)}°"
        cv2.putText(img, elbow_text,
                   (panel.x + 10, panel.y + y_offset),
                   self.fonts["small"]["face"],
                   self.fonts["small"]["scale"],
                   elbow_color,
                   self.fonts["small"]["thickness"] + 1,
                   cv2.LINE_AA)
        
        # Elbow angle bar
        if show_all:
            bar_width = 100
            bar_height = 8
            bar_x = panel.x + 150
            bar_y = panel.y + y_offset - 5
            
            # Background bar
            cv2.rectangle(img,
                         (bar_x, bar_y),
                         (bar_x + bar_width, bar_y + bar_height),
                         COLOR_DARK_GRAY,
                         cv2.FILLED)
            
            # Fill based on angle (normalized to 60-150 range)
            fill_width = int((min(150, max(60, metrics.elbow_angle)) - 60) / 90 * bar_width)
            cv2.rectangle(img,
                         (bar_x, bar_y),
                         (bar_x + fill_width, bar_y + bar_height),
                         elbow_color,
                         cv2.FILLED)
        
        y_offset += line_height
        
        # 2. Torso Deviation
        torso_percentage = min(100, (metrics.torso_angle / metrics.torso_threshold) * 100)
        torso_color = self.get_color_for_value(
            torso_percentage, 0, 100, reverse=True
        )
        
        torso_text = f"Swing: {int(metrics.torso_angle)}°"
        cv2.putText(img, torso_text,
                   (panel.x + 10, panel.y + y_offset),
                   self.fonts["small"]["face"],
                   self.fonts["small"]["scale"],
                   torso_color,
                   self.fonts["small"]["thickness"] + 1,
                   cv2.LINE_AA)
        
        # Warning indicator if swing is high
        if metrics.torso_angle > metrics.torso_threshold * 0.8:
            cv2.putText(img, "⚠",
                       (panel.x + panel.width - 30, panel.y + y_offset),
                       self.fonts["medium"]["face"],
                       self.fonts["medium"]["scale"],
                       self.colors["warning"],
                       self.fonts["medium"]["thickness"],
                       cv2.LINE_AA)
        
        y_offset += line_height
        
        if show_all:
            # 3. Form Quality
            quality_color = self.get_color_for_value(metrics.form_quality)
            quality_text = f"Form: {int(metrics.form_quality)}%"
            cv2.putText(img, quality_text,
                       (panel.x + 10, panel.y + y_offset),
                       self.fonts["small"]["face"],
                       self.fonts["small"]["scale"],
                       quality_color,
                       self.fonts["small"]["thickness"] + 1,
                       cv2.LINE_AA)
            
            # Quality indicator dots
            dot_count = 5
            for i in range(dot_count):
                dot_x = panel.x + 150 + i * 15
                filled = metrics.form_quality >= ((i + 1) / dot_count * 100)
                dot_color = quality_color if filled else COLOR_DARK_GRAY
                cv2.circle(img, (dot_x, panel.y + y_offset - 5), 4, dot_color, -1)
            
            y_offset += line_height
            
            # 4. Stage & Arm
            stage_text = f"{metrics.stage.upper()} | {metrics.arm_side}"
            cv2.putText(img, stage_text,
                       (panel.x + 10, panel.y + y_offset),
                       self.fonts["small"]["face"],
                       self.fonts["small"]["scale"],
                       self.colors["accent"],
                       self.fonts["small"]["thickness"],
                       cv2.LINE_AA)
        
        return img
    
    def draw_stats_panel(self, img: np.ndarray, stats: Dict,
                        layout: str = "default") -> np.ndarray:
        """
        Draw comprehensive statistics panel
        
        Args:
            img: Input image
            stats: Dictionary of statistics
            layout: Layout configuration
        """
        layout_config = self.layouts.get(layout, self.layouts["default"])
        if "stats_panel" not in layout_config:
            return img
        
        panel = layout_config["stats_panel"]
        
        # Draw panel
        cv2.rectangle(img,
                     (panel.x, panel.y),
                     (panel.x + panel.width, panel.y + panel.height),
                     self.colors["background"],
                     cv2.FILLED)
        
        # Add gradient header
        header_height = 40
        for i in range(header_height):
            alpha = i / header_height
            color = tuple(
                int(self.colors["primary"][j] * alpha + 
                    self.colors["background"][j] * (1 - alpha))
                for j in range(3)
            )
            cv2.line(img,
                    (panel.x, panel.y + i),
                    (panel.x + panel.width, panel.y + i),
                    color, 1)
        
        # Panel title
        cv2.putText(img, "STATISTICS",
                   (panel.x + 15, panel.y + 28),
                   self.fonts["bold"]["face"],
                   self.fonts["bold"]["scale"] * 0.9,
                   self.colors["text"],
                   self.fonts["bold"]["thickness"],
                   cv2.LINE_AA)
        
        # Draw stats
        y_offset = 60
        line_height = 25
        
        stat_items = [
            ("Total Reps", f"{stats.get('total_reps', 0)}", None),
            ("Good Form", f"{stats.get('good_form_reps', 0)}", self.colors["success"]),
            ("Bad Form", f"{stats.get('bad_form_reps', 0)}", self.colors["error"]),
            ("Form Acc", f"{stats.get('form_accuracy', 0):.1f}%", 
             self.get_color_for_value(stats.get('form_accuracy', 0))),
            ("Avg Rep Time", f"{stats.get('avg_rep_time', 0):.2f}s", None),
            ("Best Rep", f"{stats.get('best_rep', 0):.0f}%", self.colors["success"]),
            ("FPS", f"{stats.get('fps', 0):.1f}", 
             self.get_color_for_value(stats.get('fps', 0), 15, 60))
        ]
        
        for label, value, color in stat_items:
            if color is None:
                color = self.colors["text"]
            
            # Label
            cv2.putText(img, f"{label}:",
                       (panel.x + 15, panel.y + y_offset),
                       self.fonts["small"]["face"],
                       self.fonts["small"]["scale"],
                       self.colors["secondary"],
                       self.fonts["small"]["thickness"],
                       cv2.LINE_AA)
            
            # Value
            value_x = panel.x + panel.width - 100
            cv2.putText(img, value,
                       (value_x, panel.y + y_offset),
                       self.fonts["small"]["face"],
                       self.fonts["small"]["scale"],
                       color,
                       self.fonts["small"]["thickness"] + 1,
                       cv2.LINE_AA)
            
            y_offset += line_height
        
        return img
    
    def draw_progress_bar(self, img: np.ndarray, value: float, max_value: float = 100,
                         label: str = "Progress", layout: str = "default",
                         show_percentage: bool = True) -> np.ndarray:
        """
        Draw animated progress bar
        
        Args:
            img: Input image
            value: Current value
            max_value: Maximum value
            label: Bar label
            layout: Layout configuration
            show_percentage: Show percentage text
        """
        layout_config = self.layouts.get(layout, self.layouts["default"])
        bar = layout_config["progress_bar"]
        
        # Calculate fill
        percentage = min(1.0, max(0.0, value / max_value))
        fill_width = int(bar.width * percentage)
        
        # Get color based on percentage
        bar_color = self.get_color_for_value(percentage * 100)
        
        # Draw background
        cv2.rectangle(img,
                     (bar.x, bar.y),
                     (bar.x + bar.width, bar.y + bar.height),
                     COLOR_DARK_GRAY,
                     cv2.FILLED)
        
        # Draw fill with gradient
        if fill_width > 0:
            # Simple gradient effect
            for i in range(fill_width):
                x_pos = bar.x + i
                gradient_factor = i / bar.width
                
                # Adjust color intensity
                adjusted_color = tuple(
                    int(c * (0.7 + 0.3 * gradient_factor))
                    for c in bar_color
                )
                
                cv2.line(img,
                        (x_pos, bar.y),
                        (x_pos, bar.y + bar.height),
                        adjusted_color, 1)
        
        # Add border
        cv2.rectangle(img,
                     (bar.x, bar.y),
                     (bar.x + bar.width, bar.y + bar.height),
                     self.colors["text"],
                     2)
        
        # Draw label
        if label:
            label_size = cv2.getTextSize(label,
                                        self.fonts["small"]["face"],
                                        self.fonts["small"]["scale"],
                                        self.fonts["small"]["thickness"])[0]
            
            label_x = bar.x + (bar.width - label_size[0]) // 2
            cv2.putText(img, label,
                       (label_x, bar.y - 5),
                       self.fonts["small"]["face"],
                       self.fonts["small"]["scale"],
                       self.colors["text"],
                       self.fonts["small"]["thickness"],
                       cv2.LINE_AA)
        
        # Draw percentage
        if show_percentage:
            percent_text = f"{int(percentage * 100)}%"
            percent_size = cv2.getTextSize(percent_text,
                                          self.fonts["small"]["face"],
                                          self.fonts["small"]["scale"],
                                          self.fonts["small"]["thickness"])[0]
            
            percent_x = bar.x + (bar.width - percent_size[0]) // 2
            percent_y = bar.y + bar.height // 2 + 5
            
            cv2.putText(img, percent_text,
                       (percent_x, percent_y),
                       self.fonts["small"]["face"],
                       self.fonts["small"]["scale"],
                       self.colors["text"],
                       self.fonts["small"]["thickness"],
                       cv2.LINE_AA)
        
        return img
    
    def draw_fps(self, img: np.ndarray, fps: float, layout: str = "default") -> np.ndarray:
        """
        Draw FPS counter with performance indicator
        
        Args:
            img: Input image
            fps: Current FPS
            layout: Layout configuration
        """
        layout_config = self.layouts.get(layout, self.layouts["default"])
        fps_pos = layout_config["fps_display"]
        
        # Color based on FPS performance
        if fps > 30:
            fps_color = self.colors["success"]
        elif fps > 20:
            fps_color = self.colors["warning"]
        else:
            fps_color = self.colors["error"]
        
        # Draw background
        cv2.rectangle(img,
                     (fps_pos.x, fps_pos.y),
                     (fps_pos.x + fps_pos.width, fps_pos.y + fps_pos.height),
                     self.colors["background"],
                     cv2.FILLED)
        
        # Draw FPS text
        fps_text = f"{fps:.1f} FPS"
        text_size = cv2.getTextSize(fps_text,
                                   self.fonts["small"]["face"],
                                   self.fonts["small"]["scale"],
                                   self.fonts["small"]["thickness"])[0]
        
        text_x = fps_pos.x + (fps_pos.width - text_size[0]) // 2
        text_y = fps_pos.y + (fps_pos.height + text_size[1]) // 2
        
        cv2.putText(img, fps_text,
                   (text_x, text_y),
                   self.fonts["small"]["face"],
                   self.fonts["small"]["scale"],
                   fps_color,
                   self.fonts["small"]["thickness"] + 1,
                   cv2.LINE_AA)
        
        # Performance indicator dot
        dot_radius = 4
        dot_x = fps_pos.x + 10
        dot_y = fps_pos.y + fps_pos.height // 2
        
        cv2.circle(img, (dot_x, dot_y), dot_radius, fps_color, -1)
        
        return img
    
    def draw_pose_skeleton(self, img: np.ndarray, landmarks, connections,
                          landmark_color: Optional[Tuple] = None,
                          connection_color: Optional[Tuple] = None,
                          show_indices: bool = False) -> np.ndarray:
        """
        Draw pose skeleton with enhanced visualization
        
        Args:
            img: Input image
            landmarks: List of landmarks
            connections: List of connection pairs
            landmark_color: Color for landmarks
            connection_color: Color for connections
            show_indices: Show landmark indices (for debugging)
        """
        if landmark_color is None:
            landmark_color = self.colors["primary"]
        if connection_color is None:
            connection_color = self.colors["accent"]
        
        img_h, img_w = img.shape[:2]
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            if (0 <= start_idx < len(landmarks) and 
                0 <= end_idx < len(landmarks)):
                
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]
                
                # Only draw if both landmarks are visible
                if (hasattr(start_lm, 'visibility') and 
                    hasattr(end_lm, 'visibility') and
                    start_lm.visibility > 0.5 and 
                    end_lm.visibility > 0.5):
                    
                    start_pt = (int(start_lm.x * img_w), int(start_lm.y * img_h))
                    end_pt = (int(end_lm.x * img_w), int(end_lm.y * img_h))
                    
                    # Draw connection with gradient based on confidence
                    alpha = min(start_lm.visibility, end_lm.visibility)
                    color = tuple(int(c * alpha) for c in connection_color)
                    
                    cv2.line(img, start_pt, end_pt, color, 3)
        
        # Draw landmarks
        for idx, landmark in enumerate(landmarks):
            if hasattr(landmark, 'visibility') and landmark.visibility > 0.3:
                x = int(landmark.x * img_w)
                y = int(landmark.y * img_h)
                
                # Size based on landmark importance/visibility
                radius = int(6 * landmark.visibility)
                
                # Draw outer circle
                cv2.circle(img, (x, y), radius + 2, COLOR_BLACK, -1)
                
                # Draw inner circle with color based on visibility
                intensity = int(255 * landmark.visibility)
                color = tuple(int(c * landmark.visibility) for c in landmark_color)
                cv2.circle(img, (x, y), radius, color, -1)
                
                # Draw index if requested
                if show_indices:
                    cv2.putText(img, str(idx),
                               (x + 10, y),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.4, COLOR_WHITE, 1)
        
        return img
    
    def draw_angle_arc(self, img: np.ndarray, joint: Tuple[int, int],
                      angle: float, radius: int = 50,
                      color: Optional[Tuple] = None) -> np.ndarray:
        """
        Draw angle arc visualization at a joint
        
        Args:
            img: Input image
            joint: (x, y) coordinates of the joint
            angle: Angle in degrees to visualize
            radius: Arc radius
            color: Arc color
        """
        if color is None:
            color = self.colors["accent"]
        
        # Convert angle to radians for drawing
        start_angle = -90  # Starting from vertical
        end_angle = start_angle + angle
        
        # Draw arc
        cv2.ellipse(img, joint, (radius, radius), 0,
                   start_angle, end_angle,
                   color, 3)
        
        # Draw angle text
        angle_text = f"{int(angle)}°"
        text_size = cv2.getTextSize(angle_text,
                                   self.fonts["small"]["face"],
                                   self.fonts["small"]["scale"],
                                   self.fonts["small"]["thickness"])[0]
        
        # Position text at the arc
        text_angle = math.radians(start_angle + angle / 2)
        text_x = int(joint[0] + (radius + 20) * math.cos(text_angle))
        text_y = int(joint[1] + (radius + 20) * math.sin(text_angle))
        
        cv2.putText(img, angle_text,
                   (text_x - text_size[0] // 2, text_y + text_size[1] // 2),
                   self.fonts["small"]["face"],
                   self.fonts["small"]["scale"],
                   color,
                   self.fonts["small"]["thickness"] + 1,
                   cv2.LINE_AA)
        
        return img
    
    def draw_instructions(self, img: np.ndarray, instructions: List[str],
                         position: str = "bottom") -> np.ndarray:
        """
        Draw instructional text
        
        Args:
            img: Input image
            instructions: List of instruction strings
            position: "top", "bottom", or "center"
        """
        img_h, img_w = img.shape[:2]
        
        if position == "top":
            y_start = 40
            y_increment = 30
        elif position == "center":
            total_height = len(instructions) * 30
            y_start = (img_h - total_height) // 2
            y_increment = 30
        else:  # bottom
            y_start = img_h - 40 - (len(instructions) * 30)
            y_increment = 30
        
        # Draw semi-transparent background
        bg_height = len(instructions) * 30 + 20
        overlay = img.copy()
        cv2.rectangle(overlay,
                     (img_w // 2 - 200, y_start - 10),
                     (img_w // 2 + 200, y_start + bg_height),
                     (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        
        # Draw instructions
        for i, instruction in enumerate(instructions):
            text_size = cv2.getTextSize(instruction,
                                       self.fonts["small"]["face"],
                                       self.fonts["small"]["scale"],
                                       self.fonts["small"]["thickness"])[0]
            
            text_x = img_w // 2 - text_size[0] // 2
            text_y = y_start + i * y_increment
            
            cv2.putText(img, instruction,
                       (text_x, text_y),
                       self.fonts["small"]["face"],
                       self.fonts["small"]["scale"],
                       self.colors["text"],
                       self.fonts["small"]["thickness"],
                       cv2.LINE_AA)
        
        return img
    
    def draw_countdown(self, img: np.ndarray, seconds: float,
                      position: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Draw animated countdown timer
        
        Args:
            img: Input image
            seconds: Seconds remaining
            position: Center position of countdown
        """
        if position is None:
            position = (self.resolution[0] // 2, self.resolution[1] // 2)
        
        # Draw background circle
        radius = 80
        cv2.circle(img, position, radius, self.colors["background"], -1)
        cv2.circle(img, position, radius, self.colors["primary"], 3)
        
        # Draw countdown text
        count_text = str(int(seconds))
        text_size = cv2.getTextSize(count_text,
                                   self.fonts["large"]["face"],
                                   self.fonts["large"]["scale"] * 2,
                                   self.fonts["large"]["thickness"] + 2)[0]
        
        text_x = position[0] - text_size[0] // 2
        text_y = position[1] + text_size[1] // 2
        
        # Pulsing effect
        pulse = abs(math.sin(self.pulse_phase * 2)) * 50
        text_color = tuple(
            min(255, c + pulse) for c in self.colors["primary"]
        )
        
        cv2.putText(img, count_text,
                   (text_x, text_y),
                   self.fonts["large"]["face"],
                   self.fonts["large"]["scale"] * 2,
                   text_color,
                   self.fonts["large"]["thickness"] + 2,
                   cv2.LINE_AA)
        
        return img
    
    def update_animation(self, dt: float = 0.1):
        """
        Update animation state
        
        Args:
            dt: Time delta for animation
        """
        self.pulse_phase += dt
        if self.pulse_phase > 2 * math.pi:
            self.pulse_phase -= 2 * math.pi

# ===========================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ===========================================
def draw_status_box(img, counter, feedback, feedback_color):
    """
    Legacy function for backward compatibility
    """
    visualizer = VisualFeedback()
    return visualizer.draw_status_box(img, counter, feedback, feedback_color, "default")

def draw_debug_panel(img, elbow_angle, torso_angle, torso_threshold):
    """
    Legacy function for backward compatibility
    """
    visualizer = VisualFeedback()
    metrics = DisplayMetrics(
        elbow_angle=elbow_angle,
        torso_angle=torso_angle,
        torso_threshold=torso_threshold
    )
    return visualizer.draw_debug_panel(img, metrics, "default", show_all=False)

# ===========================================
# QUICK UTILITY FUNCTIONS
# ===========================================
def add_text_with_background(img, text, position,
                            font_face=cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale=1.0, thickness=2,
                            text_color=COLOR_WHITE,
                            bg_color=COLOR_BLACK,
                            padding=5):
    """
    Add text with background for better readability
    
    Args:
        img: Input image
        text: Text to display
        position: (x, y) position for text
        font_face: OpenCV font face
        font_scale: Font scale
        thickness: Font thickness
        text_color: Text color
        bg_color: Background color
        padding: Padding around text
    """
    # Get text size
    text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
    
    # Calculate background rectangle
    bg_rect = (
        position[0] - padding,
        position[1] - text_size[1] - padding,
        position[0] + text_size[0] + padding,
        position[1] + padding
    )
    
    # Draw background
    cv2.rectangle(img, bg_rect[:2], bg_rect[2:], bg_color, cv2.FILLED)
    
    # Draw text
    cv2.putText(img, text, position, font_face, font_scale, text_color, thickness)
    
    return img

def create_gradient_background(width, height, color1, color2, direction='vertical'):
    """
    Create a gradient background
    
    Args:
        width: Image width
        height: Image height
        color1: Start color (B, G, R)
        color2: End color (B, G, R)
        direction: 'vertical' or 'horizontal'
    """
    background = np.zeros((height, width, 3), dtype=np.uint8)
    
    if direction == 'vertical':
        for i in range(height):
            alpha = i / height
            color = tuple(
                int(color1[j] * (1 - alpha) + color2[j] * alpha)
                for j in range(3)
            )
            background[i, :] = color
    else:  # horizontal
        for i in range(width):
            alpha = i / width
            color = tuple(
                int(color1[j] * (1 - alpha) + color2[j] * alpha)
                for j in range(3)
            )
            background[:, i] = color
    
    return background

# ===========================================
# EXPORTS
# ===========================================
__all__ = [
    # Colors
    'COLOR_RED', 'COLOR_GREEN', 'COLOR_BLUE', 'COLOR_WHITE', 'COLOR_BLACK',
    'COLOR_YELLOW', 'COLOR_ORANGE', 'COLOR_PURPLE', 'COLOR_CYAN',
    'COLOR_GRAY', 'COLOR_DARK_GRAY', 'COLOR_LIGHT_GRAY',
    
    # Classes
    'VisualFeedback', 'ColorPalette', 'UIPosition', 'DisplayMetrics',
    
    # Legacy functions
    'draw_status_box', 'draw_debug_panel',
    
    # Utility functions
    'add_text_with_background', 'create_gradient_background',
    
    # Color schemes
    'THEME_COLORS', 'QUALITY_GRADIENT'
]

# ===========================================
# USAGE EXAMPLE
# ===========================================
if __name__ == "__main__":
    # Example usage
    print("Visual Utilities Module - Enhanced Version")
    print("=========================================")
    
    # Create a test image
    width, height = 800, 600
    test_img = np.zeros((height, width, 3), dtype=np.uint8)
    test_img[:] = (40, 40, 40)  # Dark gray background
    
    # Initialize visualizer
    visualizer = VisualFeedback(ColorPalette.FITNESS, (width, height))
    
    # Test status box
    visualizer.draw_status_box(test_img, 15, "✓ GOOD FORM", COLOR_GREEN)
    
    # Test debug panel
    metrics = DisplayMetrics(
        elbow_angle=75.5,
        torso_angle=12.3,
        torso_threshold=25,
        form_quality=85.2,
        rep_count=15,
        stage="down",
        arm_side="RIGHT",
        fps=29.7
    )
    visualizer.draw_debug_panel(test_img, metrics)
    
    # Test stats panel
    stats = {
        'total_reps': 15,
        'good_form_reps': 12,
        'bad_form_reps': 3,
        'form_accuracy': 80.0,
        'avg_rep_time': 2.1,
        'best_rep': 95,
        'fps': 29.7
    }
    visualizer.draw_stats_panel(test_img, stats)
    
    # Test progress bar
    visualizer.draw_progress_bar(test_img, 75, 100, "Form Quality")
    
    # Test FPS display
    visualizer.draw_fps(test_img, 29.7)
    
    # Save test image
    cv2.imwrite("test_visualization.jpg", test_img)
    print("Test visualization saved as 'test_visualization.jpg'")