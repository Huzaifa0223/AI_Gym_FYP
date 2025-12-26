import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
import json
import datetime
import sys
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Import your new visual engine
from visual_utils import VisualFeedback, ColorPalette, DisplayMetrics
from bicep_curl import BicepCurlExercise

# ===========================================
# CONFIGURATION - USER ADJUSTABLE
# ===========================================
MODEL_PATH = 'bicep_model.pkl'
STRICT_MODE_THRESHOLD = 25  # Maximum allowed upper arm movement (degrees)
CALIBRATION_TIME = 5  # Longer calibration for better accuracy
MIN_REP_ANGLE = 50   # Minimum angle (adjustable per user)
MAX_REP_ANGLE = 170  # Maximum angle (adjustable per user)
MIN_REP_TIME = 1.0   # Minimum time for a rep (seconds)
MAX_REP_TIME = 4.0   # Maximum time for a rep (seconds)
TARGET_FPS = 30      # Target frames per second
RESOLUTION = (640, 480)  # Lower resolution for better performance

# ===========================================
# FEEDBACK SYSTEM
# ===========================================
FEEDBACK_TYPES = {
    "good_form": {"text": "✓ GOOD FORM", "color": (0, 255, 0)},
    "elbow_swing": {"text": "✗ ELBOW SWING!", "color": (0, 0, 255)},
    "partial_rep": {"text": "⚠ PARTIAL REP", "color": (0, 165, 255)},
    "too_fast": {"text": "⚡ TOO FAST!", "color": (0, 255, 255)},
    "too_slow": {"text": "🐌 TOO SLOW", "color": (255, 165, 0)},
    "full_lockout": {"text": "⚠ FULL LOCKOUT", "color": (255, 255, 0)},
    "no_rep": {"text": "NO REP", "color": (100, 100, 100)},
    "bad_lighting": {"text": "⚠ POOR LIGHTING", "color": (255, 100, 0)},
    "wrong_angle": {"text": "↻ SIDE VIEW REQUIRED", "color": (255, 200, 0)},
    "occlusion": {"text": "👕 BAGGY CLOTHING", "color": (180, 180, 180)}
}

# ===========================================
# CAMERA HANDLER - Solves Simulation 1
# ===========================================
class CameraHandler:
    def __init__(self):
        self.cap = None
        self.camera_index = 0
        self.max_camera_tries = 5
        
    def find_working_camera(self):
        """Try multiple camera indices to find a working one"""
        print("[INFO] Searching for available cameras...")
        
        for i in range(self.max_camera_tries):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DSHOW on Windows for better compatibility
            
            if cap.isOpened():
                # Try to read a frame
                success, frame = cap.read()
                if success:
                    print(f"[SUCCESS] Found camera at index {i}")
                    cap.release()
                    return i
                cap.release()
                
            print(f"[INFO] Camera index {i} not available")
            
        print("[ERROR] No working camera found!")
        return None
    
    def open_camera(self, index=None):
        """Open camera with specified index or find one"""
        if index is None:
            index = self.find_working_camera()
            if index is None:
                return None
        
        self.camera_index = index
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Could not open camera at index {index}")
            return False
        
        # Set standard resolution - Solves Simulation 3
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        print(f"[SUCCESS] Camera {index} opened at {RESOLUTION[0]}x{RESOLUTION[1]}")
        return True
    
    def release(self):
        """Release camera resources - Solves Simulation 2"""
        if self.cap:
            self.cap.release()
            print("[INFO] Camera released")

# ===========================================
# IMAGE PROCESSING - Solves Edge Cases 2 & 6
# ===========================================
def adjust_gamma(image, gamma=1.0):
    """Gamma correction for poor lighting conditions - Edge Case 2"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                     for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_histogram_equalization(image):
    """Improve contrast in poor lighting"""
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return cv2.equalizeHist(image)

def detect_background_clutter(frame, prev_frame):
    """Simple background motion detection - Edge Case 3"""
    if prev_frame is None:
        return 0
    
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    motion_percentage = np.sum(thresh) / (thresh.size * 255)
    return motion_percentage * 100

def apply_motion_blur_reduction(frame, prev_frames):
    """Reduce motion blur effects - Edge Case 6"""
    if len(prev_frames) < 3:
        return frame
    
    # FIX: Convert deque to list first, as deque does not support slicing
    frames_list = list(prev_frames)
    
    # Simple temporal smoothing
    weights = np.array([0.5, 0.3, 0.2])  # Weight recent frames more
    blended = np.zeros_like(frame, dtype=np.float32)
    
    # Use the list for slicing
    for i, prev_frame in enumerate(frames_list[-3:]):
        blended += prev_frame.astype(np.float32) * weights[i]
    
    return blended.astype(np.uint8)

# ===========================================
# POSE ANALYSIS - Solves Edge Cases 1, 4, 5
# ===========================================
def calculate_angle(a, b, c):
    """Robust angle calculation with error checking"""
    try:
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        c = np.array(c, dtype=np.float32)
        
        ba = a - b
        bc = c - b
        
        dot_product = np.dot(ba, bc)
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba < 0.001 or norm_bc < 0.001:
            return 0.0
        
        cosine_angle = dot_product / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
    except:
        return 0.0

def check_side_view(landmarks):
    """Detect if user is in correct side view - Edge Case 1"""
    if not landmarks:
        return False, 0.0
    
    # Check shoulder-elbow-wrist alignment in 2D
    # In side view, these should be roughly in a straight line when arm is down
    shoulder = np.array([landmarks[12].x, landmarks[12].y])
    elbow = np.array([landmarks[14].x, landmarks[14].y])
    wrist = np.array([landmarks[16].x, landmarks[16].y])
    
    # Calculate angle when arm is supposed to be straight
    angle = calculate_angle(shoulder, elbow, wrist)
    
    # In side view with straight arm, angle should be close to 180 degrees
    side_view_score = min(100, max(0, (180 - abs(angle - 180)) * 100 / 180))
    
    return side_view_score > 60, side_view_score

def detect_occlusion(landmarks, confidence_threshold=0.5):
    """Detect if joints are occluded by clothing - Edge Case 4"""
    if not landmarks:
        return True, 0.0
    
    # Key joints to check
    key_joints = [12, 14, 16]  # Shoulder, elbow, wrist
    visibilities = []
    
    for joint in key_joints:
        if hasattr(landmarks[joint], 'visibility'):
            visibilities.append(landmarks[joint].visibility)
        else:
            visibilities.append(0.5)  # Default if not available
    
    avg_visibility = np.mean(visibilities)
    is_occluded = avg_visibility < confidence_threshold
    
    return is_occluded, avg_visibility

def adaptive_angle_thresholds(baseline_angles, flexibility_factor=0.2):
    """Adapt thresholds to user's flexibility - Edge Case 5"""
    if not baseline_angles:
        return MIN_REP_ANGLE, MAX_REP_ANGLE
    
    min_angle = np.min(baseline_angles) * (1 - flexibility_factor)
    max_angle = np.max(baseline_angles) * (1 + flexibility_factor)
    
    # Ensure reasonable bounds
    min_angle = max(30, min(min_angle, 80))
    max_angle = min(180, max(max_angle, 150))
    
    return min_angle, max_angle

def check_form(shoulder, elbow):
    """Calculate the angle of the upper arm relative to vertical"""
    # Vector from shoulder to elbow
    arm_vector = [elbow[0] - shoulder[0], elbow[1] - shoulder[1]]
    # Vertical vector (assuming y increases downward)
    vertical = [0, 1]  # Downward vertical
    # Dot product and magnitudes
    dot = arm_vector[0] * vertical[0] + arm_vector[1] * vertical[1]
    mag_arm = (arm_vector[0]**2 + arm_vector[1]**2)**0.5
    mag_vert = (vertical[0]**2 + vertical[1]**2)**0.5
    if mag_arm == 0:
        return 0  # Avoid division by zero
    cos_angle = dot / (mag_arm * mag_vert)
    angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi  # In degrees
    return angle

# ===========================================
# PERFORMANCE OPTIMIZATION - Solves Edge Case 7
# ===========================================
class PerformanceOptimizer:
    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.frame_times = deque(maxlen=30)
        self.last_time = time.time()
        self.current_fps = target_fps
        self.low_perf_mode = False
        
    def update(self):
        """Update FPS calculation"""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) > 0:
            self.current_fps = 1.0 / np.mean(self.frame_times)
        
        # Enter low performance mode if FPS drops too much
        if self.current_fps < self.target_fps * 0.5:
            self.low_perf_mode = True
        elif self.current_fps > self.target_fps * 0.8:
            self.low_perf_mode = False
            
        self.last_time = current_time
        return self.current_fps
    
    def should_skip_rendering(self):
        """Determine if we should skip heavy rendering"""
        return self.low_perf_mode and self.current_fps < 15

# ===========================================
# MAIN APPLICATION CLASS
# ===========================================
class AIGymTrainer:
    def __init__(self):
        self.camera = CameraHandler()
        self.performance = PerformanceOptimizer(TARGET_FPS)
        self.pose = None
        self.mp_pose = None
        self.mp_draw = None
        self.model = None
        self.stats = None
        self.calibrated = False
        self.paused = False
        self.personal_min = MIN_REP_ANGLE
        self.personal_max = MAX_REP_ANGLE
        self.stage = ""  # For rep counting state
        
        # Initialize the Visual Engine (Choose your theme!)
        self.visualizer = VisualFeedback(theme=ColorPalette.FITNESS)
        
    def initialize(self):
        """Initialize all components"""
        print("=" * 60)
        print("AI GYM TRAINER - ENHANCED EDGE CASE PROTECTION")
        print("=" * 60)
        
        # 1. Check model
        if not os.path.exists(MODEL_PATH):
            print(f"[ERROR] Model file '{MODEL_PATH}' not found!")
            print("[INFO] Please train the model first")
            return False
        
        # 2. Load model
        print("[INFO] Loading AI model...")
        try:
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            print("[SUCCESS] Model loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False
        
        # 3. Initialize MediaPipe
        print("[INFO] Initializing pose detection...")
        self.mp_pose = mp.solutions.pose  # type: ignore
        self.mp_draw = mp.solutions.drawing_utils  # type: ignore
        
        # Higher confidence thresholds to avoid false positives - Edge Case 3
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1  # Balance between accuracy and speed
        )
        
        # 4. Open camera
        if not self.camera.open_camera():
            return False
        
        return True
    
    def calibrate_user(self):
        """Enhanced calibration with a visual waiting room"""
        print("\n" + "=" * 40)
        print("PERSONALIZED CALIBRATION")
        print("=" * 40)
        print("1. Stand in SIDE VIEW of camera")
        print("2. Wear fitted clothing")
        print("3. Ensure good lighting")
        print("=" * 40)
        
        # VISUAL WAITING ROOM
        print("[INFO] Look at the camera window and press ENTER to start...")
        
        while True:
            success, frame = self.camera.cap.read()  # type: ignore
            if not success:
                continue
            
            frame = cv2.resize(frame, RESOLUTION)
            
            # Draw instructions on the screen
            cv2.putText(frame, "STEP 1: STAND IN SIDE VIEW", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "Press ENTER to Start Calibration", (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("Calibration", frame)
            
            # Wait for Enter key (13) or 'c'
            key = cv2.waitKey(1) & 0xFF
            if key == 13 or key == ord('c'): 
                break
        
        calibration_data = {
            "angles": [],
            "flexibility": [],
            "lighting": [],
            "start_time": time.time()
        }
        
        print("\n[INFO] Calibration Started! Hold still...")
        
        # Phase 1: Resting position
        frames_collected = 0
        while frames_collected < 60:  # Increased to 60 frames (~2 seconds)
            success, frame = self.camera.cap.read()  # type: ignore
            if not success:
                continue
                
            frame = cv2.resize(frame, RESOLUTION)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Adjust for lighting if needed
            brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # type: ignore # Convert to grayscale first
            if brightness < 50:
                frame_rgb = adjust_gamma(frame_rgb, gamma=2.0)
            
            results = self.pose.process(frame_rgb)  # type: ignore
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[12].x, landmarks[12].y]
                elbow = [landmarks[14].x, landmarks[14].y]
                wrist = [landmarks[16].x, landmarks[16].y]
                
                angle = calculate_angle(shoulder, elbow, wrist)
                calibration_data["angles"].append(angle)
                calibration_data["lighting"].append(brightness)
                
                frames_collected += 1
                
                # Show progress bar
                bar_width = 300
                fill = int((frames_collected / 60) * bar_width)
                cv2.rectangle(frame, (50, 200), (50 + bar_width, 230), (50, 50, 50), -1)
                cv2.rectangle(frame, (50, 200), (50 + fill, 230), (0, 255, 255), -1)
                cv2.putText(frame, "Calibrating...", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            else:
                 cv2.putText(frame, "NO PERSON DETECTED", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Calibration", frame)
            cv2.waitKey(1)
        
        # Calculate personalized thresholds
        if calibration_data["angles"]:
            avg_resting_angle = np.mean(calibration_data["angles"])
            self.personal_min, self.personal_max = adaptive_angle_thresholds(
                calibration_data["angles"], 
                flexibility_factor=0.15
            )
            
            print(f"\n[SUCCESS] Calibration Complete!")
            print(f"  Resting Angle: {avg_resting_angle:.1f}°")
            print(f"  Personal Min Angle: {self.personal_min:.1f}°")
            print(f"  Personal Max Angle: {self.personal_max:.1f}°")
            
            self.calibrated = True
            cv2.destroyWindow("Calibration")
            return self.personal_min, self.personal_max
        
        cv2.destroyWindow("Calibration")
        self.personal_min, self.personal_max = MIN_REP_ANGLE, MAX_REP_ANGLE
        return self.personal_min, self.personal_max
    
    def run(self):
        """Main application loop"""
        try:
            if not self.initialize():
                return
            
            # Initialize variables
            counter = 0
            stage = None
            feedback = FEEDBACK_TYPES["no_rep"]
            form_history = deque(maxlen=30)
            angle_history = deque(maxlen=5)
            
            # Statistics
            self.stats = {
                "total_reps": 0,
                "good_form_reps": 0,
                "bad_form_reps": 0,
                "partial_reps": 0,
                "form_accuracy": 0,
                "avg_rep_time": 0,
                "start_time": time.time(),
                "fps": TARGET_FPS,
                "lighting_warnings": 0,
                "occlusion_warnings": 0
            }
            
            # Timing variables
            rep_start_time = None
            rep_times = []
            
            # Performance tracking
            prev_frame = None
            frame_buffer = deque(maxlen=3)
            
            # Get personalized thresholds
            print("\n[INFO] Starting calibration...")
            self.calibrate_user()
            
            print("\n[INFO] Starting AI Gym Trainer...")
            print("[CONTROLS] Q: quit | R: reset | S: save | C: recalibrate | P: pause")
            print("[TIPS] Stand in SIDE VIEW, wear fitted clothing, ensure good lighting")
            
            # --- WINDOW STRETCHING SETUP ---
            # This tells OpenCV: "Allow this window to be resized and stretched"
            window_name = 'AI Gym Trainer Final'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # Optional: Force it to open nicely (e.g., 1280x720) even if video is small
            cv2.resizeWindow(window_name, 1280, 720)
            
            while True:
                # Skip if paused
                if self.paused:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('p'):
                        self.paused = False
                        print("[INFO] Resumed")
                    continue
                
                # Performance monitoring
                fps = self.performance.update()
                self.stats["fps"] = fps
                
                # Capture frame
                success, frame = self.camera.cap.read()  # type: ignore
                if not success:
                    print("[WARNING] Frame capture failed, skipping...")
                    time.sleep(0.1)
                    continue
                
                # Force standard resolution
                frame = cv2.resize(frame, RESOLUTION)
                
                # Store for motion blur reduction
                frame_buffer.append(frame.copy())
                
                # Apply image enhancements based on conditions
                enhanced_frame = frame.copy()
                
                # Check lighting - Edge Case 2 (ENABLED)
                brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # type: ignore # Convert to grayscale first
                if brightness < 60:  # Enable if lighting is poor (lowered from 100)
                    enhanced_frame = adjust_gamma(enhanced_frame, gamma=1.5)
                    # enhanced_frame = apply_histogram_equalization(enhanced_frame)  # Optional
                    if brightness < 80:
                        feedback = FEEDBACK_TYPES["bad_lighting"]
                        self.stats["lighting_warnings"] += 1
                
                # Apply motion blur reduction if needed - Edge Case 6
                if fps < 20 and len(frame_buffer) >= 3:
                    enhanced_frame = apply_motion_blur_reduction(enhanced_frame, frame_buffer)
                
                # Check background clutter - Edge Case 3
                clutter_score = detect_background_clutter(frame, prev_frame)
                if clutter_score > 40:  # High background motion
                    if fps > 20:  # Only warn if FPS is good
                        cv2.putText(enhanced_frame, f"Busy Background: {clutter_score:.1f}%", 
                                   (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                prev_frame = frame.copy()
                
                # Process pose
                frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)  # type: ignore
                
                # Display frame (with or without enhancements based on FPS)
                display_frame = enhanced_frame if fps > 15 else frame
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Check side view - Edge Case 1 (ENABLED but with warning only)
                    is_side_view, side_view_score = check_side_view(landmarks)
                    # Allow all views but warn if not side view
                    if not is_side_view and side_view_score < 40 and fps > 20:
                        cv2.putText(display_frame, "TURN TO SIDE FOR BEST RESULTS", 
                                   (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Check occlusion - Edge Case 4
                    is_occluded, visibility_score = detect_occlusion(landmarks, 0.6)
                    if is_occluded:
                        feedback = FEEDBACK_TYPES["occlusion"]
                        self.stats["occlusion_warnings"] += 1
                    
                    # Select arm (auto-detect)
                    left_vis = landmarks[13].visibility if hasattr(landmarks[13], 'visibility') else 0
                    right_vis = landmarks[14].visibility if hasattr(landmarks[14], 'visibility') else 0
                    
                    if left_vis > right_vis:
                        shoulder = [landmarks[11].x, landmarks[11].y]
                        elbow = [landmarks[13].x, landmarks[13].y]
                        wrist = [landmarks[15].x, landmarks[15].y]
                        side = "LEFT"
                    else:
                        shoulder = [landmarks[12].x, landmarks[12].y]
                        elbow = [landmarks[14].x, landmarks[14].y]
                        wrist = [landmarks[16].x, landmarks[16].y]
                        side = "RIGHT"
                    
                    # Calculate angles with smoothing
                    raw_angle = calculate_angle(shoulder, elbow, wrist)
                    angle_history.append(raw_angle)
                    elbow_angle = np.median(list(angle_history))  # Use median for robustness
                    
                    upper_arm_angle = check_form(shoulder, elbow)
                    
                    # ---------------------------------------------------------
                    # UNIVERSAL PREDICTION BLOCK (Works with Old & New Models)
                    # ---------------------------------------------------------
                    
                    # 1. Calculate Swing (Horizontal distance)
                    swing_dist = abs(shoulder[0] - elbow[0])
                    
                    # 2. Smart Prediction (Try 2 inputs, fallback to 1)
                    try:
                        # Try passing BOTH Angle and Swing (For new model)
                        prediction = self.model.predict([[elbow_angle, swing_dist]])[0]  # type: ignore
                    except ValueError:
                        # If model crashes, it means it's the OLD model (expects 1 input)
                        prediction = self.model.predict([[elbow_angle]])[0]  # type: ignore

                    # 3. Logic Mapping
                    # Old Model: 0=Down, 1=Up
                    # New Model: 1=Down, 2=Up, 3=Bad Form
                    
                    if prediction == 2 or (prediction == 1 and swing_dist < 0.1): 
                        # '1' is UP in old model, '2' is UP in new model
                        current_state = "UP"
                    elif prediction == 1 or prediction == 0: 
                        current_state = "DOWN"
                    elif prediction == 3:
                        current_state = "BAD"
                        # The AI detected cheating!
                        feedback = FEEDBACK_TYPES["elbow_swing"]

                    # ---------------------------------------------------------
                    # FEEDBACK PRIORITY SYSTEM
                    # ---------------------------------------------------------
                    feedback_type = "no_rep" # Default

                    # Priority 1: Physical issues
                    if is_occluded:
                        feedback_type = "occlusion"
                    elif brightness < 50:
                        feedback_type = "bad_lighting"
                    
                    # Priority 2: AI Detected Errors (Only works with New Model)
                    elif prediction == 3:
                        feedback_type = "elbow_swing"
                        
                    # Priority 3: Manual Safety Checks (Backup for Old Model)
                    elif upper_arm_angle > STRICT_MODE_THRESHOLD:
                        # This catches swinging even if the Old Model misses it
                        feedback_type = "elbow_swing"
                    elif elbow_angle < self.personal_min:
                        feedback_type = "partial_rep"
                    elif elbow_angle > self.personal_max:
                        feedback_type = "full_lockout"
                    else:
                        feedback_type = "good_form"
                    
                    # Apply the decision
                    if feedback_type != "elbow_swing": # Don't overwrite if AI caught swing
                        feedback = FEEDBACK_TYPES[feedback_type]
                    
                    # Track form quality
                    if feedback_type == "good_form":
                        form_history.append(100)
                    elif feedback_type in ["elbow_swing", "partial_rep", "full_lockout"]:
                        form_history.append(50)
                    else:
                        form_history.append(0)
                    
                    form_quality = np.mean(form_history) if form_history else 0
                    
                    # Rep counting (only with good conditions)
                    if (feedback_type == "good_form" and not is_occluded and brightness > 60):
                        
                        if current_state == "UP" and stage != "up":
                            stage = "up"
                            if rep_start_time is not None:
                                rep_duration = time.time() - rep_start_time
                                rep_times.append(rep_duration)
                                
                                # Update statistics
                                self.stats["total_reps"] += 1
                                self.stats["good_form_reps"] += 1
                                
                                # Update form accuracy
                                if self.stats["total_reps"] > 0:
                                    self.stats["form_accuracy"] = (
                                        self.stats["good_form_reps"] / self.stats["total_reps"] * 100
                                    )
                                
                                # Update average rep time
                                if rep_times:
                                    self.stats["avg_rep_time"] = np.mean(rep_times)
                                
                                print(f"[REP {counter}] Angle: {elbow_angle:.1f}°, "
                                      f"Time: {rep_duration:.2f}s, FPS: {fps:.1f}")
                            
                            rep_start_time = time.time()
                        
                        elif current_state == "DOWN" and stage == "up":
                            stage = "down"
                            counter += 1
                    
                    # Draw skeleton (always draw)
                    # Convert MediaPipe landmarks to list for your visualizer
                    if results.pose_landmarks:
                        self.visualizer.draw_pose_skeleton(
                            display_frame, 
                            results.pose_landmarks.landmark, 
                            self.mp_pose.POSE_CONNECTIONS  # type: ignore
                        )
                    
                    # Draw UI (simplified if low FPS)
                    self.draw_ui(display_frame, counter, feedback, elbow_angle, 
                               upper_arm_angle, side, form_quality, fps)
                
                else:
                    # No pose detected
                    cv2.putText(display_frame, "NO POSE DETECTED", (150, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(display_frame, "Check: Lighting | Camera View | Distance", 
                               (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display FPS
                fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 10 else (0, 0, 255)
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
                
                # Show frame
                cv2.imshow(window_name, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(max(1, int(1000/TARGET_FPS))) & 0xFF
                counter, rep_times = self.handle_keyboard(key, counter, rep_times)
                
                if key == ord('q'):
                    print("[INFO] Exiting...")
                    break
        
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # SOLVES SIMULATION 2: Always clean up resources
            self.cleanup()
    
    def draw_ui(self, frame, counter, feedback, elbow_angle, upper_arm_angle, side, form_quality, fps):
        """Draw UI using the new VisualFeedback engine"""
        
        # 1. Pack the data into the Metrics structure
        metrics = DisplayMetrics(
            elbow_angle=elbow_angle,
            torso_angle=upper_arm_angle,
            torso_threshold=STRICT_MODE_THRESHOLD,
            fps=fps,
            form_quality=form_quality,
            rep_count=counter,
            stage=self.stage,
            arm_side=side
        )

        # 2. Draw Status Box (Top Left)
        # Note: Feedback dictionary needs to be unpacked or handled. 
        # Assuming feedback is a dict like {'text': '...', 'color': ...}
        fb_text = feedback['text'] if isinstance(feedback, dict) else str(feedback)
        fb_color = feedback['color'] if isinstance(feedback, dict) else (0, 255, 0)
        
        frame = self.visualizer.draw_status_box(
            frame, counter, fb_text, fb_color
        )

        # 3. Draw Debug Panel (Bottom Left) - Only if FPS is good
        if fps > 15:
            frame = self.visualizer.draw_debug_panel(frame, metrics)

        # 4. Draw Stats Panel (Right Side) - Optional
        # self.visualizer.draw_stats_panel(frame, self.stats)

        # 5. Draw Progress Bar (Center)
        frame = self.visualizer.draw_progress_bar(
            frame, form_quality, 100, "Form Quality"
        )
        
        # 6. Draw FPS (Top Right)
        frame = self.visualizer.draw_fps(frame, fps)

        return frame
    
    def handle_keyboard(self, key, counter, rep_times):
        """Handle keyboard input and return updated values"""
        if key == ord('q'):
            raise KeyboardInterrupt
        elif key == ord('r'):
            counter = 0
            rep_times.clear()
            self.stats = {k: 0 if isinstance(v, (int, float)) else v for k, v in self.stats.items()}  # type: ignore
            self.stats["start_time"] = time.time()
            self.stats["form_accuracy"] = 0
            self.stats["avg_rep_time"] = 0
            print("[INFO] Counter and stats reset!")
        elif key == ord('s'):
            self.save_session(counter, rep_times)
        elif key == ord('c'):
            print("[INFO] Recalibrating...")
            self.calibrate_user()
        elif key == ord('p'):
            self.paused = not self.paused
            print(f"[INFO] {'Paused' if self.paused else 'Resumed'}")
        
        return counter, rep_times
    
    def save_session(self, counter, rep_times):
        """Save workout session to file"""
        if counter == 0:
            print("[INFO] No reps to save")
            return
        
        workout_data = {
            "date": datetime.datetime.now().isoformat(),
            "exercise": "bicep_curl",
            "total_reps": counter,
            "stats": self.stats,
            "rep_times": list(rep_times),
            "duration": time.time() - self.stats["start_time"],  # type: ignore
            "fps_avg": np.mean([self.stats["fps"]] * len(rep_times)) if rep_times else 0,  # type: ignore
            "warnings": {
                "lighting": self.stats["lighting_warnings"],  # type: ignore
                "occlusion": self.stats["occlusion_warnings"]  # type: ignore
            },
            "personal_thresholds": {
                "min_angle": self.personal_min,
                "max_angle": self.personal_max
            }
        }
        
        filename = f"workout_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(workout_data, f, indent=2)
        
        print(f"[SUCCESS] Workout saved to {filename}")
        return filename
    
    def cleanup(self):
        """Clean up all resources - SOLVES SIMULATION 2"""
        print("\n[INFO] Cleaning up resources...")
        
        # Release camera
        self.camera.release()
        
        # Close pose detection
        if self.pose:
            self.pose.close()
        
        # Destroy all windows
        cv2.destroyAllWindows()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Final statistics
        if self.stats:
            duration = time.time() - self.stats.get("start_time", time.time())
            print("\n" + "=" * 60)
            print("SESSION SUMMARY")
            print("=" * 60)
            print(f"Total Reps: {self.stats.get('total_reps', 0)}")
            print(f"Good Form Reps: {self.stats.get('good_form_reps', 0)}")
            print(f"Bad Form Reps: {self.stats.get('bad_form_reps', 0)}")
            print(f"Form Accuracy: {self.stats.get('form_accuracy', 0):.1f}%")
            print(f"Average FPS: {self.stats.get('fps', 0):.1f}")
            print(f"Lighting Warnings: {self.stats.get('lighting_warnings', 0)}")
            print(f"Occlusion Warnings: {self.stats.get('occlusion_warnings', 0)}")
            print(f"Session Duration: {duration:.1f}s")
            
            if self.stats.get('total_reps', 0) > 0 and duration > 0:
                print(f"Reps per Minute: {(self.stats['total_reps'] / duration * 60):.1f}")
        
        print("\n[INFO] AI Gym Trainer closed successfully!")

# ===========================================
# MAIN ENTRY POINT
# ===========================================
def main():
    """Main entry point with exception handling"""
    trainer = None
    
    try:
        trainer = AIGymTrainer()
        trainer.run()
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    except Exception as e:
        print(f"\n[ERROR] Critical error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Double safety: ensure cleanup even if exception occurs during cleanup
        if trainer:
            try:
                trainer.cleanup()
            except:
                pass
        
        # Force exit if window hangs
        cv2.destroyAllWindows()
        for i in range(5):
            cv2.waitKey(1)
        
        print("\n[INFO] System shutdown complete")

if __name__ == "__main__":
    # Add Windows-specific fix for camera access
    if sys.platform == "win32":
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
    
    main()