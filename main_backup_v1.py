import cv2
import mediapipe as mp
import numpy as np
import pickle
import math
import os
import time
import json
import datetime
from collections import deque

# ===========================================
# CONFIGURATION
# ===========================================
# FIXED: Pointing to the correct folder location
MODEL_PATH = os.path.join("data", "models", "bicep_model.pkl")

STRICT_MODE_THRESHOLD = 25  # Maximum allowed upper arm movement (degrees)
CALIBRATION_TIME = 3  # Seconds for calibration
MIN_REP_ANGLE = 60  # Minimum angle for a valid rep
MAX_REP_ANGLE = 160  # Maximum angle for a valid rep
MIN_REP_TIME = 1.0  # Minimum time for a rep (seconds)
MAX_REP_TIME = 3.0  # Maximum time for a rep (seconds)

# ===========================================
# FEEDBACK SYSTEM
# ===========================================
FEEDBACK_TYPES = {
    "good_form": {"text": "GOOD FORM", "color": (0, 255, 0)},
    "elbow_swing": {"text": "ELBOW SWING!", "color": (0, 0, 255)},
    "partial_rep": {"text": "PARTIAL REP", "color": (0, 165, 255)},
    "too_fast": {"text": "TOO FAST!", "color": (0, 255, 255)},
    "too_slow": {"text": "TOO SLOW", "color": (255, 165, 0)},
    "full_lockout": {"text": "FULL LOCKOUT", "color": (255, 255, 0)},
    "no_rep": {"text": "NO REP", "color": (100, 100, 100)}
}

# ===========================================
# HELPER FUNCTIONS
# ===========================================
def calculate_angle(a, b, c):
    """Calculates angle between three points (Shoulder-Elbow-Wrist)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def check_form(shoulder, elbow):
    """Checks if the upper arm is vertical"""
    vertical_point = [shoulder[0], shoulder[1] + 0.5]
    vec_arm = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
    vec_vertical = np.array([vertical_point[0] - shoulder[0], vertical_point[1] - shoulder[1]])
    
    cosine_angle = np.dot(vec_arm, vec_vertical) / (np.linalg.norm(vec_arm) * np.linalg.norm(vec_vertical))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle

def validate_angles(shoulder, elbow, wrist):
    if any(coord == 0 for coord in shoulder + elbow + wrist):
        return False
    angle = calculate_angle(shoulder, elbow, wrist)
    return MIN_REP_ANGLE <= angle <= MAX_REP_ANGLE

def save_workout_session(counter, stats, rep_times):
    """Save workout data to JSON file inside data/logs"""
    # Create logs folder if it doesn't exist
    log_dir = os.path.join("data", "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    workout_data = {
        "date": datetime.datetime.now().isoformat(),
        "exercise": "bicep_curl",
        "total_reps": counter,
        "strict_mode_threshold": STRICT_MODE_THRESHOLD,
        "stats": stats,
        "rep_times": rep_times,
        "duration": time.time() - stats.get("start_time", time.time())
    }
    
    filename = f"workout_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(log_dir, filename)
    
    with open(filepath, "w") as f:
        json.dump(workout_data, f, indent=2)
    print(f"\n[INFO] Workout saved to: {filepath}")
    return filename

def draw_progress_bar(img, quality_percent, position=(500, 50), width=200, height=20):
    fill_width = int(width * (quality_percent / 100))
    cv2.rectangle(img, position, (position[0] + width, position[1] + height), (50, 50, 50), -1)
    cv2.rectangle(img, position, (position[0] + fill_width, position[1] + height), (0, 255, 0), -1)
    cv2.rectangle(img, position, (position[0] + width, position[1] + height), (255, 255, 255), 2)
    
    color = (0, 255, 0) if quality_percent > 70 else (0, 255, 255) if quality_percent > 40 else (0, 0, 255)
    cv2.putText(img, f"Form: {int(quality_percent)}%", (position[0], position[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def calibrate_position(cap, pose, duration=3):
    print(f"\n[INFO] Calibration starting in {duration} seconds...")
    print("Please stand in your starting position (arms down)")
    
    calibration_angles = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        success, img = cap.read()
        if not success: continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Use right arm for calibration
            shoulder = [landmarks[12].x, landmarks[12].y]
            elbow = [landmarks[14].x, landmarks[14].y]
            wrist = [landmarks[16].x, landmarks[16].y]
            
            angle = calculate_angle(shoulder, elbow, wrist)
            calibration_angles.append(angle)
            
            remaining = duration - (time.time() - start_time)
            cv2.putText(img, f"Calibrating: {remaining:.1f}s", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("AI Gym Trainer - Calibration", img)
            cv2.waitKey(1)
    
    if calibration_angles:
        avg_angle = sum(calibration_angles) / len(calibration_angles)
        print(f"[INFO] Calibration complete. Baseline angle: {avg_angle:.1f}")
        return avg_angle
    return None

# ===========================================
# MAIN FUNCTION
# ===========================================
def main():
    print("AI GYM TRAINER - BICEP CURL ANALYZER")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file '{MODEL_PATH}' not found!")
        return
    
    print("[INFO] Loading AI model...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("[SUCCESS] Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        return
    
    # Optional Calibration
    print("Press 'c' during workout to calibrate.")
    
    counter = 0
    stage = None
    feedback = FEEDBACK_TYPES["no_rep"]
    form_history = deque(maxlen=30)
    
    stats = {
        "total_reps": 0, "good_form_reps": 0, "bad_form_reps": 0,
        "partial_reps": 0, "form_accuracy": 0, "avg_rep_time": 0,
        "start_time": time.time()
    }
    
    rep_start_time = None
    rep_times = []
    angle_history = deque(maxlen=5)
    
    paused = False
    
    try:
        while True:
            if paused:
                cv2.putText(img, "PAUSED", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                cv2.imshow("AI Gym Trainer - STRICT MODE", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('p'): paused = False
                elif key == ord('q'): break
                continue
            
            success, img = cap.read()
            if not success: continue
            
            img = cv2.resize(img, (1280, 720))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_vis = landmarks[13].visibility
                right_vis = landmarks[14].visibility
                
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
                
                # Calculations
                raw_angle = calculate_angle(shoulder, elbow, wrist)
                angle_history.append(raw_angle)
                elbow_angle = sum(angle_history) / len(angle_history)
                upper_arm_angle = check_form(shoulder, elbow)
                is_valid_angle = validate_angles(shoulder, elbow, wrist)
                
                # AI Prediction
                prediction = model.predict([[elbow_angle]])[0]
                current_state = "UP" if prediction == 1 else "DOWN"
                
                # Logic
                feedback_type = "no_rep"
                if not is_valid_angle: feedback_type = "no_rep"
                elif upper_arm_angle > STRICT_MODE_THRESHOLD: feedback_type = "elbow_swing"
                elif elbow_angle < MIN_REP_ANGLE: feedback_type = "partial_rep"
                elif elbow_angle > MAX_REP_ANGLE: feedback_type = "full_lockout"
                else: feedback_type = "good_form"
                
                feedback = FEEDBACK_TYPES[feedback_type]
                box_color = feedback["color"]
                
                # Form Quality Math
                if feedback_type == "good_form": form_history.append(100)
                elif feedback_type in ["elbow_swing", "partial_rep", "full_lockout"]: form_history.append(50)
                else: form_history.append(0)
                form_quality = sum(form_history) / len(form_history) if form_history else 0
                
                # Rep Counting
                if feedback_type == "good_form":
                    if current_state == "UP" and stage != "up":
                        stage = "up"
                        if rep_start_time is not None:
                            rep_duration = time.time() - rep_start_time
                            rep_times.append(rep_duration)
                            
                            if rep_duration < MIN_REP_TIME: feedback_type = "too_fast"
                            elif rep_duration > MAX_REP_TIME: feedback_type = "too_slow"
                            
                            stats["total_reps"] += 1
                            stats["good_form_reps"] += 1
                            if stats["total_reps"] > 0:
                                stats["form_accuracy"] = (stats["good_form_reps"] / stats["total_reps"]) * 100
                            if rep_times:
                                stats["avg_rep_time"] = sum(rep_times) / len(rep_times)
                        
                        rep_start_time = time.time()
                    
                    elif current_state == "DOWN" and stage == "up":
                        stage = "down"
                        counter += 1
                
                elif feedback_type in ["elbow_swing", "partial_rep", "full_lockout"] and current_state == "UP" and stage != "up":
                     # Count bad reps too but don't increment main counter
                     stats["bad_form_reps"] += 1
                     stage = "up"

                # Drawing
                mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Info Box
                cv2.rectangle(img, (0, 0), (400, 120), box_color, cv2.FILLED)
                cv2.rectangle(img, (0, 0), (400, 120), (255, 255, 255), 2)
                cv2.putText(img, f"REPS: {counter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(img, feedback["text"], (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Live Data
                cv2.rectangle(img, (0, 600), (450, 720), (40, 40, 40), cv2.FILLED)
                cv2.putText(img, f"Angle: {int(elbow_angle)}", (10, 640), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
                cv2.putText(img, f"Swing: {int(upper_arm_angle)}", (10, 670), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255) if upper_arm_angle > STRICT_MODE_THRESHOLD else (0,255,0), 2)
                
                # Stats Panel
                cv2.rectangle(img, (880, 0), (1280, 150), (40, 40, 40), cv2.FILLED)
                cv2.putText(img, f"Accuracy: {stats['form_accuracy']:.1f}%", (900, 60), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,255,255), 2)
                cv2.putText(img, f"Avg Time: {stats['avg_rep_time']:.2f}s", (900, 90), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,255,255), 2)
                
                draw_progress_bar(img, int(form_quality), (500, 50))
                
                cv2.putText(img, "Q=Quit R=Reset S=Save C=Calib P=Pause", (10, 730), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)

            else:
                 cv2.putText(img, "NO POSE DETECTED", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            
            cv2.imshow("AI Gym Trainer - STRICT MODE", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'): 
                counter = 0
                stats = {k: 0 if isinstance(v, (int, float)) else v for k, v in stats.items()}
                print("[INFO] Reset!")
            elif key == ord('s'): 
                save_workout_session(counter, stats, rep_times)
            elif key == ord('c'): 
                calibrate_position(cap, pose)
            elif key == ord('p'): 
                paused = True
    
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

if __name__ == "__main__":
    main()