import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# --- CONFIG ---
FILE_NAME = "training_data.csv"
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

print("[DEBUG] Attempting to open camera...")

# Try DSHOW (DirectShow) for Windows compatibility
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

if not cap.isOpened():
    print("[ERROR] Could not open camera at index 0.")
    print("[TIP] Make sure no other apps (Zoom, Teams, previous python script) are using it.")
    exit()

print("[DEBUG] Camera opened successfully!")
print("[DEBUG] Waking up camera sensors...")
time.sleep(2.0) # Give camera time to warm up

# --- FUNCTIONS ---
def calculate_angle(a, b, c):
    a = np.array(a) # Shoulder
    b = np.array(b) # Elbow
    c = np.array(c) # Wrist
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

def get_body_swing(shoulder, hip, elbow):
    return abs(shoulder[0] - elbow[0])

# --- CSV SETUP ---
if not os.path.exists(FILE_NAME):
    with open(FILE_NAME, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'elbow_angle', 'swing_dist'])
    print(f"[DEBUG] Created new file: {FILE_NAME}")
else:
    print(f"[DEBUG] Appending to existing file: {FILE_NAME}")

print("="*30)
print(" CONTROLS:")
print(" Press '1' -> Save DOWN (Straight Arm)")
print(" Press '2' -> Save UP (Curled Arm)")
print(" Press '3' -> Save BAD FORM")
print(" Press 'q' -> QUIT")
print("="*30)

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret: 
        print("[ERROR] Failed to read frame from camera.")
        break
    
    frame_count += 1
    
    # Process Frame
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates
        shoulder = [landmarks[12].x, landmarks[12].y]
        elbow = [landmarks[14].x, landmarks[14].y]
        wrist = [landmarks[16].x, landmarks[16].y]
        hip = [landmarks[24].x, landmarks[24].y]
        
        # Calculate Features
        angle = calculate_angle(shoulder, elbow, wrist)
        swing = get_body_swing(shoulder, hip, elbow)
        
        # Visuals
        cv2.putText(frame, f"Angle: {int(angle)}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Swing: {swing:.2f}", (50, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        
        # Draw skeleton
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Collect Data on Key Press
        k = cv2.waitKey(1)
        if k == ord('1'): # DOWN
            with open(FILE_NAME, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([1, angle, swing])
                print(f"Saved DOWN: {angle:.1f}")
                
        elif k == ord('2'): # UP
            with open(FILE_NAME, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([2, angle, swing])
                print(f"Saved UP: {angle:.1f}")

        elif k == ord('3'): # BAD FORM
            with open(FILE_NAME, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([3, angle, swing])
                print(f"Saved BAD: {angle:.1f}")

        elif k == ord('q'):
            print("[INFO] Quitting...")
            break
    else:
        # Just show the frame if no person detected yet
        cv2.imshow("Data Collector", frame)
        if cv2.waitKey(1) == ord('q'): break
        continue

    cv2.imshow("Data Collector", frame)

cap.release()
cv2.destroyAllWindows()