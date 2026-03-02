import cv2
import mediapipe as mp
import csv
import os

# --- CONFIGURATION ---
VIDEO_FOLDER = "videos"  # The folder where you put your 62 videos
OUTPUT_FILE = "bicep_dataset.csv"
# ---------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Create CSV Header
header = [
    'video_name', 'frame', 
    'shoulder_x', 'shoulder_y', 'shoulder_z', 'shoulder_v',
    'elbow_x', 'elbow_y', 'elbow_z', 'elbow_v',
    'wrist_x', 'wrist_y', 'wrist_z', 'wrist_v',
    'label'
]

# Get list of all video files
video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
print(f"Found {len(video_files)} videos. Starting processing...")

with open(OUTPUT_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for video_name in video_files:
        video_path = os.path.join(VIDEO_FOLDER, video_name)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        print(f"Processing: {video_name}...")

        while True:
            success, img = cap.read()
            if not success:
                break 

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Auto-detect Left vs Right arm
                right_vis = landmarks[16].visibility
                left_vis = landmarks[15].visibility
                
                if right_vis > left_vis:
                    s_idx, e_idx, w_idx = 12, 14, 16 
                else:
                    s_idx, e_idx, w_idx = 11, 13, 15

                shoulder = landmarks[s_idx]
                elbow = landmarks[e_idx]
                wrist = landmarks[w_idx]

                row = [
                    video_name, frame_count,
                    shoulder.x, shoulder.y, shoulder.z, shoulder.visibility,
                    elbow.x, elbow.y, elbow.z, elbow.visibility,
                    wrist.x, wrist.y, wrist.z, wrist.visibility,
                    "correct" # Assuming these are all correct forms for now
                ]
                writer.writerow(row)

            frame_count += 1
            
            # Show progress (Optional - comment out if it's too slow)
            cv2.imshow("Processing...", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

cv2.destroyAllWindows()
print(f"Done! All data saved to {OUTPUT_FILE}")