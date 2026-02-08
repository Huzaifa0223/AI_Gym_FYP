"""
AI GYM - SKELETON-ONLY VIDEO RECORDER
Records workout session and saves skeleton-only video (no person visible)
Perfect for analyzing body joint movements and pose tracking
"""

import cv2
import mediapipe as mp
import time
import os
from datetime import datetime
import numpy as np

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

class SkeletonRecorder:
    def __init__(self, output_dir="session_recordings"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create session-specific folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, f"session_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.skeleton_writer = None
        self.recording = False
        self.frame_count = 0
        
    def start_recording(self, width, height, fps=30):
        """Start recording skeleton-only video"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        skeleton_path = os.path.join(self.session_dir, f"skeleton_only_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.skeleton_writer = cv2.VideoWriter(skeleton_path, fourcc, fps, (width, height))
        self.recording = True
        self.frame_count = 0
        
        print(f"[INFO] Recording started: {skeleton_path}")
        return skeleton_path
    
    def record_frame(self, landmarks, width, height):
        """Record a frame with skeleton only (black background)"""
        if not self.recording or self.skeleton_writer is None:
            return
        
        # Create black background
        skeleton_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw skeleton on black background
        if landmarks:
            mp_drawing.draw_landmarks(
                skeleton_frame,
                landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),  # Green landmarks
                    thickness=3,
                    circle_radius=4
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 255),  # Cyan connections
                    thickness=3
                )
            )
        
        self.skeleton_writer.write(skeleton_frame)
        self.frame_count += 1
    
    def stop_recording(self):
        """Stop recording and save video"""
        if self.skeleton_writer is not None:
            self.skeleton_writer.release()
            self.recording = False
            print(f"[SUCCESS] Skeleton video saved ({self.frame_count} frames)")
            print(f"[INFO] Session folder: {self.session_dir}")
            return self.session_dir
        return None

def main():
    print("\n" + "="*60)
    print("AI GYM - SKELETON RECORDER")
    print("Records skeleton-only video for movement analysis")
    print("="*60)
    print("Controls:")
    print("  R - Start/Stop Recording")
    print("  Q - Quit")
    print("  S - Screenshot")
    print("="*60 + "\n")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    width, height = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[SUCCESS] Camera opened at {actual_width}x{actual_height}")
    
    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    
    # Initialize recorder
    recorder = SkeletonRecorder()
    
    # FPS calculation
    prev_time = time.time()
    fps_list = []
    screenshot_count = 0
    recording_start_time = None
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break
            
            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process pose
            results = pose.process(rgb_frame)
            
            # Convert back to BGR
            rgb_frame.flags.writeable = True
            display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Draw skeleton if detected
            skeleton_detected = False
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                skeleton_detected = True
                
                # Record frame if recording
                if recorder.recording:
                    recorder.record_frame(results.pose_landmarks, actual_width, actual_height)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
            prev_time = current_time
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)
            
            # Display overlay
            overlay = display_frame.copy()
            
            # Recording indicator
            if recorder.recording:
                # Red recording circle
                cv2.circle(overlay, (30, 30), 15, (0, 0, 255), -1)
                
                # Recording time
                rec_duration = int(time.time() - recording_start_time)
                mins, secs = divmod(rec_duration, 60)
                cv2.putText(
                    overlay,
                    f"REC {mins:02d}:{secs:02d}",
                    (60, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
                
                # Frame count
                cv2.putText(
                    overlay,
                    f"Frames: {recorder.frame_count}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
            else:
                cv2.putText(
                    overlay,
                    "Press 'R' to Record",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
            
            # Pose status
            status_text = "✓ POSE DETECTED" if skeleton_detected else "NO POSE"
            status_color = (0, 255, 0) if skeleton_detected else (0, 0, 255)
            cv2.putText(
                overlay,
                status_text,
                (20, actual_height - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2
            )
            
            # FPS
            cv2.putText(
                overlay,
                f"FPS: {avg_fps:.1f}",
                (actual_width - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            # Instructions
            cv2.putText(
                overlay,
                "R: Record | Q: Quit | S: Screenshot",
                (20, actual_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Show frame
            cv2.imshow('AI Gym - Skeleton Recorder', overlay)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[INFO] Quitting...")
                break
            elif key == ord('r') or key == ord('R'):
                if not recorder.recording:
                    recorder.start_recording(actual_width, actual_height, int(avg_fps))
                    recording_start_time = time.time()
                else:
                    session_dir = recorder.stop_recording()
                    print(f"[INFO] Video saved to: {session_dir}")
                    # Create new recorder for next session
                    recorder = SkeletonRecorder()
            elif key == ord('s') or key == ord('S'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"[INFO] Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Stop recording if still active
        if recorder.recording:
            session_dir = recorder.stop_recording()
            print(f"[INFO] Video saved to: {session_dir}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("\n[INFO] Cleanup complete")
        print(f"Average FPS: {avg_fps:.1f}")
        print("="*60)

if __name__ == "__main__":
    main()
