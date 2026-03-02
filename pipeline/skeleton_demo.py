"""
AI GYM - SKELETON POSE VISUALIZATION ONLY
Simple demo to show MediaPipe pose detection without exercise tracking
Perfect for demonstrations and testing
"""

import cv2
import mediapipe as mp
import time

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def draw_skeleton(frame, landmarks):
    """Draw skeleton on frame"""
    mp_drawing.draw_landmarks(
        frame,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    return frame

def main():
    print("\n" + "="*60)
    print("AI GYM - SKELETON VISUALIZATION")
    print("="*60)
    print("Press 'Q' to quit")
    print("Press 'S' to save screenshot")
    print("="*60 + "\n")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    
    print(f"[SUCCESS] Camera opened at {int(cap.get(3))}x{int(cap.get(4))}")
    
    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    
    # FPS calculation
    prev_time = time.time()
    fps_list = []
    screenshot_count = 0
    
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
            if results.pose_landmarks:
                display_frame = draw_skeleton(display_frame, results.pose_landmarks)
                
                # Add status text
                cv2.putText(
                    display_frame,
                    "✓ POSE DETECTED",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    display_frame,
                    "NO POSE DETECTED",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
            prev_time = current_time
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)
            
            # Display FPS
            cv2.putText(
                display_frame,
                f"FPS: {avg_fps:.1f}",
                (display_frame.shape[1] - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            # Display instructions
            cv2.putText(
                display_frame,
                "Press 'Q' to quit | 'S' to screenshot",
                (20, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Show frame
            cv2.imshow('AI Gym - Skeleton Demo', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[INFO] Quitting...")
                break
            elif key == ord('s') or key == ord('S'):
                screenshot_count += 1
                filename = f"skeleton_screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"[INFO] Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("\n[INFO] Cleanup complete")
        print(f"Average FPS: {avg_fps:.1f}")
        print("="*60)

if __name__ == "__main__":
    main()
