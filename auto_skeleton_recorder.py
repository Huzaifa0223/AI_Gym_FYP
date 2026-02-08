"""
AI GYM - AUTO SKELETON RECORDER
Automatically records skeleton-only video during your exercise session
Perfect for post-session analysis of body movements
"""

import cv2
import mediapipe as mp
import time
import os
import argparse
from datetime import datetime
import numpy as np

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def main():
    parser = argparse.ArgumentParser(description='AI Gym - Auto Skeleton Recorder')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Recording duration in seconds (default: 60)')
    parser.add_argument('--output', type=str, default='skeleton_recordings',
                       help='Output directory for recordings')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("AI GYM - AUTO SKELETON RECORDER")
    print("="*70)
    print(f"Recording Duration: {args.duration} seconds")
    print(f"Output Directory: {args.output}")
    print("="*70)
    print("\n🎬 Get ready! Recording will start automatically...")
    print("Stand in view of camera and perform your exercise")
    print("Press 'Q' to stop early")
    print("="*70 + "\n")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    print(f"[SUCCESS] Camera opened at {actual_width}x{actual_height}\n")
    
    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    
    # Setup video writer for skeleton-only video
    skeleton_path = os.path.join(args.output, f"skeleton_only_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    skeleton_writer = cv2.VideoWriter(skeleton_path, fourcc, 30, (actual_width, actual_height))
    
    # Countdown
    print("Starting in...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("🔴 RECORDING!\n")
    
    start_time = time.time()
    frame_count = 0
    fps_list = []
    prev_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break
            
            # Check duration
            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                print(f"\n✅ Recording complete ({args.duration}s)")
                break
            
            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process pose
            results = pose.process(rgb_frame)
            
            # Create black background for skeleton-only video
            skeleton_frame = np.zeros((actual_height, actual_width, 3), dtype=np.uint8)
            
            # Create display frame (with person)
            rgb_frame.flags.writeable = True
            display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Draw skeleton on both frames
            if results.pose_landmarks:
                # Draw on display frame (with person)
                mp_drawing.draw_landmarks(
                    display_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Draw on skeleton-only frame (black background)
                mp_drawing.draw_landmarks(
                    skeleton_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0),  # Green landmarks
                        thickness=4,
                        circle_radius=5
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 255),  # Cyan connections
                        thickness=4
                    )
                )
            
            # Write skeleton frame to video
            skeleton_writer.write(skeleton_frame)
            frame_count += 1
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
            prev_time = current_time
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)
            
            # Time remaining
            time_left = args.duration - elapsed
            mins, secs = divmod(int(time_left), 60)
            
            # Add overlay to display frame
            # Recording indicator
            cv2.circle(display_frame, (30, 30), 15, (0, 0, 255), -1)
            cv2.putText(
                display_frame,
                f"REC {mins:02d}:{secs:02d}",
                (60, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )
            
            # Frame count
            cv2.putText(
                display_frame,
                f"Frames: {frame_count}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # FPS
            cv2.putText(
                display_frame,
                f"FPS: {avg_fps:.1f}",
                (actual_width - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            # Pose status
            if results.pose_landmarks:
                cv2.putText(
                    display_frame,
                    "✓ POSE DETECTED",
                    (20, actual_height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    display_frame,
                    "NO POSE - Stand in view!",
                    (20, actual_height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
            
            # Instructions
            cv2.putText(
                display_frame,
                "Press 'Q' to stop early",
                (20, actual_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Show live preview
            cv2.imshow('AI Gym - Recording Session', display_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print(f"\n⚠️  Recording stopped early ({elapsed:.1f}s)")
                break
            
            # Progress indicator in console (every 10 seconds)
            if int(elapsed) % 10 == 0 and int(elapsed) > 0 and frame_count % 300 == 0:
                print(f"  ⏱️  {int(elapsed)}s / {args.duration}s - {frame_count} frames")
    
    except KeyboardInterrupt:
        print("\n[INFO] Recording interrupted by user")
    
    finally:
        # Cleanup
        skeleton_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        
        # Summary
        print("\n" + "="*70)
        print("SESSION COMPLETE")
        print("="*70)
        print(f"✅ Skeleton video saved: {skeleton_path}")
        print(f"📊 Total frames: {frame_count}")
        print(f"⏱️  Duration: {elapsed:.1f}s")
        print(f"🎬 Average FPS: {avg_fps:.1f}")
        print(f"📁 Output folder: {args.output}")
        print("="*70)
        print("\n💡 TIP: Play the skeleton video to analyze your body movements!")
        print(f"   Command: vlc \"{skeleton_path}\" (or open in any video player)\n")

if __name__ == "__main__":
    main()
