"""
AI GYM - AUTO SKELETON RECORDER
Automatically records skeleton-only video during your exercise session.
Delegates recording logic to SkeletonRecorder (skeleton_recorder.py)
to eliminate code duplication.
"""

import cv2
import mediapipe as mp
import time
import argparse

from skeleton_recorder import SkeletonRecorder   # reuse — no duplicate code

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
    print(f"Output Directory:   {args.output}")
    print("="*70)
    print("\nGet ready! Recording starts automatically after countdown.")
    print("Stand in view of the camera and perform your exercise.")
    print("Press Q to stop early.")
    print("="*70 + "\n")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    actual_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[SUCCESS] Camera opened at {actual_width}x{actual_height}\n")

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

    # SkeletonRecorder handles the black-background video file — no duplicate code
    recorder = SkeletonRecorder(output_dir=args.output)

    # Countdown
    print("Starting in...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("[REC] RECORDING!\n")

    skeleton_path = recorder.start_recording(actual_width, actual_height, fps=30)
    start_time = time.time()
    elapsed = 0.0
    fps_list = []
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break

            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                print(f"\n[INFO] Recording complete ({args.duration}s)")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True
            display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # Draw skeleton overlay on the live preview
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                # Delegate black-background skeleton frame to SkeletonRecorder
                recorder.record_frame(results.pose_landmarks, actual_width, actual_height)

            # FPS
            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)

            # Overlay — recording indicator
            time_left = args.duration - elapsed
            mins, secs = divmod(int(time_left), 60)
            cv2.circle(display_frame, (30, 30), 15, (0, 0, 255), -1)
            cv2.putText(display_frame, f"REC {mins:02d}:{secs:02d}",
                        (60, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Frames: {recorder.frame_count}",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}",
                        (actual_width - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            pose_color = (0, 255, 0) if results.pose_landmarks else (0, 0, 255)
            pose_text  = "POSE DETECTED" if results.pose_landmarks else "NO POSE - Stand in view!"
            cv2.putText(display_frame, pose_text,
                        (20, actual_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, pose_color, 2)
            cv2.putText(display_frame, "Press Q to stop early",
                        (20, actual_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('AI Gym - Recording Session', display_frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                print(f"\n[INFO] Recording stopped early ({elapsed:.1f}s)")
                break

            if int(elapsed) % 10 == 0 and int(elapsed) > 0 and recorder.frame_count % 300 == 0:
                print(f"  {int(elapsed)}s / {args.duration}s  |  {recorder.frame_count} frames")

    except KeyboardInterrupt:
        print("\n[INFO] Recording interrupted by user")

    finally:
        recorder.stop_recording()
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

        print("\n" + "="*70)
        print("SESSION COMPLETE")
        print("="*70)
        print(f"Skeleton video: {skeleton_path}")
        print(f"Total frames:   {recorder.frame_count}")
        print(f"Duration:       {elapsed:.1f}s")
        print(f"Average FPS:    {avg_fps:.1f}")
        print("="*70)

if __name__ == "__main__":
    main()
