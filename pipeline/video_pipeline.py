"""
Video Processing Pipeline for Training Data Generation
Industrial-Grade Implementation with Data Augmentation
- Folder-based automatic labeling (good_form=1, bad_form=0)
- Mirroring augmentation (left/right hand detection)
- Automatic age_group and exercise extraction from paths
"""
# -*- coding: utf-8 -*-
import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import argparse

class VideoProcessor:
    """Process exercise videos to extract training data with augmentation"""
    
    def __init__(self, output_dir='data/training', log_file='processing_log.txt'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Log file for persistent progress tracking
        self.log_file = log_file

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose  # type: ignore[attr-defined]
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils  # type: ignore[attr-defined]

        # Valid exercises and age groups
        self.valid_exercises = ['bicep', 'back', 'chest']
        self.valid_age_groups = ['children', 'adult', 'senior']
        self.label_mapping = {
            'good_form': 1,
            'bad_form': 0
        }

    def log_progress(self, message: str):
        """Append a progress message to the log file with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def extract_metadata_from_path(self, video_path: str) -> Optional[Dict]:
        """
        Extract exercise_type, age_group, and label from folder structure
        Expected: videos/{exercise}/{age_group}/{form_type}/video.mp4
        
        Args:
            video_path: Full path to video file
            
        Returns:
            Dict with exercise_type, age_group, label, or None if invalid
        """
        try:
            parts = Path(video_path).parts
            # Find 'videos' index
            if 'videos' not in parts:
                return None
            
            videos_idx = parts.index('videos')
            
            # Extract: videos/[exercise]/[age_group]/[form_type]/[video_name]
            if videos_idx + 3 >= len(parts):
                return None
            
            exercise = parts[videos_idx + 1].lower()
            age_group = parts[videos_idx + 2].lower()
            form_type = parts[videos_idx + 3].lower()
            
            # Validate
            if exercise not in self.valid_exercises:
                return None
            if age_group not in self.valid_age_groups:
                return None
            if form_type not in self.label_mapping:
                return None
            
            return {
                'exercise_type': exercise,
                'age_group': age_group,
                'label': self.label_mapping[form_type],
                'form_type': form_type
            }
        except (IndexError, ValueError) as e:
            print(f"[WARNING] Could not extract metadata from {video_path}: {e}")
            return None
    
    def mirror_landmarks(self, landmarks: Dict) -> Dict:
        """
        Create mirrored version of landmarks (flip X-coordinates)
        Converts right-hand to left-hand or vice versa
        
        Args:
            landmarks: Dictionary with landmark_N_x, landmark_N_y, etc.
            
        Returns:
            New dictionary with mirrored X-coordinates
        """
        mirrored = {}
        
        for key, value in landmarks.items():
            if '_x' in key:
                # Flip X coordinate: x_new = 1 - x_old
                mirrored[key] = 1.0 - float(value) if isinstance(value, (int, float)) else value
            else:
                # Keep Y, Z, visibility as-is
                mirrored[key] = value
        
        return mirrored
    
    def extract_landmarks_from_video(self, video_path: str, metadata: Dict) -> List[Dict]:
        """
        Extract pose landmarks from a video file with augmentation
        
        Args:
            video_path: Path to video file
            metadata: Dictionary with exercise_type, age_group, label
            
        Returns:
            List of dictionaries containing frame data (original + mirrored)
        """
        cap = cv2.VideoCapture(video_path)  # type: ignore[attr-defined]
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return []
        
        frame_data = []
        frame_count = 0
        
        print(f"    Processing: {os.path.basename(video_path)}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore[attr-defined]
            
            # Process with MediaPipe
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Extract landmark coordinates
                landmarks = {}
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmarks[f'landmark_{idx}_x'] = landmark.x
                    landmarks[f'landmark_{idx}_y'] = landmark.y
                    landmarks[f'landmark_{idx}_z'] = landmark.z
                    landmarks[f'landmark_{idx}_visibility'] = landmark.visibility
                
                # Add metadata
                landmarks['frame'] = frame_count
                landmarks['video_path'] = video_path
                landmarks['video_file'] = os.path.basename(video_path)
                landmarks['exercise_type'] = metadata['exercise_type']
                landmarks['age_group'] = metadata['age_group']
                landmarks['label'] = metadata['label']
                landmarks['form_type'] = metadata['form_type']
                landmarks['augmentation'] = 'original'
                
                frame_data.append(landmarks)
                
                # ===== DATA AUGMENTATION: MIRRORING =====
                mirrored = self.mirror_landmarks(landmarks.copy())
                mirrored['augmentation'] = 'mirrored'
                frame_data.append(mirrored)
                # ==========================================
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:
                print(f"      Frame {frame_count}...", end='\r')
        
        cap.release()
        print(f"    ✓ Extracted {len(frame_data)} frames (with augmentation)")
        return frame_data
    
    def process_video_batch(self, video_dir: str, exercise_type: str, 
                           age_group: str, form_type: str) -> Optional[str]:
        """
        Process all videos in a directory
        
        Args:
            video_dir: Directory containing videos
            exercise_type: 'bicep', 'back', or 'chest'
            age_group: 'children', 'adult', or 'senior'
            form_type: 'good_form' or 'bad_form'
            
        Returns:
            Path to saved CSV file, or None if no videos found
        """
        video_files = list(Path(video_dir).glob('*.mp4')) + \
                     list(Path(video_dir).glob('*.avi')) + \
                     list(Path(video_dir).glob('*.mov')) + \
                     list(Path(video_dir).glob('*.mkv'))

        if not video_files:
            return None

        print(f"\n  Processing {len(video_files)} {form_type} videos...")

        all_data = []

        for i, video_path in enumerate(video_files, 1):
            # Progress print for each video
            video_name = os.path.basename(video_path)
            print(f"[{i}/{len(video_files)}] Processing: {video_name}")
            self.log_progress(f"STARTED: {video_name} ({exercise_type}, {age_group}, {form_type})")

            metadata = {
                'exercise_type': exercise_type,
                'age_group': age_group,
                'label': self.label_mapping[form_type],
                'form_type': form_type
            }

            frame_data = self.extract_landmarks_from_video(str(video_path), metadata)
            all_data.extend(frame_data)
            self.log_progress(f"COMPLETED: {video_name} ({exercise_type}, {age_group}, {form_type}), Frames: {len(frame_data)}")

        # Save to CSV
        if all_data:
            df = pd.DataFrame(all_data)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                self.output_dir,
                f'{exercise_type}_{age_group}_{form_type}_{timestamp}.csv'
            )
            df.to_csv(output_file, index=False)

            print(f"\n  ✅ Saved: {output_file}")
            print(f"     Total frames (with augmentation): {len(df)}")
            print(f"     Age groups: {df['age_group'].unique().tolist()}")
            print(f"     Labels: {df['label'].unique().tolist()}")
            self.log_progress(f"BATCH COMPLETE: {output_file} | Total frames: {len(df)}")
            return output_file

        return None
    
    def process_exercise_with_auto_labeling(self, base_path: str = 'videos') -> Dict[str, List[str]]:
        """
        Auto-discover and process all videos using folder structure for labeling
        Expected: videos/{exercise}/{age_group}/{form_type}/video.mp4
        
        Args:
            base_path: Base videos directory
            
        Returns:
            Dictionary with results: {exercise: [csv_files]}
        """
        results = {}
        
        videos_root = Path(base_path)
        if not videos_root.exists():
            print(f"[ERROR] Videos directory not found: {base_path}")
            return results
        
        # Find all exercise directories
        for exercise_dir in videos_root.iterdir():
            if not exercise_dir.is_dir():
                continue
            
            exercise_name = exercise_dir.name.lower()
            if exercise_name not in self.valid_exercises:
                continue
            
            results[exercise_name] = []
            
            print(f"\n{'='*60}")
            print(f"EXERCISE: {exercise_name.upper()}")
            print(f"{'='*60}")
            
            # Find all age group directories
            for age_dir in exercise_dir.iterdir():
                if not age_dir.is_dir():
                    continue
                
                age_group = age_dir.name.lower()
                if age_group not in self.valid_age_groups:
                    continue
                
                print(f"\nAge Group: {age_group}")
                
                # Process good_form
                good_form_dir = age_dir / 'good_form'
                if good_form_dir.exists():
                    csv_file = self.process_video_batch(
                        str(good_form_dir), exercise_name, age_group, 'good_form'
                    )
                    if csv_file:
                        results[exercise_name].append(csv_file)
                
                # Process bad_form
                bad_form_dir = age_dir / 'bad_form'
                if bad_form_dir.exists():
                    csv_file = self.process_video_batch(
                        str(bad_form_dir), exercise_name, age_group, 'bad_form'
                    )
                    if csv_file:
                        results[exercise_name].append(csv_file)
        
        return results
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'pose'):
            self.pose.close()


def main():
    parser = argparse.ArgumentParser(description='Process exercise videos for training')
    parser.add_argument('--exercise', choices=['bicep', 'back', 'chest'],
                       help='Process specific exercise only')
    parser.add_argument('--age', choices=['children', 'adult', 'senior'],
                       help='Process specific age group only')
    parser.add_argument('--base-path', default='videos',
                       help='Base path for videos directory')
    
    args = parser.parse_args()
    
    processor = VideoProcessor()
    
    print("\n" + "="*60)
    print("AI GYM - VIDEO PROCESSING PIPELINE")
    print("Industrial Grade with Data Augmentation")
    print("="*60)
    
    if args.exercise and args.age:
        # Process specific exercise/age combination
        print(f"\nProcessing: {args.exercise.upper()} - {args.age.upper()}")
        
        good_form_dir = f"{args.base_path}/{args.exercise}/{args.age}/good_form"
        bad_form_dir = f"{args.base_path}/{args.exercise}/{args.age}/bad_form"
        
        if Path(good_form_dir).exists():
            processor.process_video_batch(good_form_dir, args.exercise, args.age, 'good_form')
        if Path(bad_form_dir).exists():
            processor.process_video_batch(bad_form_dir, args.exercise, args.age, 'bad_form')
    else:
        # Process all exercises
        results = processor.process_exercise_with_auto_labeling(args.base_path)
        
        print(f"\n\n{'='*60}")
        print("PROCESSING COMPLETE!")
        print(f"{'='*60}")
        for exercise, csv_files in results.items():
            print(f"\n{exercise.upper()}: {len(csv_files)} CSV files")
            for csv_file in csv_files:
                print(f"  ✓ {csv_file}")


if __name__ == '__main__':
    main()
