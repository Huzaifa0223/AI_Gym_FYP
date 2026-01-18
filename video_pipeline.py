"""
Video Processing Pipeline for Training Data Generation
Processes videos to extract pose data for ML training
Supports batch processing for multiple exercises and age groups
"""
# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

class VideoProcessor:
    """Process exercise videos to extract training data"""
    
    def __init__(self, output_dir='data/training'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose  # type: ignore
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_draw = mp.solutions.drawing_utils  # type: ignore
        
    def extract_landmarks_from_video(self, video_path: str) -> List[Dict]:
        """
        Extract pose landmarks from a video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of dictionaries containing frame data
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return []
        
        frame_data = []
        frame_count = 0
        
        print(f"[INFO] Processing video: {os.path.basename(video_path)}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
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
                
                landmarks['frame'] = frame_count
                landmarks['video_path'] = video_path
                frame_data.append(landmarks)
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames...")
        
        cap.release()
        print(f"[SUCCESS] Extracted {len(frame_data)} frames with pose data")
        return frame_data
    
    def process_video_batch(self, video_dir: str, exercise_type: str, 
                           age_group: str, label: str = 'unknown') -> str | None:
        """
        Process all videos in a directory
        
        Args:
            video_dir: Directory containing videos
            exercise_type: 'bicep', 'back', or 'chest'
            age_group: 'children', 'adult', or 'senior'
            label: Label for the exercise (e.g., 'good_form', 'bad_form')
            
        Returns:
            Path to saved CSV file
        """
        video_files = list(Path(video_dir).glob('*.mp4')) + \
                     list(Path(video_dir).glob('*.avi')) + \
                     list(Path(video_dir).glob('*.mov'))
        
        if not video_files:
            print(f"[WARNING] No videos found in {video_dir}")
            return None
        
        print(f"\n[INFO] Processing {len(video_files)} videos...")
        print(f"  Exercise: {exercise_type}")
        print(f"  Age Group: {age_group}")
        print(f"  Label: {label}")
        
        all_data = []
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")
            
            frame_data = self.extract_landmarks_from_video(str(video_path))
            
            # Add metadata to each frame
            for data in frame_data:
                data['exercise_type'] = exercise_type
                data['age_group'] = age_group
                data['label'] = label
                data['video_file'] = video_path.name
            
            all_data.extend(frame_data)
        
        # Save to CSV
        if all_data:
            df = pd.DataFrame(all_data)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                self.output_dir, 
                f'{exercise_type}_{age_group}_{label}_{timestamp}.csv'
            )
            df.to_csv(output_file, index=False)
            print(f"\n[SUCCESS] Saved training data: {output_file}")
            print(f"  Total frames: {len(df)}")
            print(f"  Features: {len(df.columns)}")
            return output_file
        
        return None
    
    def create_training_config(self, exercise_type: str, age_group: str,
                              good_form_videos: str, bad_form_videos: str | None = None) -> Dict:
        """
        Create configuration for training pipeline
        
        Args:
            exercise_type: 'bicep', 'back', or 'chest'
            age_group: 'children', 'adult', or 'senior'
            good_form_videos: Directory with good form videos
            bad_form_videos: Directory with bad form videos (optional)
            
        Returns:
            Configuration dictionary
        """
        config = {
            'exercise_type': exercise_type,
            'age_group': age_group,
            'timestamp': datetime.now().isoformat(),
            'good_form_videos': good_form_videos,
            'bad_form_videos': bad_form_videos,
            'model_output': f'data/models/{exercise_type}_{age_group}.pkl'
        }
        
        # Save config
        config_file = os.path.join(
            self.output_dir,
            f'config_{exercise_type}_{age_group}.json'
        )
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[INFO] Training config saved: {config_file}")
        return config
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'pose'):
            self.pose.close()


def process_all_exercises():
    """
    Example batch processing script
    Organize your videos like:
    
    videos/
        bicep/
            children/
                good_form/
                bad_form/
            adult/
                good_form/
                bad_form/
            senior/
                good_form/
                bad_form/
        back/
            [same structure]
        chest/
            [same structure]
    """
    processor = VideoProcessor()
    
    exercises = ['bicep', 'back', 'chest']
    age_groups = ['children', 'adult', 'senior']
    
    for exercise in exercises:
        for age_group in age_groups:
            print(f"\n{'='*60}")
            print(f"Processing: {exercise.upper()} - {age_group.upper()}")
            print(f"{'='*60}")
            
            # Process good form videos
            good_dir = f'videos/{exercise}/{age_group}/good_form'
            if os.path.exists(good_dir):
                processor.process_video_batch(
                    good_dir, exercise, age_group, 'good_form'
                )
            
            # Process bad form videos
            bad_dir = f'videos/{exercise}/{age_group}/bad_form'
            if os.path.exists(bad_dir):
                processor.process_video_batch(
                    bad_dir, exercise, age_group, 'bad_form'
                )
    
    print("\n[COMPLETE] All videos processed!")


if __name__ == '__main__':
    # Example usage
    processor = VideoProcessor()
    
    # Process current bicep videos
    print("Processing existing bicep curl videos...")
    processor.process_video_batch(
        video_dir='videos',
        exercise_type='bicep',
        age_group='adult',  # Assuming current videos are adults
        label='good_form'
    )
