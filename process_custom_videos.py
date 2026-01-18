#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video Processing Pipeline for Custom Folder Structure
Processes videos from BACK/chest/legs/etc folders with age and goal subfolders
Maps to training data format
"""
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

class CustomVideoProcessor:
    """Process exercise videos from custom folder structure"""
    
    def __init__(self, output_dir='data/training'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Map custom age ranges to system age groups
        self.age_mapping = {
            'AGE 10–16 (Youth)': 'children',
            'AGE 10-16 (Youth)': 'children',
            'AGE 17–45 (Adults)': 'adult',
            'AGE 17-45 (Adults)': 'adult',
            'AGE 46–85 (Seniors)': 'senior',
            'AGE 46-85 (Seniors)': 'senior'
        }
        
        # Map exercise folder names
        self.exercise_mapping = {
            'BACK': 'back',
            'back': 'back',
            'chest': 'chest',
            'legs': 'legs',
            'shoulders': 'shoulders',
            'arms': 'arms',
            'bicep': 'bicep'
        }
    
    def extract_landmarks_from_video(self, video_path: str, metadata: Dict) -> List[Dict]:
        """
        Extract pose landmarks from a video file with metadata
        
        Args:
            video_path: Path to video file
            metadata: Dictionary with exercise_type, age_group, goal_type
            
        Returns:
            List of dictionaries containing frame data
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return []
        
        frame_data = []
        frame_count = 0
        
        print(f"  Processing: {os.path.basename(video_path)}")
        
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
                
                # Add metadata
                landmarks['frame'] = frame_count
                landmarks['video_path'] = video_path
                landmarks['video_file'] = os.path.basename(video_path)
                landmarks['exercise_type'] = metadata['exercise_type']
                landmarks['age_group'] = metadata['age_group']
                landmarks['goal_type'] = metadata.get('goal_type', 'general')
                landmarks['label'] = 'good_form'  # Assume good form
                
                frame_data.append(landmarks)
            
            frame_count += 1
            
            # Progress indicator (every second at 30fps)
            if frame_count % 30 == 0:
                print(f"    Frame {frame_count}...", end='\r')
        
        cap.release()
        print(f"    ✓ Extracted {len(frame_data)} frames")
        return frame_data
    
    def process_exercise_folder(self, exercise_name: str, base_path: str = 'videos') -> str:
        """
        Process all videos for a specific exercise
        
        Args:
            exercise_name: Exercise folder name (e.g., 'BACK', 'chest')
            base_path: Base videos directory
            
        Returns:
            Path to saved CSV file
        """
        exercise_path = os.path.join(base_path, exercise_name)
        
        if not os.path.exists(exercise_path):
            print(f"[ERROR] Exercise folder not found: {exercise_path}")
            return None
        
        exercise_type = self.exercise_mapping.get(exercise_name, exercise_name.lower())
        
        print(f"\n{'='*60}")
        print(f"Processing: {exercise_name.upper()}")
        print(f"{'='*60}")
        
        all_data = []
        video_count = 0
        
        # Walk through age group folders
        for age_folder in os.listdir(exercise_path):
            age_path = os.path.join(exercise_path, age_folder)
            
            if not os.path.isdir(age_path):
                continue
            
            # Map age folder to system age group
            age_group = self.age_mapping.get(age_folder, 'adult')
            
            print(f"\n📁 {age_folder} → {age_group}")
            
            # Walk through goal folders (or straight to videos if none)
            goal_folders = [f for f in os.listdir(age_path) if os.path.isdir(os.path.join(age_path, f))]
            
            if not goal_folders:
                goal_folders = ['.']  # Current directory
            
            for goal_folder in goal_folders:
                if goal_folder == '.':
                    goal_path = age_path
                    display_goal = "Root"
                else:
                    goal_path = os.path.join(age_path, goal_folder)
                    display_goal = goal_folder
                
                if not os.path.isdir(goal_path):
                    continue
                
                print(f"  📂 {display_goal}")
                
                # Find all video files recursively (search deeper for exercise folders)
                video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV', '.mkv', '.MKV']
                video_files = []
                
                # Recursive search for video files
                for ext in video_extensions:
                    video_files.extend(Path(goal_path).rglob(f'*{ext}'))
                
                if not video_files:
                    print(f"    ⚠ No videos found")
                    continue
                
                print(f"    Found {len(video_files)} videos")
                
                # Process each video
                for video_path in video_files:
                    metadata = {
                        'exercise_type': exercise_type,
                        'age_group': age_group,
                        'goal_type': goal_folder
                    }
                    
                    frame_data = self.extract_landmarks_from_video(
                        str(video_path), metadata
                    )
                    
                    all_data.extend(frame_data)
                    video_count += 1
        
        # Save to CSV
        if all_data:
            df = pd.DataFrame(all_data)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                self.output_dir,
                f'{exercise_type}_all_ages_{timestamp}.csv'
            )
            df.to_csv(output_file, index=False)
            
            print(f"\n{'='*60}")
            print(f"✅ SUCCESS!")
            print(f"{'='*60}")
            print(f"Output: {output_file}")
            print(f"Videos processed: {video_count}")
            print(f"Total frames: {len(df)}")
            print(f"Features: {len(df.columns)}")
            print(f"Age groups: {df['age_group'].unique().tolist()}")
            print(f"{'='*60}\n")
            
            return output_file
        else:
            print(f"\n[WARNING] No data extracted from {exercise_name}")
            return None
    
    def process_all_exercises(self, exercise_list: List[str] = None, base_path: str = 'videos'):
        """
        Process multiple exercises
        
        Args:
            exercise_list: List of exercise folders to process (None = all)
            base_path: Base videos directory
        """
        if exercise_list is None:
            # Auto-detect available exercises
            exercise_list = []
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path) and item in self.exercise_mapping:
                    exercise_list.append(item)
        
        print("\n" + "="*60)
        print("PROCESSING ALL EXERCISES")
        print("="*60)
        print(f"Exercises to process: {', '.join(exercise_list)}")
        print("="*60 + "\n")
        
        results = {}
        
        for exercise in exercise_list:
            output_file = self.process_exercise_folder(exercise, base_path)
            results[exercise] = output_file
        
        # Summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE!")
        print("="*60)
        for exercise, output_file in results.items():
            if output_file:
                print(f"✅ {exercise}: {output_file}")
            else:
                print(f"❌ {exercise}: No data")
        print("="*60 + "\n")
        
        return results
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'pose'):
            self.pose.close()


# Main execution
if __name__ == '__main__':
    import sys
    
    processor = CustomVideoProcessor()
    
    if len(sys.argv) > 1:
        # Process specific exercise
        exercise_name = sys.argv[1]
        print(f"\n[*] Processing {exercise_name} only...")
        processor.process_exercise_folder(exercise_name)
    else:
        # Process all available exercises
        print("\n[*] Processing ALL available exercises...")
        print("\nNote: This will process videos from:")
        print("  - BACK folder")
        print("  - chest folder (if has videos)")
        print("  - legs folder (if has videos)")
        print("  - shoulders folder (if has videos)")
        print("  - arms folder (if has videos)")
        print("  - bicep folder (if has videos)")
        print("\n")
        
        # Only process folders that exist and have videos
        available = ['BACK', 'bicep']  # Known to have videos
        
        response = input(f"Process {', '.join(available)}? (y/n): ").strip().lower()
        if response == 'y':
            processor.process_all_exercises(available)
        else:
            print("Cancelled.")
