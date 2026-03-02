import numpy as np
import pickle
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class AdvancedExerciseModel:
    def __init__(self):
        self.model = None
        self.best_model = None

    def load_data_from_json(self, filepath):
        """Load real data collected by data_collector.py"""
        if not os.path.exists(filepath):
            print(f"[WARN] No data file found at {filepath}")
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"[INFO] Found {len(data)} recorded reps")
        
        features = []
        labels = []
        
        # Convert JSON structure to Training Features
        for rep in data:
            # We assume the user recorded "Good Form" reps
            # We extract the angles from the rep
            rep_angles = [frame['elbow_angle'] for frame in rep]
            
            # Simple Feature Extraction:
            # We define "Good Form" as the full range of motion (approx 160 -> 50 -> 160)
            # We label frames based on angle:
            # > 140 = DOWN (Class 0)
            # < 70 = UP (Class 1)
            # In between = TRANSITION (Class 2)
            
            for angle in rep_angles:
                # 1. Feature: The angle itself
                # 2. Feature: Distance from shoulder (Swing) - simplified as 0 for basic training
                features.append([angle, 0.0]) 
                
                # Auto-Labeling
                if angle > 130:
                    labels.append(1) # DOWN
                elif angle < 80:
                    labels.append(2) # UP
                else:
                    labels.append(1) # Moving (treat as DOWN/Safe for now)

        return np.array(features), np.array(labels)

    def train_model(self, X, y):
        print(f"[INFO] Training on {len(X)} data points...")
        
        # Train Random Forest
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10)
        self.model.fit(X, y)
        
        score = self.model.score(X, y)
        print(f"[SUCCESS] Model Accuracy: {score*100:.1f}%")
        self.best_model = self.model
        return self.model

    def save_model(self, filename='advanced_bicep_model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"[INFO] Model saved to {filename}")

    def load_model(self, filename='advanced_bicep_model.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.best_model = pickle.load(f)
            return True
        return False

def main():
    print("=== Advanced Model Training ===")
    
    trainer = AdvancedExerciseModel()
    
    # 1. Try to load REAL data
    json_path = 'training_data/bicep_curls.json'
    data = trainer.load_data_from_json(json_path)
    
    if data is not None and len(data[0]) > 0:
        X, y = data
        print("[INFO] Training on REAL data collected by you!")
        trainer.train_model(X, y)
    else:
        # 2. Fallback to SYNTHETIC data if no file found
        print("[WARN] No real data found. Generating SYNTHETIC data...")
        # Create fake "perfect" reps (180 -> 50 -> 180)
        X_syn = []
        y_syn = []
        for _ in range(100):
            # Down Phase (Class 1)
            for ang in range(130, 180, 5): 
                X_syn.append([ang, 0])
                y_syn.append(1)
            # Up Phase (Class 2)
            for ang in range(50, 80, 5):
                X_syn.append([ang, 0])
                y_syn.append(2)
            # Bad Form Swing (Class 3)
            for ang in range(90, 120, 5):
                X_syn.append([ang, 0.5]) # High swing distance
                y_syn.append(3)
                
        trainer.train_model(X_syn, y_syn)

    trainer.save_model()
    print("=== Process Complete ===")

if __name__ == "__main__":
    main()