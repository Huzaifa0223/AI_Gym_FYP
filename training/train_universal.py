"""
Universal Training Script for All Exercises
Industrial-Grade Implementation with Data Augmentation
- Noise injection for robustness
- Age-group specific training
- Automatic CSV discovery
"""
import pandas as pd
import numpy as np
import pickle
import os
import argparse
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no GUI windows during training
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

class ExerciseModelTrainer:
    """Train ML models for exercise detection"""
    
    def __init__(self, exercise_type: str, age_group: str):
        """
        Initialize trainer
        
        Args:
            exercise_type: 'bicep', 'back', or 'chest'
            age_group: 'children', 'adult', or 'senior'
        """
        self.exercise_type = exercise_type
        self.age_group = age_group
        self.model = None
        self.angle_data = None
        
        # Landmark indices for different exercises
        self.landmark_configs = {
            'bicep': {
                'shoulder': 12,
                'elbow': 14,
                'wrist': 16,
                'hip': 24
            },
            'back': {
                'shoulder': 12,
                'elbow': 14,
                'wrist': 16,
                'hip': 24
            },
            'chest': {
                'shoulder': 12,
                'elbow': 14,
                'wrist': 16,
                'hip': 24,
                'knee': 26,
                'ankle': 28
            }
        }
    
    def calculate_angle(self, row, point_a_idx, point_b_idx, point_c_idx):
        """Calculate angle between three points"""
        try:
            a = np.array([
                row[f'landmark_{point_a_idx}_x'],
                row[f'landmark_{point_a_idx}_y']
            ])
            b = np.array([
                row[f'landmark_{point_b_idx}_x'],
                row[f'landmark_{point_b_idx}_y']
            ])
            c = np.array([
                row[f'landmark_{point_c_idx}_x'],
                row[f'landmark_{point_c_idx}_y']
            ])
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cosine_angle))
            
            return angle
        except:
            return 0.0
    
    def inject_noise(self, X: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        """
        Inject Gaussian noise into training data for robustness
        
        Args:
            X: Feature matrix
            noise_std: Standard deviation of Gaussian noise (default 0.01)
            
        Returns:
            Noisy feature matrix
        """
        noise = np.random.normal(0, noise_std, X.shape)
        X_noisy = X + noise
        print(f"[INFO] Noise injection: std={noise_std}, shape={X_noisy.shape}")
        return X_noisy
    
    def load_training_data(self, csv_path: str):
        """Load training data from one or more CSVs (pass a glob pattern or single path)"""
        from pathlib import Path as _Path

        # Support passing a glob pattern or a list of files
        matched = list(_Path('.').glob(csv_path)) if '*' in str(csv_path) else [_Path(csv_path)]
        if not matched:
            raise FileNotFoundError(f"No training data found matching: {csv_path}")

        frames = []
        for f in matched:
            print(f"\n[INFO] Loading: {f.name}")
            tmp = pd.read_csv(f)
            print(f"  -> {len(tmp)} rows")
            frames.append(tmp)

        df = pd.concat(frames, ignore_index=True)
        print(f"[INFO] Combined: {len(df)} total rows")

        # Filter by age_group if specified
        if 'age_group' in df.columns:
            df = df[df['age_group'] == self.age_group]
            print(f"  Filtered to {len(df)} frames for age_group: {self.age_group}")
        
        # Calculate angles based on exercise type
        landmarks = self.landmark_configs[self.exercise_type]
        
        print(f"[INFO] Calculating angles for {self.exercise_type}...")
        
        if self.exercise_type == 'bicep':
            # Primary: Elbow angle (shoulder→elbow→wrist)
            df['primary_angle'] = df.apply(
                lambda row: self.calculate_angle(
                    row, landmarks['shoulder'], landmarks['elbow'], landmarks['wrist']
                ), axis=1
            )
            # Secondary: Upper arm deviation (hip→shoulder→elbow) — detects elbow swing
            df['secondary_angle'] = df.apply(
                lambda row: self.calculate_angle(
                    row, landmarks['hip'], landmarks['shoulder'], landmarks['elbow']
                ), axis=1
            )
            # Tertiary: Full arm deviation (hip→shoulder→wrist) — wrist swing complements secondary
            df['tertiary_angle'] = df.apply(
                lambda row: self.calculate_angle(
                    row, landmarks['hip'], landmarks['shoulder'], landmarks['wrist']
                ), axis=1
            )

        elif self.exercise_type == 'back':
            # Primary: Elbow angle (arm pull)
            df['primary_angle'] = df.apply(
                lambda row: self.calculate_angle(
                    row, landmarks['shoulder'], landmarks['elbow'], landmarks['wrist']
                ), axis=1
            )
            # Secondary: Back/torso lean (hip→shoulder→elbow)
            df['secondary_angle'] = df.apply(
                lambda row: self.calculate_angle(
                    row, landmarks['hip'], landmarks['shoulder'], landmarks['elbow']
                ), axis=1
            )
            # Tertiary: Full reach deviation (hip→shoulder→wrist)
            df['tertiary_angle'] = df.apply(
                lambda row: self.calculate_angle(
                    row, landmarks['hip'], landmarks['shoulder'], landmarks['wrist']
                ), axis=1
            )

        elif self.exercise_type == 'chest':
            # Primary: Elbow angle (push-up depth)
            df['primary_angle'] = df.apply(
                lambda row: self.calculate_angle(
                    row, landmarks['shoulder'], landmarks['elbow'], landmarks['wrist']
                ), axis=1
            )
            # Secondary: Body alignment (shoulder→hip→knee)
            df['secondary_angle'] = df.apply(
                lambda row: self.calculate_angle(
                    row, landmarks['shoulder'], landmarks['hip'], landmarks['knee']
                ), axis=1
            )
            # Tertiary: Leg straightness (hip→knee→ankle) — detects bent knees in plank
            df['tertiary_angle'] = df.apply(
                lambda row: self.calculate_angle(
                    row, landmarks['hip'], landmarks['knee'], landmarks['ankle']
                ), axis=1
            )
        
        self.angle_data = df
        print(f"[SUCCESS] Calculated angles for {len(df)} frames")
        
        return df
    
    def train_unsupervised_model(self):
        """Train unsupervised model (K-Means clustering)"""
        print(f"\n[INFO] Training unsupervised model (K-Means)...")
        
        # Use primary angle for clustering
        X = self.angle_data[['primary_angle']].values
        
        # K-Means with 2 clusters: "up" and "down"
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.angle_data['cluster'] = kmeans.fit_predict(X)
        
        # Identify which cluster is "up" and "down"
        center_0 = kmeans.cluster_centers_[0][0]
        center_1 = kmeans.cluster_centers_[1][0]
        
        if center_0 < center_1:
            up_cluster = 0
            down_cluster = 1
            print(f"  Cluster 0 is UP (Avg Angle: {center_0:.1f}°)")
            print(f"  Cluster 1 is DOWN (Avg Angle: {center_1:.1f}°)")
        else:
            up_cluster = 1
            down_cluster = 0
            print(f"  Cluster 0 is DOWN (Avg Angle: {center_0:.1f}°)")
            print(f"  Cluster 1 is UP (Avg Angle: {center_1:.1f}°)")
        
        self.model = {
            'type': 'kmeans',
            'model': kmeans,
            'up_cluster': up_cluster,
            'down_cluster': down_cluster,
            'exercise_type': self.exercise_type,
            'age_group': self.age_group
        }
        
        print(f"[SUCCESS] Unsupervised model trained")
        return self.model
    
    def train_supervised_model(self):
        """Train supervised model with noise injection"""
        if 'label' not in self.angle_data.columns:
            print("[WARNING] No labels found, cannot train supervised model")
            return None
        
        print(f"\n[INFO] Training supervised model (Random Forest with Noise Injection)...")
        
        # Features — 3 angles per exercise for richer form discrimination
        feature_cols = ['primary_angle', 'secondary_angle', 'tertiary_angle']
        X = self.angle_data[feature_cols].values
        y = self.angle_data['label'].values

        print(f"  Original dataset size: {len(X)}")
        print(f"  Class distribution: {np.bincount(y.astype(int))}")

        # ===== DATA AUGMENTATION: NOISE INJECTION =====
        # std=2.0 degrees is meaningful for angle-valued features (was 0.01, essentially zero)
        X_noisy = self.inject_noise(X, noise_std=2.0)
        X_combined = np.vstack([X, X_noisy])
        y_combined = np.hstack([y, y])
        print(f"  After noise augmentation: {len(X_combined)} samples")
        # ===============================================

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )

        # Train Random Forest — class_weight='balanced' handles imbalanced datasets
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n  Accuracy: {accuracy:.2%}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Confusion Matrix:")
        if cm.shape == (1, 1):
            # Only one class present
            print(f"    Only one class present in test data")
            print(f"    Correctly classified: {cm[0,0]}")
        else:
            print(f"    True Negatives:  {cm[0,0]}")
            print(f"    False Positives: {cm[0,1]}")
            print(f"    False Negatives: {cm[1,0]}")
            print(f"    True Positives:  {cm[1,1]}")
        
        self.model = {
            'type': 'supervised',
            'model': rf,
            'feature_cols': feature_cols,
            'exercise_type': self.exercise_type,
            'age_group': self.age_group,
            'accuracy': accuracy,
            'noise_injection': True
        }
        self.rf_model = rf
        
        print(f"[SUCCESS] Supervised model trained with noise injection")
        return self.model
    
    def save_model(self, output_dir='data/models'):
        """Save trained model, backing up any existing version first."""
        import shutil
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, f'{self.exercise_type}_{self.age_group}.pkl')

        # Version backup: rename existing model before overwriting
        if os.path.exists(model_path):
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(
                output_dir, f'{self.exercise_type}_{self.age_group}_backup_{ts}.pkl'
            )
            shutil.copy2(model_path, backup_path)
            print(f"[INFO] Backed up previous model -> {os.path.basename(backup_path)}")

        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        print(f"[SUCCESS] Model saved: {model_path}")
        return model_path
    
    def visualize_results(self):
        """Visualize training results"""
        if self.angle_data is None:
            print("[WARNING] No data to visualize")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Angle distribution
        plt.subplot(1, 2, 1)
        plt.hist(self.angle_data['primary_angle'], bins=50, alpha=0.7)
        plt.xlabel('Primary Angle (degrees)')
        plt.ylabel('Frequency')
        plt.title(f'{self.exercise_type.capitalize()} - {self.age_group.capitalize()}\nAngle Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Clusters (if available)
        if 'cluster' in self.angle_data.columns:
            plt.subplot(1, 2, 2)
            colors = ['red', 'blue']
            for cluster in [0, 1]:
                data = self.angle_data[self.angle_data['cluster'] == cluster]
                plt.scatter(
                    data['primary_angle'],
                    data['secondary_angle'],
                    c=colors[cluster],
                    label=f'Cluster {cluster}',
                    alpha=0.5
                )
            plt.xlabel('Primary Angle (degrees)')
            plt.ylabel('Secondary Angle (degrees)')
            plt.title('K-Means Clusters')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'data/models/{self.exercise_type}_{self.age_group}_analysis.png')
        plt.close()
        print(f"[INFO] Visualization saved: data/models/{self.exercise_type}_{self.age_group}_analysis.png")


def train_all_models():
    """
    Train models for all exercises and age groups
    Requires training data to be organized as:
    data/training/{exercise}_{age_group}_{label}_{timestamp}.csv
    """
    exercises = ['bicep', 'back', 'chest']
    age_groups = ['children', 'adult', 'senior']
    
    print("="*60)
    print("TRAINING ALL EXERCISE MODELS")
    print("="*60)
    
    for exercise in exercises:
        for age_group in age_groups:
            print(f"\n{'='*60}")
            print(f"Training: {exercise.upper()} - {age_group.upper()}")
            print(f"{'='*60}")
            
            # Find training data (may be multiple files — good + bad form)
            pattern = f'data/training/{exercise}_{age_group}_*.csv'
            csv_files = list(Path('.').glob(pattern))

            if not csv_files:
                print(f"[WARNING] No training data found for {exercise} - {age_group}")
                print(f"  Expected pattern: {pattern}")
                continue

            print(f"  Found {len(csv_files)} CSV file(s):")
            for f in csv_files:
                print(f"    {f.name}")

            try:
                # Train model — pass glob pattern so all files are loaded & combined
                trainer = ExerciseModelTrainer(exercise, age_group)
                trainer.load_training_data(pattern)
                
                # Try supervised first, fallback to unsupervised
                model = trainer.train_supervised_model()
                if model is None:
                    model = trainer.train_unsupervised_model()
                
                # Save model
                trainer.save_model()
                
                # Visualize
                trainer.visualize_results()
                
            except Exception as e:
                print(f"[ERROR] Failed to train {exercise} - {age_group}: {e}")
                continue
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train exercise models')
    parser.add_argument('--exercise', choices=['bicep', 'back', 'chest'],
                       help='Train specific exercise only')
    parser.add_argument('--age', choices=['children', 'adult', 'senior'],
                       help='Train specific age group only')
    parser.add_argument('--csv', help='Path to CSV file (optional)')
    
    args = parser.parse_args()
    
    if args.exercise and args.age:
        # Train specific model
        if args.csv:
            csv_path = args.csv
        else:
            # Use glob pattern so ALL matching CSVs (good + bad form) are loaded
            csv_path = f'data/training/{args.exercise}_{args.age}_*.csv'
            if not list(Path('.').glob(csv_path)):
                print(f"[ERROR] No CSV found for {args.exercise} - {args.age}")
                print(f"  Expected: {csv_path}")
                exit(1)

        trainer = ExerciseModelTrainer(args.exercise, args.age)
        trainer.load_training_data(csv_path)
        
        model = trainer.train_supervised_model()
        if model is None:
            model = trainer.train_unsupervised_model()
        
        trainer.save_model()
        trainer.visualize_results()
    else:
        # Train all models
        train_all_models()
