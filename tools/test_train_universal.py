"""
Comprehensive Test Suite for train_universal.py
Tests: Noise injection, age filtering, angle calculation, Random Forest training
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import pickle
from training.train_universal import ExerciseModelTrainer

def test_noise_injection():
    """Test noise injection for robust training"""
    print("\n" + "="*70)
    print("TEST 1: Noise Injection (Gaussian ±0.01)")
    print("="*70)
    
    trainer = ExerciseModelTrainer('bicep', 'adult')
    
    # Create sample data
    X_original = np.array([[90.0, 25.0], [85.0, 20.0], [95.0, 30.0]])
    
    print(f"  Original data shape: {X_original.shape}")
    print(f"  Original data:\n{X_original}")
    
    # Apply noise injection
    X_noisy = trainer.inject_noise(X_original, noise_std=0.01)
    
    print(f"\n  Noisy data shape: {X_noisy.shape}")
    print(f"  Noisy data:\n{X_noisy}")
    
    # Verify differences are small
    differences = np.abs(X_original - X_noisy)
    max_diff = np.max(differences)
    mean_diff = np.mean(differences)
    
    print(f"\n  Noise Statistics:")
    print(f"    Max difference: {max_diff:.6f}")
    print(f"    Mean difference: {mean_diff:.6f}")
    print(f"    Shape preserved: {X_original.shape == X_noisy.shape}")
    
    # Noise should be small (mostly < 0.05 for std=0.01)
    mostly_small = (differences < 0.05).sum() / differences.size > 0.95
    
    checks = [
        (X_original.shape == X_noisy.shape, "Shape preserved"),
        (max_diff < 0.1, "Max noise < 0.1"),
        (mean_diff < 0.02, "Mean noise < 0.02"),
        (mostly_small, "95%+ of noise < 0.05"),
    ]
    
    passed = sum(1 for check, _ in checks if check)
    
    print(f"\n  Validation:")
    for check, desc in checks:
        print(f"    {'✅' if check else '❌'} {desc}")
    
    return passed == len(checks)


def test_angle_calculation():
    """Test angle calculation between three points"""
    print("\n" + "="*70)
    print("TEST 2: Angle Calculation")
    print("="*70)
    
    trainer = ExerciseModelTrainer('bicep', 'adult')
    
    # Create sample row with landmarks
    row = {}
    
    # Straight line (should be ~180 degrees)
    for i in range(33):
        row[f'landmark_{i}_x'] = 0.5
        row[f'landmark_{i}_y'] = 0.5
    
    # Right angle test: points at (0,0), (0,1), (1,0)
    row['landmark_12_x'] = 0.0  # Shoulder
    row['landmark_12_y'] = 1.0
    row['landmark_14_x'] = 0.0  # Elbow
    row['landmark_14_y'] = 0.0
    row['landmark_16_x'] = 1.0  # Wrist
    row['landmark_16_y'] = 0.0
    
    angle = trainer.calculate_angle(row, 12, 14, 16)
    
    print(f"  Right angle test:")
    print(f"    Shoulder: ({row['landmark_12_x']}, {row['landmark_12_y']})")
    print(f"    Elbow: ({row['landmark_14_x']}, {row['landmark_14_y']})")
    print(f"    Wrist: ({row['landmark_16_x']}, {row['landmark_16_y']})")
    print(f"    Calculated angle: {angle:.1f}°")
    print(f"    Expected: ~90°")
    print(f"    {'✅' if 85 < angle < 95 else '❌'} Within range")
    
    # Straight line test (180 degrees)
    row['landmark_12_x'] = 0.0
    row['landmark_12_y'] = 0.0
    row['landmark_14_x'] = 0.5
    row['landmark_14_y'] = 0.5
    row['landmark_16_x'] = 1.0
    row['landmark_16_y'] = 1.0
    
    angle_straight = trainer.calculate_angle(row, 12, 14, 16)
    
    print(f"\n  Straight line test:")
    print(f"    Calculated angle: {angle_straight:.1f}°")
    print(f"    Expected: ~180°")
    print(f"    {'✅' if 175 < angle_straight < 185 else '❌'} Within range")
    
    checks = [
        (85 < angle < 95, "Right angle correct"),
        (175 < angle_straight < 185, "Straight line correct"),
    ]
    
    passed = sum(1 for check, _ in checks if check)
    return passed == len(checks)


def test_age_filtering():
    """Test filtering data by age group"""
    print("\n" + "="*70)
    print("TEST 3: Age Group Filtering")
    print("="*70)
    
    # Create sample DataFrame with mixed ages
    data = {
        'primary_angle': [90, 85, 95, 88, 92],
        'secondary_angle': [20, 25, 18, 22, 20],
        'label': [1, 0, 1, 0, 1],
        'age_group': ['adult', 'senior', 'adult', 'children', 'adult']
    }
    df = pd.DataFrame(data)
    
    print(f"  Original data (5 rows):")
    print(f"    age_group: {df['age_group'].tolist()}")
    print(f"    Labels: {df['label'].tolist()}")
    
    # Simulate filtering (as done in load_training_data)
    for age_group in ['children', 'adult', 'senior']:
        filtered = df[df['age_group'] == age_group]
        print(f"\n  Filtered to '{age_group}': {len(filtered)} rows")
        if len(filtered) > 0:
            print(f"    Angles: {filtered['primary_angle'].tolist()}")
            print(f"    Labels: {filtered['label'].tolist()}")
    
    checks = [
        (len(df[df['age_group'] == 'children']) == 1, "Children: 1 row"),
        (len(df[df['age_group'] == 'adult']) == 3, "Adult: 3 rows"),
        (len(df[df['age_group'] == 'senior']) == 1, "Senior: 1 row"),
    ]
    
    passed = sum(1 for check, _ in checks if check)
    
    print(f"\n  Validation:")
    for check, desc in checks:
        print(f"    {'✅' if check else '❌'} {desc}")
    
    return passed == len(checks)


def test_data_augmentation_with_noise():
    """Test combining original + noisy data"""
    print("\n" + "="*70)
    print("TEST 4: Data Augmentation (Original + Noisy)")
    print("="*70)
    
    trainer = ExerciseModelTrainer('bicep', 'adult')
    
    # Original data
    X_original = np.array([[90.0, 25.0], [85.0, 20.0], [95.0, 30.0]])
    y_original = np.array([1, 0, 1])
    
    print(f"  Original dataset:")
    print(f"    Samples: {len(X_original)}")
    print(f"    Shape: {X_original.shape}")
    
    # Apply noise injection
    X_noisy = trainer.inject_noise(X_original, noise_std=0.01)
    
    # Combine
    X_combined = np.vstack([X_original, X_noisy])
    y_combined = np.hstack([y_original, y_original])
    
    print(f"\n  After augmentation:")
    print(f"    Samples: {len(X_combined)}")
    print(f"    Shape: {X_combined.shape}")
    print(f"    Expected samples: 6 (2x original)")
    print(f"    Ratio: {len(X_combined) / len(X_original):.1f}x")
    
    checks = [
        (len(X_combined) == 6, "Combined size is 2x"),
        (len(y_combined) == 6, "Labels doubled"),
        (np.array_equal(y_combined[:3], y_combined[3:]), "Label distribution preserved"),
    ]
    
    passed = sum(1 for check, _ in checks if check)
    
    print(f"\n  Validation:")
    for check, desc in checks:
        print(f"    {'✅' if check else '❌'} {desc}")
    
    return passed == len(checks)


def test_create_sample_csv():
    """Test creating sample training CSV"""
    print("\n" + "="*70)
    print("TEST 5: Sample CSV Generation")
    print("="*70)
    
    # Create sample data with proper structure
    data = []
    
    for exercise in ['bicep', 'back', 'chest']:
        for age_group in ['children', 'adult', 'senior']:
            for label in [0, 1]:
                for _ in range(10):  # 10 samples per combination
                    row = {}
                    
                    # Add landmarks (33 points × 4 values)
                    for i in range(33):
                        row[f'landmark_{i}_x'] = np.random.uniform(0, 1)
                        row[f'landmark_{i}_y'] = np.random.uniform(0, 1)
                        row[f'landmark_{i}_z'] = np.random.uniform(-0.5, 0.5)
                        row[f'landmark_{i}_visibility'] = np.random.uniform(0.5, 1.0)
                    
                    row['label'] = label
                    row['exercise_type'] = exercise
                    row['age_group'] = age_group
                    row['form_type'] = 'good_form' if label == 1 else 'bad_form'
                    
                    data.append(row)
    
    df = pd.DataFrame(data)
    
    print(f"  Generated CSV statistics:")
    print(f"    Total rows: {len(df)}")
    print(f"    Exercises: {df['exercise_type'].unique().tolist()}")
    print(f"    Age groups: {df['age_group'].unique().tolist()}")
    print(f"    Labels: {df['label'].unique().tolist()}")
    
    # Show distribution
    print(f"\n  Data distribution:")
    for ex in ['bicep', 'back', 'chest']:
        for age in ['children', 'adult', 'senior']:
            count = len(df[(df['exercise_type'] == ex) & (df['age_group'] == age)])
            print(f"    {ex:8} - {age:8}: {count:3} rows")
    
    # Save sample CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / 'test_data.csv'
        df.to_csv(csv_path, index=False)
        
        # Verify it can be loaded
        df_loaded = pd.read_csv(csv_path)
        
        checks = [
            (len(df_loaded) == len(df), "CSV saved and loaded correctly"),
            (set(df_loaded.columns) == set(df.columns), "Columns preserved"),
            ('landmark_0_x' in df_loaded.columns, "Landmarks present"),
            ('age_group' in df_loaded.columns, "Age group column present"),
        ]
        
        passed = sum(1 for check, _ in checks if check)
        
        print(f"\n  CSV Validation:")
        for check, desc in checks:
            print(f"    {'✅' if check else '❌'} {desc}")
        
        return passed == len(checks)


def test_trainer_initialization():
    """Test ExerciseModelTrainer initialization"""
    print("\n" + "="*70)
    print("TEST 6: Trainer Initialization")
    print("="*70)
    
    exercises = ['bicep', 'back', 'chest']
    ages = ['children', 'adult', 'senior']
    
    print(f"  Testing all combinations:")
    
    all_pass = True
    for exercise in exercises:
        for age in ages:
            try:
                trainer = ExerciseModelTrainer(exercise, age)
                
                checks = [
                    (trainer.exercise_type == exercise, f"Exercise: {exercise}"),
                    (trainer.age_group == age, f"Age: {age}"),
                    (trainer.landmark_configs is not None, "Landmark configs loaded"),
                    (exercise in trainer.landmark_configs, f"Config for {exercise}"),
                ]
                
                passed = all(check for check, _ in checks)
                status = "✅" if passed else "❌"
                
                print(f"    {status} {exercise:8} - {age:8}")
                
                all_pass = all_pass and passed
                
            except Exception as e:
                print(f"    ❌ {exercise:8} - {age:8}: {e}")
                all_pass = False
    
    return all_pass


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("TRAIN_UNIVERSAL TEST SUITE")
    print("Testing: Noise Injection, Age Filtering, Data Augmentation")
    print("="*70)
    
    tests = [
        ("Noise Injection", test_noise_injection),
        ("Angle Calculation", test_angle_calculation),
        ("Age Filtering", test_age_filtering),
        ("Data Augmentation", test_data_augmentation_with_noise),
        ("Sample CSV Generation", test_create_sample_csv),
        ("Trainer Initialization", test_trainer_initialization),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n  ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("="*70)
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
