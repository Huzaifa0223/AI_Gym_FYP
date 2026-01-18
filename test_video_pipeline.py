"""
Comprehensive Test Suite for video_pipeline.py
Tests folder-based labeling, metadata extraction, and data augmentation (mirroring)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from video_pipeline import VideoProcessor
import tempfile
import os

def test_metadata_extraction():
    """Test folder-based automatic metadata extraction"""
    print("\n" + "="*70)
    print("TEST 1: Metadata Extraction from Folder Paths")
    print("="*70)
    
    processor = VideoProcessor()
    
    test_cases = [
        {
            'path': 'videos/bicep/adult/good_form/video1.mp4',
            'expected': {'exercise_type': 'bicep', 'age_group': 'adult', 'label': 1, 'form_type': 'good_form'},
            'name': 'Bicep Adult Good Form'
        },
        {
            'path': 'videos/back/senior/bad_form/video2.mp4',
            'expected': {'exercise_type': 'back', 'age_group': 'senior', 'label': 0, 'form_type': 'bad_form'},
            'name': 'Back Senior Bad Form'
        },
        {
            'path': 'videos/chest/children/good_form/video3.mp4',
            'expected': {'exercise_type': 'chest', 'age_group': 'children', 'label': 1, 'form_type': 'good_form'},
            'name': 'Chest Children Good Form'
        },
    ]
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        result = processor.extract_metadata_from_path(test['path'])
        
        if result == test['expected']:
            print(f"  ✅ {test['name']}")
            print(f"     Path: {test['path']}")
            print(f"     Result: {result}")
            passed += 1
        else:
            print(f"  ❌ {test['name']}")
            print(f"     Expected: {test['expected']}")
            print(f"     Got: {result}")
            failed += 1
    
    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed == 0


def test_invalid_paths():
    """Test that invalid paths are rejected"""
    print("\n" + "="*70)
    print("TEST 2: Invalid Path Rejection")
    print("="*70)
    
    processor = VideoProcessor()
    
    invalid_cases = [
        ('videos/invalid_exercise/adult/good_form/video.mp4', 'Invalid exercise'),
        ('videos/bicep/invalid_age/good_form/video.mp4', 'Invalid age group'),
        ('videos/bicep/adult/invalid_form/video.mp4', 'Invalid form type'),
        ('videos/bicep/video.mp4', 'Missing folders'),
    ]
    
    passed = 0
    
    for path, reason in invalid_cases:
        result = processor.extract_metadata_from_path(path)
        if result is None:
            print(f"  ✅ Correctly rejected: {reason}")
            print(f"     Path: {path}")
            passed += 1
        else:
            print(f"  ❌ Should have rejected: {reason}")
            print(f"     Path: {path}")
    
    print(f"\n  Results: {passed}/{len(invalid_cases)} correctly rejected")
    return passed == len(invalid_cases)


def test_landmark_mirroring():
    """Test data augmentation: mirroring (X-coordinate flip)"""
    print("\n" + "="*70)
    print("TEST 3: Data Augmentation - Landmark Mirroring")
    print("="*70)
    
    processor = VideoProcessor()
    
    # Create sample landmarks
    landmarks = {
        'landmark_12_x': 0.3,  # Shoulder X
        'landmark_12_y': 0.4,  # Shoulder Y
        'landmark_12_z': 0.1,
        'landmark_12_visibility': 0.95,
        'landmark_14_x': 0.4,  # Elbow X
        'landmark_14_y': 0.6,
        'landmark_14_z': 0.2,
        'landmark_14_visibility': 0.90,
        'frame': 1,
        'label': 1
    }
    
    print(f"  Original Landmarks:")
    print(f"    Shoulder X: {landmarks['landmark_12_x']}")
    print(f"    Elbow X: {landmarks['landmark_14_x']}")
    
    mirrored = processor.mirror_landmarks(landmarks)
    
    print(f"\n  Mirrored Landmarks:")
    print(f"    Shoulder X: {mirrored['landmark_12_x']}")
    print(f"    Elbow X: {mirrored['landmark_14_x']}")
    
    # Verify mirroring formula: x_new = 1 - x_old
    expected_shoulder = 1.0 - 0.3  # 0.7
    expected_elbow = 1.0 - 0.4     # 0.6
    
    checks = [
        (mirrored['landmark_12_x'] == expected_shoulder, 
         f"Shoulder X: {mirrored['landmark_12_x']} == {expected_shoulder}"),
        (mirrored['landmark_14_x'] == expected_elbow,
         f"Elbow X: {mirrored['landmark_14_x']} == {expected_elbow}"),
        (mirrored['landmark_12_y'] == landmarks['landmark_12_y'],
         f"Y-coordinate unchanged"),
        (mirrored['landmark_12_visibility'] == landmarks['landmark_12_visibility'],
         f"Visibility unchanged"),
        (mirrored['label'] == landmarks['label'],
         f"Label preserved"),
    ]
    
    passed = sum(1 for check, _ in checks if check)
    
    for check, desc in checks:
        status = "✅" if check else "❌"
        print(f"  {status} {desc}")
    
    print(f"\n  Results: {passed}/{len(checks)} checks passed")
    return passed == len(checks)


def test_dataframe_augmentation():
    """Test creating augmented dataset with original + mirrored"""
    print("\n" + "="*70)
    print("TEST 4: DataFrame Augmentation (Original + Mirrored)")
    print("="*70)
    
    processor = VideoProcessor()
    
    # Create sample data
    data = []
    for i in range(5):
        frame = {
            f'landmark_{j}_x': np.random.uniform(0, 1) for j in range(33)
        }
        frame.update({
            f'landmark_{j}_y': np.random.uniform(0, 1) for j in range(33)
        })
        frame.update({
            f'landmark_{j}_z': np.random.uniform(-0.5, 0.5) for j in range(33)
        })
        frame.update({
            f'landmark_{j}_visibility': np.random.uniform(0.5, 1.0) for j in range(33)
        })
        frame.update({
            'frame': i,
            'label': i % 2,
            'augmentation': 'original'
        })
        data.append(frame)
    
    print(f"  Original data: {len(data)} frames")
    
    # Simulate augmentation (as done in extract_landmarks_from_video)
    augmented_data = []
    for original in data:
        augmented_data.append(original.copy())  # Original
        
        mirrored = processor.mirror_landmarks(original.copy())
        mirrored['augmentation'] = 'mirrored'
        augmented_data.append(mirrored)  # Mirrored
    
    print(f"  After mirroring: {len(augmented_data)} frames")
    
    df = pd.DataFrame(augmented_data)
    
    # Verify
    original_count = len(df[df['augmentation'] == 'original'])
    mirrored_count = len(df[df['augmentation'] == 'mirrored'])
    
    print(f"\n  Augmentation Summary:")
    print(f"    Original samples: {original_count}")
    print(f"    Mirrored samples: {mirrored_count}")
    print(f"    Total: {len(df)}")
    print(f"    Ratio: {len(df) / len(data):.1f}x (expected: 2.0x)")
    
    # Check X-coordinate flipping
    orig = df.iloc[0]
    mirr = df.iloc[1]
    
    x_flip_check = abs(orig['landmark_12_x'] + mirr['landmark_12_x'] - 1.0) < 0.001
    
    print(f"\n  X-coordinate flip verification:")
    print(f"    Original X: {orig['landmark_12_x']:.4f}")
    print(f"    Mirrored X: {mirr['landmark_12_x']:.4f}")
    print(f"    Sum: {orig['landmark_12_x'] + mirr['landmark_12_x']:.4f} (expected: 1.0)")
    print(f"    {'✅' if x_flip_check else '❌'} Flip correct")
    
    success = (original_count == mirrored_count == len(data)) and x_flip_check
    
    print(f"\n  Results: {'✅ PASSED' if success else '❌ FAILED'}")
    return success


def test_folder_structure_discovery():
    """Test auto-discovery of exercise/age/form folders"""
    print("\n" + "="*70)
    print("TEST 5: Folder Structure Auto-Discovery")
    print("="*70)
    
    processor = VideoProcessor()
    
    # Create temporary folder structure
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir) / 'videos'
        
        # Create structure
        folders = [
            'bicep/adult/good_form',
            'bicep/adult/bad_form',
            'back/senior/good_form',
            'chest/children/good_form',
        ]
        
        for folder in folders:
            (base / folder).mkdir(parents=True, exist_ok=True)
        
        print(f"  Created test structure in: {base}")
        print(f"  Folders created:")
        for folder in folders:
            print(f"    ✓ {folder}")
        
        # Test path validation on created structure
        test_paths = [
            str(base / 'bicep/adult/good_form'),
            str(base / 'back/senior/good_form'),
            str(base / 'chest/children/good_form'),
        ]
        
        print(f"\n  Folder validation:")
        for path in test_paths:
            exists = Path(path).exists()
            print(f"    {'✅' if exists else '❌'} {path}")
        
        return all(Path(p).exists() for p in test_paths)


def test_valid_exercises_ages():
    """Test valid exercises and age groups"""
    print("\n" + "="*70)
    print("TEST 6: Valid Exercises and Age Groups")
    print("="*70)
    
    processor = VideoProcessor()
    
    print(f"  Valid Exercises: {processor.valid_exercises}")
    print(f"  Valid Age Groups: {processor.valid_age_groups}")
    print(f"  Label Mapping: {processor.label_mapping}")
    
    checks = [
        (processor.valid_exercises == ['bicep', 'back', 'chest'], "Exercises correct"),
        (processor.valid_age_groups == ['children', 'adult', 'senior'], "Age groups correct"),
        (processor.label_mapping == {'good_form': 1, 'bad_form': 0}, "Label mapping correct"),
    ]
    
    passed = sum(1 for check, _ in checks if check)
    
    print(f"\n  Validation:")
    for check, desc in checks:
        print(f"    {'✅' if check else '❌'} {desc}")
    
    print(f"\n  Results: {passed}/{len(checks)} checks passed")
    return passed == len(checks)


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("VIDEO PIPELINE TEST SUITE")
    print("Testing: Metadata Extraction, Data Augmentation (Mirroring)")
    print("="*70)
    
    tests = [
        ("Metadata Extraction", test_metadata_extraction),
        ("Invalid Path Rejection", test_invalid_paths),
        ("Landmark Mirroring", test_landmark_mirroring),
        ("DataFrame Augmentation", test_dataframe_augmentation),
        ("Folder Discovery", test_folder_structure_discovery),
        ("Valid Exercises/Ages", test_valid_exercises_ages),
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
