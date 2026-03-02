"""
Quick Setup Script - Creates folder structure for videos and models
"""
import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create all necessary directories"""
    
    directories = [
        # Training videos structure
        'videos/bicep/children/good_form',
        'videos/bicep/children/bad_form',
        'videos/bicep/adult/good_form',
        'videos/bicep/adult/bad_form',
        'videos/bicep/senior/good_form',
        'videos/bicep/senior/bad_form',
        
        'videos/back/children/good_form',
        'videos/back/children/bad_form',
        'videos/back/adult/good_form',
        'videos/back/adult/bad_form',
        'videos/back/senior/good_form',
        'videos/back/senior/bad_form',
        
        'videos/chest/children/good_form',
        'videos/chest/children/bad_form',
        'videos/chest/adult/good_form',
        'videos/chest/adult/bad_form',
        'videos/chest/senior/good_form',
        'videos/chest/senior/bad_form',
        
        # Data directories
        'data/models',
        'data/training',
        
        # Backup
        'videos/backup_original'
    ]
    
    print("="*60)
    print("CREATING DIRECTORY STRUCTURE")
    print("="*60)
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {directory}")
        else:
            print(f"  Exists: {directory}")
    
    print("\n" + "="*60)
    print("DIRECTORY STRUCTURE COMPLETE")
    print("="*60)
    
    return True


def move_existing_videos():
    """Move existing bicep videos to proper location"""
    
    source_dir = Path('videos')
    target_dir = Path('videos/bicep/adult/good_form')
    backup_dir = Path('videos/backup_original')
    
    # Find all video files in root videos directory
    video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(source_dir.glob(f'*{ext}'))
    
    if not video_files:
        print("\n[INFO] No videos found in videos/ root directory")
        return
    
    print(f"\n[INFO] Found {len(video_files)} video files")
    print("\nOptions:")
    print("1. Move to videos/bicep/adult/good_form/ (recommended)")
    print("2. Copy to videos/bicep/adult/good_form/ (keep originals)")
    print("3. Skip (do nothing)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        # Move files
        print(f"\nMoving {len(video_files)} videos...")
        for video in video_files:
            target = target_dir / video.name
            print(f"  Moving: {video.name}")
            shutil.move(str(video), str(target))
        print("✓ All videos moved!")
        
    elif choice == '2':
        # Copy files
        print(f"\nCopying {len(video_files)} videos...")
        
        # Also backup originals
        for video in video_files:
            # Copy to target
            target = target_dir / video.name
            print(f"  Copying: {video.name}")
            shutil.copy2(str(video), str(target))
            
            # Backup original
            backup = backup_dir / video.name
            shutil.copy2(str(video), str(backup))
        
        print("✓ All videos copied!")
        print(f"✓ Originals backed up to {backup_dir}")
    
    else:
        print("Skipped video migration")


def create_readme_files():
    """Create README files in each video directory with instructions"""
    
    exercise_instructions = {
        'bicep': {
            'children': 'Modified bicep curls with light weights or resistance bands. 5-10 reps per video.',
            'adult': 'Standard bicep curls with dumbbells or barbell. Full range of motion. 5-10 reps per video.',
            'senior': 'Seated or supported bicep curls with light weights. Controlled movement. 5-10 reps per video.'
        },
        'back': {
            'children': 'Light resistance band rows or assisted pull movements. Focus on form.',
            'adult': 'Bent-over rows, lat pulldowns, or pull-ups. Full range of motion.',
            'senior': 'Seated rows with light resistance. Supported movements.'
        },
        'chest': {
            'children': 'Wall push-ups or incline push-ups. Modified form acceptable.',
            'adult': 'Standard push-ups or bench press. Full range of motion.',
            'senior': 'Wall push-ups or seated chest press with light weights.'
        }
    }
    
    exercises = ['bicep', 'back', 'chest']
    age_groups = ['children', 'adult', 'senior']
    
    for exercise in exercises:
        for age_group in age_groups:
            readme_path = Path(f'videos/{exercise}/{age_group}/README.txt')
            
            content = f"""
====================================
{exercise.upper()} EXERCISE - {age_group.upper()}
====================================

Place your {exercise} exercise videos here.

INSTRUCTIONS:
{exercise_instructions[exercise][age_group]}

VIDEO REQUIREMENTS:
- Duration: 10-30 seconds
- Reps: 5-10 repetitions
- View: Side view (90° angle to camera)
- Lighting: Good, consistent lighting
- Background: Minimal clutter, solid color preferred
- Quality: At least 480p, 720p preferred

GOOD FORM CRITERIA:
- Proper posture maintained
- Full or appropriate range of motion
- Controlled movement (not too fast)
- No excessive swinging or compensation

BAD FORM (for bad_form folder):
- Poor posture
- Partial range of motion
- Too fast/jerky movements
- Incorrect technique

NAMING:
- Use descriptive names: {exercise}_form1.mp4, {exercise}_form2.mp4
- Or keep default names from camera

QUANTITY:
- Minimum: 10 videos per category
- Recommended: 20-30 videos
- More variety = better ML model

After adding videos:
1. Run: python video_pipeline.py
2. Then: python train_universal.py
"""
            
            with open(readme_path, 'w') as f:
                f.write(content.strip())
    
    print("\n[INFO] Created README files in all video directories")


def show_next_steps():
    """Display next steps after setup"""
    
    print("\n" + "="*60)
    print("SETUP COMPLETE! 🎉")
    print("="*60)
    
    print("\n📁 Directory structure created:")
    print("   videos/{exercise}/{age_group}/good_form/")
    print("   videos/{exercise}/{age_group}/bad_form/")
    
    print("\n📝 Next Steps:")
    print("   1. Add your exercise videos to the appropriate folders")
    print("      - Read README.txt files in each folder for guidance")
    print("   ")
    print("   2. Process videos:")
    print("      python video_pipeline.py")
    print("   ")
    print("   3. Train models:")
    print("      python train_universal.py")
    print("   ")
    print("   4. Test the API:")
    print("      python api_backend.py")
    print("      python example_client.py test")
    print("   ")
    print("   5. Read full guide:")
    print("      Open: DEPLOYMENT_SUMMARY.md")
    
    print("\n💡 Tips:")
    print("   - You need videos for ALL exercises and age groups")
    print("   - Minimum 10 videos per category")
    print("   - Side view works best for detection")
    print("   - Good lighting is essential")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("AI GYM - QUICK SETUP")
    print("="*60)
    
    # Step 1: Create directories
    create_directory_structure()
    
    # Step 2: Move existing videos
    move_existing_videos()
    
    # Step 3: Create README files
    create_readme_files()
    
    # Step 4: Show next steps
    show_next_steps()
