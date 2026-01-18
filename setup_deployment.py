"""
Setup Guide and Requirements for AI Gym Project
"""

# Required packages
REQUIREMENTS = """
# Core ML and CV
mediapipe==0.10.9
opencv-python==4.8.1.78
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0

# Backend API
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pydantic==2.5.0

# Visualization
matplotlib==3.7.2

# Utilities
python-dateutil==2.8.2
"""

# Directory structure
DIRECTORY_STRUCTURE = """
AI_Gym_FYP/
├── api_backend.py              # FastAPI backend (NEW)
├── video_pipeline.py           # Video processing for training (NEW)
├── train_universal.py          # Universal training script (NEW)
├── main.py                     # Original main app
├── bicep_curl.py              # Bicep detection
├── pose_utils.py              # Pose utilities
├── visual_utils.py            # Visualization
├── expert_coach.py            # Coaching system
│
├── exercises/                  # Exercise modules (NEW)
│   ├── __init__.py
│   ├── base_exercise.py       # Base class & configs
│   ├── back_exercise.py       # Back exercise detection
│   └── chest_exercise.py      # Chest exercise detection
│
├── data/
│   ├── models/                # Trained models
│   │   ├── bicep_children.pkl
│   │   ├── bicep_adult.pkl
│   │   ├── bicep_senior.pkl
│   │   ├── back_children.pkl
│   │   ├── back_adult.pkl
│   │   ├── back_senior.pkl
│   │   ├── chest_children.pkl
│   │   ├── chest_adult.pkl
│   │   └── chest_senior.pkl
│   └── training/              # Training datasets
│
└── videos/                    # Training videos (ORGANIZE)
    ├── bicep/
    │   ├── children/
    │   │   ├── good_form/
    │   │   └── bad_form/
    │   ├── adult/
    │   │   ├── good_form/
    │   │   └── bad_form/
    │   └── senior/
    │       ├── good_form/
    │       └── bad_form/
    ├── back/
    │   └── [same structure]
    └── chest/
        └── [same structure]
"""

SETUP_INSTRUCTIONS = """
====================================
AI GYM - SETUP INSTRUCTIONS
====================================

STEP 1: Install Requirements
-----------------------------
pip install -r requirements.txt

STEP 2: Organize Training Videos
---------------------------------
1. Create folder structure:
   videos/
     ├── bicep/children/good_form/
     ├── bicep/children/bad_form/
     ├── bicep/adult/good_form/
     ... (etc)

2. Place your videos in appropriate folders
   - Current bicep videos → videos/bicep/adult/good_form/
   - Add back exercise videos → videos/back/[age]/good_form/
   - Add chest exercise videos → videos/chest/[age]/good_form/

STEP 3: Process Videos & Generate Training Data
-----------------------------------------------
python video_pipeline.py

This will:
- Extract pose landmarks from all videos
- Save training data as CSV files in data/training/
- Create one CSV per exercise/age/label combination

STEP 4: Train ML Models
-----------------------
python train_universal.py

This will:
- Train models for all 9 combinations (3 exercises × 3 age groups)
- Save models to data/models/
- Generate visualization plots

STEP 5: Test Locally
--------------------
# Test specific exercise
python main.py --exercise bicep --age 25

# Or use the old main.py for biceps only

STEP 6: Run Backend API
-----------------------
python api_backend.py

API will be available at:
- http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health

STEP 7: Backend Integration
---------------------------
Your backend should send POST requests to:
/api/process-frame

Example request:
{
  "exercise_type": "bicep",
  "age": 25,
  "frame_data": "<base64_encoded_image>"
}

Example response:
{
  "success": true,
  "counter": 5,
  "feedback": "✓ GOOD FORM",
  "stage": "up",
  "primary_angle": 145.2,
  "secondary_angle": 85.3,
  "rep_quality": 95.0,
  "confidence": 0.98,
  "is_valid_rep": true,
  "timestamp": "2026-01-16T10:30:45"
}

====================================
DEPLOYMENT CHECKLIST
====================================

Before deploying to production:

□ All 9 models trained and tested
□ API endpoints tested with Postman/curl
□ CORS configured for your frontend domain
□ Environment variables set (API keys, etc.)
□ Error handling tested
□ Load testing completed
□ Monitoring/logging configured
□ Docker containerization (optional)
□ SSL/HTTPS certificate (production)

====================================
VIDEO RECORDING TIPS
====================================

For best training results:

Children (8-15):
- Modified exercises (e.g., wall push-ups)
- Slower pace
- Clear side view

Adults (20-50):
- Standard form
- Full range of motion
- Side view preferred

Seniors (60-80):
- Gentler variations
- Chair support if needed
- Safety-focused form

General:
- Good lighting
- Minimal background clutter
- Consistent camera angle
- 5-10 reps per video
- Multiple people for variety

====================================
API ENDPOINTS SUMMARY
====================================

GET  /                     - API info
GET  /health              - Health check
GET  /api/exercises       - List exercises
GET  /api/age-groups      - Age group info
POST /api/start-session   - Start workout session
POST /api/process-frame   - Process single frame
POST /api/end-session     - End session & get summary

====================================
"""

if __name__ == '__main__':
    print(SETUP_INSTRUCTIONS)
    
    # Write requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write(REQUIREMENTS.strip())
    print("\n[✓] requirements.txt created")
    
    # Create README.md
    with open('README_DEPLOYMENT.md', 'w', encoding='utf-8') as f:
        f.write("# AI Gym - Deployment Guide\n\n")
        f.write("## Overview\n")
        f.write("Multi-exercise, age-adaptive AI fitness trainer with backend API.\n\n")
        f.write("## Features\n")
        f.write("- 3 Exercise Types: Bicep, Back, Chest\n")
        f.write("- 3 Age Groups: Children (8-15), Adults (16-59), Seniors (60+)\n")
        f.write("- Real-time form detection\n")
        f.write("- RESTful API for backend integration\n\n")
        f.write("## Directory Structure\n```\n")
        f.write(DIRECTORY_STRUCTURE.strip())
        f.write("\n```\n\n")
        f.write("## Setup\n")
        f.write(SETUP_INSTRUCTIONS)
    
    print("[✓] README_DEPLOYMENT.md created")
    print("\n" + "="*60)
    print("Setup files created! Read README_DEPLOYMENT.md for instructions.")
    print("="*60)
