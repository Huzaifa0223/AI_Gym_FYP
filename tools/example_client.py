"""
Example Client for Testing the AI Gym Backend API
Demonstrates how to integrate with the FastAPI backend
"""
import requests
import cv2
import base64
import json
import time
from typing import Optional

class AIGymClient:
    """Client for interacting with AI Gym API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id: Optional[str] = None
    
    def check_health(self):
        """Check if API is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def start_session(self, user_id: str, exercise_type: str, age: int):
        """Start a new workout session"""
        try:
            response = requests.post(
                f"{self.base_url}/api/start-session",
                data={
                    "user_id": user_id,
                    "exercise_type": exercise_type,
                    "age": age
                }
            )
            result = response.json()
            if result.get("success"):
                self.session_id = result["session_id"]
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def process_frame(self, frame, exercise_type: str, age: int):
        """
        Send a frame to the API for processing
        
        Args:
            frame: OpenCV frame (numpy array)
            exercise_type: 'bicep_curl', 'bent_over_row', or 'push_up'
            age: User's age
            
        Returns:
            API response with rep count and feedback
        """
        try:
            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send to API
            response = requests.post(
                f"{self.base_url}/api/process-frame",
                json={
                    "exercise_type": exercise_type,
                    "age": age,
                    "frame_data": frame_base64
                }
            )
            
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def end_session(self):
        """End the current session"""
        if not self.session_id:
            return {"success": False, "error": "No active session"}
        
        try:
            response = requests.post(
                f"{self.base_url}/api/end-session",
                data={"session_id": self.session_id}
            )
            result = response.json()
            self.session_id = None
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_exercises(self):
        """Get list of available exercises"""
        try:
            response = requests.get(f"{self.base_url}/api/exercises")
            return response.json()
        except Exception as e:
            return {"error": str(e)}


def demo_webcam_integration(exercise_type='bicep_curl', age=25):
    """
    Demo: Process webcam feed through API
    Shows real-time integration example
    """
    print("="*60)
    print("AI GYM API - WEBCAM DEMO")
    print("="*60)
    
    # Initialize client
    client = AIGymClient()
    
    # Check health
    health = client.check_health()
    print(f"\nAPI Status: {health.get('status', 'unknown')}")
    
    if health.get('status') != 'healthy':
        print("[ERROR] API is not running!")
        print("Start the API first: python api_backend.py")
        return
    
    # Start session
    print(f"\nStarting session...")
    session = client.start_session("demo_user", exercise_type, age)
    print(f"Session ID: {session.get('session_id', 'N/A')}")
    print(f"Age Group: {session.get('age_group', 'N/A')}")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return
    
    print(f"\n[INFO] Processing {exercise_type} exercise for age {age}")
    print("[INFO] Press 'q' to quit\n")
    
    frame_count = 0
    last_process_time = time.time()
    process_interval = 0.1  # Process every 100ms
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame at intervals (not every frame to reduce API load)
        current_time = time.time()
        if current_time - last_process_time >= process_interval:
            result = client.process_frame(frame, exercise_type, age)
            last_process_time = current_time
            
            if result.get('success'):
                # Display results
                cv2.putText(
                    frame,
                    f"Reps: {result['counter']}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"{result['feedback']}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    f"Quality: {result['rep_quality']:.0f}%",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                
                # Print to console
                if result.get('is_valid_rep'):
                    print(f"✓ Rep {result['counter']} - Quality: {result['rep_quality']:.0f}%")
        
        # Display frame
        cv2.imshow('AI Gym - API Demo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # End session
    print("\n[INFO] Ending session...")
    summary = client.end_session()
    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)
    print(f"Total Reps: {summary.get('total_reps', 0)}")
    print(f"Frames Processed: {summary.get('frames_processed', 0)}")
    print(f"Duration: {summary.get('start_time', 'N/A')} to {summary.get('end_time', 'N/A')}")
    print("="*60)


def test_api_endpoints():
    """Test all API endpoints"""
    print("="*60)
    print("TESTING AI GYM API ENDPOINTS")
    print("="*60)
    
    client = AIGymClient()
    
    # Test 1: Health Check
    print("\n1. Health Check")
    health = client.check_health()
    print(f"   Status: {health.get('status')}")
    print(f"   Active Sessions: {health.get('active_sessions', 0)}")
    
    # Test 2: Get Exercises
    print("\n2. Available Exercises")
    exercises = client.get_exercises()
    for ex in exercises.get('exercises', []):
        print(f"   - {ex['name']} ({ex['id']})")
    
    # Test 3: Get Age Groups
    print("\n3. Age Groups")
    response = requests.get(f"{client.base_url}/api/age-groups")
    age_groups = response.json()
    for ag in age_groups.get('age_groups', []):
        print(f"   - {ag['name']}: {ag['age_range']}")
    
    print("\n" + "="*60)
    print("API tests completed!")
    print("="*60)


if __name__ == '__main__':
    import sys
    
    print("\n" + "="*60)
    print("AI GYM - CLIENT EXAMPLE")
    print("="*60)
    print("\nUsage:")
    print("  python example_client.py test          - Test API endpoints")
    print("  python example_client.py demo          - Webcam demo (bicep)")
    print("  python example_client.py demo back 30  - Webcam demo (back, age 30)")
    print("="*60 + "\n")
    
    if len(sys.argv) < 2:
        print("Please specify 'test' or 'demo'")
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == 'test':
        test_api_endpoints()
    
    elif command == 'demo':
        exercise = sys.argv[2] if len(sys.argv) > 2 else 'bicep_curl'
        age = int(sys.argv[3]) if len(sys.argv) > 3 else 25
        demo_webcam_integration(exercise, age)
    
    else:
        print(f"Unknown command: {command}")
