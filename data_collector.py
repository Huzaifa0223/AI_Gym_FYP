# utils/data_collector.py
import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime

class DataCollector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.collected_data = []
        
    def collect_single_rep(self, output_file='collected_data.json'):
        """
        Collect data for one bicep curl rep
        Press 's' to start, 'e' to end, 'q' to quit
        """
        cap = cv2.VideoCapture(0)
        print("=== Data Collection Mode ===")
        print("Press 's' to START recording a rep")
        print("Press 'e' to END recording")
        print("Press 'q' to QUIT")
        
        recording = False
        rep_data = []
        
        while True:
            success, frame = cap.read()
            if not success:
                continue
                
            frame = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            # Display instructions
            status = "🔴 RECORDING" if recording else "🟢 READY"
            cv2.putText(frame, f"Status: {status}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "s=Start, e=End, q=Quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Extract right arm landmarks
                shoulder = [landmarks[12].x, landmarks[12].y]
                elbow = [landmarks[14].x, landmarks[14].y]
                wrist = [landmarks[16].x, landmarks[16].y]
                hip = [landmarks[24].x, landmarks[24].y]
                
                # Calculate angle
                def calculate_angle(a, b, c):
                    a = np.array(a)
                    b = np.array(b)
                    c = np.array(c)
                    ba = a - b
                    bc = c - b
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle = np.degrees(np.arccos(cosine_angle))
                    return angle
                
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                
                # Display angle
                cv2.putText(frame, f"Elbow Angle: {int(elbow_angle)}°", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if recording:
                    rep_data.append({
                        'timestamp': datetime.now().isoformat(),
                        'elbow_angle': float(elbow_angle),
                        'shoulder': shoulder,
                        'elbow': elbow,
                        'wrist': wrist
                    })
                    cv2.putText(frame, f"Frames: {len(rep_data)}", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow("Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                recording = True
                rep_data = []  # Clear previous data
                print("Started recording...")
                
            elif key == ord('e'):
                if recording and rep_data:
                    recording = False
                    self.collected_data.append(rep_data)
                    print(f"Stopped recording. Collected {len(rep_data)} frames.")
                    
                    # Save to file
                    self.save_data(output_file)
                    
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.collected_data:
            print(f"\n=== Collection Complete ===")
            print(f"Collected {len(self.collected_data)} reps")
            print(f"Data saved to {output_file}")
        
    def save_data(self, filename):
        """Save collected data to JSON file"""
        os.makedirs('training_data', exist_ok=True)
        
        with open(f'training_data/{filename}', 'w') as f:
            json.dump(self.collected_data, f, indent=2)
    
    def load_data(self, filename):
        """Load previously collected data"""
        if os.path.exists(f'training_data/{filename}'):
            with open(f'training_data/{filename}', 'r') as f:
                self.collected_data = json.load(f)
            print(f"Loaded {len(self.collected_data)} reps from {filename}")
            return True
        return False

def main():
    """Run the data collector"""
    collector = DataCollector()
    collector.collect_single_rep('bicep_curls.json')

if __name__ == "__main__":
    main()