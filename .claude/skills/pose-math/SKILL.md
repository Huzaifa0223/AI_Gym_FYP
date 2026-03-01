---
name: pose-math
description: Helps calculate and verify joint angles using MediaPipe landmarks.
---
# Pose Math Protocol
When the user asks to calculate an angle:
1. Use the Law of Cosines or `atan2` to calculate the angle between three points.
2. Points should be identified by MediaPipe Landmark IDs (e.g., 11, 13, 15 for left arm).
3. Ensure the result is in degrees (0-180).
4. Suggest "Thresholds" for exercise states (e.g., "Up" state < 30°, "Down" state > 160°).
