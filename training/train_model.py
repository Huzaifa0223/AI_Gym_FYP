import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle

# 1. Load the Data
print("Loading dataset...")
df = pd.read_csv('bicep_dataset.csv')

# 2. Feature Engineering: Calculate the Elbow Angle
# We can't just feed raw X,Y coordinates because the user might stand in different spots.
# The ANGLE is universal.
def calculate_angle(row):
    # Get coordinates for Shoulder (A), Elbow (B), Wrist (C)
    # Using indices from your CSV columns
    a = np.array([row['shoulder_x'], row['shoulder_y']])
    b = np.array([row['elbow_x'], row['elbow_y']])
    c = np.array([row['wrist_x'], row['wrist_y']])
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate cosine of angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

print("Calculating angles for all frames...")
df['angle'] = df.apply(calculate_angle, axis=1)

# 3. Train the Model (Unsupervised K-Means)
# We ask the AI to find 2 clusters: "Arm Up" and "Arm Down"
X = df[['angle']].values
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# 4. Analyze the Clusters
# Find which cluster represents "UP" (Smaller angle) and "DOWN" (Larger angle)
center_0 = kmeans.cluster_centers_[0][0]
center_1 = kmeans.cluster_centers_[1][0]

if center_0 < center_1:
    up_cluster = 0
    down_cluster = 1
    print(f"Cluster 0 is UP (Avg Angle: {center_0:.1f})")
    print(f"Cluster 1 is DOWN (Avg Angle: {center_1:.1f})")
else:
    up_cluster = 1
    down_cluster = 0
    print(f"Cluster 1 is UP (Avg Angle: {center_1:.1f})")
    print(f"Cluster 0 is DOWN (Avg Angle: {center_0:.1f})")

# 5. Save the Brain (The Model)
model_filename = "bicep_model.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(kmeans, f)
print(f"Success! Model saved to {model_filename}")

# 6. Visualize the Result (Optional but recommended)
plt.scatter(df.index, df['angle'], c=df['cluster'], cmap='viridis', s=1)
plt.title("AI Automatically Detecting Up (Purple) vs Down (Yellow)")
plt.xlabel("Frame Number")
plt.ylabel("Elbow Angle")
plt.show()