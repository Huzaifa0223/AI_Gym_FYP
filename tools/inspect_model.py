#!/usr/bin/env python
"""
Simple script to inspect the bicep_model.pkl file
"""
import pickle
from pathlib import Path
import sys

model_path = Path('data/models/bicep_model.pkl')

print("=" * 70)
print("BICEP MODEL INSPECTION")
print("=" * 70)
print()

if not model_path.exists():
    print(f"[ERROR] File not found: {model_path}")
    sys.exit(1)

# File info
file_size_kb = model_path.stat().st_size / 1024
file_mtime = model_path.stat().st_mtime

print(f"[File]     : {model_path}")
print(f"[Size]     : {file_size_kb:.2f} KB ({model_path.stat().st_size} bytes)")
print(f"[Modified] : {file_mtime}")
print()
print("=" * 70)
print("LOADING MODEL...")
print("=" * 70)
print()

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print("[SUCCESS] Model loaded successfully!")
    print()
    print("[Type]   :", type(model).__name__)
    print("[Module] :", type(model).__module__)
    print()
    print("[Model Attributes]:")
    print("-" * 70)
    
    if hasattr(model, '__dict__'):
        attrs = model.__dict__
        for key in sorted(attrs.keys()):
            val = attrs[key]
            val_type = type(val).__name__
            
            # Show shape if it's a numpy array
            if hasattr(val, 'shape'):
                print(f"  {key:30} : {val_type:20} Shape: {val.shape}")
            else:
                val_str = str(val)[:50]  # Truncate long values
                print(f"  {key:30} : {val_type:20} = {val_str}")
    
    print()
    print("=" * 70)
    print("[SUMMARY]")
    print("=" * 70)
    print("This is a K-Means clustering model trained on bicep exercise data.")
    print("Uses: sklearn.cluster.KMeans")
    print()
    print("Common used attributes:")
    print("  - cluster_centers_ : The cluster centers (e.g., 'up' and 'down' positions)")
    print("  - labels_         : Cluster assignments for training data")
    print("  - inertia_        : Sum of squared distances to nearest cluster")
    print()
    print("[READY] Model is ready for inference/prediction!")
    print("=" * 70)
    
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print(f"[ERROR] Exception type: {type(e).__name__}")
    sys.exit(1)
