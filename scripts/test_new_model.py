"""
Quick test of the new model file
"""
import sys
import os
BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"
sys.path.append(os.path.join(BASE_DIR, "backend"))

import torch

model_path = os.path.join(BASE_DIR, "models", "image", "cnn_qnn_model_state.pkl")

print("="*80)
print("TESTING NEW MODEL FILE")
print("="*80)

print(f"\nModel path: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")
print(f"File size: {os.path.getsize(model_path) / 1024**2:.2f} MB")

print("\nAttempting to load...")
try:
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    print("✅ Successfully loaded!")
    
    if isinstance(checkpoint, dict):
        print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
        if 'model_state_dict' in checkpoint:
            print("✅ Found 'model_state_dict' - this is correct format!")
            state_dict = checkpoint['model_state_dict']
            print(f"\nModel state_dict keys (first 5):")
            for i, key in enumerate(list(state_dict.keys())[:5]):
                print(f"  {i+1}. {key}: {state_dict[key].shape}")
    else:
        print(f"Checkpoint type: {type(checkpoint)}")
        
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
