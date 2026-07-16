"""
Direct test of image model with a single known image
"""
import sys
import os
import torch
import numpy as np

BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"
sys.path.append(os.path.join(BASE_DIR, "backend"))

from models.image import ImageModel

print("="*80)
print("TESTING IMAGE MODEL DIRECTLY")
print("="*80)

# Load model - using default path now
model = ImageModel()

# Check if label mapping was loaded
print(f"\nLabel mapping loaded: {model.label_mapping is not None}")
if model.label_mapping is not None:
    print(f"Mapping shape: {model.label_mapping.shape}")
    print(f"Mapping (first 5): {model.label_mapping[:5]}")

# Test with a known Tomato_healthy image
test_image = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT\dataset\Image Dataset\PlantVillage\Tomato_healthy"

# Get first image from folder
import os
images = [f for f in os.listdir(test_image) if f.endswith('.JPG')][:1]
if images:
    img_path = os.path.join(test_image, images[0])
    print(f"\nTesting image: {images[0]}")
    print(f"True class folder: Tomato_healthy")
    
    try:
        probs = model.predict(img_path)
        predicted_class = np.argmax(probs)
        confidence = np.max(probs)
        
        print(f"\n✅ Prediction successful!")
        print(f"Predicted class index: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        
        # Load text label encoder to see actual class name
        import pickle
        with open(os.path.join(BASE_DIR, "data", "preprocessed_data.pkl"), 'rb') as f:
            data = pickle.load(f)
        
        label_encoder = data['label_encoder']
        print(f"Predicted class name: {label_encoder.classes_[predicted_class]}")
        print(f"Expected: Tomato_healthy (index 14)")
        
        print(f"\nTop 5 predictions:")
        top5_idx = np.argsort(probs[0])[::-1][:5]
        for i, idx in enumerate(top5_idx, 1):
            print(f"  {i}. {label_encoder.classes_[idx]}: {probs[0][idx]:.4f}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No images found in Tomato_healthy folder!")
