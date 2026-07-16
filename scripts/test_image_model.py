"""
Test image model predictions directly to diagnose the issue
"""
import sys
import os
BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"
sys.path.append(os.path.join(BASE_DIR, "backend"))

from models.image import ImageModel
import torch
import numpy as np

# Load model
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "image", "cnn_qnn_best.pt")
model = ImageModel(model_path=IMAGE_MODEL_PATH)

# Test with a known image
test_image = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT\dataset\Image Dataset\PlantVillage\Tomato_healthy\0a31a3ef-3c08-46f4-9c13-cbb27c6d2f22___GH_HL Leaf 279.JPG"

print("Testing image prediction...")
print(f"Image: {os.path.basename(test_image)}")
print(f"True class: Tomato_healthy (should be class 14)")

probs = model.predict(test_image)
print(f"\nPrediction shape: {probs.shape}")
print(f"Predicted class: {np.argmax(probs)}")
print(f"Confidence: {np.max(probs):.4f}")

print("\nTop 5 predictions:")
top5_idx = np.argsort(probs[0])[::-1][:5]
for i, idx in enumerate(top5_idx, 1):
    print(f"  {i}. Class {idx}: {probs[0][idx]:.4f}")

# Check label mapping
print("\n" + "="*80)
print("Checking label mapping...")
label_mapping_path = os.path.join(BASE_DIR, "models", "image", "label_mapping.pkl")
if os.path.exists(label_mapping_path):
    import pickle
    with open(label_mapping_path, 'rb') as f:
        mapping_data = pickle.load(f)
        print("Label mapping exists:")
        print(f"  Image labels: {len(mapping_data['image_labels'])}")
        print(f"  Text labels: {len(mapping_data['text_labels'])}")
        print("\nMapping (first 5):")
        for i in range(min(5, len(mapping_data['image_to_text']))):
            img_label = mapping_data['image_labels'][i]
            text_idx = mapping_data['image_to_text'][i]
            text_label = mapping_data['text_labels'][text_idx]
            print(f"  Image[{i}] '{img_label}' -> Text[{text_idx}] '{text_label}'")
else:
    print("No label mapping found!")
