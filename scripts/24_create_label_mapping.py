"""
Create Label Mapping Between Image Model and Text Model
========================================================

The image model was trained on PlantVillage dataset with labels in alphabetical order.
The text model uses a different label encoding from the preprocessing step.

This script creates a mapping to convert image model predictions to text label space.
"""

import pickle
import os
import numpy as np

BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"

# Load text label encoder
with open(os.path.join(BASE_DIR, "data", "preprocessed_data.pkl"), 'rb') as f:
    data = pickle.load(f)

text_labels = data['label_encoder'].classes_

# Get image dataset folder names (sorted alphabetically - this is how image model was trained)
image_dataset_dir = os.path.join(BASE_DIR, "dataset", "Image Dataset", "PlantVillage")
all_folders = sorted([f for f in os.listdir(image_dataset_dir) 
                      if os.path.isdir(os.path.join(image_dataset_dir, f))])

# Filter out non-disease folders (like nested 'PlantVillage' directory)
image_labels = [f for f in all_folders if f != 'PlantVillage']

print("=" * 80)
print("CREATING LABEL MAPPING")
print("=" * 80)

print(f"\nText labels ({len(text_labels)} classes):")
for i, label in enumerate(text_labels):
    print(f"  {i:2d}. {label}")

print(f"\nImage labels ({len(image_labels)} classes, alphabetical order):")
for i, label in enumerate(image_labels):
    print(f"  {i:2d}. {label}")

# Create mapping: image_model_idx -> text_model_idx
image_to_text_mapping = np.zeros(len(image_labels), dtype=int)

print("\n" + "=" * 80)
print("MAPPING (Image_idx -> Text_idx)")
print("=" * 80)

for img_idx, img_label in enumerate(image_labels):
    # Find corresponding text label
    text_idx = np.where(text_labels == img_label)[0]
    if len(text_idx) > 0:
        image_to_text_mapping[img_idx] = text_idx[0]
        print(f"  Image[{img_idx:2d}] '{img_label}' -> Text[{text_idx[0]:2d}] '{text_labels[text_idx[0]]}'")
    else:
        print(f"  ⚠️  Image[{img_idx:2d}] '{img_label}' -> NO MATCH")

# Save mapping
mapping_path = os.path.join(BASE_DIR, "models", "image", "label_mapping.pkl")
with open(mapping_path, 'wb') as f:
    pickle.dump({
        'image_to_text': image_to_text_mapping,
        'image_labels': image_labels,
        'text_labels': text_labels
    }, f)

print(f"\n✅ Mapping saved to: {mapping_path}")
print("\nThis mapping will be used to convert image model predictions to text label space.")
