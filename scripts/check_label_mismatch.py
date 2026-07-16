"""
Diagnostic script to check label encoder mismatch
"""
import pickle
import os

BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"

# Load preprocessed data
with open(os.path.join(BASE_DIR, "data", "preprocessed_data.pkl"), 'rb') as f:
    data = pickle.load(f)

print("=" * 80)
print("LABEL ENCODER FROM TEXT PREPROCESSING")
print("=" * 80)
print(f"\nNumber of classes: {len(data['label_encoder'].classes_)}")
print("\nClass labels (in order):")
for i, label in enumerate(data['label_encoder'].classes_):
    print(f"  {i:2d}. {label}")

# Get image dataset classes
image_dataset_dir = os.path.join(BASE_DIR, "dataset", "Image Dataset", "PlantVillage")
image_folders = sorted([f for f in os.listdir(image_dataset_dir) 
                       if os.path.isdir(os.path.join(image_dataset_dir, f))])

print("\n" + "=" * 80)
print("IMAGE DATASET FOLDERS (PlantVillage)")
print("=" * 80)
print(f"\nNumber of folders: {len(image_folders)}")
print("\nFolder names (in alphabetical order):")
for i, folder in enumerate(image_folders):
    print(f"  {i:2d}. {folder}")

print("\n" + "=" * 80)
print("MISMATCH ANALYSIS")
print("=" * 80)

text_labels = set(data['label_encoder'].classes_)
image_labels = set(image_folders)

# Look for potential matches
print("\nAttempting to match text labels to image folders:")
for text_label in sorted(text_labels):
    matched = None
    for img_folder in image_folders:
        if text_label.lower().replace(" ", "_") in img_folder.lower():
            matched = img_folder
            break
    
    if matched:
        print(f"  ✓ '{text_label}' -> '{matched}'")
    else:
        print(f"  ✗ '{text_label}' -> NO MATCH FOUND")

print("\n" + "=" * 80)
print("RECOMMENDED FIX")
print("=" * 80)
print("""
The image model was likely trained with labels in alphabetical order from PlantVillage,
but the text model uses a different label encoding.

We need to create a mapping between the two label spaces.
""")
