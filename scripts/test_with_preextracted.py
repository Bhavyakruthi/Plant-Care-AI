"""
Test classifier with pre-extracted features to isolate the root cause
This bypasses live image processing to test if the issue is in the classifier or preprocessing
"""
import sys
import os
import torch
import numpy as np
BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"
sys.path.append(os.path.join(BASE_DIR, "backend"))

from models.image import ImageModel

print("="*80)
print("TESTING CLASSIFIER WITH PRE-EXTRACTED FEATURES")
print("="*80)

# Load pre-extracted test features (same ones used during training)
print("\n📂 Loading pre-extracted features...")
feature_path = os.path.join(BASE_DIR, "dataset", "Image Dataset", "Features", "CNN_resnet18_test.pt")
test_data = torch.load(feature_path, map_location='cpu', weights_only=False)

features = test_data['features']
labels = test_data['labels']
n_classes = test_data['n_classes']

print(f"✅ Loaded {len(features)} test samples")
print(f"   Features shape: {features.shape}")
print(f"   Labels shape: {labels.shape}")
print(f"   Number of classes: {n_classes}")

# Load the image model (just need the classifier)
print("\n🤖 Loading classifier model...")
model = ImageModel()

# Test classifier directly with pre-extracted features
print("\n🧪 Testing classifier with pre-extracted features...")
device = model.device
features = features.to(device)
labels_cpu = labels.cpu().numpy()

with torch.no_grad():
    # Pass features directly to classifier (no preprocessing, no remapping)
    logits = model.classifier(features)
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1).cpu().numpy()

# Calculate accuracy WITHOUT label remapping (testing raw classifier)
raw_accuracy = (preds == labels_cpu).mean()

print(f"\n📊 RAW CLASSIFIER RESULTS (no label mapping):")
print(f"   Accuracy: {raw_accuracy*100:.2f}%")
print(f"   Expected: ~90% if model trained correctly")

# Now test with label mapping
if model.label_mapping is not None:
    print(f"\n🔄 Applying label mapping...")
    print(f"   Mapping shape: {model.label_mapping.shape}")
    
    # Remap predictions
    remapped_preds = np.zeros_like(preds)
    for img_idx, text_idx in enumerate(model.label_mapping):
        mask = preds == img_idx
        remapped_preds[mask] = text_idx
    
    mapped_accuracy = (remapped_preds == labels_cpu).mean()
    print(f"\n📊 MAPPED CLASSIFIER RESULTS (with label mapping):")
    print(f"   Accuracy: {mapped_accuracy*100:.2f}%")
else:
    print(f"\n⚠️  No label mapping found!")

# Show sample predictions
print(f"\n📝 Sample Predictions (first 10):")
print(f"   True labels: {labels_cpu[:10]}")
print(f"   Predictions: {preds[:10]}")
if model.label_mapping is not None:
    print(f"   Remapped:    {remapped_preds[:10]}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

if raw_accuracy > 0.85:
    print("✅ Classifier works correctly with pre-extracted features!")
    print("   Issue is likely in LIVE image preprocessing pipeline")
elif raw_accuracy < 0.15:
    print("❌ Classifier performs poorly even with pre-extracted features!")
    print("   The trained model checkpoint may be corrupted or from wrong epoch")
else:
    print("⚠️  Classifier has moderate accuracy")
    print(f"   This suggests label alignment issues")
