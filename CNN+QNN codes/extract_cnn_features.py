"""
Complete CNN Feature Extraction for Training
=============================================
Extracts CNN features from all images and saves them for classifier training
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

print("=" * 80)
print("  COMPLETE CNN FEATURE EXTRACTION")
print("  Extracting features for train/val/test splits")
print("=" * 80)
print()

# Create directories
os.makedirs('features', exist_ok=True)
os.makedirs('experiments_stage1', exist_ok=True)

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
batch_size = 32
samples_per_class = None  # None = Use ALL data (full dataset)

if samples_per_class is None:
    print(f"Using FULL DATASET (all available images)")
else:
    print(f"Samples per class: {samples_per_class}")
print()

# Load ResNet18 feature extractor
print("Loading ResNet18...")
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
model = model.to(device)
model.eval()
print("✓ ResNet18 loaded (512-dimensional features)")
print()

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load dataset
print("Loading PlantVillage dataset...")
dataset = ImageFolder('PlantVillage', transform=transform)
n_classes = len(dataset.classes)
print(f"✓ Loaded {len(dataset)} images from {n_classes} classes")
print()

# Create stratified splits (train/val/test: 70%/15%/15%)
print("Creating stratified splits...")
train_indices = []
val_indices = []
test_indices = []

# Group indices by class
class_indices = {i: [] for i in range(n_classes)}
for idx in range(len(dataset)):
    label = dataset.targets[idx]
    class_indices[label].append(idx)

# Split each class
for label in range(n_classes):
    indices = class_indices[label]
    n_samples = len(indices)
    
    if samples_per_class is not None:
        # Use subset
        n_samples = min(n_samples, samples_per_class)
        indices = indices[:n_samples]
    
    # Split: 70% train, 15% val, 15% test
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)
    
    train_indices.extend(indices[:n_train])
    val_indices.extend(indices[n_train:n_train + n_val])
    test_indices.extend(indices[n_train + n_val:])

print(f"✓ Created splits:")
print(f"  Train: {len(train_indices)} samples")
print(f"  Val:   {len(val_indices)} samples")
print(f"  Test:  {len(test_indices)} samples")
print(f"  Test:  {len(test_indices)} samples")
print()

# Create data loaders
train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=False)
val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False)

# Function to extract features
def extract_features(loader, split_name):
    """Extract features from a data loader"""
    all_features = []
    all_labels = []
    
    print(f"Extracting {split_name} features...")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"{split_name}"):
            images = images.to(device)
            features = model(images)
            features = features.squeeze(-1).squeeze(-1)  # Remove spatial dims
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    print(f"✓ Extracted {split_name}: {features.shape}")
    return features, labels

# Extract features for all splits
print("=" * 80)
print("  EXTRACTING FEATURES")
print("=" * 80)
print()

train_features, train_labels = extract_features(train_loader, "train")
val_features, val_labels = extract_features(val_loader, "val")
test_features, test_labels = extract_features(test_loader, "test")

print()

# Save features
print("=" * 80)
print("  SAVING FEATURES")
print("=" * 80)
print()

# Save as PyTorch tensors
torch.save({
    'features': train_features,
    'labels': train_labels,
    'n_classes': n_classes,
    'feature_dim': train_features.shape[1]
}, 'features/CNN_resnet18_train.pt')
print("✓ Saved: features/CNN_resnet18_train.pt")

torch.save({
    'features': val_features,
    'labels': val_labels,
    'n_classes': n_classes,
    'feature_dim': val_features.shape[1]
}, 'features/CNN_resnet18_val.pt')
print("✓ Saved: features/CNN_resnet18_val.pt")

torch.save({
    'features': test_features,
    'labels': test_labels,
    'n_classes': n_classes,
    'feature_dim': test_features.shape[1]
}, 'features/CNN_resnet18_test.pt')
print("✓ Saved: features/CNN_resnet18_test.pt")

print()

# Compute and save statistics
print("=" * 80)
print("  COMPUTING STATISTICS")
print("=" * 80)
print()

stats = {
    'extractor': 'CNN_resnet18',
    'feature_dim': int(train_features.shape[1]),
    'n_samples': {
        'train': int(len(train_features)),
        'val': int(len(val_features)),
        'test': int(len(test_features))
    },
    'n_classes': n_classes,
    'samples_per_class': samples_per_class,
    'statistics': {
        'mean': float(train_features.mean()),
        'std': float(train_features.std()),
        'min': float(train_features.min()),
        'max': float(train_features.max()),
        'sparsity': float((torch.abs(train_features) < 0.01).float().mean())
    },
    'per_feature_stats': {
        'mean': train_features.mean(dim=0).tolist(),
        'std': train_features.std(dim=0).tolist()
    },
    'timestamp': datetime.now().isoformat(),
    'device': device
}

with open('experiments_stage1/CNN_resnet18_stats.json', 'w') as f:
    json.dump(stats, f, indent=4)

print("✓ Saved: experiments_stage1/CNN_resnet18_stats.json")
print()

# Print summary
print("=" * 80)
print("  ✅ FEATURE EXTRACTION COMPLETE!")
print("=" * 80)
print()
print("Saved Features:")
print("  📁 features/CNN_resnet18_train.pt")
print(f"     Shape: {train_features.shape} (samples × features)")
print("  📁 features/CNN_resnet18_val.pt")
print(f"     Shape: {val_features.shape} (samples × features)")
print("  📁 features/CNN_resnet18_test.pt")
print(f"     Shape: {test_features.shape} (samples × features)")
print()
print("Statistics:")
print(f"  Feature Dimension: {stats['feature_dim']}")
print(f"  Total Samples: {stats['n_samples']['train'] + stats['n_samples']['val'] + stats['n_samples']['test']}")
print(f"  Classes: {stats['n_classes']}")
print(f"  Mean: {stats['statistics']['mean']:.4f}")
print(f"  Std: {stats['statistics']['std']:.4f}")
print(f"  Sparsity: {stats['statistics']['sparsity']:.2%}")
print()
print("=" * 80)
print()
print("✅ Features ready for training!")
print()
print("Next steps:")
print("  1. These features can be used to train DNN classifiers")
print("  2. Run Stage 2 to train classifiers on these features")
print("  3. Or use your own training script with the extracted features")
print()
print("=" * 80)
