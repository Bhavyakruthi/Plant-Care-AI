"""
Quick CNN Feature Visualization (No Quantum)
============================================
Extract and visualize CNN features only for speed
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

print("=" * 80)
print("  CNN FEATURE EXTRACTION & VISUALIZATION")
print("=" * 80)
print()

# Create directories
os.makedirs('features', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print()

# Load ResNet18
print("Loading ResNet18...")
model = models.resnet18(pretrained=True)
# Remove final FC layer to get features
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()
print("✓ ResNet18 loaded (512-dimensional features)")
print()

# Data loading
print("Loading PlantVillage dataset...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder('PlantVillage', transform=transform)

# Use subset (10 samples per class = 150 total)
samples_per_class = 10
indices = []
class_counts = {}

for idx in range(len(dataset)):
    label = dataset.targets[idx]
    if label not in class_counts:
        class_counts[label] = 0
    
    if class_counts[label] < samples_per_class:
        indices.append(idx)
        class_counts[label] += 1

subset = Subset(dataset, indices)
loader = DataLoader(subset, batch_size=32, shuffle=False)

print(f"✓ Loaded {len(subset)} samples ({samples_per_class} per class)")
print(f"  Classes: {len(class_counts)}")
print()

# Extract features
print("Extracting CNN features...")
all_features = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(loader, desc="Extracting"):
        images = images.to(device)
        features = model(images)
        features = features.squeeze(-1).squeeze(-1)  # Remove spatial dims
        all_features.append(features.cpu())
        all_labels.append(labels)

features = torch.cat(all_features, dim=0).numpy()
labels = torch.cat(all_labels, dim=0).numpy()

print(f"✓ Extracted features: {features.shape}")
print()

# Save features
torch.save({'features': features, 'labels': labels}, 
           'features/CNN_features.pt')
print("✓ Saved to features/CNN_features.pt")
print()

# === VISUALIZATIONS ===
print("=" * 80)
print("  CREATING VISUALIZATIONS")
print("=" * 80)
print()

# 1. t-SNE
print("[1/5] Creating t-SNE plot...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
features_2d = tsne.fit_transform(features)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                     c=labels, cmap='tab20', alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Disease Class')
plt.title('CNN Features - t-SNE Visualization\\n(ResNet18 Features Projected to 2D)', 
          fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/CNN_tsne.png', dpi=150, bbox_inches='tight')
print("✓ Saved: visualizations/CNN_tsne.png")
plt.close()

# 2. PCA
print("[2/5] Creating PCA plot...")
pca = PCA(n_components=min(50, features.shape[1]))
features_pca = pca.fit_transform(features)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PCA scatter
scatter = axes[0].scatter(features_pca[:, 0], features_pca[:, 1],
                         c=labels, cmap='tab20', alpha=0.7, s=100, 
                         edgecolors='black', linewidth=0.5)
axes[0].set_title('PCA Projection (First 2 Components)', fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
axes[0].grid(True, alpha=0.3)

# Explained variance
cumsum = np.cumsum(pca.explained_variance_ratio_)
axes[1].plot(range(1, len(cumsum)+1), cumsum, marker='o', markersize=4, linewidth=2)
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold', linewidth=2)
axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
axes[1].set_xlabel('Number of Components', fontsize=11)
axes[1].set_ylabel('Cumulative Variance Explained', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

# Add text showing 95% components
n_components_95 = np.argmax(cumsum >= 0.95) + 1
axes[1].text(0.5, 0.1, f'{n_components_95} components explain 95% variance',
            transform=axes[1].transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('visualizations/CNN_pca.png', dpi=150, bbox_inches='tight')
print("✓ Saved: visualizations/CNN_pca.png")
plt.close()

# 3. Correlation heatmap
print("[3/5] Creating correlation heatmap...")
# Sample 50 features
sample_features = features[:, :50]
corr = np.corrcoef(sample_features.T)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', center=0, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap\\n(First 50 Features)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/CNN_correlation.png', dpi=150, bbox_inches='tight')
print("✓ Saved: visualizations/CNN_correlation.png")
plt.close()

# 4. Feature statistics
print("[4/5] Creating feature statistics plot...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Mean
axes[0, 0].hist(features.mean(axis=0), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_title('Distribution of Feature Means', fontweight='bold', fontsize=11)
axes[0, 0].set_xlabel('Mean Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# Std
axes[0, 1].hist(features.std(axis=0), bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_title('Distribution of Feature Std Dev', fontweight='bold', fontsize=11)
axes[0, 1].set_xlabel('Std Dev')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Sample feature
axes[1, 0].hist(features[:, 0], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[1, 0].set_title('Sample Feature Distribution (Feature 0)', fontweight='bold', fontsize=11)
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# Sparsity
sparsity = (np.abs(features) < 0.01).sum(axis=0) / features.shape[0]
axes[1, 1].hist(sparsity, bins=30, edgecolor='black', alpha=0.7, color='red')
axes[1, 1].set_title('Feature Sparsity', fontweight='bold', fontsize=11)
axes[1, 1].set_xlabel('Sparsity Ratio')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/CNN_statistics.png', dpi=150, bbox_inches='tight')
print("✓ Saved: visualizations/CNN_statistics.png")
plt.close()

# 5. Feature heatmap
print("[5/5] Creating feature heatmap...")
# Show all samples vs first 100 features
plt.figure(figsize=(12, 6))
im = plt.imshow(features[:, :100].T, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(im, label='Feature Value')
plt.title('Feature Heatmap\\n(All Samples × First 100 Features)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Feature Dimension', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/CNN_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Saved: visualizations/CNN_heatmap.png")
plt.close()

print()
print("=" * 80)
print("  ✅ VISUALIZATION COMPLETE!")
print("=" * 80)
print()
print("Created visualizations:")
print("  1. visualizations/CNN_tsne.png          - t-SNE 2D projection")
print("  2. visualizations/CNN_pca.png           - PCA analysis")
print("  3. visualizations/CNN_correlation.png   - Feature correlations")
print("  4. visualizations/CNN_statistics.png    - Statistical distributions")
print("  5. visualizations/CNN_heatmap.png       - Feature value heatmap")
print()
print(f"Feature Statistics:")
print(f"  Shape: {features.shape} (samples × features)")
print(f"  Mean: {features.mean():.4f}")
print(f"  Std: {features.std():.4f}")
print(f"  Min: {features.min():.4f}")
print(f"  Max: {features.max():.4f}")
print()
print(f"PCA Analysis:")
print(f"  {n_components_95} components explain 95% of variance")
print(f"  First 10 components: {cumsum[9]:.1%} variance")
print()
print("=" * 80)
print()
print("Open the visualizations folder to view the feature maps!")
print("=" * 80)
