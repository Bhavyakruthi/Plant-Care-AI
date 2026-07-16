"""
Visualize Feature Maps from CNN Layers
======================================
Shows the activation maps from different convolutional layers for individual images
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("  CNN FEATURE MAP VISUALIZATION")
print("  (Convolutional Layer Activations)")
print("=" * 80)
print()

# Create output directory
os.makedirs('feature_maps', exist_ok=True)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print()

# Load ResNet18
print("Loading ResNet18 model...")
model = models.resnet18(pretrained=True)
model = model.to(device)
model.eval()
print("✓ ResNet18 loaded")
print()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load dataset
print("Loading PlantVillage dataset...")
dataset = ImageFolder('PlantVillage', transform=transform)
print(f"✓ Loaded {len(dataset)} images from {len(dataset.classes)} classes")
print()

# Select sample images (one from each class)
print("Selecting sample images (one per class)...")
class_samples = {}
for idx in range(len(dataset)):
    label = dataset.targets[idx]
    if label not in class_samples:
        class_samples[label] = idx
    if len(class_samples) == len(dataset.classes):
        break

sample_indices = list(class_samples.values())[:5]  # First 5 classes
print(f"✓ Selected {len(sample_indices)} sample images")
print()

# Define hook to capture intermediate layer outputs
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Register hooks on convolutional layers
print("Registering hooks on convolutional layers...")
layers_to_visualize = {
    'conv1': model.conv1,           # First conv layer (64 filters, 7x7)
    'layer1': model.layer1,         # First residual block
    'layer2': model.layer2,         # Second residual block
    'layer3': model.layer3,         # Third residual block
    'layer4': model.layer4,         # Fourth residual block
}

for name, layer in layers_to_visualize.items():
    layer.register_forward_hook(get_activation(name))
    print(f"  ✓ Registered hook on {name}")

print()

# Process each sample image
for sample_idx, img_idx in enumerate(sample_indices):
    img, label = dataset[img_idx]
    class_name = dataset.classes[label]
    
    print(f"[{sample_idx+1}/{len(sample_indices)}] Processing: {class_name}")
    
    # Get original image for display
    img_path = dataset.imgs[img_idx][0]
    original_img = Image.open(img_path).convert('RGB')
    original_img = original_img.resize((224, 224))
    
    # Forward pass
    with torch.no_grad():
        img_batch = img.unsqueeze(0).to(device)
        _ = model(img_batch)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Original image
    plt.subplot(4, 6, 1)
    plt.imshow(original_img)
    plt.title(f'Original Image\n{class_name}', fontweight='bold', fontsize=10)
    plt.axis('off')
    
    # Visualize feature maps from each layer
    layer_names = list(activation.keys())
    
    for layer_idx, layer_name in enumerate(layer_names):
        feature_maps = activation[layer_name].squeeze(0).cpu()
        
        # Show first 5 channels from each layer
        n_channels_to_show = min(5, feature_maps.shape[0])
        
        for ch in range(n_channels_to_show):
            subplot_idx = (layer_idx * 6) + ch + 2
            if subplot_idx <= 24:  # Max 24 subplots (4x6 grid)
                plt.subplot(4, 6, subplot_idx)
                
                fmap = feature_maps[ch].numpy()
                plt.imshow(fmap, cmap='viridis')
                plt.title(f'{layer_name}\nChannel {ch}', fontsize=8)
                plt.axis('off')
    
    plt.suptitle(f'Feature Maps: {class_name}', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    safe_class_name = class_name.replace('/', '_').replace(' ', '_')
    output_path = f'feature_maps/{sample_idx+1}_{safe_class_name}_feature_maps.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()
    
    # Create detailed layer-by-layer visualization
    fig, axes = plt.subplots(len(layer_names), 8, figsize=(20, 3*len(layer_names)))
    
    for layer_idx, layer_name in enumerate(layer_names):
        feature_maps = activation[layer_name].squeeze(0).cpu()
        n_channels = feature_maps.shape[0]
        
        # Show first 8 channels
        for ch in range(min(8, n_channels)):
            ax = axes[layer_idx, ch] if len(layer_names) > 1 else axes[ch]
            
            fmap = feature_maps[ch].numpy()
            im = ax.imshow(fmap, cmap='viridis')
            ax.axis('off')
            
            if ch == 0:
                ax.text(-0.1, 0.5, f'{layer_name}\n({n_channels} ch)',
                       transform=ax.transAxes, fontsize=10, fontweight='bold',
                       verticalalignment='center', rotation=90)
            
            ax.set_title(f'Ch {ch}', fontsize=8)
    
    plt.suptitle(f'Detailed Feature Maps: {class_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = f'feature_maps/{sample_idx+1}_{safe_class_name}_detailed.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()
    
    print()

print("=" * 80)
print("  ✅ FEATURE MAP VISUALIZATION COMPLETE!")
print("=" * 80)
print()
print("Created visualizations:")
print(f"  Location: feature_maps/")
print(f"  Files: {len(sample_indices) * 2} images")
print()
print("File types:")
print("  *_feature_maps.png  - Overview of all layers")
print("  *_detailed.png      - Detailed layer-by-layer view")
print()
print("=" * 80)

# Now create a comparison visualization showing how features evolve
print()
print("Creating feature evolution comparison...")
print()

# Take first sample
img, label = dataset[sample_indices[0]]
class_name = dataset.classes[label]
img_path = dataset.imgs[sample_indices[0]][0]
original_img = Image.open(img_path).convert('RGB').resize((224, 224))

# Forward pass
with torch.no_grad():
    img_batch = img.unsqueeze(0).to(device)
    _ = model(img_batch)

# Create evolution visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Original
axes[0, 0].imshow(original_img)
axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=12)
axes[0, 0].axis('off')

# Feature maps from different layers
layer_order = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

for (layer_name, pos) in zip(layer_order, positions):
    feature_maps = activation[layer_name].squeeze(0).cpu()
    
    # Average across channels to show overall activation
    avg_fmap = feature_maps.mean(dim=0).numpy()
    
    ax = axes[pos]
    im = ax.imshow(avg_fmap, cmap='hot')
    ax.set_title(f'{layer_name}\n({feature_maps.shape[0]} channels, {avg_fmap.shape})',
                fontweight='bold', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle(f'Feature Evolution Through CNN Layers\n{class_name}',
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_maps/feature_evolution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: feature_maps/feature_evolution.png")
plt.close()

print()
print("=" * 80)
print()
print("Open the 'feature_maps' folder to view:")
print("  1. Individual feature maps for each sample")
print("  2. Detailed layer-by-layer visualizations")
print("  3. Feature evolution comparison")
print()
print("=" * 80)
