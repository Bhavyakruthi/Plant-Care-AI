"""
CNN Filter/Kernel Visualization
================================

Visualizes the learned convolutional filters to understand what patterns
the CNN layers detect (edges, textures, shapes, disease-specific features).

What it shows: Visual patterns each filter learned to detect
Why it matters: Shows progression from low-level (edges) to high-level (symptoms) features

Author: NLP Project Team
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add backend path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from models.image import ImageModel

def visualize_conv_filters(model, layer_name, output_path, max_filters=64):
    """Visualize convolutional filters from a specific layer"""
    # Get the layer
    layer = None
    for name, module in model.feature_extractor.named_modules():
        if isinstance(module, torch.nn.Conv2d) and layer_name in name:
            layer = module
            break
    
    if layer is None:
        print(f"  ⚠️  Layer {layer_name} not found")
        return
    
    # Get filter weights
    weights = layer.weight.data.cpu().numpy()  # Shape: (out_channels, in_channels, height, width)
    num_filters = min(weights.shape[0], max_filters)
    
    # Normalize weights for visualization
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx in range(num_filters):
        # For RGB input (first layer), show as RGB
        if weights_norm.shape[1] == 3:
            # Transpose to (height, width, channels)
            filter_img = np.transpose(weights_norm[idx], (1, 2, 0))
        else:
            # For other layers, show first channel
            filter_img = weights_norm[idx, 0]
        
        axes[idx].imshow(filter_img, cmap='viridis' if weights_norm.shape[1] != 3 else None)
        axes[idx].set_title(f'Filter {idx}', fontsize=8)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_filters, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Convolutional Filters - {layer_name}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved {layer_name} filters to {output_path}")

def visualize_filter_statistics(model, output_dir):
    """Visualize filter weight distributions across layers"""
    conv_layers = []
    layer_names = []
    
    #Extract all conv layers
    for name, module in model.feature_extractor.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(module)
            layer_names.append(name)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Filter count per layer
    filter_counts = [layer.out_channels for layer in conv_layers]
    axes[0, 0].bar(range(len(filter_counts)), filter_counts, color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('Layer Index', fontweight='bold')
    axes[0, 0].set_ylabel('Number of Filters', fontweight='bold')
    axes[0, 0].set_title('Filter Count per Layer', fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Kernel sizes
    kernel_sizes = [layer.kernel_size[0] for layer in conv_layers]
    axes[0, 1].bar(range(len(kernel_sizes)), kernel_sizes, color='coral', alpha=0.7)
    axes[0, 1].set_xlabel('Layer Index', fontweight='bold')
    axes[0, 1].set_ylabel('Kernel Size', fontweight='bold')
    axes[0, 1].set_title('Kernel Size per Layer', fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Weight distribution (first layer)
    first_layer_weights = conv_layers[0].weight.data.cpu().numpy().flatten()
    axes[1, 0].hist(first_layer_weights, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Weight Value', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('First Layer Weight Distribution', fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Weight distribution (last layer)
    last_layer_weights = conv_layers[-1].weight.data.cpu().numpy().flatten()
    axes[1, 1].hist(last_layer_weights, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Weight Value', fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontweight='bold')
    axes[1, 1].set_title('Last Layer Weight Distribution', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    stats_path = os.path.join(output_dir, 'filter_statistics.png')
    plt.savefig(stats_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved filter statistics to {stats_path}")

def visualize_feature_maps(model, sample_image_path, output_dir, max_layers=10):
    """Pass an image through the model and visualize the activations (feature maps)"""
    device = model.device
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess image
    from PIL import Image
    try:
        pil_img = Image.open(sample_image_path).convert('RGB')
        img_tensor = model.transform(pil_img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"  ⚠️ Error loading sample image: {e}")
        return

    # To capture activations, we'll register hooks or just iterate
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    hooks = []
    # Register hooks for conv layers
    conv_count = 0
    for name, module in model.feature_extractor.named_modules():
        if isinstance(module, torch.nn.Conv2d) and conv_count < max_layers:
            hooks.append(module.register_forward_hook(get_activation(f"conv_{conv_count}_{name.replace('.', '_')}")))
            conv_count += 1

    # Forward pass
    with torch.no_grad():
        model.feature_extractor(img_tensor)

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"  Visualizing {len(activations)} layer activations...")
    
    for name, act in activations.items():
        # act shape: (1, channels, h, w)
        act = act.squeeze().cpu().numpy()
        num_maps = min(act.shape[0], 64)
        grid_size = int(np.ceil(np.sqrt(num_maps)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        # Normalize for display
        act_min, act_max = act.min(), act.max()
        act = (act - act_min) / (act_max - act_min + 1e-8)
        
        for i in range(num_maps):
            axes[i].imshow(act[i], cmap='magma')
            axes[i].axis('off')
        
        for i in range(num_maps, len(axes)):
            axes[i].axis('off')
            
        plt.suptitle(f'Feature Maps - {name}', fontsize=16)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'feature_map_{name}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    print(f"  ✅ Saved feature maps to {output_dir}")

def main():
    # Configuration
    DATA_PATH = 'data/cleaned_multimodal_plant_data.csv'
    MODEL_PATH = 'models/image/cnn_qnn_best.pt'
    LABEL_MAPPING_PATH = 'models/image/label_mapping.pkl'
    OUTPUT_DIR = 'outputs/interpretability'
    DATASET_ROOT = 'dataset/Image Dataset'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("CNN INTERPRETABILITY - FILTERS & FEATURE MAPS")
    print("="*70)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Using device: {device}")
    
    print("\nLoading model...")
    model = ImageModel(model_path=MODEL_PATH, label_mapping_path=LABEL_MAPPING_PATH, device=device)
    print("✅ Model loaded")
    
    # Create output directories
    cnn_filters_dir = os.path.join(OUTPUT_DIR, 'cnn_filters')
    feature_maps_dir = os.path.join(OUTPUT_DIR, 'feature_maps')
    os.makedirs(cnn_filters_dir, exist_ok=True)
    os.makedirs(feature_maps_dir, exist_ok=True)
    
    # 1. Visualize kernels (Filters)
    print("\n[1/3] Visualizing Convolutional Kernels...")
    layers_to_visualize = [
        ('0', 'conv1_first_layer'),
        ('4.0.conv1', 'layer1_block0_conv1'),
        ('4.1.conv1', 'layer1_block1_conv1'),
        ('5.0.conv1', 'layer2_block0_conv1'),
        ('6.0.conv1', 'layer3_block0_conv1'),
        ('7.0.conv1', 'layer4_block0_conv1'),
    ]
    
    for layer_name, output_name in layers_to_visualize:
        output_path = os.path.join(cnn_filters_dir, f'{output_name}.png')
        visualize_conv_filters(model, layer_name, output_path, max_filters=64)
    
    # 2. Visualize Feature Maps (Activations)
    print("\n[2/3] Visualizing Feature Maps (Activations)...")
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    # Pick a sample image (e.g., Tomato Bacterial Spot)
    sample_row = df[df['LABEL'].str.contains('Tomato', na=False)].iloc[0]
    sample_img_path = os.path.join(DATASET_ROOT, 'PlantVillage', sample_row['LABEL'], sample_row['FILENAME'])
    
    visualize_feature_maps(model, sample_img_path, feature_maps_dir)
    
    # 3. Visualize filter statistics
    print("\n[3/3] Generating weight statistics...")
    visualize_filter_statistics(model, cnn_filters_dir)
    
    print(f"\n✅ All CNN interpretability assets saved to: {OUTPUT_DIR}/")
    
    print("\n" + "="*70)
    print("✅ COMPREHENSIVE CNN INTERPRETABILITY COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()
