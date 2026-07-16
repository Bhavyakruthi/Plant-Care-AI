"""
Grad-CAM Visualization for CNN+QNN Image Model
==============================================

Generates Class Activation Maps to visualize which image regions
the model focuses on when making predictions.

What it shows: Heatmap overlay showing important regions for each class prediction
Why it matters: Validates that model looks at disease symptoms, not background

Author: NLP Project Team
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path

# Add backend path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from models.image import ImageModel

class GradCAM:
    """Generate Grad-CAM heatmaps for CNN models"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate CAM for specific class"""
        # Forward pass
        model_output = self.model.classifier(
            self.model.feature_extractor(input_tensor).flatten(1)
        )
        
        if class_idx is None:
            class_idx = model_output.argmax(dim=1)
        
        # Backward pass
        self.model.feature_extractor.zero_grad()
        self.model.classifier.zero_grad()
        
        target = model_output[0, class_idx]
        target.backward()
        
        # Generate CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, class_idx.item()

def apply_colormap_on_image(org_img, activation, colormap_name='jet'):
    """Overlay heatmap on original image"""
    # Resize activation to match image
    heatmap = cv2.applyColorMap(np.uint8(255 * activation), getattr(cv2, f'COLORMAP_{colormap_name.upper()}'))
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlaid = cv2.addWeighted(org_img, 0.6, heatmap, 0.4, 0)
    
    return heatmap, overlaid

def visualize_gradcam_samples(model, df, dataset_root, output_dir, n_samples=20):
    """Generate Grad-CAM visualizations for sample images"""
    print(f"Generating Grad-CAM visualizations for {n_samples} samples...")
    
    # Get the last conv layer from ResNet feature extractor
    # ResNet18: layer4 is the final conv block
    target_layer = model.feature_extractor[-2]  # layer4
    
    gradcam = GradCAM(model, target_layer)
    
    # Create output directory
    gradcam_dir = os.path.join(output_dir, 'gradcam_examples')
    os.makedirs(gradcam_dir, exist_ok=True)
    
    # Sample images from each class
    classes = df['LABEL'].unique()
    samples_per_class = max(1, n_samples // len(classes))
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    sample_idx = 0
    
    for class_name in classes:
        class_df = df[df['LABEL'] == class_name].sample(min(samples_per_class, len(df[df['LABEL'] == class_name])))
        
        for _, row in class_df.iterrows():
            if sample_idx >= n_samples:
                break
                
            image_path = os.path.join(dataset_root, 'PlantVillage', row['LABEL'], row['FILENAME'])
            
            try:
                # Load and preprocess image
                pil_img = Image.open(image_path).convert('RGB')
                
                # Apply segmentation like in backend service
                from services.explainability import XAIService
                segmented_pil, leaf_mask = XAIService._segment_leaf(pil_img.resize((224, 224)))
                
                img_tensor = model.transform(segmented_pil).unsqueeze(0).to(model.device)
                
                # Get prediction
                with torch.no_grad():
                    probs = model.predict(image_path)[0]
                    pred_idx = np.argmax(probs)
                    pred_label = list(df['LABEL'].unique())[pred_idx]
                    confidence = probs[pred_idx] * 100
                
                # Generate Grad-CAM
                cam, _ = gradcam.generate_cam(img_tensor, class_idx=torch.tensor([pred_idx]).to(model.device))
                
                # Prepare images for visualization
                org_img = np.array(pil_img.resize((224, 224)))
                # Overlay on segmented image for cleaner view
                seg_np = cv2.cvtColor(np.array(segmented_pil), cv2.COLOR_RGB2BGR)
                heatmap, overlaid = apply_colormap_on_image(seg_np, cam)
                # Convert back to RGB for matplotlib
                overlaid = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # Plot
                axes[sample_idx, 0].imshow(org_img)
                axes[sample_idx, 0].set_title(f'Original\nTrue: {row["LABEL"]}', fontsize=10)
                axes[sample_idx, 0].axis('off')
                
                axes[sample_idx, 1].imshow(heatmap)
                axes[sample_idx, 1].set_title('Grad-CAM Heatmap', fontsize=10)
                axes[sample_idx, 1].axis('off')
                
                axes[sample_idx, 2].imshow(overlaid)
                axes[sample_idx, 2].set_title('Overlay', fontsize=10)
                axes[sample_idx, 2].axis('off')
                
                # Confidence bar
                axes[sample_idx, 3].barh(['Pred'], [confidence], color='green' if row['LABEL'] == pred_label else 'red')
                axes[sample_idx, 3].set_xlim(0, 100)
                axes[sample_idx, 3].set_title(f'Predicted: {pred_label}\nConf: {confidence:.1f}%', fontsize=10)
                axes[sample_idx, 3].set_xlabel('Confidence (%)')
                
                sample_idx += 1
                
            except Exception as e:
                print(f"  Skipped {image_path}: {e}")
                continue
    
    # Remove empty subplots
    for idx in range(sample_idx, n_samples):
        for col in range(4):
            fig.delaxes(axes[idx, col])
    
    plt.tight_layout()
    plt.savefig(os.path.join(gradcam_dir, 'gradcam_visualization.png'), dpi=200, bbox_inches='tight')
    print(f"  ✅ Grad-CAM visualization saved to {gradcam_dir}/gradcam_visualization.png")
    plt.close()
    
    return gradcam_dir

def main():
    # Configuration
    DATA_PATH = 'data/cleaned_multimodal_plant_data.csv'
    DATASET_ROOT = 'dataset/Image Dataset'
    MODEL_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/models/image/cnn_qnn_best.pt'
    LABEL_MAPPING_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/models/image/label_mapping.pkl'
    OUTPUT_DIR = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/outputs/interpretability'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("GRAD-CAM VISUALIZATION - IMAGE MODEL")
    print("="*70)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Using device: {device}")
    
    print("\nLoading model...")
    model = ImageModel(model_path=MODEL_PATH, label_mapping_path=LABEL_MAPPING_PATH, device=device)
    model.feature_extractor.eval()
    model.classifier.eval()
    print("✅ Model loaded")
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Dataset loaded: {len(df)} samples")
    
    # Generate Grad-CAM visualizations
    visualize_gradcam_samples(model, df, DATASET_ROOT, OUTPUT_DIR, n_samples=15)
    
    print("\n" + "="*70)
    print("✅ GRAD-CAM VISUALIZATION COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()
