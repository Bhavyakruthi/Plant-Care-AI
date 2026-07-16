"""
Feature Space Visualization (t-SNE/UMAP)
========================================

Visualizes how the model separates different disease classes in the learned feature space.
Shows whether classes form distinct clusters or overlap.

What it shows: 2D projection of high-dimensional learned features
Why it matters: Demonstrates model's ability to create linearly separable representations

Author: NLP Project Team
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.getcwd())
from backend.models.image import ImageModel
from backend.models.text import TextModel

def extract_image_features(model, df, dataset_root, max_samples=1000):
    """Extract features from penultimate layer of image model"""
    print(f"Extracting image features from {min(max_samples, len(df))} samples...")
    
    features = []
    labels = []
    
    # Sample uniformly from all classes
    sampled_dfs = []
    unique_labels = df['LABEL'].unique()
    samples_per_class = max_samples // len(unique_labels)
    
    for label in unique_labels:
        class_df = df[df['LABEL'] == label]
        sampled_dfs.append(class_df.sample(min(samples_per_class, len(class_df)), random_state=42))
        
    sampled_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    model.feature_extractor.eval()
    
    with torch.no_grad():
        for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Extracting features"):
            image_path = os.path.join(dataset_root, 'PlantVillage', row['LABEL'], row['FILENAME'])
            
            try:
                # Load image
                from PIL import Image
                pil_img = Image.open(image_path).convert('RGB')
                img_tensor = model.transform(pil_img).unsqueeze(0).to(model.device)
                
                # Extract features (before classifier)
                feature_vec = model.feature_extractor(img_tensor).flatten(1)
                
                features.append(feature_vec.cpu().numpy()[0])
                labels.append(row['LABEL'])
                
            except Exception as e:
                continue
    
    return np.array(features), np.array(labels)

def extract_text_features(model, df, max_samples=1000):
    """Extract features from BERT pooler output"""
    print(f"Extractingtext features from {min(max_samples, len(df))} samples...")
    
    features = []
    labels = []
    
    # Sample uniformly from all classes
    sampled_dfs = []
    unique_labels = df['LABEL'].unique()
    samples_per_class = max_samples // len(unique_labels)
    
    for label in unique_labels:
        class_df = df[df['LABEL'] == label]
        sampled_dfs.append(class_df.sample(min(samples_per_class, len(class_df)), random_state=42))
        
    sampled_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    model.model.eval()
    
    with torch.no_grad():
        for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Extracting features"):
            # Use cleaned text if available
            text = row['cleaned_text'] if 'cleaned_text' in row else str(row['DESCRIPTION'])
            
            try:
                # Get structured features
                from backend.utils.feature_engineering import extract_structured_features
                temp_df = pd.DataFrame([row])
                struct_feats = extract_structured_features(temp_df)
                struct_tensor = torch.tensor(struct_feats, dtype=torch.float).to(model.device)
                
                # Tokenize
                inputs = model.tokenizer(text, return_tensors='pt', truncation=True, 
                                       padding='max_length', max_length=128).to(model.device)
                
                # Extract Hybrid features
                bert_output = model.model.bert(inputs['input_ids'], inputs['attention_mask'])
                pooled_output = bert_output.pooler_output
                
                # Combine with structured features as the model does
                combined_vec = torch.cat((pooled_output, struct_tensor), dim=1)
                
                features.append(combined_vec.cpu().numpy()[0])
                labels.append(row['LABEL'])
                
            except Exception as e:
                print(f"Error extracting features: {e}")
                continue
    
    return np.array(features), np.array(labels)

def visualize_tsne(features, labels, title, output_path):
    """Create t-SNE visualization"""
    print(f"Computing t-SNE projection...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    features_2d = tsne.fit_transform(features)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get unique classes
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                  c=[colors[idx]], label=label, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved t-SNE visualization to {output_path}")

def main():
    # Configuration
    DATA_PATH = 'data/cleaned_multimodal_plant_data.csv'
    DATASET_ROOT = 'dataset/Image Dataset'
    IMAGE_MODEL_PATH = 'models/image/cnn_qnn_best.pt'
    LABEL_MAPPING_PATH = 'models/image/label_mapping.pkl'
    TEXT_MODEL_PATH = 'models/text/overhaul/best_biobert_overhaul.pth'
    PREPROCESSED_DATA_PATH = 'models/text/overhaul/overhaul_metadata.pkl'
    OUTPUT_DIR = 'outputs/interpretability'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("FEATURE SPACE VISUALIZATION (t-SNE)")
    print("="*70)
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Dataset loaded: {len(df)} samples")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Using device: {device}")
    
    # Create output directory
    feature_space_dir = os.path.join(OUTPUT_DIR, 'feature_spaces')
    os.makedirs(feature_space_dir, exist_ok=True)
    
    # === IMAGE MODEL ===
    print("\n" + "-"*70)
    print("IMAGE MODEL FEATURE SPACE")
    print("-"*70)
    
    print("\nLoading image model...")
    image_model = ImageModel(model_path=IMAGE_MODEL_PATH, label_mapping_path=LABEL_MAPPING_PATH, device=device)
    print("✅ Model loaded")
    
    image_features, image_labels = extract_image_features(image_model, df, DATASET_ROOT, max_samples=1000)
    print(f"✅ Extracted {len(image_features)} image features")
    
    visualize_tsne(
        image_features, image_labels,
        "Image Model Feature Space (CNN+QNN) - t-SNE",
        os.path.join(feature_space_dir, 'image_tsne.png')
    )
    
    # === TEXT MODEL ===
    print("\n" + "-"*70)
    print("TEXT MODEL FEATURE SPACE")
    print("-"*70)
    
    print("\nLoading text model...")
    text_model = TextModel(model_path=TEXT_MODEL_PATH, data_path=PREPROCESSED_DATA_PATH, device=device)
    print("✅ Model loaded")
    
    text_features, text_labels = extract_text_features(text_model, df, max_samples=1000)
    print(f"✅ Extracted {len(text_features)} text features")
    
    visualize_tsne(
        text_features, text_labels,
        "Text Model Feature Space (BioBERT) - t-SNE",
        os.path.join(feature_space_dir, 'text_tsne.png')
    )
    
    print(f"\n✅ All visualizations saved to: {feature_space_dir}/")
    
    print("\n" + "="*70)
    print("✅ FEATURE SPACE VISUALIZATION COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()
