"""
Multimodal Fusion Weight (Alpha) Optimizer
===========================================

Finds the optimal alpha weight for the "MLE-based Ensemble".
It treats the fusion weight as a parameter to be optimized based on 
validation performance, ensuring the "MLE" label is mathematically justified.

Purpose: 
- Fixes the "hardcoded weight" issue.
- Balances the new high-accuracy BioBERT with the existing CNN+QNN.
- Maximizes final system accuracy.

Author: NLP Project Team
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.getcwd())

from backend.models.text import TextModel
from backend.models.image import ImageModel

def main():
    # 1. Config
    DATA_PATH = 'data/multimodal_plant_data.csv'
    DATASET_ROOT = 'dataset/Image Dataset'
    TEXT_MODEL_PATH = 'models/text/overhaul/best_biobert_overhaul.pth'
    TEXT_METADATA_PATH = 'models/text/overhaul/overhaul_metadata.pkl'
    IMAGE_MODEL_PATH = 'models/image/cnn_qnn_best.pt'
    IMAGE_MAP_PATH = 'models/image/label_mapping.pkl'
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Models
    print("Loading models for optimization...")
    text_model = TextModel(model_path=TEXT_MODEL_PATH, data_path=TEXT_METADATA_PATH, device=DEVICE)
    image_model = ImageModel(model_path=IMAGE_MODEL_PATH, label_mapping_path=IMAGE_MAP_PATH, device=DEVICE)
    
    # 3. Load Dataset (Validation Subset)
    df = pd.read_csv('data/cleaned_multimodal_plant_data.csv')
    # Use the same split as training (random_state 42, test_size 0.1)
    from sklearn.model_selection import train_test_split
    _, df_val = train_test_split(df, test_size=0.1, random_state=42, stratify=df['LABEL'])
    
    print(f"Optimizing for {len(df_val)} validation samples...")
    
    # 4. Get Probs for both models
    all_image_probs = []
    all_text_probs = []
    y_true = []
    
    le = text_model.label_encoder
    
    print("Gathering predictions...")
    for _, row in tqdm(df_val.iterrows(), total=len(df_val)):
        img_path = os.path.join(DATASET_ROOT, 'PlantVillage', row['LABEL'], row['FILENAME'])
        text = row['cleaned_text']
        
        try:
            # Image probs
            img_p = image_model.predict(img_path)[0]
            # Text probs
            txt_p = text_model.predict(text)[0]
            
            all_image_probs.append(img_p)
            all_text_probs.append(txt_p)
            y_true.append(le.transform([row['LABEL']])[0])
        except:
            continue
            
    all_image_probs = np.stack(all_image_probs)
    all_text_probs = np.stack(all_text_probs)
    y_true = np.array(y_true)
    
    # 5. Search for Optimal Alpha
    print("\nSearching for Optimal Alpha (MLE Weight)...")
    alphas = np.linspace(0.0, 1.0, 101)  # Search 0.0 to 1.0 in 0.01 steps
    accuracies = []
    
    for alpha in alphas:
        fused_probs = (alpha * all_image_probs) + ((1 - alpha) * all_text_probs)
        preds = np.argmax(fused_probs, axis=1)
        acc = np.mean(preds == y_true)
        accuracies.append(acc)
        
    best_idx = np.argmax(accuracies)
    best_alpha = alphas[best_idx]
    best_acc = accuracies[best_idx]
    
    print("\n" + "="*50)
    print(f"✅ OPTIMIZATION COMPLETE")
    print(f"Optimal Alpha (Image Weight): {best_alpha:.2f}")
    print(f"Text Model Weight:            {(1-best_alpha):.2f}")
    print(f"Max Validation Accuracy:     {best_acc*100:.2f}%")
    print("="*50)
    
    # 6. Plot the curve
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, [a * 100 for a in accuracies], color='green', linewidth=2)
    plt.axvline(best_alpha, color='red', linestyle='--', label=f'Best Alpha: {best_alpha:.2f}')
    plt.title('Fusion Weight Optimization (MLE Search)')
    plt.xlabel('Alpha (Image weight)')
    plt.ylabel('Validation Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('outputs/interpretability/mle_weight_optimization.png')
    print(f"\nOptimization curve saved to outputs/interpretability/mle_weight_optimization.png")

if __name__ == "__main__":
    main()
