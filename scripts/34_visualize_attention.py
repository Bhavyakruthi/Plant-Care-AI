"""
BioBERT Attention Heatmap Visualization
=======================================

Generates visual heatmaps of word-level attention for disease descriptions.
This script demonstrates how the overhauled BioBERT model focuses on diagnostic keywords.

Author: NLP Project Team
"""

import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from backend.models.text import TextModel
from backend.services.explainability import XAIService

def plot_attention(token_weights, title, output_path):
    tokens = [tw[0] for tw in token_weights]
    weights = [tw[1] for tw in token_weights]
    
    # Filter special tokens for cleaner visualization (keep [CLS] but maybe skip others)
    # Actually, keep it simple first
    
    plt.figure(figsize=(12, 2))
    sns.heatmap([weights], annot=[tokens], fmt="", cmap='YlGnBu', cbar=False)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # 1. Config
    TEXT_MODEL_PATH = 'models/text/overhaul/best_biobert_overhaul.pth'
    METADATA_PATH = 'models/text/overhaul/overhaul_metadata.pkl'
    OUTPUT_DIR = 'outputs/interpretability/attention'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Model
    print("Loading Overhauled BioBERT Model...")
    text_model = TextModel(model_path=TEXT_MODEL_PATH, data_path=METADATA_PATH, device=DEVICE)
    
    # 3. Sample Descriptions (Scientific Samples)
    samples = [
        "The tomato leaf shows severe late blight with large dark brown spots and fuzzy white growth on the underside.",
        "Corn leaf affected by Common Rust showing yellow-orange elongated pustules on both surfaces.",
        "Grape leaf with Black Rot symptoms: small brown circular lesions with dark borders."
    ]
    
    # 4. Generate Visualizations
    print("Generating attention heatmaps...")
    for i, text in enumerate(samples):
        print(f"  Processing sample {i+1}...")
        result = XAIService.generate_attention_map(text_model, text)
        
        safe_title = f"Attention: {text[:40]}..."
        plot_attention(result['token_weights'], safe_title, os.path.join(OUTPUT_DIR, f'attention_sample_{i+1}.png'))
        
    print(f"\n✅ Attention heatmaps saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
