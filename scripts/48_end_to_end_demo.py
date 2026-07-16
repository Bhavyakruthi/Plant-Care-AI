"""
End-to-End Multimodal Diagnostic Demo
======================================

This script demonstrates the full diagnostic pipeline:
1. Random Sample Selection from Dataset
2. Visual AI Description Generation (via Gemini, purely morphological)
3. Multimodal Fusion Prediction (Image + Text)
4. Explainable AI Visualization (Original | Prediction | Word Attention)

Author: NLP Project Team
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.getcwd())

# Load environment variables from backend/.env
env_path = os.path.join(os.getcwd(), 'backend', '.env')
load_dotenv(dotenv_path=env_path)

from backend.models.image import ImageModel
from backend.models.text import TextModel
from backend.services.inference import EnsembleInference
from backend.services.explainability import XAIService
from backend.services.gemini import GeminiService

def main():
    # 1. Config & Initialization
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing System Demo on {DEVICE}...")

    TEXT_MODEL_PATH = 'models/text/overhaul/best_biobert_overhaul.pth'
    TEXT_DATA_PATH = 'models/text/overhaul/overhaul_metadata.pkl'
    IMAGE_MODEL_PATH = 'models/image/cnn_qnn_best.pt'
    IMAGE_MAP_PATH = 'models/image/label_mapping.pkl'
    CLEANED_DATA_PATH = 'data/cleaned_multimodal_plant_data.csv'
    DATASET_ROOT = 'dataset/Image Dataset/PlantVillage'

    # Get API Key from environment (or placeholder)
    API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
    
    # Initialize Models
    text_model = TextModel(model_path=TEXT_MODEL_PATH, data_path=TEXT_DATA_PATH, device=DEVICE)
    image_model = ImageModel(model_path=IMAGE_MODEL_PATH, label_mapping_path=IMAGE_MAP_PATH, device=DEVICE)
    ensemble = EnsembleInference(alpha=0.65)
    gemini = GeminiService(api_key=API_KEY)

    # 2. Select Random Sample
    print("\nSelect random sample from high-quality dataset...")
    df = pd.read_csv(CLEANED_DATA_PATH)
    sample = df.sample(1).iloc[0]
    img_path = os.path.join(DATASET_ROOT, sample['LABEL'], sample['FILENAME'])
    true_label = sample['LABEL']

    print(f"  Selected: {sample['FILENAME']} (True Class: {true_label})")

    # 3. Generate Description (AI Vision)
    print("\nGenerating morphological description via Gemini AI...")
    if API_KEY == "YOUR_API_KEY_HERE":
        print("  ⚠️ No API Key found. Using fallback description for demo purposes.")
        description = sample['cleaned_text']
    else:
        description = gemini.generate_description(img_path)
    
    print("-" * 50)
    print(f"AL DESCRIPTION: {description}")
    print("-" * 50)

    # 4. Multimodal Inference
    print("\nRunning Multimodal Fusion Inference...")
    img_probs = image_model.predict(img_path)[0]
    txt_probs = text_model.predict(description)[0]
    
    fusion_result = ensemble.fuse(img_probs, txt_probs)
    pred_idx = fusion_result["predicted_idx"]
    pred_label = text_model.label_encoder.classes_[pred_idx]
    confidence = fusion_result["confidence"]
    is_correct = (pred_label == true_label)

    print(f"  Prediction: {pred_label} ({confidence:.1%})")
    print(f"  Status: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")

    # 5. Explainability (Attention)
    print("\nExtracting Word Attention Heatmap...")
    attention = XAIService.generate_attention_map(text_model, description)
    token_weights = attention['token_weights']

    # 6. Final Visualization Layout
    print("\nRendering final diagnostic dashboard...")
    fig = plt.figure(figsize=(16, 10))
    
    # Subplot A: Original Image
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.imshow(Image.open(img_path))
    ax1.set_title(f"Visual Diagnosis\n(True: {true_label})", fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Subplot B: Status & Confidence
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    colors = ['#2ecc71' if is_correct else '#e74c3c']
    ax2.barh(['Diagnosis Confidence'], [confidence * 100], color=colors)
    ax2.set_xlim(0, 100)
    ax2.set_title(f"Predicted Result: {pred_label}", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Confidence (%)")
    
    # Subplot C: Word Attention Map
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    tokens = [tw[0] for tw in token_weights]
    weights = [tw[1] for tw in token_weights]
    
    # We'll split the tokens into rows for better readability if too long
    # But for a demo, 128 max tokens might be messy, so we take the top N or wrap
    sns.heatmap([weights], annot=[tokens], fmt="", cmap='YlGnBu', cbar=False, ax=ax3)
    ax3.set_title("Scientific Logic (BioBERT Attention over Symptoms)", fontsize=12, fontweight='bold', pad=15)
    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.tight_layout()
    output_path = 'outputs/end_to_end_demo_result.png'
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"\n✅ DEMO COMPLETE! Result saved to: {output_path}")

if __name__ == "__main__":
    main()
