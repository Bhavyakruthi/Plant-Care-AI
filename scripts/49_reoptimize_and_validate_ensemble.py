"""
Final Ensemble Validation & Logic Verification
===============================================

Verifies that the new "Confidence-Aware Dynamic Fusion" outperforms 
the Image-Only baseline (96.48%) and the legacy Linear Ensemble (96.20%).

Author: NLP Project Team
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.getcwd())

from backend.models.text import TextModel
from backend.models.image import ImageModel
from backend.services.inference import EnsembleInference

def main():
    # 1. Config
    CLEANED_DATA_PATH = 'data/cleaned_multimodal_plant_data.csv'
    DATASET_ROOT = 'dataset/Image Dataset'
    TEXT_MODEL_PATH = 'models/text/overhaul/best_biobert_overhaul.pth'
    TEXT_METADATA_PATH = 'models/text/overhaul/overhaul_metadata.pkl'
    IMAGE_MODEL_PATH = 'models/image/cnn_qnn_best.pt'
    IMAGE_MAP_PATH = 'models/image/label_mapping.pkl'
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Models
    print(f"🚀 Initializing Validation on {DEVICE}...")
    text_model = TextModel(model_path=TEXT_MODEL_PATH, data_path=TEXT_METADATA_PATH, device=DEVICE)
    image_model = ImageModel(model_path=IMAGE_MODEL_PATH, label_mapping_path=IMAGE_MAP_PATH, device=DEVICE)
    ensemble_dynamic = EnsembleInference(alpha=0.65)
    
    # 3. Load Validation Set (10% slice)
    df = pd.read_csv(CLEANED_DATA_PATH)
    _, df_val = train_test_split(df, test_size=0.1, random_state=42, stratify=df['LABEL'])
    
    # 4. Run Inference & Statistics
    results = []
    skipped = 0
    
    print("\nStarting inference loop...")
    for idx, row in tqdm(df_val.iterrows(), total=len(df_val), desc="Verifying Fusion"):
        # Correct Path Construction
        img_path = os.path.normpath(os.path.join(DATASET_ROOT, 'PlantVillage', row['LABEL'], row['FILENAME']))
        text = row['cleaned_text']
        true_label = row['LABEL']
        
        if not os.path.exists(img_path):
            if idx < 5: print(f"  ❌ Missing file: {img_path}")
            skipped += 1
            continue
            
        try:
            # Model Probs
            img_probs = image_model.predict(img_path)[0]
            txt_probs = text_model.predict(text)[0]
            
            # Strategies
            # a) Image Only
            img_pred_idx = np.argmax(img_probs)
            # Use text_model's label_encoder as it's the source of truth for labels
            img_pred = text_model.label_encoder.classes_[img_pred_idx]
            
            # b) Legacy Linear (alpha=0.65)
            legacy_probs = (0.65 * np.array(img_probs)) + (0.35 * np.array(txt_probs))
            legacy_pred = text_model.label_encoder.classes_[np.argmax(legacy_probs)]
            
            # c) New Dynamic Logic
            dynamic_res = ensemble_dynamic.fuse(img_probs, txt_probs)
            dynamic_pred = text_model.label_encoder.classes_[dynamic_res["predicted_idx"]]
            
            results.append({
                "true": true_label,
                "img_only": img_pred,
                "legacy_linear": legacy_pred,
                "dynamic_fusion": dynamic_pred,
                "strategy": dynamic_res["fusion_strategy"]
            })
        except Exception as e:
            if skipped < 5: print(f"  ❌ Error on sample {idx}: {str(e)}")
            skipped += 1
            continue

    if not results:
        print(f"\nFATAL: No results generated. All {len(df_val)} samples skipped. (Skipped: {skipped})")
        return

    # 5. Report Accuracies
    res_df = pd.DataFrame(results)
    
    acc_img = (res_df['img_only'] == res_df['true']).mean() * 100
    acc_legacy = (res_df['legacy_linear'] == res_df['true']).mean() * 100
    acc_dynamic = (res_df['dynamic_fusion'] == res_df['true']).mean() * 100
    
    print("\n" + "="*50)
    print("🎯 FINAL FUSION ACCURACY COMPARISON")
    print("="*50)
    print(f"Image-Only Baseline:   {acc_img:.2f}%")
    print(f"Legacy Linear Fusion:  {acc_legacy:.2f}%")
    print(f"NEW DYNAMIC FUSION:    {acc_dynamic:.2f}%  <-- 🚀")
    print("="*50)
    
    # Analyze Strategy usage
    strategy_counts = res_df['strategy'].value_counts()
    print("\nFusion Strategy Utilization:")
    for strat, count in strategy_counts.items():
        print(f"- {strat}: {count} samples ({count/len(res_df):.1%})")
        
    # Check if Dynamic actually fixed any Image failures
    rescued = res_df[(res_df['img_only'] != res_df['true']) & (res_df['dynamic_fusion'] == res_df['true'])]
    print(f"\nSamples 'Rescued' by Text Model: {len(rescued)}")
    
    if acc_dynamic >= acc_img:
        print("\n✅ SUCCESS: Multimodal system is now constructive (>= Image-Only)!")
    else:
        print("\n⚠️ WARNING: Multimodal system still regressing. Refine logic further.")

if __name__ == "__main__":
    main()
