import torch
import numpy as np
import pickle
import os
import sys
import cv2
import base64
from PIL import Image
from tqdm import tqdm
import json
from dotenv import load_dotenv

BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"
GALLERY_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT\Journal_XAI_Gallery"
sys.path.append(os.path.join(BASE_DIR, "backend"))

# Load environment variables
load_dotenv(os.path.join(BASE_DIR, "backend", ".env"))

from models.text import TextModel
from models.image import ImageModel
from services.explainability import XAIService
from services.inference import EnsembleInference
from services.gemini import GeminiService

import pandas as pd

def main():
    print("GENERATING XAI GALLERY (HYBRID MODE)...")
    os.makedirs(GALLERY_DIR, exist_ok=True)
    
    # Load Gemini Service (as fallback)
    api_key = os.getenv("GEMINI_API_KEY")
    gemini_service = GeminiService(api_key=api_key) if api_key else None
    
    # Load pre-cleaned descriptions
    print("Loading cleaned descriptions...")
    csv_path = os.path.join(BASE_DIR, "data", "cleaned_multimodal_plant_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Create mapping from filename to cleaned_text
        desc_map = dict(zip(df['FILENAME'], df['cleaned_text']))
        print(f"  Loaded {len(desc_map)} cleaned descriptions from CSV.")
    else:
        # Fallback to old CSV if cleaned not found
        old_csv = os.path.join(BASE_DIR, "data", "processed_data_with_features.csv")
        df = pd.read_csv(old_csv)
        desc_map = dict(zip(df['FILENAME'], df['combined_text']))
        print(f"  ⚠️ Cleaned CSV not found, using old descriptions from {old_csv}")

    # Load Models
    print("Loading models...")
    # Update to overhaul path by default
    overhaul_model = os.path.join(BASE_DIR, "models", "text", "overhaul", "best_biobert_overhaul.pth")
    if os.path.exists(overhaul_model):
        model_path = overhaul_model
        # Use overhaul metadata
        data_path = os.path.join(BASE_DIR, "models", "text", "overhaul", "overhaul_metadata.pkl")
    else:
        model_path = os.path.join(BASE_DIR, "models", "text", "best_hybrid_bert_model.pth")
        data_path = os.path.join(BASE_DIR, "data", "preprocessed_data.pkl")
    
    image_map_path = os.path.join(BASE_DIR, "models", "image", "label_mapping.pkl")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    label_encoder = data['label_encoder']
    classes = label_encoder.classes_
    
    text_model = TextModel(model_path=model_path, data_path=data_path)
    image_model = ImageModel(model_path=os.path.join(BASE_DIR, "models", "image", "cnn_qnn_best.pt"), label_mapping_path=os.path.join(BASE_DIR, "models", "image", "label_mapping.pkl"))
    ensemble = EnsembleInference(alpha=0.65)
    
    # Organize samples by class
    class_indices = {label: [] for label in classes}
    for idx, row in df.iterrows():
        class_indices[row['LABEL']].append(idx)
            
    stats = []
    
    dataset_root = os.path.join(BASE_DIR, 'dataset', 'Image Dataset', 'PlantVillage')

    for cls_idx, label_name in enumerate(classes):
        print(f"\nProcessing class: {label_name}")
        cls_dir = os.path.join(GALLERY_DIR, label_name)
        os.makedirs(cls_dir, exist_ok=True)
        
        # Take up to 2 samples per class for a balanced overview
        samples_idx = class_indices[label_name][:2]
        
        for s_idx, test_idx in enumerate(samples_idx):
            row = df.iloc[test_idx]
            filename = row['FILENAME']
            img_path = os.path.normpath(os.path.join(dataset_root, label_name, filename))
            
            if not os.path.exists(img_path):
                continue
                
            true_label = label_name
            print(f"  Sample {s_idx+1}/2 - {filename}")
            
            # 1. Get Description (CSV -> Gemini -> Fallback)
            desc = desc_map.get(filename)
            
            if not desc:
                if gemini_service:
                    print(f"    Generating Gemini description...")
                    try:
                        desc = gemini_service.generate_description(img_path)
                    except Exception as e:
                        print(f"    ⚠️ API Exception: {e}")
                
            if not desc or "Error" in desc:
                print(f"    ⚠️ No description found, using generic fallback")
                desc = f"Technical leaf morphology showing symptoms characteristic of {label_name}."
            
            # 2. Inference
            text_probs = text_model.predict(desc)
            image_probs = image_model.predict(img_path)
            
            fusion = ensemble.fuse(image_probs, text_probs)
            pred_idx = fusion["predicted_idx"]
            pred_label = classes[pred_idx]
            is_correct = (pred_idx == cls_idx)
            confidence = fusion["confidence"]
            
            # 3. XAI - Grad-CAM
            gradcam_b64 = XAIService.generate_gradcam(image_model, img_path, target_idx=pred_idx)
            
            # 4. XAI - Text Attention
            attention_result = XAIService.generate_attention_map(text_model, desc)
            importance = attention_result['token_weights']
            
            # 5. Save visualization
            if gradcam_b64:
                gradcam_bytes = base64.b64decode(gradcam_b64)
                gradcam_img = cv2.imdecode(np.frombuffer(gradcam_bytes, np.uint8), cv2.IMREAD_COLOR)
                orig_img = cv2.imread(img_path)
                orig_img = cv2.resize(orig_img, (224, 224))
                
                h_concat = cv2.hconcat([orig_img, gradcam_img])
                status_text = f"Pred: {pred_label} ({confidence:.1%})"
                color = (0, 255, 0) if is_correct else (0, 0, 255)
                cv2.putText(h_concat, status_text, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(h_concat, f"True: {true_label}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                save_path = os.path.join(cls_dir, f"sample_{s_idx+1}_xai.png")
                cv2.imwrite(save_path, h_concat)
            
            # 6. Save metadata
            metadata = {
                "sample": s_idx + 1,
                "file": filename,
                "true_label": true_label,
                "pred_label": pred_label,
                "is_correct": is_correct,
                "confidence": float(confidence),
                "description": desc,
                "symptom_importance": importance
            }
            with open(os.path.join(cls_dir, f"sample_{s_idx+1}_meta.json"), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
            
            stats.append(is_correct)
            
    accuracy = sum(stats) / len(stats) if stats else 0
    print(f"\nHybrid XAI Gallery Generation Complete!")
    print(f"Accuracy on gallery samples: {accuracy*100:.2f}%")
    
    with open(os.path.join(GALLERY_DIR, "summary.md"), 'w', encoding='utf-8') as f:
        f.write("# Hybrid XAI Gallery Summary\n\n")
        f.write(f"**Total Samples:** {len(stats)}\n")
        f.write(f"**Gallery Accuracy:** {accuracy*100:.2f}%\n\n")
        f.write("Generated using a mix of CSV-stored technical descriptions and live API synthesis.\n")

if __name__ == "__main__":
    main()
