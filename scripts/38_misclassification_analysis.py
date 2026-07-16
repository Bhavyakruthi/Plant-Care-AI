"""
Detailed Misclassification and Confusion Analysis
================================================

Performs in-depth analysis of model failures to understand where the model
is confused and why. Generates enhanced confusion matrices and visual examples
of misclassified images and text.

What it shows: Normalized confusion matrices, per-class error profiles, and failure examples
Why it matters: Helps identify subtle disease similarities that confuse the model

Author: NLP Project Team
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

# Add backend path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from models.image import ImageModel
from models.text import TextModel

def plot_enhanced_confusion_matrix(y_true, y_pred, labels, output_path, title="Enhanced Confusion Matrix"):
    """Plot normalized confusion matrix with percentages and color mapping"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Recall (%)'})
    
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def analyze_misclassifications(model, df, dataset_root, output_dir, n_examples=10, type='image'):
    """Find and save visual examples of misclassified samples"""
    print(f"Analyzing {type} misclassifications...")
    
    misclassified = []
    
    # Get predictions for subset of data
    test_df = df.sample(min(2000, len(df)), random_state=42)
    
    for _, row in test_df.iterrows():
        try:
            if type == 'image':
                img_path = os.path.join(dataset_root, 'PlantVillage', row['LABEL'], row['FILENAME'])
                probs = model.predict(img_path)[0]
            else:
                # Text prediction logic
                from utils.text_cleaner import combine_and_clean_features as combine_text_features
                text = combine_text_features(row)
                
                inputs = model.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128).to(model.device)
                struct_feats = torch.zeros((1, model.num_struct)).to(model.device).float()
                with torch.no_grad():
                    logits = model.model(inputs['input_ids'], inputs['attention_mask'], struct_feats)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            pred_idx = np.argmax(probs)
            if type == 'image':
                pred_label = model.label_encoder.inverse_transform([pred_idx])[0]
            else:
                pred_label = model.label_encoder.classes_[pred_idx]
                
            if pred_label != row['LABEL']:
                misclassified.append({
                    'True': row['LABEL'],
                    'Pred': pred_label,
                    'Conf': probs[pred_idx] * 100,
                    'Path': img_path if type == 'image' else row['FILENAME'],
                    'Item': row
                })
        except Exception:
            continue
            
    # Save examples
    if misclassified:
        print(f"  Found {len(misclassified)} misclassified {type} samples")
        
        # Sort by confidence (highest confidence errors are most interesting)
        misclassified.sort(key=lambda x: x['Conf'], reverse=True)
        
        fig, axes = plt.subplots(min(len(misclassified), n_examples), 2, figsize=(12, 5 * min(len(misclassified), n_examples)))
        if min(len(misclassified), n_examples) == 1:
            axes = axes.reshape(1, -1)
            
        for i in range(min(len(misclassified), n_examples)):
            item = misclassified[i]
            
            if type == 'image':
                img = Image.open(item['Path']).convert('RGB')
                axes[i, 0].imshow(img)
                axes[i, 0].axis('off')
            else:
                axes[i, 0].text(0.1, 0.5, f"Text: {item['Item']['MORPHOLOGY']}\n{item['Item']['LESIONS']}", 
                               wrap=True, fontsize=10)
                axes[i, 0].axis('off')
                
            axes[i, 1].text(0.1, 0.5, f"True Label: {item['True']}\nPredicted: {item['Pred']}\nConfidence: {item['Conf']:.2f}%",
                           fontsize=12, fontweight='bold', color='red')
            axes[i, 1].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'misclassified_{type}_examples.png'), dpi=150)
        plt.close()
    
    return misclassified

def main():
    # Configuration
    DATA_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/data/multimodal_plant_data.csv'
    DATASET_ROOT = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/dataset/Image Dataset'
    IMAGE_MODEL_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/models/image/cnn_qnn_best.pt'
    LABEL_MAPPING_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/models/image/label_mapping.pkl'
    TEXT_MODEL_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/models/text/overhaul/best_biobert_overhaul.pth'
    PREPROCESSED_DATA_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/models/text/overhaul/overhaul_metadata.pkl'
    OUTPUT_DIR = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/outputs/interpretability/misclassifications'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    labels = sorted(df['LABEL'].unique())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Using device: {device}")
    
    # Analyze Image Model
    print("\nLoading Image Model...")
    img_model = ImageModel(model_path=IMAGE_MODEL_PATH, label_mapping_path=LABEL_MAPPING_PATH, device=device)
    analyze_misclassifications(img_model, df, DATASET_ROOT, OUTPUT_DIR, type='image')
    
    # Analyze Text Model
    print("\nLoading Text Model...")
    txt_model = TextModel(model_path=TEXT_MODEL_PATH, data_path=PREPROCESSED_DATA_PATH, device=device)
    analyze_misclassifications(txt_model, df, DATASET_ROOT, OUTPUT_DIR, type='text')
    
    print("\n✅ Misclassification analysis complete in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
