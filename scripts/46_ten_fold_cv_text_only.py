"""
10-Fold Cross-Validation Testing - Text-Only BioBERT
====================================================

Tests the pre-trained overhauled BioBERT model (Text + Structured) across 10 stratified folds.
This establishes the standalone textual accuracy benchmark for the final report.

Author: NLP Project Team
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.models.text import TextModel
from backend.utils.nlp_pipeline import preprocess_text
from backend.utils.feature_engineering import extract_structured_features

def load_dataset(data_path):
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} samples")
    return df

def create_stratified_folds(df, n_folds=10, random_state=42):
    print(f"\nCreating {n_folds} stratified folds...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    labels = df['LABEL'].values
    indices = np.arange(len(df))
    
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(indices, labels)):
        folds.append({
            'fold': fold_idx + 1,
            'test_indices': test_idx,
            'test_labels': labels[test_idx]
        })
    return folds

def evaluate_fold(text_model, df, test_indices, label_encoder):
    predictions = []
    true_labels = []
    
    for idx in tqdm(test_indices, desc="Processing samples", leave=False):
        row = df.iloc[idx]
        true_label = row['LABEL']
        
        if 'cleaned_text' in row and pd.notna(row['cleaned_text']):
            text = row['cleaned_text']
        else:
            text = preprocess_text(str(row['DESCRIPTION']))
        
        temp_df = pd.DataFrame([row])
        struct_feats = extract_structured_features(temp_df)
        
        try:
            # Get text prediction probabilities
            text_probs = text_model.model(
                text_model.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)['input_ids'].to(text_model.device),
                text_model.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)['attention_mask'].to(text_model.device),
                torch.tensor(struct_feats, dtype=torch.float).to(text_model.device)
            )
            text_probs = torch.softmax(text_probs, dim=1).detach().cpu().numpy()[0]
            
            pred_idx = np.argmax(text_probs)
            pred_label = label_encoder.classes_[pred_idx]
            
            predictions.append(pred_label)
            true_labels.append(true_label)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
            
    return true_labels, predictions

def main():
    # Config
    CLEANED_DATA_PATH = 'data/cleaned_multimodal_plant_data.csv'
    TEXT_MODEL_PATH = 'models/text/overhaul/best_biobert_overhaul.pth'
    METADATA_PATH = 'models/text/overhaul/overhaul_metadata.pkl'
    OUTPUT_DIR = 'outputs/text_cv'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {DEVICE}")

    # Load Data & Model
    df = load_dataset(CLEANED_DATA_PATH)
    text_model = TextModel(model_path=TEXT_MODEL_PATH, data_path=METADATA_PATH, device=DEVICE)
    le = text_model.label_encoder
    
    folds = create_stratified_folds(df)
    
    all_metrics = []
    all_true = []
    all_pred = []
    
    print("\nStarting 10-Fold Text-Only Cross-Validation...")
    for fold_info in folds:
        print(f"\nEvaluating Fold {fold_info['fold']}/10...")
        true, pred = evaluate_fold(text_model, df, fold_info['test_indices'], le)
        
        acc = accuracy_score(true, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='weighted')
        
        all_metrics.append({
            'fold': fold_info['fold'],
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        all_true.extend(true)
        all_pred.extend(pred)
        print(f"Fold {fold_info['fold']} Accuracy: {acc*100:.2f}%")

    # Aggregate Results
    results_df = pd.DataFrame(all_metrics)
    avg_acc = results_df['accuracy'].mean()
    std_acc = results_df['accuracy'].std()
    
    print("\n" + "="*50)
    print("📈 FINAL TEXT-ONLY CV RESULTS")
    print(f"Average Accuracy: {avg_acc*100:.2f}% (+/- {std_acc*100:.2f}%)")
    print("="*50)
    
    # Save results
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'text_only_cv_metrics.csv'), index=False)
    
    # Full classification report
    final_report = classification_report(all_true, all_pred, target_names=le.classes_)
    with open(os.path.join(OUTPUT_DIR, 'text_only_full_report.txt'), 'w') as f:
        f.write(final_report)
        
    # Confusion Matrix
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Text-Only (Overhauled BioBERT)')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'text_only_confusion_matrix.png'))
    
    print(f"\n✅ All results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
