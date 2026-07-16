"""
Plant Disease Prediction - Comprehensive Test Evaluation
========================================================

This script provides a deep dive into the performance of all trained models
on the held-out test dataset. 

Features:
1. Loads all baseline and transformer models.
2. Per-class performance breakdown.
3. "Hard Class" analysis.
4. Visual sample predictions.

Author: NLP Project Team
"""

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
import random
import time
import os

# Import the model class from the training script
# Note: In a real project, this would be in a shared 'models.py'
import importlib.util
import sys

# Dynamic import for module starting with a number
spec = importlib.util.spec_from_file_location("bert_module", "04_train_bert_model.py")
bert_module = importlib.util.module_from_spec(spec)
sys.modules["bert_module"] = bert_module
spec.loader.exec_module(bert_module)

BioBERTPlantDiseaseClassifier = bert_module.BioBERTPlantDiseaseClassifier
PlantDiseaseDataset = bert_module.PlantDiseaseDataset

def load_data():
    print("Loading test data...")
    with open('preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print("Test data loaded.")
    return data

def load_models(num_classes, device):
    print("\nLoading trained models...")
    
    # Load Baselines
    with open('baseline_models.pkl', 'rb') as f:
        baselines = pickle.load(f)
    
    # Load BERT
    bert_model = BioBERTPlantDiseaseClassifier(num_classes=num_classes)
    bert_model.load_state_dict(torch.load('best_bert_model.pth', map_location=device))
    bert_model.to(device)
    bert_model.eval()
    
    print("All models loaded successfully.")
    return baselines, bert_model

def get_bert_predictions(model, loader, device):
    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
    return np.array(all_preds)

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data = load_data()
    X_test_tfidf = data['X_test_tfidf']
    test_encodings = data['test_encodings']
    y_test = data['y_test']
    label_encoder = data['label_encoder']
    class_names = label_encoder.classes_
    
    # Load models
    baselines, bert_model = load_models(len(class_names), device)
    
    # 1. GENERATE PREDICTIONS
    print("\nGenerating predictions on test set...")
    
    # Baseline Predictions
    svm_preds = baselines['svm'].predict(X_test_tfidf)
    rf_preds = baselines['random_forest'].predict(X_test_tfidf)
    lr_preds = baselines['logistic_regression'].predict(X_test_tfidf)
    
    # BERT Predictions
    test_dataset = PlantDiseaseDataset(test_encodings, y_test)
    test_loader = DataLoader(test_dataset, batch_size=16)
    bert_preds = get_bert_predictions(bert_model, test_loader, device)
    
    # 2. ANALYSIS TABLE
    results = {
        'SVM': svm_preds,
        'Random Forest': rf_preds,
        'Logistic Regression': lr_preds,
        'BERT': bert_preds
    }
    
    print("\n" + "="*60)
    print("OVERALL ACCURACY COMPARISON")
    print("="*60)
    
    final_metrics = []
    for name, preds in results.items():
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        final_metrics.append({'Model': name, 'Accuracy (%)': acc*100, 'F1-Score (%)': f1*100})
        print(f"📊 {name:<20}: {acc*100:.2f}%")
        
    # 3. PER-CLASS PERFORMANCE COMPARISON
    print("\n" + "="*60)
    print("PER-CLASS F1-SCORE COMPARISON (%)")
    print("="*60)
    
    class_results = []
    
    for class_idx, class_name in enumerate(class_names):
        row = {'Class': class_name}
        for model_name, preds in results.items():
            # Calculate F1 for this specific class
            f1 = f1_score(y_test == class_idx, preds == class_idx, zero_division=0)
            row[model_name] = f1 * 100
        class_results.append(row)
    
    df_class_comp = pd.DataFrame(class_results)
    print(df_class_comp.to_string(index=False))
    
    # 4. IDENTIFY HARD CLASSES
    print("\n" + "="*60)
    print("TOP 3 HARDEST CLASSES (Lowest Avg F1)")
    print("="*60)
    
    df_class_comp['Avg_F1'] = df_class_comp[list(results.keys())].mean(axis=1)
    hard_classes = df_class_comp.sort_values(by='Avg_F1').head(3)
    
    for idx, row in hard_classes.iterrows():
        print(f"🔴 {row['Class']} (Avg F1: {row['Avg_F1']:.2f}%)")
        # Find which model did worst
        worst_model = min(results.keys(), key=lambda k: row[k])
        print(f"   • Most difficult for: {worst_model} ({row[worst_model]:.2f}%)")

    # 5. RANDOM SAMPLE PREDICTIONS
    print("\n" + "="*60)
    print("VISUAL SAMPLE PREDICTIONS (Random 5)")
    print("="*60)
    
    indices = random.sample(range(len(y_test)), 5)
    
    for idx in indices:
        true_label = class_names[y_test[idx]]
        print(f"\n📂 Sample Index: {idx}")
        print(f"🎯 True Class: {true_label}")
        print("-" * 30)
        for name, preds in results.items():
            pred_label = class_names[preds[idx]]
            icon = "✅" if pred_label == true_label else "❌"
            print(f"{icon} {name:<20}: {pred_label}")

if __name__ == "__main__":
    main()
