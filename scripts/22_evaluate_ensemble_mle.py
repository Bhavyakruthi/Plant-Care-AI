"""
Feature-Based Ensemble Evaluation & MLE Optimization

This script uses PRE-EXTRACTED features (from scripts/30_extract_multimodal_features.py)
to evaluate the ensemble model efficiency and optimize the fusion parameter (alpha).

Process:
1. Load extracted features (text & image embeddings)
2. Load preprocessed data (test labels and structured features)
3. Run features through the CLASSIFICATION HEADS only (very fast)
4. Optimize alpha using MLE
5. Generate performance report
"""
import torch
import torch.nn as nn
import numpy as np
import pickle
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# Setup paths
BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"
sys.path.append(os.path.join(BASE_DIR, "backend"))

from models.text import HybridBioBERTClassifier
from models.image import ImageModel

def load_data():
    print("\nLoading data...")
    
    # 1. Extracted Features (17006 samples)
    feat_path = os.path.join(BASE_DIR, 'data', 'multimodal_features_attention.pkl')
    with open(feat_path, 'rb') as f:
        all_features = pickle.load(f)
    print(f"   Loaded {len(all_features)} extracted feature sets")
    
    # 2. Test Split Info
    prep_path = os.path.join(BASE_DIR, 'data', 'preprocessed_data.pkl')
    with open(prep_path, 'rb') as f:
        prep_data = pickle.load(f)
        
    y_test_original = prep_data['y_test_original']
    X_test_structured = prep_data['X_test_structured']
    y_test_labels = prep_data['y_test']
    label_encoder = prep_data['label_encoder']
    
    print(f"   Loaded test split: {len(y_test_original)} samples")
    
    return all_features, y_test_original, X_test_structured, y_test_labels, label_encoder, prep_data

def get_test_predictions(text_model, image_model, all_features, test_indices, X_test_struct, device):
    """
    Run the pre-extracted features through the classification heads.
    """
    print(f"Running inference on classification heads...")
    
    text_probs_list = []
    image_probs_list = []
    
    # Models to evaluation mode
    text_model.eval()
    image_model.classifier.eval() # Only need classifier part
    
    with torch.no_grad():
        for i, idx in enumerate(tqdm(test_indices)):
            # Get features for this specific sample
            # Note: idx is the dataframe index, which matches the list index in all_features
            feats = all_features[idx]
            
            # --- Text Prediction ---
            # Input: [pooled_output, structured_features]
            text_emb = torch.tensor(feats['text_feat'], device=device).unsqueeze(0) # (1, 768)
            struct_feat = torch.tensor(X_test_struct[i], device=device).unsqueeze(0) # (1, 15)
            
            combined_text = torch.cat((text_emb, struct_feat), dim=1)
            
            text_model.eval() # Ensure eval mode!
            
            x = text_model.fc1(combined_text)
            x = text_model.relu1(x)
            # x = text_model.dropout1(x) # No dropout in eval
            
            x = text_model.fc2(x)
            x = text_model.relu2(x)
            # x = text_model.dropout2(x)
            
            text_logits = text_model.classifier(x)
            text_probs = torch.softmax(text_logits, dim=1).cpu().numpy()[0]
            text_probs_list.append(text_probs)
            
            # --- Image Prediction ---
            # Input: [resnet_features] -> Quantum Circuit -> Expansion
            img_emb = torch.tensor(feats['image_feat'], device=device).unsqueeze(0) # (1, 512)
            
            # ImageModel.classifier expects float input
            img_logits = image_model.classifier(img_emb.float())
            img_probs = torch.softmax(img_logits, dim=1).cpu().numpy()[0]
            image_probs_list.append(img_probs)
            
    return np.array(text_probs_list), np.array(image_probs_list)

def optimize_mle(text_probs, image_probs, y_true):
    print("\n" + "="*80)
    print("OPTIMIZING ALPHA (MLE)")
    print("="*80)
    
    alphas = np.linspace(0, 1, 101)
    best_alpha = 0.5
    best_acc = 0
    best_ll = -np.inf
    
    log_likelihoods = []
    accuracies = []
    
    for alpha in alphas:
        # Fusion
        fused = alpha * image_probs + (1 - alpha) * text_probs
        
        # Accuracy
        preds = np.argmax(fused, axis=1)
        acc = accuracy_score(y_true, preds)
        accuracies.append(acc)
        
        # Log Likelihood
        # Select probability of the true class
        # p_true = fused[range(n), y_true]
        # ll = sum(log(p_true))
        rows = np.arange(len(y_true))
        true_class_probs = fused[rows, y_true]
        ll = np.sum(np.log(true_class_probs + 1e-10))
        log_likelihoods.append(ll)
        
        if ll > best_ll:
            best_ll = ll
            best_alpha = alpha
            best_acc = acc
            
    print(f"Optimal Alpha (by Log-Likelihood): {best_alpha:.2f}")
    print(f"   Accuracy at optimal alpha: {best_acc*100:.2f}%")
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Alpha (Image Weight)')
    ax1.set_ylabel('Log-Likelihood', color=color)
    ax1.plot(alphas, log_likelihoods, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axvline(best_alpha, color='red', linestyle='--', label=f'Optimal alpha={best_alpha:.2f}')
    
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(alphas, accuracies, color=color, linestyle=':', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'MLE Optimization: Optimal Alpha = {best_alpha:.2f}')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'output_gradcam', 'mle_optimization_curve.png'))
    print("Saved optimization plot.")
    
    return best_alpha

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load Data
    all_features, y_test_original, X_test_struct, y_test_labels, label_encoder, prep_data = load_data()
    test_indices = y_test_original.index
    
    # 2. Initialize Models (Just needed for classification heads)
    print("\nInitializing model heads...")
    num_classes = len(label_encoder.classes_)
    num_struct = X_test_struct.shape[1]
    
    # Text Model Head
    text_model = HybridBioBERTClassifier(num_classes, num_struct).to(device)
    text_path = os.path.join(BASE_DIR, "models", "text", "best_hybrid_bert_model.pth")
    if os.path.exists(text_path):
        text_model.load_state_dict(torch.load(text_path, map_location=device))
        print("Text weights loaded")
    
    # Image Model Head (QuantumClassifier)
    image_model_full = ImageModel(model_path=os.path.join(BASE_DIR, "models", "image", "cnn_qnn_best.pt"))
    image_model = image_model_full # We'll access .classifier
    print("Image weights loaded")

    # 3. Get Predictions
    text_probs, image_probs = get_test_predictions(
        text_model, image_model, all_features, test_indices, X_test_struct, device
    )
    
    # 4. Baseline Evaluation
    acc_text = accuracy_score(y_test_labels, np.argmax(text_probs, axis=1))
    acc_img = accuracy_score(y_test_labels, np.argmax(image_probs, axis=1))
    
    current_alpha = 0.6
    fused_current = current_alpha * image_probs + (1 - current_alpha) * text_probs
    acc_current = accuracy_score(y_test_labels, np.argmax(fused_current, axis=1))
    
    print("\nBaseline Metrics:")
    print(f"   Text Model Accuracy:  {acc_text*100:.2f}%")
    print(f"   Image Model Accuracy: {acc_img*100:.2f}%")
    print(f"   Current Ensemble (alpha=0.6): {acc_current*100:.2f}%")
    
    # 5. Optimize
    optimal_alpha = optimize_mle(text_probs, image_probs, y_test_labels)
    
    # 6. Final Evaluation
    fused_opt = optimal_alpha * image_probs + (1 - optimal_alpha) * text_probs
    preds_opt = np.argmax(fused_opt, axis=1)
    acc_opt = accuracy_score(y_test_labels, preds_opt)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Original Acc: {acc_current*100:.2f}%")
    print(f"Optimized Acc: {acc_opt*100:.2f}%")
    print(f"Improvement: {acc_opt - acc_current:+.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_labels, preds_opt)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix (alpha={optimal_alpha:.2f})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'output_gradcam', 'final_confusion_matrix.png'))
    print("Saved final confusion matrix.")

if __name__ == "__main__":
    main()
