import numpy as np
import pickle
import torch
from sklearn.metrics import accuracy_score
import os
import sys

# Mock probabilities for demonstration if files not available
def optimize_alpha(val_image_preds, val_text_preds, true_labels):
    best_alpha = 0.5
    best_acc = 0
    
    print("Optimizing Ensemble Alpha (MLE weighting)...")
    for alpha in np.linspace(0, 1, 101):
        fused_probs = (alpha * val_image_preds) + ((1 - alpha) * val_text_preds)
        preds = np.argmax(fused_probs, axis=1)
        acc = accuracy_score(true_labels, preds)
        
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
            
    print(f"🌟 Optimal Alpha found: {best_alpha:.2f}")
    print(f"🌟 Best Validation Accuracy: {best_acc*100:.2f}%")
    return best_alpha

if __name__ == "__main__":
    # In a real scenario, you run your models on the validation set 
    # and save the numpy probability arrays.
    print("This script helps you find the optimal 'alpha' for your ensemble.")
    print("Formula: P = alpha * P_image + (1 - alpha) * P_text")
    
    # Placeholder for user to run:
    # 1. Run inference on validation set for both models.
    # 2. Pass the results to optimize_alpha.
    pass
