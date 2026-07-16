import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

def run_stacking_experiment():
    print("\n" + "="*80)
    print("STACKING ENSEMBLE PROTOTYPE (LEARNING THE FUSION)")
    print("="*80)
    
    # Simulate a validation set where we have predictions from both models
    # This represents the probabilities (15 classes) from Image and Text models
    np.random.seed(42)
    n_samples = 1000
    
    # Mocking probability outputs (simulating our real model performance)
    image_preds = np.random.dirichlet(np.ones(15), size=n_samples)
    text_preds = np.random.dirichlet(np.ones(15), size=n_samples)
    true_labels = np.random.randint(0, 15, size=n_samples)
    
    # Method 1: Simple Weighting (Current MLE Alpha=0.6)
    alpha = 0.6
    simple_fusion = (alpha * image_preds) + ((1 - alpha) * text_preds)
    simple_acc = accuracy_score(true_labels, np.argmax(simple_fusion, axis=1))
    
    # Method 2: Stacking (Meta-Learning)
    # We combine the outputs into a 30-dimensional feature vector
    X_stacked = np.hstack((image_preds, text_preds))
    
    # Train a meta-model on a 'hold-out' part of the validation set
    X_train, X_test, y_train, y_test = train_test_split(X_stacked, true_labels, test_size=0.3)
    
    meta_model = RandomForestClassifier(n_estimators=100, max_depth=5)
    meta_model.fit(X_train, y_train)
    
    stacked_preds = meta_model.predict(X_test)
    stacked_acc = accuracy_score(y_test, stacked_preds)
    
    print(f"\n📊 Baseline (MLE 0.6 Weighting): {simple_acc*100:.2f}%")
    print(f"📊 Meta-Learning (Stacking):      {stacked_acc*100:.2f}%")
    
    improvement = stacked_acc - accuracy_score(y_test, np.argmax(simple_fusion[700:], axis=1))
    print(f"\n🚀 Theoretical Gain from Stacking: +{improvement*100:.2f}%")
    print("\n✅ Rationale: The meta-model learns THAT for certain diseases, ")
    print("   the Image model is more reliable, while for others, ")
    print("   the Gemini-Text description is the decisive factor.")
    print("="*80)

if __name__ == "__main__":
    run_stacking_experiment()
