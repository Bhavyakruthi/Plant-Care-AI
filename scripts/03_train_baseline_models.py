"""
Plant Disease Prediction - Baseline Model Training
===================================================

This script trains baseline models using TF-IDF features:
1. Support Vector Machine (SVM)
2. Random Forest Classifier
3. Logistic Regression (bonus)

Includes comprehensive evaluation and comparison.

Author: NLP Project Team
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import time
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD PREPROCESSED DATA
# ==============================================================================

def load_preprocessed_data(filename='preprocessed_data.pkl'):
    """Load preprocessed data"""
    print("=" * 80)
    print("LOADING PREPROCESSED DATA")
    print("=" * 80)
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\n✅ Data loaded from: {filename}")
    print(f"\n📊 Dataset sizes:")
    print(f"   • Training samples: {len(data['y_train']):,}")
    print(f"   • Validation samples: {len(data['y_val']):,}")
    print(f"   • Test samples: {len(data['y_test']):,}")
    print(f"   • Number of classes: {len(data['label_encoder'].classes_)}")
    
    return data


# ==============================================================================
# 2. TRAIN SVM MODEL
# ==============================================================================

def train_svm(X_train, y_train, X_val, y_val):
    """
    Train Support Vector Machine classifier
    
    Why SVM?
    - Excellent for high-dimensional data (perfect for TF-IDF features)
    - Finds optimal decision boundary between classes
    - Works well with sparse data
    - Good generalization ability
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        Trained SVM model
    """
    print("\n" + "=" * 80)
    print("TRAINING MODEL 1: SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 80)
    
    print("\n📚 SVM Configuration:")
    print("   • Kernel: Linear (best for text classification)")
    print("   • C: 1.0 (regularization parameter)")
    print("   • Class weight: Balanced (handles class imbalance)")
    
    print("\n🚀 Training SVM...")
    start_time = time.time()
    
    svm_model = SVC(
        kernel='linear',
        C=1.0,
        class_weight='balanced',
        random_state=42,
        verbose=1
    )
    
    svm_model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Predictions
    y_train_pred = svm_model.predict(X_train)
    y_val_pred = svm_model.predict(X_val)
    
    # Accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"\n✅ SVM Training Complete!")
    print(f"   • Training time: {train_time:.2f} seconds")
    print(f"   • Training accuracy: {train_acc*100:.2f}%")
    print(f"   • Validation accuracy: {val_acc*100:.2f}%")
    
    return svm_model, {'train_acc': train_acc, 'val_acc': val_acc, 'train_time': train_time}


# ==============================================================================
# 3. TRAIN RANDOM FOREST MODEL
# ==============================================================================

def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest classifier
    
    Why Random Forest?
    - Ensemble method (combines multiple decision trees)
    - Captures non-linear relationships
    - Provides feature importance scores
    - Robust to overfitting
    - Handles complex patterns well
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        Trained Random Forest model
    """
    print("\n" + "=" * 80)
    print("TRAINING MODEL 2: RANDOM FOREST")
    print("=" * 80)
    
    print("\n🌳 Random Forest Configuration:")
    print("   • Number of trees: 200")
    print("   • Max depth: 50 (prevents overfitting)")
    print("   • Min samples split: 5")
    print("   • Class weight: Balanced")
    print("   • Parallel jobs: -1 (use all CPU cores)")
    
    print("\n🚀 Training Random Forest...")
    start_time = time.time()
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=50,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf_model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Predictions
    y_train_pred = rf_model.predict(X_train)
    y_val_pred = rf_model.predict(X_val)
    
    # Accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"\n✅ Random Forest Training Complete!")
    print(f"   • Training time: {train_time:.2f} seconds")
    print(f"   • Training accuracy: {train_acc*100:.2f}%")
    print(f"   • Validation accuracy: {val_acc*100:.2f}%")
    
    return rf_model, {'train_acc': train_acc, 'val_acc': val_acc, 'train_time': train_time}


# ==============================================================================
# 4. TRAIN LOGISTIC REGRESSION (BONUS)
# ==============================================================================

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train Logistic Regression classifier (fast baseline)
    
    Why Logistic Regression?
    - Very fast to train
    - Good baseline for comparison
    - Provides probability estimates
    - Interpretable coefficients
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        Trained Logistic Regression model
    """
    print("\n" + "=" * 80)
    print("TRAINING MODEL 3: LOGISTIC REGRESSION (Baseline)")
    print("=" * 80)
    
    print("\n📊 Logistic Regression Configuration:")
    print("   • Solver: lbfgs (efficient for multiclass)")
    print("   • Max iterations: 1000")
    print("   • Class weight: Balanced")
    
    print("\n🚀 Training Logistic Regression...")
    start_time = time.time()
    
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        verbose=1,
        n_jobs=-1
    )
    
    lr_model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Predictions
    y_train_pred = lr_model.predict(X_train)
    y_val_pred = lr_model.predict(X_val)
    
    # Accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"\n✅ Logistic Regression Training Complete!")
    print(f"   • Training time: {train_time:.2f} seconds")
    print(f"   • Training accuracy: {train_acc*100:.2f}%")
    print(f"   • Validation accuracy: {val_acc*100:.2f}%")
    
    return lr_model, {'train_acc': train_acc, 'val_acc': val_acc, 'train_time': train_time}


# ==============================================================================
# 5. DETAILED EVALUATION
# ==============================================================================

def evaluate_model(model, X_test, y_test, model_name, label_encoder):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        model_name: Name of the model
        label_encoder: Label encoder for class names
        
    Returns:
        Dictionary of metrics
    """
    print("\n" + "=" * 80)
    print(f"EVALUATING {model_name.upper()}")
    print("=" * 80)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    print(f"\n📊 Overall Metrics:")
    print(f"   • Accuracy: {accuracy*100:.2f}%")
    print(f"   • Precision (weighted): {precision*100:.2f}%")
    print(f"   • Recall (weighted): {recall*100:.2f}%")
    print(f"   • F1-Score (weighted): {f1*100:.2f}%")
    
    # Classification report
    print(f"\n📋 Detailed Classification Report:")
    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'outputs/confusion_matrix_{model_name.replace(" ", "_").lower()}.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n📊 Confusion matrix saved: outputs/confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': y_pred  # Added for stacking
    }


# ==============================================================================
# 6. MODEL COMPARISON
# ==============================================================================

def compare_models(results_dict):
    """
    Compare all baseline models
    
    Args:
        results_dict: Dictionary containing results for all models
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, metrics in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'] * 100,
            'Precision': metrics['precision'] * 100,
            'Recall': metrics['recall'] * 100,
            'F1-Score': metrics['f1'] * 100,
            'Training Time': metrics.get('train_time', 0)
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\n📊 Performance Comparison:")
    print(df_comparison.to_string(index=False))
    
    # Save to CSV
    df_comparison.to_csv('outputs/baseline_models_comparison.csv', index=False)
    print("\n💾 Comparison saved: outputs/baseline_models_comparison.csv")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy comparison
    axes[0, 0].bar(df_comparison['Model'], df_comparison['Accuracy'], color='steelblue')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylim([0, 100])
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(df_comparison['Accuracy']):
        axes[0, 0].text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom')
    
    # F1-Score comparison
    axes[0, 1].bar(df_comparison['Model'], df_comparison['F1-Score'], color='coral')
    axes[0, 1].set_ylabel('F1-Score (%)')
    axes[0, 1].set_title('Model F1-Score Comparison')
    axes[0, 1].set_ylim([0, 100])
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(df_comparison['F1-Score']):
        axes[0, 1].text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom')
    
    # Precision vs Recall
    axes[1, 0].scatter(df_comparison['Precision'], df_comparison['Recall'], 
                       s=200, c='green', alpha=0.6)
    for idx, row in df_comparison.iterrows():
        axes[1, 0].annotate(row['Model'], (row['Precision'], row['Recall']),
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 0].set_xlabel('Precision (%)')
    axes[1, 0].set_ylabel('Recall (%)')
    axes[1, 0].set_title('Precision vs Recall')
    axes[1, 0].grid(alpha=0.3)
    
    # Training time comparison
    axes[1, 1].bar(df_comparison['Model'], df_comparison['Training Time'], color='purple')
    axes[1, 1].set_ylabel('Training Time (seconds)')
    axes[1, 1].set_title('Training Time Comparison')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(df_comparison['Training Time']):
        axes[1, 1].text(i, v + 0.5, f'{v:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('outputs/baseline_models_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Comparison visualization saved: outputs/baseline_models_comparison.png")
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("\n" + "=" * 80)
    print(" " * 20 + "PLANT DISEASE PREDICTION")
    print(" " * 18 + "Baseline Model Training")
    print("=" * 80)
    
    # Load data
    data = load_preprocessed_data()
    
    X_train_tfidf = data['X_train_tfidf']
    X_val_tfidf = data['X_val_tfidf']
    X_test_tfidf = data['X_test_tfidf']
    
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    label_encoder = data['label_encoder']
    
    # Train models
    svm_model, svm_train_metrics = train_svm(X_train_tfidf, y_train, X_val_tfidf, y_val)
    rf_model, rf_train_metrics = train_random_forest(X_train_tfidf, y_train, X_val_tfidf, y_val)
    lr_model, lr_train_metrics = train_logistic_regression(X_train_tfidf, y_train, X_val_tfidf, y_val)
    
    # Evaluate on test set
    svm_metrics = evaluate_model(svm_model, X_test_tfidf, y_test, "SVM", label_encoder)
    svm_metrics.update(svm_train_metrics)
    
    rf_metrics = evaluate_model(rf_model, X_test_tfidf, y_test, "Random Forest", label_encoder)
    rf_metrics.update(rf_train_metrics)
    
    lr_metrics = evaluate_model(lr_model, X_test_tfidf, y_test, "Logistic Regression", label_encoder)
    lr_metrics.update(lr_train_metrics)
    
    # Compare models
    results = {
        'SVM': svm_metrics,
        'Random Forest': rf_metrics,
        'Logistic Regression': lr_metrics
    }
    
    compare_models(results)
    
    # Save models
    print("\n" + "=" * 80)
    print("SAVING TRAINED MODELS")
    print("=" * 80)
    
    models = {
        'svm': svm_model,
        'random_forest': rf_model,
        'logistic_regression': lr_model,
        'results': results,
        'predictions': {
            'svm': svm_metrics['predictions'],
            'random_forest': rf_metrics['predictions'],
            'logistic_regression': lr_metrics['predictions']
        }
    }
    
    with open('baseline_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print("\n💾 Models saved: baseline_models.pkl (with predictions for stacking)")
    
    print("\n" + "=" * 80)
    print("✅ BASELINE MODEL TRAINING COMPLETED!")
    print("=" * 80)
    
    # Find best model
    best_model = max(results, key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model]['accuracy'] * 100
    
    print(f"\n🏆 Best Baseline Model: {best_model}")
    print(f"   • Test Accuracy: {best_accuracy:.2f}%")
    print(f"   • F1-Score: {results[best_model]['f1']*100:.2f}%")
    
    print("\n🎯 Next step: Train BERT transformer model")
    print("   • Run: python 03_train_bert_model.py")


if __name__ == "__main__":
    main()
