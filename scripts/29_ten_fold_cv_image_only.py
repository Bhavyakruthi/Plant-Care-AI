"""
10-Fold Cross-Validation Testing - Image-Only Model
====================================================

Tests the pre-trained CNN+QNN image model across 10 stratified folds
WITHOUT retraining. Evaluates model performance on different test splits
to obtain robust metrics with confidence intervals.

Author: NLP Project Team
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import pickle
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add backend path for model imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from models.image import ImageModel

def load_dataset(data_path):
    """Load the multimodal dataset"""
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} samples")
    return df

def create_stratified_folds(df, n_folds=10, random_state=42):
    """Create stratified folds for cross-validation"""
    print(f"\nCreating {n_folds} stratified folds...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Get labels
    labels = df['LABEL'].values
    indices = np.arange(len(df))
    
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(indices, labels)):
        folds.append({
            'fold': fold_idx + 1,
            'test_indices': test_idx,
            'test_labels': labels[test_idx]
        })
    
    print(f"Created {n_folds} folds with stratified class distribution")
    return folds

def evaluate_fold(model, df, test_indices, dataset_root, label_encoder):
    """Evaluate model on a single fold"""
    predictions = []
    true_labels = []
    
    for idx in tqdm(test_indices, desc="Processing images", leave=False):
        row = df.iloc[idx]
        image_filename = row['FILENAME']
        true_label = row['LABEL']
        
        # Construct full image path - images are in PlantVillage/{LABEL}/{FILENAME}
        image_path = os.path.normpath(os.path.join(dataset_root, 'PlantVillage', true_label, image_filename))
        
        try:
            # Get prediction probabilities (returns numpy array)
            probs = model.predict(image_path)
            
            # Get predicted class from probabilities
            pred_idx = np.argmax(probs[0])  # probs is shape (1, num_classes)
            pred_label = label_encoder.classes_[pred_idx]
            
            predictions.append(pred_label)
            true_labels.append(true_label)
        except Exception as e:
            # Skip failed predictions silently to avoid spam
            continue
    
    return np.array(predictions), np.array(true_labels)

def calculate_metrics(y_true, y_pred, class_names):
    """Calculate comprehensive metrics"""
    # Check if we have any predictions
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError(f"Cannot calculate metrics: y_true has {len(y_true)} samples, y_pred has {len(y_pred)} samples")
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average=None, zero_division=0
    )
    
    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    return {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'confusion_matrix': cm
    }

def main():
    # Configuration
    DATA_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/data/multimodal_plant_data.csv'
    DATASET_ROOT = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/dataset/Image Dataset'
    MODEL_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/models/image/cnn_qnn_best.pt'
    LABEL_MAPPING_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/models/image/label_mapping.pkl'
    OUTPUT_DIR = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/outputs'
    N_FOLDS = 10
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup CUDA device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("\n" + "="*70)
    print("10-FOLD CROSS-VALIDATION - IMAGE-ONLY MODEL (CNN+QNN)")
    print("="*70)
    
    # Load dataset
    df = load_dataset(DATA_PATH)
    
    # Get unique class names
    class_names = sorted(df['LABEL'].unique())
    print(f"\nNumber of classes: {len(class_names)}")
    
    # Load model
    print("\nLoading pre-trained CNN+QNN model...")
    model = ImageModel(model_path=MODEL_PATH, label_mapping_path=LABEL_MAPPING_PATH, device=device)
    
    # Create a label encoder from the model's label mapping
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(class_names)
    
    print("Model loaded successfully!")
    
    # Create folds
    folds = create_stratified_folds(df, n_folds=N_FOLDS)
    
    # Store results for each fold
    fold_results = []
    all_confusion_matrices = []
    
    # Evaluate each fold
    print("\n" + "="*70)
    print("EVALUATING FOLDS")
    print("="*70)
    
    for fold_info in folds:
        fold_num = fold_info['fold']
        test_indices = fold_info['test_indices']
        
        print(f"\n📊 Fold {fold_num}/{N_FOLDS} - Testing on {len(test_indices)} samples")
        
        # Evaluate
        predictions, true_labels = evaluate_fold(model, df, test_indices, DATASET_ROOT, label_encoder)
        
        # Check if we got any predictions
        if len(predictions) == 0:
            print(f"   ⚠️  WARNING: No successful predictions in this fold! Skipping...")
            continue
        
        print(f"   ✅ Successfully processed {len(predictions)}/{len(test_indices)} samples")
        
        # Calculate metrics
        try:
            metrics = calculate_metrics(true_labels, predictions, class_names)
        except Exception as e:
            print(f"   ⚠️  ERROR calculating metrics: {str(e)}")
            continue
        
        # Store results
        fold_results.append({
            'fold': fold_num,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision_weighted'],
            'recall': metrics['recall_weighted'],
            'f1_score': metrics['f1_weighted'],
            'n_samples': len(predictions)  # Use actual processed count
        })
        
        all_confusion_matrices.append(metrics['confusion_matrix'])
        
        # Print fold results
        print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"   Precision: {metrics['precision_weighted']*100:.2f}%")
        print(f"   Recall:    {metrics['recall_weighted']*100:.2f}%")
        print(f"   F1-Score:  {metrics['f1_weighted']*100:.2f}%")
    
    # Aggregate results
    print("\n" + "="*70)
    print("AGGREGATED RESULTS ACROSS ALL FOLDS")
    print("="*70)
    
    results_df = pd.DataFrame(fold_results)
    
    # Calculate statistics
    stats = {
        'Accuracy': {
            'mean': results_df['accuracy'].mean() * 100,
            'std': results_df['accuracy'].std() * 100,
            'min': results_df['accuracy'].min() * 100,
            'max': results_df['accuracy'].max() * 100
        },
        'Precision': {
            'mean': results_df['precision'].mean() * 100,
            'std': results_df['precision'].std() * 100,
            'min': results_df['precision'].min() * 100,
            'max': results_df['precision'].max() * 100
        },
        'Recall': {
            'mean': results_df['recall'].mean() * 100,
            'std': results_df['recall'].std() * 100,
            'min': results_df['recall'].min() * 100,
            'max': results_df['recall'].max() * 100
        },
        'F1-Score': {
            'mean': results_df['f1_score'].mean() * 100,
            'std': results_df['f1_score'].std() * 100,
            'min': results_df['f1_score'].min() * 100,
            'max': results_df['f1_score'].max() * 100
        }
    }
    
    # Print summary
    print("\n📊 SUMMARY STATISTICS (10-Fold CV):")
    print("-" * 70)
    for metric, values in stats.items():
        print(f"\n{metric}:")
        print(f"  Mean ± Std:  {values['mean']:.2f}% ± {values['std']:.2f}%")
        print(f"  Range:       [{values['min']:.2f}%, {values['max']:.2f}%]")
    
    # Calculate 95% confidence interval
    confidence_95 = 1.96 * stats['Accuracy']['std'] / np.sqrt(N_FOLDS)
    print(f"\n95% Confidence Interval for Accuracy:")
    print(f"  {stats['Accuracy']['mean']:.2f}% ± {confidence_95:.2f}%")
    
    # Save detailed results
    output_csv = os.path.join(OUTPUT_DIR, '10fold_cv_image_results.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"\n✅ Detailed results saved to: {output_csv}")
    
    # Save aggregated confusion matrix
    avg_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
    cm_output = os.path.join(OUTPUT_DIR, '10fold_cv_image_confusion_matrix.npy')
    np.save(cm_output, avg_confusion_matrix)
    print(f"✅ Average confusion matrix saved to: {cm_output}")
    
    # Save summary statistics
    summary_output = os.path.join(OUTPUT_DIR, '10fold_cv_image_summary.txt')
    with open(summary_output, 'w') as f:
        f.write("="*70 + "\n")
        f.write("10-FOLD CROSS-VALIDATION RESULTS - IMAGE-ONLY MODEL (CNN+QNN)\n")
        f.write("="*70 + "\n\n")
        
        for metric, values in stats.items():
            f.write(f"{metric}:\n")
            f.write(f"  Mean ± Std:  {values['mean']:.2f}% ± {values['std']:.2f}%\n")
            f.write(f"  Range:       [{values['min']:.2f}%, {values['max']:.2f}%]\n\n")
        
        f.write(f"95% Confidence Interval for Accuracy:\n")
        f.write(f"  {stats['Accuracy']['mean']:.2f}% ± {confidence_95:.2f}%\n")
    
    print(f"✅ Summary saved to: {summary_output}")
    
    print("\n" + "="*70)
    print("✅ IMAGE-ONLY 10-FOLD CV TESTING COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()
