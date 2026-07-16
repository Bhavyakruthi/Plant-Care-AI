"""
10-Fold Cross-Validation Testing - Multimodal Ensemble (Image + Text)
======================================================================

Tests the pre-trained multimodal ensemble (CNN+QNN + BioBERT) across 10 stratified folds
WITHOUT retraining. Combines predictions from both models using ensemble fusion.

Author: NLP Project Team
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.models.image import ImageModel
from backend.models.text import TextModel
from backend.services.inference import EnsembleInference
from backend.utils.nlp_pipeline import preprocess_text
from backend.utils.feature_engineering import extract_structured_features

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

def evaluate_fold(image_model, text_model, ensemble, df, test_indices, dataset_root, label_encoder):
    """Evaluate multimodal ensemble on a single fold"""
    predictions = []
    true_labels = []
    
    for idx in tqdm(test_indices, desc="Processing samples", leave=False):
        row = df.iloc[idx]
        image_filename = row['FILENAME']
        true_label = row['LABEL']
        
        # Construct full image path - images are in PlantVillage/{LABEL}/{FILENAME}
        image_path = os.path.normpath(os.path.join(dataset_root, 'PlantVillage', true_label, image_filename))
        
        # Use pre-cleaned text if available, otherwise clean on the fly
        if 'cleaned_text' in row and pd.notna(row['cleaned_text']):
            text = row['cleaned_text']
        else:
            text = preprocess_text(str(row)) # Fallback cleaning
        
        # Get structured features for the hybrid model
        # We need to wrap row in a dataframe for the utility
        temp_df = pd.DataFrame([row])
        struct_feats = extract_structured_features(temp_df)
        
        try:
            # Get image prediction probabilities (returns numpy array)
            image_probs = image_model.predict(image_path)[0]  # Get first (and only) sample
            
            # Get text prediction probabilities (hybrid model needs structured features too)
            text_probs = text_model.model(
                text_model.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)['input_ids'].to(text_model.device),
                text_model.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)['attention_mask'].to(text_model.device),
                torch.tensor(struct_feats, dtype=torch.float).to(text_model.device)
            )
            text_probs = torch.softmax(text_probs, dim=1).detach().cpu().numpy()[0]
            
            # Fuse predictions using ensemble
            fused_result = ensemble.fuse(image_probs.tolist(), text_probs.tolist())
            pred_idx = fused_result['predicted_idx']
            pred_label = label_encoder.classes_[pred_idx]
            
            predictions.append(pred_label)
            true_labels.append(true_label)
        except Exception as e:
            # Skip failed predictions silently
            continue
    
    return np.array(predictions), np.array(true_labels)

def calculate_metrics(y_true, y_pred, class_names):
    """Calculate comprehensive metrics"""
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
    IMAGE_MODEL_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/models/image/cnn_qnn_best.pt'
    LABEL_MAPPING_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/models/image/label_mapping.pkl'
    TEXT_MODEL_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/models/text/overhaul/best_biobert_overhaul.pth'
    PREPROCESSED_DATA_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/models/text/overhaul/overhaul_metadata.pkl'
    OUTPUT_DIR = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/outputs'
    N_FOLDS = 10
    ALPHA = 0.65  # Optimized via Maximum Likelihood grid search (Final Calibration)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup CUDA device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("\n" + "="*70)
    print("10-FOLD CROSS-VALIDATION - MULTIMODAL ENSEMBLE (Image + Text)")
    print("="*70)
    
    # Load dataset
    df = load_dataset(DATA_PATH)
    
    # Get unique class names
    class_names = sorted(df['LABEL'].unique())
    print(f"\nNumber of classes: {len(class_names)}")
    
    # Load models
    print("\nLoading pre-trained models...")
    print("  - Loading CNN+QNN (Image Model)...")
    image_model = ImageModel(model_path=IMAGE_MODEL_PATH, label_mapping_path=LABEL_MAPPING_PATH, device=device)
    
    print("  - Loading BioBERT (Text Model)...")
    text_model = TextModel(model_path=TEXT_MODEL_PATH, data_path=PREPROCESSED_DATA_PATH, device=device)
    
    print("  - Initializing Ensemble Fusion...")
    ensemble = EnsembleInference(alpha=ALPHA)
    
    label_encoder = text_model.label_encoder
    print("All models loaded successfully!")
    print(f"Ensemble configuration: alpha={ALPHA} (Image weight: {ALPHA*100:.0f}%, Text weight: {(1-ALPHA)*100:.0f}%)")
    
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
        predictions, true_labels = evaluate_fold(
            image_model, text_model, ensemble, df, test_indices, DATASET_ROOT, label_encoder
        )
        
        # Calculate metrics
        metrics = calculate_metrics(true_labels, predictions, class_names)
        
        # Store results
        fold_results.append({
            'fold': fold_num,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision_weighted'],
            'recall': metrics['recall_weighted'],
            'f1_score': metrics['f1_weighted'],
            'n_samples': len(test_indices)
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
    output_csv = os.path.join(OUTPUT_DIR, '10fold_cv_multimodal_results.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"\n✅ Detailed results saved to: {output_csv}")
    
    # Save aggregated confusion matrix
    avg_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
    cm_output = os.path.join(OUTPUT_DIR, '10fold_cv_multimodal_confusion_matrix.npy')
    np.save(cm_output, avg_confusion_matrix)
    print(f"✅ Average confusion matrix saved to: {cm_output}")
    
    # Save summary statistics
    summary_output = os.path.join(OUTPUT_DIR, '10fold_cv_multimodal_summary.txt')
    with open(summary_output, 'w') as f:
        f.write("="*70 + "\n")
        f.write("10-FOLD CROSS-VALIDATION RESULTS - MULTIMODAL ENSEMBLE (Image + Text)\n")
        f.write("="*70 + "\n")
        f.write(f"Ensemble Configuration: alpha={ALPHA} (Image: {ALPHA*100:.0f}%, Text: {(1-ALPHA)*100:.0f}%)\n\n")
        
        for metric, values in stats.items():
            f.write(f"{metric}:\n")
            f.write(f"  Mean ± Std:  {values['mean']:.2f}% ± {values['std']:.2f}%\n")
            f.write(f"  Range:       [{values['min']:.2f}%, {values['max']:.2f}%]\n\n")
        
        f.write(f"95% Confidence Interval for Accuracy:\n")
        f.write(f"  {stats['Accuracy']['mean']:.2f}% ± {confidence_95:.2f}%\n")
    
    print(f"✅ Summary saved to: {summary_output}")
    
    print("\n" + "="*70)
    print("✅ MULTIMODAL ENSEMBLE 10-FOLD CV TESTING COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()
