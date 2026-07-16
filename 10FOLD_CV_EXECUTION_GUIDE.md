# 10-Fold Cross-Validation Testing - Execution Guide

## Overview

This guide explains how to run the 10-fold cross-validation testing on your existing trained models **without retraining**. The testing will evaluate three modalities:

1. **Image-Only** (CNN+QNN model)
2. **Text-Only** (BioBERT model)
3. **Multimodal** (Image + Text ensemble)

## Prerequisites

✅ Pre-trained models are available in `models/` directory  
✅ Dataset is available at `data/multimodal_plant_data.csv`  
✅ Images are available in `dataset/` directory  
✅ All dependencies are installed (see `requirements.txt`)

## Execution Steps

### Step 1: Run Image-Only 10-Fold CV Testing

```bash
cd "d:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT"
python scripts/29_ten_fold_cv_image_only.py
```

**Expected Duration**: 20-40 minutes (depends on dataset size and GPU availability)

**Output Files**:
- `outputs/10fold_cv_image_results.csv` - Detailed per-fold results
- `outputs/10fold_cv_image_confusion_matrix.npy` - Average confusion matrix
- `outputs/10fold_cv_image_summary.txt` - Summary statistics

### Step 2: Run Text-Only 10-Fold CV Testing

```bash
python scripts/30_ten_fold_cv_text_only.py
```

**Expected Duration**: 15-30 minutes (depends on GPU availability)

**Output Files**:
- `outputs/10fold_cv_text_results.csv` - Detailed per-fold results
- `outputs/10fold_cv_text_confusion_matrix.npy` - Average confusion matrix
- `outputs/10fold_cv_text_summary.txt` - Summary statistics

### Step 3: Run Multimodal Ensemble 10-Fold CV Testing

```bash
python scripts/31_ten_fold_cv_multimodal.py
```

**Expected Duration**: 30-60 minutes (processes both image and text)

**Output Files**:
- `outputs/10fold_cv_multimodal_results.csv` - Detailed per-fold results
- `outputs/10fold_cv_multimodal_confusion_matrix.npy` - Average confusion matrix
- `outputs/10fold_cv_multimodal_summary.txt` - Summary statistics

### Step 4: Generate Comprehensive Report

After all three tests complete, run the report generator:

```bash
python scripts/32_generate_cv_report.py
```

**Expected Duration**: < 1 minute

**Output Files**:
- `outputs/10fold_cv_comparison_table.csv` - Side-by-side comparison
- `outputs/10fold_cv_statistical_tests.csv` - Statistical significance tests
- `outputs/10fold_cv_final_report.txt` - Comprehensive text report
- `outputs/cv_comparison_barplot.png` - Bar chart comparison
- `outputs/cv_boxplot_distribution.png` - Distribution across folds
- `outputs/cv_per_fold_comparison.png` - Line plot showing all folds
- `outputs/cv_confusion_heatmaps.png` - Confusion matrix visualizations

## Quick Run (All Tests)

To run all tests sequentially:

```bash
python scripts/29_ten_fold_cv_image_only.py && python scripts/30_ten_fold_cv_text_only.py && python scripts/31_ten_fold_cv_multimodal.py && python scripts/32_generate_cv_report.py
```

## Understanding the Results

### Metrics Reported

For each modality and each fold:
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: True positive rate (sensitivity)
- **F1-Score**: Harmonic mean of precision and recall

### Summary Statistics

- **Mean ± Std**: Average performance across 10 folds with standard deviation
- **Min-Max Range**: Range of performance across folds
- **95% CI**: 95% Confidence interval for accuracy

### Statistical Tests

The report includes paired t-tests to determine if differences between modalities are statistically significant (α = 0.05).

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: The scripts use the same pre-trained models, so this shouldn't be an issue. If it occurs, close other GPU-using applications.

### Issue: Image file not found
**Solution**: Verify that the `dataset/` directory contains all images referenced in `multimodal_plant_data.csv`.

### Issue: Model file not found
**Solution**: Ensure the following model files exist:
- `models/image/cnn_qnn_best.pt`
- `models/image/label_mapping.pkl`
- `models/text/best_hybrid_bert_model.pth`

### Issue: Different number of samples in folds
**Solution**: This is normal due to stratification. Each fold will have approximately 10% of the data while maintaining class distribution.

## Notes

- The models are NOT retrained during this process
- Stratified folding ensures class distribution is maintained in each fold
- Results may vary slightly from single train/test split due to different data distributions
- Processing time depends on dataset size (17,008 samples reported)
- GPU acceleration recommended but not required

## Contact

For questions or issues, refer to the project documentation or contact the NLP Project Team.
