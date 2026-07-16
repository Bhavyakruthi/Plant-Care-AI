# Two-Stage Training Quick Reference

## Overview
The improved two-stage approach addresses the inefficiency of training models from scratch with random initialization. Instead:

**Stage 1**: Extract features once (expensive, especially quantum)  
**Stage 2**: Train many classifiers on pre-extracted features (fast)

This enables fair comparison and rapid experimentation.

---

## Quick Start

### Full Pipeline (Both Stages)
```bash
python main_two_stage.py
```

### Quick Test (Small Subset)
```bash
# Use only 100 samples per class for fast testing
python main_two_stage.py --subset-size 100
```

### Run Only Stage 1 (Feature Extraction)
```bash
python main_two_stage.py --skip-stage2
```

### Run Only Stage 2 (Classifier Training)
```bash
# If features already extracted
python main_two_stage.py --skip-stage1
```

---

## What Gets Extracted (Stage 1)

### CNN Features
- **Extractor**: ResNet18 (pretrained)
- **Input**: 224×224 RGB images
- **Output**: 512-dimensional features per sample
- **Speed**: Fast (~10ms per sample)
- **Files Created**:
  - `features/CNN_resnet18_train.pt`
  - `features/CNN_resnet18_val.pt`
  - `features/CNN_resnet18_test.pt`

### QCNN Features (3 Encodings)

#### Amplitude Encoding
- **Input**: 8×8 grayscale (64 pixels → 6 qubits)
- **Encoding**: Features as probability amplitudes
- **Output**: Low-dimensional quantum features
- **Speed**: Slow (~100ms per sample)
- **Best For**: High-dimensional data compression
- **Files**: `QCNN_amplitude_q6_*.pt`

#### Angle Encoding  
- **Input**: 8×8 grayscale (64 pixels → 6 qubits)
- **Encoding**: Features as rotation angles
- **Output**: Low-dimensional quantum features
- **Speed**: Slow (~100ms per sample)
- **Best For**: Continuous value preservation
- **Files**: `QCNN_angle_q6_*.pt`

#### Basis Encoding
- **Input**: 8×8 grayscale (64 pixels → 6 qubits)
- **Encoding**: Binary features as computational basis states
- **Output**: Low-dimensional quantum features
- **Speed**: Slow (~100ms per sample)
- **Best For**: Binary/categorical data
- **Files**: `QCNN_basis_q6_*.pt`

### Visualizations Created
- **t-SNE plots**: 2D projections colored by disease class
- **PCA plots**: Principal components with explained variance
- **Correlation heatmaps**: Feature correlations
- **Distribution histograms**: Feature value distributions

All saved to `./visualizations/`

---

## What Gets Trained (Stage 2)

### 8 Classifier Combinations

| # | Feature Extractor | Classifier | Description |
|---|------------------|------------|-------------|
| 1 | CNN_resnet18 | DNN | Classical baseline |
| 2 | CNN_resnet18 | QNN | Classical features → Quantum classifier |
| 3 | QCNN_amplitude_q6 | DNN | Quantum features → Classical classifier |
| 4 | QCNN_amplitude_q6 | QNN | Fully quantum (amplitude) |
| 5 | QCNN_angle_q6 | DNN | Quantum features → Classical classifier |
| 6 | QCNN_angle_q6 | QNN | Fully quantum (angle) |
| 7 | QCNN_basis_q6 | DNN | Quantum features → Classical classifier |
| 8 | QCNN_basis_q6 | QNN | Fully quantum (basis) |

### Classifiers

#### DNN Classifier
- **Architecture**: [input_dim → 256 → 128 → 15 classes]
- **Regularization**: Dropout (0.5), Batch Normalization
- **Speed**: Fast training (~1-2 min per model)

#### QNN Classifier
- **Architecture**: Projection layer + VQC (5 qubits, 3 layers)
- **Encoding**: Amplitude encoding for features
- **Speed**: Slower training (~5-10 min per model)

### Results Generated
- Training history (loss, accuracy per epoch)
- Best validation accuracy
- Test set metrics (accuracy, F1-score, precision, recall)
- Confusion matrices
- Comparison table (all 8 models ranked)

All saved to `./experiments_stage2/`

---

## Expected Timeline

### Full Dataset (~15,000 samples total)

**Stage 1: Feature Extraction**
- CNN (ResNet18): ~2-3 minutes
- QCNN Amplitude: ~25-30 minutes  
- QCNN Angle: ~25-30 minutes
- QCNN Basis: ~25-30 minutes
- Visualization: ~5 minutes
- **Total Stage 1**: ~80-100 minutes

**Stage 2: Classifier Training**
- Each DNN: ~1-2 minutes
- Each QNN: ~5-10 minutes
- 4 DNNs + 4 QNNs: ~30-50 minutes
- **Total Stage 2**: ~30-50 minutes

**Grand Total**: ~2-2.5 hours

### Subset (100 samples per class = 1,500 total)

**Stage 1**: ~10-15 minutes  
**Stage 2**: ~5-10 minutes  
**Total**: ~20-25 minutes

---

## Output Directory Structure

```
FINAL PROJECT/
├── features/                         # Pre-extracted features (Stage 1)
│   ├── CNN_resnet18_train.pt
│   ├── CNN_resnet18_val.pt
│   ├── CNN_resnet18_test.pt
│   ├── QCNN_amplitude_q6_train.pt
│   ├── QCNN_amplitude_q6_val.pt
│   ├── QCNN_amplitude_q6_test.pt
│   ├── QCNN_angle_q6_train.pt
│   ├── QCNN_angle_q6_val.pt
│   ├── QCNN_angle_q6_test.pt
│   ├── QCNN_basis_q6_train.pt
│   ├── QCNN_basis_q6_val.pt
│   └── QCNN_basis_q6_test.pt
│
├── experiments_stage1/               # Feature extraction logs
│   ├── config_stage1.json
│   ├── CNN_resnet18_stats.json
│   ├── QCNN_amplitude_q6_stats.json
│   ├── QCNN_angle_q6_stats.json
│   └── QCNN_basis_q6_stats.json
│
├── experiments_stage2/               # Classifier training results
│   ├── config_stage2.json
│   ├── comparison_results.txt        # ⭐ Main results table
│   ├── CNN_resnet18_DNN/
│   │   ├── training_history.png
│   │   ├── confusion_matrix.png
│   │   └── metrics.json
│   ├── CNN_resnet18_QNN/
│   │   └── ...
│   └── ... (8 model folders total)
│
└── visualizations/                   # Feature analysis plots
    ├── CNN_resnet18_tsne.png
    ├── CNN_resnet18_pca.png
    ├── CNN_resnet18_correlation.png
    ├── QCNN_amplitude_q6_tsne.png
    └── ... (4 extractors × 4 plots each)
```

---

## Interpreting Results

### Key Files to Check

1. **`experiments_stage2/comparison_results.txt`**
   - Ranked table of all 8 models
   - Shows which combination works best

2. **`visualizations/*_tsne.png`**
   - 2D projection of features colored by class
   - Good separation = features are discriminative

3. **`experiments_stage2/<model>/confusion_matrix.png`**
   - Shows which diseases are confused
   - Diagonal values should be high

4. **`experiments_stage2/<model>/metrics.json`**
   - Detailed accuracy, F1, precision, recall
   - Per-class metrics

### What to Look For

**Good Signs**:
- ✅ Test accuracy > 85% (excellent)
- ✅ Test accuracy > 75% (good)
- ✅ Validation and test accuracy similar (not overfitting)
- ✅ Clear class separation in t-SNE plots
- ✅ High diagonal in confusion matrix

**Warning Signs**:
- ⚠️ Test accuracy << validation accuracy (overfitting)
- ⚠️ Test accuracy << 70% (underfitting or bad hyperparameters)
- ⚠️ No separation in t-SNE plots (features not discriminative)
- ⚠️ Off-diagonal entries in confusion matrix (class confusion)

---

## Advantages of Two-Stage Approach

### 1. Efficiency
- Extract expensive quantum features **once**
- Train many classifiers quickly (minutes vs hours)
- Easy to experiment with new classifiers

### 2. Fair Comparison
- Same features for different classifiers
- Isolates classifier performance
- No bias from different random initializations

### 3. Modularity
- Add new feature extractors without retraining classifiers
- Add new classifiers without re-extracting features
- Mix and match any combination

### 4. Analysis
- Timing breakdown shows bottlenecks
- Feature statistics reveal quality
- Visualizations aid understanding

### 5. Reproducibility
- Features saved to disk (deterministic)
- Same features every time
- Easy to share and compare

---

## Hyperparameter Tuning

After running the baseline, tune hyperparameters to improve performance.

### Priority 1: Learning Rate
```bash
# Test different learning rates
python stage2_classifier_training.py --learning-rate 0.001
python stage2_classifier_training.py --learning-rate 0.0005
python stage2_classifier_training.py --learning-rate 0.0001
```

### Priority 2: Quantum Encoding
Already tested! Stage 1 extracts all three encodings, Stage 2 trains on all.

### Priority 3: Circuit Depth
Edit `config.py`:
```python
qcnn_depth = 4  # Increase from 3
vqc_depth = 5   # Increase from 3
```
Then re-run Stage 1.

See `HYPERPARAMETER_GUIDE.md` for complete tuning strategies.

---

## Troubleshooting

### "Dataset not found"
- Check `config.py` → `DataConfig.data_root`
- Ensure path points to `PlantVillage/` folder
- Verify 15 disease class folders exist

### "Feature files not found" (Stage 2)
- Run Stage 1 first: `python main_two_stage.py --skip-stage2`
- Or check that `./features/*.pt` files exist

### "CUDA out of memory"
- Reduce batch size in `config.py`: `batch_size = 16`
- Or use CPU: `python main_two_stage.py --device cpu`

### "Quantum circuit too slow"
- Normal! ~100ms per sample for quantum
- Use `--subset-size 100` for testing
- Stage 1 extracts once, then reuse forever

### "Poor accuracy (< 50%)"
- Check if using data augmentation (should be False for quantum)
- Verify learning rate (try 0.001, 0.0005, 0.0001)
- Ensure dataset has enough samples per class

---

## Next Steps After Running

1. **Review Results**
   - Check `comparison_results.txt` for best model
   - Look at confusion matrices
   - Analyze t-SNE visualizations

2. **Hyperparameter Tuning**
   - Follow `HYPERPARAMETER_GUIDE.md`
   - Focus on learning rate first
   - Try different circuit depths

3. **Analysis**
   - Which encoding works best? (amplitude/angle/basis)
   - Does quantum help? (compare QCNN vs CNN)
   - Which classifier is better? (DNN vs QNN)

4. **Production**
   - Train final model with best configuration
   - Save model weights
   - Deploy for inference

---

## Key Insights

### Why Two-Stage?
Original approach: Train 4 models from scratch with random weights
- ❌ Unfair comparison (different random initializations)
- ❌ Wasteful (extract features 4 times)
- ❌ Slow (quantum extraction repeated)

Two-stage approach: Extract once, train many
- ✅ Fair comparison (same features)
- ✅ Efficient (extract once, reuse forever)
- ✅ Fast (Stage 2 takes minutes)

### Which Encoding to Use?
- **Amplitude**: Best for high-dimensional → low-dimensional (CNN → QNN)
- **Angle**: Best for continuous features (QCNN features)
- **Basis**: Educational, rarely optimal

Test all three and pick best!

### Quantum vs Classical?
- **Classical (CNN+DNN)**: Fast, accurate baseline
- **Hybrid (CNN+QNN or QCNN+DNN)**: Experimental, may improve on specific tasks
- **Quantum (QCNN+QNN)**: Research, explores quantum advantage

Goal: See if quantum provides benefit on this dataset!

---

## Questions?

- Configuration: See `config.py`
- Hyperparameters: See `HYPERPARAMETER_GUIDE.md`
- Code details: See `stage1_feature_extraction.py` and `stage2_classifier_training.py`
- General info: See `README.md`
