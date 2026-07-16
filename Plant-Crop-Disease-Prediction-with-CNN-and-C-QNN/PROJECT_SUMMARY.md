# Project Summary - Plant Disease Detection
## Quantum-Classical Hybrid Deep Learning


## 📊 Dataset
- **Source:** PlantVillage
- **Total Images:** 19,622
- **Classes:** 15 plant disease categories
- **Split:** 70% Train (13,729) / 15% Val (2,936) / 15% Test (2,957)

---

## ✅ Completed Work

### 1. CNN Feature Extraction (Classical)
**Status:** ✅ COMPLETE

**Architecture:**
- ResNet18 (pretrained)
- Input: 224×224 RGB images
- Output: 512-dimensional features

**Files:**
- `features/CNN_resnet18_train.pt` (26.92 MB, 13,729 × 512)
- `features/CNN_resnet18_val.pt` (5.76 MB, 2,936 × 512)
- `features/CNN_resnet18_test.pt` (5.8 MB, 2,957 × 512)

**Visualizations:**
- `feature_maps/` - 11 CNN layer activation maps
- `visualizations/` - 5 feature analysis plots (t-SNE, PCA, correlation, etc.)

---

### 2. QCNN Feature Extraction (Quantum - Amplitude Encoding)
**Status:** ✅ COMPLETE

**Architecture:**
- Input: 224×224 RGB images (same as CNN)
- Classical reduction: 150,528 → 4,096 → 1,024
- Quantum encoding: Amplitude encoding
- Quantum circuit: 10 qubits, 2 conv layers
- Quantum output: 10 measurements
- Classical expansion: 10 → 256 → 512 features

**Files:**
- `features/QCNN_amplitude_train.pt` (26.92 MB, 13,729 × 512)
- `features/QCNN_amplitude_val.pt` (5.76 MB, 2,936 × 512)
- `features/QCNN_amplitude_test.pt` (5.8 MB, 2,957 × 512)

**Visualizations:**
- `qcnn_feature_maps/` - 5 amplitude encoding feature maps

**Performance:**
- Speed: ~4 samples/sec
- Total time: ~68 minutes for full dataset
- Memory: ~2.3 GB per model instance

---

## 📁 Project Structure

```
FINAL PROJECT/
├── Core Files
│   ├── config.py                      # Configuration system
│   ├── data_loader.py                 # Data loading utilities  
│   ├── quantum_layers.py              # Quantum circuit components
│   └── requirements.txt               # Dependencies
│
├── Feature Extraction
│   ├── extract_cnn_features.py        # ✅ CNN extraction (WORKING)
│   ├── extract_qcnn_features_full.py  # ✅ QCNN extraction (WORKING)
│   └── stage1_feature_extraction.py   # Original comprehensive version
│
├── Visualization
│   ├── visualize_feature_maps.py      # CNN layer activations
│   └── quick_visualize.py             # Feature analysis (t-SNE, PCA)
│
├── Training
│   ├── stage2_classifier_training.py  # ⏳ Classifier training
│   ├── main_two_stage.py              # Main orchestration
│   └── test_setup.py                  # Setup testing
│
├── Features (features/)
│   ├── CNN_resnet18_train.pt          # ✅ 13,729 × 512
│   ├── CNN_resnet18_val.pt            # ✅ 2,936 × 512
│   ├── CNN_resnet18_test.pt           # ✅ 2,957 × 512
│
│
├── Visualizations
│   ├── feature_maps/                  # 11 CNN layer activation maps
│   ├── qcnn_feature_maps/             # 5 QCNN feature maps
│   └── visualizations/                # 5 CNN analysis plots
│
├── Dataset
│   └── PlantVillage/                  # 19,622 images, 15 classes
│
└── Documentation
    ├── Readme/                        # Project docs
    └── FEATURE_EXTRACTION_STATUS.md   # Extraction status
```

---

## 🎯 Next Steps: Stage 2 - Classifier Training

### Classifier Combinations to Train (4 total):

1. **CNN + DNN** (Classical-Classical)
   - Features: CNN ResNet18 (512-dim)
   - Classifier: Deep Neural Network

2. **CNN + QNN** (Classical-Quantum)
   - Features: CNN ResNet18 (512-dim)
   - Classifier: Variational Quantum Classifier


### Expected Outputs:
- Training/validation curves
- Test accuracy for each combination
- Confusion matrices
- Comparison table
- Best model identification

---



## 📈 Key Metrics

### Feature Extraction Performance:
| Method | Time | Speed | Memory |
|--------|------|-------|--------|
| CNN (ResNet18) | ~35 min | 9.3 samples/sec | ~500 MB |


### Dataset Statistics:
- **Total:** 19,622 samples
- **Feature Dimension:** 512 (both CNN and QCNN)
- **Storage:** ~82 MB total (6 feature files)

---


All feature extraction complete. Project organized and cleaned. Ready to train and compare 4 quantum-classical hybrid model combinations.



**Last Updated:** October 11, 2025, 2:56 PM
