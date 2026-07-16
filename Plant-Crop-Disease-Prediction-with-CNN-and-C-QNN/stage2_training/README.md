# Quantum-Classical Hybrid Deep Learning for Plant Disease Detection
## Complete Implementation Guide

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Implementation Details](#implementation-details)
5. [Feature Extraction](#feature-extraction)
6. [Model Training](#model-training)
7. [Results](#results)
8. [Usage Guide](#usage-guide)
9. [File Structure](#file-structure)
10. [Technical Details](#technical-details)

---

## 🎯 Project Overview

This project implements a **Quantum-Classical Hybrid Deep Learning** system for plant disease classification using the PlantVillage dataset. It explores the quantum advantage by comparing pure classical, hybrid, and full quantum approaches.

### Key Innovations:
- ✅ **Two-Stage Architecture**: Feature extraction → Classification
- ✅ **Quantum Feature Extraction**: QCNN with amplitude encoding
- ✅ **Quantum Classification**: Variational Quantum Classifier (VQC)
- ✅ **Fair Comparison**: All models use same 512-dim features
- ✅ **Full Dataset**: 19,622 images (not toy examples!)

---

## 🏗️ Architecture

### Four Model Combinations:

```
1. CNN + DNN (Classical-Classical Baseline)
   └─ ResNet18 → 512 features → DNN Classifier
   
2. CNN + QNN (Classical-Quantum Hybrid)
   └─ ResNet18 → 512 features → Quantum Classifier
   
3. QCNN + DNN (Quantum-Classical Hybrid)
   └─ Quantum CNN → 512 features → DNN Classifier
   
4. QCNN + QNN (Full Quantum Pipeline)
   └─ Quantum CNN → 512 features → Quantum Classifier
```

### Two-Stage Process:

**Stage 1: Feature Extraction** (One-time, pre-compute)
- Extract features from ALL images
- Save to disk for reuse
- Avoids random initialization issues

**Stage 2: Classifier Training** (Fast, multiple experiments)
- Load pre-extracted features
- Train various classifiers
- Compare results

---

## 📊 Dataset

**PlantVillage Dataset**
- **Total Images**: 19,622
- **Classes**: 15 plant disease categories
- **Split**: 70% Train / 15% Val / 15% Test
- **Train**: 13,729 images
- **Val**: 2,936 images  
- **Test**: 2,957 images

**Disease Classes:**
```
1. Pepper bell - Bacterial spot
2. Pepper bell - healthy
3. Potato - Early blight
4. Potato - healthy
5. Potato - Late blight
6. Tomato - Target Spot
7. Tomato - Tomato mosaic virus
8. Tomato - Tomato Yellow Leaf Curl Virus
9. Tomato - Bacterial spot
10. Tomato - Early blight
11. Tomato - healthy
12. Tomato - Late blight
13. Tomato - Leaf Mold
14. Tomato - Septoria leaf spot
15. Tomato - Spider mites (Two-spotted spider mite)
```

---

## 🔬 Implementation Details

### 1. CNN Feature Extractor (Classical)

**Architecture:**
```
Input: 224×224 RGB images
  ↓
ResNet18 (pretrained on ImageNet)
  ├─ conv1: 64 channels (112×112)
  ├─ layer1: 64 channels (56×56)
  ├─ layer2: 128 channels (28×28)
  ├─ layer3: 256 channels (14×14)
  └─ layer4: 512 channels (7×7)
  ↓
Global Average Pooling
  ↓
Output: 512-dimensional features
```

**Key Features:**
- ✅ Pretrained weights (transfer learning)
- ✅ Fast: ~9 samples/sec
- ✅ Memory efficient: ~500 MB
- ✅ Time: ~35 minutes for full dataset

---

### 2. QCNN Feature Extractor (Quantum)

**Architecture:**
```
Input: 224×224 RGB images
  ↓
Classical Reduction Network
  ├─ Flatten: 150,528 features
  ├─ Linear: 150,528 → 4,096
  ├─ ReLU + Dropout(0.3)
  └─ Linear: 4,096 → 1,024
  ↓
Quantum Encoding (Amplitude)
  ├─ L2 normalization
  └─ Encode to 10 qubits (2^10 = 1,024 basis states)
  ↓
Quantum Convolutional Circuit
  ├─ 10 qubits
  ├─ 2 convolutional layers
  ├─ Pooling between layers
  ├─ Ring entanglement
  └─ Output: 10 measurements
  ↓
Classical Expansion Network
  ├─ Linear: 10 → 256
  ├─ ReLU + Dropout(0.2)
  └─ Linear: 256 → 512
  ↓
Output: 512-dimensional quantum features
```

**Quantum Encoding Types:**
1. **Amplitude Encoding** (Currently implemented)
   - Encodes features as quantum state amplitudes
   - Most information-dense encoding
   - Requires L2 normalization

2. **Angle Encoding** (Optional)
   - Encodes features as rotation angles
   - Maps to [-π, π] range

3. **Basis Encoding** (Optional)
   - Binary encoding in computational basis
   - Simplest but less expressive

**Key Features:**
- ✅ Full dataset support: 19,622 images
- ✅ Same output dimension as CNN: 512 features
- ⚠️  Slower: ~4 samples/sec (quantum simulation)
- ⚠️  Time: ~68 minutes for full dataset
- ⚠️  Memory: ~2.3 GB (large quantum circuit)

---

### 3. DNN Classifier (Classical)

**Architecture:**
```
Input: 512 features
  ↓
Linear: 512 → 256
ReLU
Dropout(0.3)
BatchNorm
  ↓
Linear: 256 → 128
ReLU
Dropout(0.3)
BatchNorm
  ↓
Linear: 128 → 15 (classes)
  ↓
Output: Class logits
```

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: Cross-Entropy
- Epochs: 50
- Batch size: 32
- Scheduler: ReduceLROnPlateau

---

### 4. QNN Classifier (Quantum)

**Architecture:**
```
Input: 512 features
  ↓
Classical Reduction
  ├─ Linear: 512 → 256
  ├─ ReLU + Dropout(0.2)
  └─ Linear: 256 → 256 (2^8 for 8 qubits)
  ↓
Variational Quantum Circuit
  ├─ 8 qubits
  ├─ 3 variational layers
  ├─ Amplitude encoding
  ├─ Parameterized rotations
  └─ Entangling gates
  ↓
Measurement + Post-processing
  └─ 15 output classes
  ↓
Output: Class logits
```

**Training Configuration:**
- Optimizer: Adam (lr=0.01)
- Loss: Cross-Entropy  
- Epochs: 30 (fewer due to quantum slowness)
- Batch size: 8 (smaller for quantum)
- Scheduler: ReduceLROnPlateau

---

## 📁 Feature Extraction

### CNN Feature Extraction

**Script:** `extract_cnn_features.py`

**Process:**
1. Load PlantVillage dataset
2. Create stratified 70/15/15 split
3. Extract ResNet18 features
4. Save to disk:
   - `features/CNN_resnet18_train.pt` (13,729 × 512)
   - `features/CNN_resnet18_val.pt` (2,936 × 512)
   - `features/CNN_resnet18_test.pt` (2,957 × 512)

**Run:**
```bash
python extract_cnn_features.py
```

**Output:**
- ✅ 3 feature files (total ~40 MB)
- ✅ ~35 minutes extraction time
- ✅ Feature visualizations (optional)

---

### QCNN Feature Extraction

**Script:** `extract_qcnn_features_full.py`

**Process:**
1. Load PlantVillage dataset
2. Create same stratified split as CNN
3. Build hybrid quantum-classical pipeline
4. Extract quantum features with amplitude encoding
5. Save to disk:
   - `features/QCNN_amplitude_train.pt` (13,729 × 512)
   - `features/QCNN_amplitude_val.pt` (2,936 × 512)
   - `features/QCNN_amplitude_test.pt` (2,957 × 512)

**Run:**
```bash
python extract_qcnn_features_full.py
```

**Output:**
- ✅ 3 feature files (total ~40 MB)
- ✅ ~68 minutes extraction time
- ✅ 5 quantum feature map visualizations
- ⚠️  Requires significant RAM (~3 GB)

**Feature Map Visualizations:**
- Shows quantum circuit outputs
- Displays pseudo activation maps
- Compares with original images

---

## 🚀 Model Training

All training scripts are in `stage2_training/` directory.

### 1. Train CNN + DNN

**Script:** `train_cnn_dnn.py`

**Command:**
```bash
cd stage2_training
python train_cnn_dnn.py
```

**What it does:**
- Loads CNN features
- Trains DNN classifier (512 → 256 → 128 → 15)
- Saves best model: `models/cnn_dnn_best.pt`
- Generates:
  - Confusion matrix
  - Training curves
  - Results JSON

**Expected time:** ~10-15 minutes

---

### 2. Train CNN + QNN

**Script:** `train_cnn_qnn.py`

**Command:**
```bash
cd stage2_training
python train_cnn_qnn.py
```

**What it does:**
- Loads CNN features
- Trains Quantum classifier (8 qubits, 3 layers)
- Saves best model: `models/cnn_qnn_best.pt`
- Generates same outputs as above

**Expected time:** ~45-60 minutes (quantum is slow)

---

### 3. Train QCNN + DNN

**Script:** `train_qcnn_dnn.py`

**Command:**
```bash
cd stage2_training
python train_qcnn_dnn.py
```

**What it does:**
- Loads QCNN amplitude features
- Trains DNN classifier
- Saves best model: `models/qcnn_dnn_best.pt`
- Generates same outputs

**Expected time:** ~10-15 minutes

---

### 4. Train QCNN + QNN

**Script:** `train_qcnn_qnn.py`

**Command:**
```bash
cd stage2_training
python train_qcnn_qnn.py
```

**What it does:**
- Loads QCNN amplitude features
- Trains Quantum classifier
- Saves best model: `models/qcnn_qnn_best.pt`
- Generates same outputs

**Expected time:** ~45-60 minutes

---

## 📊 Results Comparison

### Compare All Models

**Script:** `compare_models.py`

**Command:**
```bash
cd stage2_training
python compare_models.py
```

**What it does:**
- Loads all 4 model results
- Creates comparison table
- Generates comparison plots:
  - Test accuracy bars
  - Validation accuracy bars
  - Model complexity
  - Performance vs complexity scatter
- Identifies best models
- Calculates quantum advantage

**Output:**
- `results/model_comparison.csv`
- `results/model_comparison.png`
- Console summary

---

## 📂 File Structure

```
FINAL PROJECT/
│
├── Core Components/
│   ├── config.py                      # Configuration classes
│   ├── data_loader.py                 # Data loading utilities
│   ├── quantum_layers.py              # Quantum circuit components
│   └── requirements.txt               # Python dependencies
│
├── Feature Extraction/
│   ├── extract_cnn_features.py        # CNN extraction (WORKING)
│   ├── extract_qcnn_features_full.py  # QCNN extraction (WORKING)
│   ├── visualize_feature_maps.py      # CNN activation maps
│   └── quick_visualize.py             # Feature analysis
│
├── Stage 2 Training/ (stage2_training/)
│   ├── train_cnn_dnn.py               # Model 1: Classical baseline
│   ├── train_cnn_qnn.py               # Model 2: Quantum classifier
│   ├── train_qcnn_dnn.py              # Model 3: Quantum features
│   ├── train_qcnn_qnn.py              # Model 4: Full quantum
│   ├── compare_models.py              # Results comparison
│   │
│   ├── models/                        # Trained model checkpoints
│   │   ├── cnn_dnn_best.pt
│   │   ├── cnn_qnn_best.pt
│   │   ├── qcnn_dnn_best.pt
│   │   └── qcnn_qnn_best.pt
│   │
│   └── results/                       # Training results
│       ├── cnn_dnn_results.json
│       ├── cnn_qnn_results.json
│       ├── qcnn_dnn_results.json
│       ├── qcnn_qnn_results.json
│       ├── model_comparison.csv
│       ├── *_confusion_matrix.png
│       └── *_training_curves.png
│
├── Features/ (features/)
│   ├── CNN_resnet18_train.pt          # 13,729 × 512 (26.92 MB)
│   ├── CNN_resnet18_val.pt            # 2,936 × 512 (5.76 MB)
│   ├── CNN_resnet18_test.pt           # 2,957 × 512 (5.8 MB)
│   ├── QCNN_amplitude_train.pt        # 13,729 × 512 (26.92 MB)
│   ├── QCNN_amplitude_val.pt          # 2,936 × 512 (5.76 MB)
│   └── QCNN_amplitude_test.pt         # 2,957 × 512 (5.8 MB)
│
├── Visualizations/
│   ├── feature_maps/                  # 11 CNN layer activations
│   ├── qcnn_feature_maps/             # 5 QCNN quantum maps
│   └── visualizations/                # 5 analysis plots
│
├── Dataset/
│   └── PlantVillage/                  # 19,622 images, 15 classes
│
└── Documentation/
    ├── README.md                       # This file
    ├── PROJECT_SUMMARY.md              # Project summary
    └── Readme/                         # Additional docs
```

---

## 🔧 Technical Details

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
pennylane>=0.32.0
pennylane-lightning
numpy
matplotlib
seaborn
scikit-learn
tqdm
pandas
Pillow
```

**Install:**
```bash
pip install -r requirements.txt
```

---

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 10 GB
- OS: Windows/Linux/Mac

**Recommended for Quantum:**
- CPU: 8+ cores
- RAM: 16 GB
- Storage: 20 GB
- GPU: Optional (for CNN only)

---

### Performance Metrics

| Component | Time | Speed | Memory |
|-----------|------|-------|--------|
| CNN Extraction | 35 min | 9.3 samples/sec | 500 MB |
| QCNN Extraction | 68 min | 4.8 samples/sec | 2.3 GB |
| DNN Training (50 epochs) | 10-15 min | - | 1 GB |
| QNN Training (30 epochs) | 45-60 min | - | 2 GB |

---

### Quantum Circuit Details

**QCNN Architecture:**
```python
# 10 qubits encode 1,024 basis states
n_qubits = 10
input_dim = 2^10 = 1,024

# Convolutional layers
for layer in range(2):
    - Parameterized rotations (RX, RY, RZ)
    - Entangling gates (CNOT ring)
    - Pooling (measure & reduce)

# Output
measurements = 10 (one per qubit)
```

**VQC Architecture:**
```python
# 8 qubits for classification
n_qubits = 8
input_dim = 2^8 = 256

# Variational layers
for layer in range(3):
    - Amplitude encoding
    - Parameterized rotations
    - Entangling operations

# Readout
measurements → 15 class probabilities
```

---

## 📈 Usage Guide

### Quick Start (5 steps)

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Extract CNN features (one-time):**
```bash
python extract_cnn_features.py
```
⏱️ ~35 minutes

**3. Extract QCNN features (one-time):**
```bash
python extract_qcnn_features_full.py
```
⏱️ ~68 minutes

**4. Train all 4 models:**
```bash
cd stage2_training

# Classical baseline
python train_cnn_dnn.py

# Hybrid models
python train_cnn_qnn.py
python train_qcnn_dnn.py

# Full quantum
python train_qcnn_qnn.py
```
⏱️ ~2-3 hours total

**5. Compare results:**
```bash
python compare_models.py
```

---

### Advanced Usage

**Train single model:**
```bash
cd stage2_training
python train_cnn_dnn.py
```

**Modify hyperparameters:**
Edit the training script:
```python
# Change learning rate
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Change epochs
n_epochs = 30

# Change architecture
model = DNNClassifier(input_dim, n_classes, hidden_dims=[512, 256])
```

**Add new encoding:**
1. Edit `extract_qcnn_features_full.py`
2. Add angle/basis encoding loop
3. Run extraction
4. Train on new features

---

## 🎓 Understanding the Implementation

### Why Two-Stage Architecture?

**Problem with End-to-End:**
```python
# Bad: Random initialization every run
model = QCNN_QNN(...)
model.train()  # Features + classifier trained together
# Result: Can't compare fairly (different random seeds)
```

**Solution: Two-Stage:**
```python
# Good: Extract features once
features = extract_features()  # Fixed features
save(features)

# Train many classifiers
for classifier in [DNN, QNN, SVM, ...]:
    model = classifier(features)
    model.train()  # Fair comparison!
```

---

### Why 512 Features?

**Dimension Choice:**
- ResNet18 naturally outputs 512 features
- Good balance: not too small, not too large
- Standard dimension for transfer learning
- QCNN matches this for fair comparison

---

### Quantum Advantage Sources

**1. Feature Extraction (QCNN):**
- Encodes classical data in quantum superposition
- Exploits quantum interference patterns
- Learns features classical CNNs might miss

**2. Classification (QNN):**
- Quantum entanglement for complex relationships
- Exponential state space (2^n_qubits)
- Potentially better generalization

**3. Hybrid Benefits:**
- Best of both worlds
- Classical preprocessing + quantum processing
- Practical near-term approach

---

## 🔬 Research Questions Explored

1. **Does quantum feature extraction help?**
   - Compare CNN+DNN vs QCNN+DNN
   
2. **Does quantum classification help?**
   - Compare CNN+DNN vs CNN+QNN
   
3. **Is full quantum best?**
   - Compare all 4 models
   
4. **Quantum vs classical trade-offs?**
   - Accuracy vs training time
   - Performance vs model complexity

---

## 📝 Notes and Limitations

**Current Limitations:**
- Quantum simulation is slow (~100x slower than classical)
- Large memory requirements for quantum circuits
- Only amplitude encoding implemented (angle/basis optional)

**Future Improvements:**
- Add other quantum encodings
- Implement quantum data augmentation
- Try different quantum architectures
- Use real quantum hardware (IBM Quantum, etc.)

**Computational Notes:**
- Full dataset extraction takes ~2 hours total
- Training all models takes ~3 hours
- Total project time: ~5-6 hours
- Can be run on standard laptop (no GPU required)

---

## 🏆 Expected Results

**Typical Performance:**
```
Model            | Val Acc | Test Acc | Training Time
-----------------|---------|----------|---------------
CNN + DNN        | 94-96%  | 93-95%   | 10-15 min
CNN + QNN        | 90-94%  | 89-93%   | 45-60 min
QCNN + DNN       | 92-95%  | 91-94%   | 10-15 min
QCNN + QNN       | 88-92%  | 87-91%   | 45-60 min
```

**Note:** Quantum models might show advantage with:
- More training data
- Better hyperparameter tuning
- Real quantum hardware
- Different problem domains

---

## 📞 Support

For questions or issues:
1. Check documentation first
2. Review training logs
3. Verify feature extraction completed
4. Check file paths and dependencies

---

## 📜 Citation

If you use this code, please cite:
```
@misc{plant_disease_qml_2025,
  title={Quantum-Classical Hybrid Deep Learning for Plant Disease Detection},
  author={Your Name},
  year={2025},
  institution={Your Institution}
}
```

---

## 🙏 Acknowledgments

- **PlantVillage Dataset**: Plant disease image dataset
- **PyTorch**: Deep learning framework
- **PennyLane**: Quantum machine learning library
- **ResNet**: Pretrained CNN architecture

---

**Last Updated:** October 11, 2025

**Status:** ✅ Ready for Training

**Next Steps:** Run `train_cnn_dnn.py` to start!
