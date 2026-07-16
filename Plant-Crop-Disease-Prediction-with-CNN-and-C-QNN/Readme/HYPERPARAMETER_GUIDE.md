# Hyperparameter Analysis Guide
## Complete Guide to Tuning Quantum-Classical Hybrid Models

This document provides a comprehensive analysis of all hyperparameters in the plant disease detection system, their impact on model performance, and recommended tuning strategies.

---

## Table of Contents
1. [Overview](#overview)
2. [Critical Hyperparameters](#critical-hyperparameters)
3. [Quantum-Specific Parameters](#quantum-specific-parameters)
4. [Classical Model Parameters](#classical-model-parameters)
5. [Training Parameters](#training-parameters)
6. [Tuning Priorities](#tuning-priorities)
7. [Recommended Tuning Ranges](#recommended-tuning-ranges)
8. [HPO Strategies](#hpo-strategies)

---

## Overview

### Hyperparameter Categories
Our system has **5 main categories** of hyperparameters:

1. **Training Parameters** - Learning rate, batch size, epochs
2. **Quantum Circuit Parameters** - Encoding type, qubit count, circuit depth
3. **Classical Network Parameters** - Architecture depth, hidden dimensions
4. **Data Parameters** - Image size, augmentation strength
5. **Optimization Parameters** - Optimizer choice, regularization

### Impact Levels
- 🔴 **CRITICAL**: Major impact on performance (tune first)
- 🟡 **IMPORTANT**: Moderate impact (tune second)  
- 🟢 **MINOR**: Small impact (tune last or use defaults)

---

## Critical Hyperparameters

### 1. Learning Rate (🔴 CRITICAL)
**Parameter**: `learning_rate`  
**Location**: `config.py` - DataConfig  
**Default**: 0.001  
**Impact**: Controls convergence speed and final accuracy

**Analysis**:
- **Too High** (>0.01): Model diverges, loss explodes, no learning
- **Too Low** (<0.0001): Extremely slow convergence, gets stuck in local minima
- **Optimal Range**: 0.0001 - 0.01 (depends on model complexity)

**Recommended Values**:
- **Classical CNN models**: 0.001 - 0.005 (ResNet, MobileNet)
- **Quantum models**: 0.0001 - 0.001 (slower, more sensitive)
- **Hybrid models**: 0.0005 - 0.002 (balance classical and quantum)

**Tuning Strategy**:
```python
# Grid search (coarse to fine)
learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]

# Or use learning rate schedulers
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)
```

**Why It Matters**:
- Single most important hyperparameter
- Affects both convergence speed and final performance
- Different for classical vs quantum components (quantum is more sensitive)

---

### 2. Quantum Encoding Type (🔴 CRITICAL)
**Parameter**: `encoding_type`  
**Location**: `config.py` - QCNNConfig, QNNConfig  
**Choices**: `["amplitude", "angle", "basis"]`  
**Default**: "amplitude"  

**Analysis**:

#### **Amplitude Encoding**
- **Description**: Encodes features as probability amplitudes
- **Qubit Efficiency**: Best (log₂(features) qubits needed)
- **Requirements**: Features must be L2-normalized, length = 2^n
- **Best For**: High-dimensional features (CNN outputs)
- **Drawbacks**: Normalization can lose information

```python
# Example: 64 features → 6 qubits (2^6 = 64)
features = [0.1, 0.2, ..., 0.5]  # 64 values
normalized = features / np.linalg.norm(features)
# Encodes as: |ψ⟩ = Σᵢ normalized[i]|i⟩
```

#### **Angle Encoding**
- **Description**: Encodes features as rotation angles
- **Qubit Efficiency**: Poor (1 feature per qubit)
- **Requirements**: Features in range [-π, π]
- **Best For**: Low-dimensional features, geometric data
- **Drawbacks**: Limited capacity (max features = qubits)

```python
# Example: 6 features → 6 qubits
features = [0.5, -0.3, 1.2, ...]  # 6 values
# Encodes as: RY(features[0])|0⟩ ⊗ RY(features[1])|0⟩ ⊗ ...
```

#### **Basis Encoding**
- **Description**: Encodes binary features into computational basis
- **Qubit Efficiency**: Poor (1 binary feature per qubit)
- **Requirements**: Features binarized (0 or 1)
- **Best For**: Categorical/binary data
- **Drawbacks**: Loss of continuous information

```python
# Example: 6 features → 6 qubits
features = [0.7, 0.2, 0.9, 0.1, 0.8, 0.3]
binary = (features > median(features))  # [1, 0, 1, 0, 1, 0]
# Encodes as: X|0⟩ ⊗ |0⟩ ⊗ X|0⟩ ⊗ ... = |101010⟩
```

**Recommended Usage**:
| Feature Type | Recommended Encoding | Reason |
|-------------|---------------------|---------|
| CNN features (512-dim) | Amplitude | Efficient for high-dim |
| QCNN features (6-8 dim) | Angle | Preserves continuous values |
| Binary masks | Basis | Natural for binary data |

**Tuning Strategy**:
- Test ALL three encodings
- Amplitude usually best for CNN→QNN
- Angle often best for QCNN→QNN
- Basis rarely optimal but educational

---

### 3. Number of Qubits (🔴 CRITICAL)
**Parameter**: `n_qubits`  
**Location**: `config.py` - QCNNConfig, QNNConfig  
**Default**: 6  
**Range**: 4 - 10  

**Analysis**:

**Computational Cost**:
- **4 qubits**: Fast (2^4 = 16-dimensional state)
- **6 qubits**: Moderate (2^6 = 64-dimensional state)
- **8 qubits**: Slow (2^8 = 256-dimensional state)
- **10 qubits**: Very slow (2^10 = 1024-dimensional state)

**Expressivity**:
- More qubits → More expressive quantum circuits
- But diminishing returns after certain point
- Too many qubits → Overfitting, slow training

**Encoding Constraints**:
- **Amplitude**: Must have 2^n features (e.g., 6 qubits = 64 features)
- **Angle**: Can use up to n features (e.g., 6 qubits = up to 6 features)
- **Basis**: Can encode up to n binary features

**Recommended Values**:
| Task | Qubits | Justification |
|------|--------|---------------|
| QCNN feature extraction | 6 | Balances expressivity and speed |
| QNN classification (15 classes) | 5-6 | log₂(15) ≈ 4, add buffer |
| Small dataset experiments | 4 | Faster iteration |
| Final production model | 8 | Maximum accuracy |

**Tuning Strategy**:
```python
# Start small, increase gradually
qubit_range = [4, 5, 6, 7, 8]

# Monitor training time vs accuracy
# Stop when accuracy plateaus
```

**Warning**: Each additional qubit **doubles** simulation time!

---

## Quantum-Specific Parameters

### 4. Circuit Depth / Layers (🟡 IMPORTANT)
**Parameter**: `n_layers`  
**Location**: `config.py` - QCNNConfig.qcnn_depth, QNNConfig.vqc_depth  
**Default**: 3  
**Range**: 1 - 5  

**Analysis**:
- **Depth = 1**: Shallow circuit, underfitting risk
- **Depth = 2-3**: Sweet spot for most tasks
- **Depth = 4-5**: Deep circuit, overfitting risk, slow

**Impact**:
- More layers → More trainable parameters
- More layers → Longer circuit execution time
- More layers → Better feature abstraction

**Recommended Values**:
| Model Component | Layers | Reason |
|----------------|--------|---------|
| QCNN (conv layers) | 2-3 | Balance depth and speed |
| QCNN (pooling layers) | 2 | Matches conv layers |
| VQC (classification) | 3-4 | Needs more expressivity |

**Tuning Strategy**:
- Start with depth=2, increase if underfitting
- Use early stopping to prevent overfitting
- Monitor validation loss carefully

---

### 5. Entanglement Pattern (🟡 IMPORTANT)
**Parameter**: `entanglement_type`  
**Location**: `config.py` - QCNNConfig, QNNConfig  
**Choices**: `["linear", "circular", "full"]`  
**Default**: "linear"  

**Analysis**:

#### **Linear Entanglement**
```
q0 ─■─────────
    │
q1 ─■─■───────
      │
q2 ───■─■─────
        │
q3 ─────■─────
```
- **Connectivity**: Each qubit entangles with next
- **Gates**: n-1 CNOT gates
- **Speed**: Fastest
- **Best For**: Sequential features (images)

#### **Circular Entanglement**
```
q0 ─■─────────■
    │         │
q1 ─■─■───────┘
      │
q2 ───■─■─────
        │
q3 ─────■─────
```
- **Connectivity**: Linear + wrap around
- **Gates**: n CNOT gates
- **Speed**: Fast
- **Best For**: Periodic features

#### **Full Entanglement**
```
q0 ─■─■─■─────
    │ │ │
q1 ─■─│─│─■─■─
      │ │ │ │
q2 ───■─│─■─│─
        │   │
q3 ─────■───■─
```
- **Connectivity**: All pairs entangled
- **Gates**: n(n-1)/2 CNOT gates
- **Speed**: Slowest
- **Best For**: Highly correlated features

**Recommended Usage**:
- **Linear**: Default choice (fast, effective)
- **Circular**: When spatial periodicity exists
- **Full**: When maximum entanglement needed (rare)

**Tuning Strategy**:
- Start with linear (fastest)
- Try circular if accuracy plateaus
- Use full only for final model (slow)

---

### 6. Measurement Type (🟢 MINOR)
**Parameter**: `measurement_type`  
**Location**: `config.py` - QCNNConfig, QNNConfig  
**Choices**: `["expectation", "sample"]`  
**Default**: "expectation"  

**Analysis**:
- **Expectation**: Measure ⟨Z⟩ for each qubit (deterministic)
- **Sample**: Sample from probability distribution (stochastic)

**Recommended**: Use expectation for training (stable gradients)

---

## Classical Model Parameters

### 7. CNN Backbone (🟡 IMPORTANT)
**Parameter**: `cnn_backbone`  
**Location**: `config.py` - CNNConfig  
**Choices**: `["resnet18", "resnet34", "mobilenetv2"]`  
**Default**: "resnet18"  

**Analysis**:

| Backbone | Parameters | Speed | Accuracy | Best For |
|----------|-----------|-------|----------|----------|
| ResNet18 | 11M | Fast | Good | Baseline, rapid prototyping |
| ResNet34 | 21M | Moderate | Better | Production models |
| MobileNetV2 | 3.5M | Fastest | Lower | Mobile/edge deployment |

**Tuning Strategy**:
- Start with ResNet18 (fastest iteration)
- Upgrade to ResNet34 if accuracy insufficient
- Use MobileNetV2 only for deployment constraints

---

### 8. DNN Hidden Layers (🟢 MINOR)
**Parameter**: `dnn_hidden_dims`  
**Location**: `config.py` - DNNConfig  
**Default**: `[256, 128]`  
**Range**: `[64] to [512, 256, 128]`  

**Analysis**:
- **Shallow** [128]: Fast, may underfit
- **Medium** [256, 128]: Balanced (default)
- **Deep** [512, 256, 128]: Slow, may overfit

**Tuning Strategy**:
- Use default [256, 128] for most tasks
- Reduce if small dataset (< 5000 samples)
- Increase if very complex patterns

---

## Training Parameters

### 9. Batch Size (🟡 IMPORTANT)
**Parameter**: `batch_size`  
**Location**: `config.py` - DataConfig  
**Default**: 32  
**Range**: 16 - 128  

**Analysis**:

| Batch Size | Memory | Speed | Gradient Quality | Best For |
|-----------|--------|-------|-----------------|----------|
| 16 | Low | Slow | Noisy | Small GPU, regularization |
| 32 | Medium | Medium | Balanced | Default choice |
| 64 | High | Fast | Stable | Large datasets |
| 128 | Very High | Fastest | Very stable | Very large datasets |

**Special Considerations**:
- **Quantum models**: Always batch_size=1 (circuits can't be batched)
- **Classical models**: Use 32-64 for best balance
- **Two-stage approach**: Stage 1 uses batch_size=1, Stage 2 can use larger batches

**Tuning Strategy**:
```python
# Use largest batch that fits in GPU memory
batch_sizes = [16, 32, 64]  # Test to find max

# Adjust learning rate with batch size
# Rule of thumb: LR ∝ sqrt(batch_size)
lr_16 = 0.001
lr_32 = 0.001 * sqrt(32/16) = 0.00141
lr_64 = 0.001 * sqrt(64/16) = 0.002
```

---

### 10. Number of Epochs (🟡 IMPORTANT)
**Parameter**: `epochs`  
**Location**: `config.py` - DataConfig  
**Default**: 50  
**Range**: 20 - 100  

**Analysis**:
- **20 epochs**: Quick experiments, may underfit
- **50 epochs**: Standard training
- **100 epochs**: Deep training, use early stopping

**Recommended with Early Stopping**:
```python
early_stopping_patience = 10  # Stop if no improvement for 10 epochs
max_epochs = 100  # Upper limit
```

**Tuning Strategy**:
- Set high (100), rely on early stopping
- Monitor training vs validation loss
- Stop when validation loss increases

---

### 11. Optimizer Choice (🟡 IMPORTANT)
**Parameter**: `optimizer`  
**Location**: `config.py` - DataConfig  
**Choices**: `["adam", "sgd", "adamw"]`  
**Default**: "adam"  

**Analysis**:

| Optimizer | Speed | Stability | Best For |
|-----------|-------|-----------|----------|
| Adam | Fast | Good | General purpose, quantum models |
| SGD | Slow | Requires tuning | Classical CNNs (with momentum) |
| AdamW | Fast | Best | Large models, prevents overfitting |

**Recommended Settings**:
```python
# Adam (default)
optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999))

# SGD (for CNNs)
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)

# AdamW (best for large models)
optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.01)
```

---

### 12. Weight Decay (🟢 MINOR)
**Parameter**: `weight_decay`  
**Location**: `config.py` - DataConfig  
**Default**: 0.0001  
**Range**: 0 - 0.01  

**Analysis**:
- L2 regularization to prevent overfitting
- **0**: No regularization
- **0.0001**: Light regularization (default)
- **0.001 - 0.01**: Strong regularization

**Tuning Strategy**:
- Use default 0.0001 initially
- Increase if overfitting (train acc >> val acc)
- Decrease if underfitting

---

## Data Parameters

### 13. Image Size (🟢 MINOR)
**Parameter**: `classical_image_size`, `quantum_image_size`  
**Location**: `config.py` - DataConfig  
**Default**: (224, 224) for CNN, (8, 8) for QCNN  

**Analysis**:
- **Classical**: 224×224 is standard (pretrained model requirement)
- **Quantum**: 8×8 → 64 features → 6 qubits (power of 2)

**Tuning Strategy**:
- Keep classical at 224×224 (don't change)
- Test quantum at 4×4 (16 features, 4 qubits) or 8×8

---

### 14. Data Augmentation (🟡 IMPORTANT)
**Parameter**: `use_augmentation`  
**Location**: `config.py` - DataConfig  
**Default**: True (for classical), False (for quantum)  

**Critical Research Finding**:
> **Do NOT use augmentation for quantum models!**
> 
> Research shows quantum circuits need consistent inputs. Augmentation (random flips, rotations) introduces noise that degrades quantum feature learning.

**Recommended**:
- **Classical CNN training**: use_augmentation=True
- **QCNN training**: use_augmentation=False
- **Two-stage feature extraction**: use_augmentation=False

---

## Tuning Priorities

### Priority 1: Critical Parameters (Tune First)
1. **Learning Rate** (most important)
2. **Quantum Encoding Type** (amplitude vs angle vs basis)
3. **Number of Qubits** (4, 6, or 8)

**Why**: These have the largest impact on model performance. Getting these right is essential.

**Approach**: Grid search over these 3 parameters
```python
search_space = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'encoding_type': ['amplitude', 'angle', 'basis'],
    'n_qubits': [4, 6, 8]
}
# Total: 3 × 3 × 3 = 27 combinations
```

---

### Priority 2: Important Parameters (Tune Second)
4. **Circuit Depth** (n_layers)
5. **Batch Size**
6. **CNN Backbone** (resnet18 vs resnet34)
7. **Entanglement Pattern**

**Why**: Moderate impact, can improve performance by 5-10%.

**Approach**: Refine best model from Priority 1
```python
# After finding best LR, encoding, qubits
refine_space = {
    'n_layers': [2, 3, 4],
    'batch_size': [16, 32, 64],
    'entanglement': ['linear', 'circular']
}
```

---

### Priority 3: Minor Parameters (Tune Last)
8. **DNN Hidden Dimensions**
9. **Weight Decay**
10. **Dropout Rate**

**Why**: Small impact, fine-tuning only.

**Approach**: Use defaults or simple 2-3 value tests

---

## Recommended Tuning Ranges

### Complete HPO Search Space
```python
HPO_SEARCH_SPACE = {
    # Priority 1: Critical
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
    'qcnn_encoding': ['amplitude', 'angle', 'basis'],
    'qcnn_n_qubits': [4, 6, 8],
    
    # Priority 2: Important
    'qcnn_depth': [2, 3, 4],
    'vqc_depth': [3, 4, 5],
    'batch_size': [16, 32, 64],
    'cnn_backbone': ['resnet18', 'resnet34'],
    'entanglement': ['linear', 'circular'],
    
    # Priority 3: Minor
    'dnn_hidden_dims': [[128], [256, 128], [512, 256, 128]],
    'weight_decay': [0, 0.0001, 0.001],
    'dropout': [0.3, 0.5],
}
```

---

## HPO Strategies

### Strategy 1: Grid Search (Systematic)
**Best For**: Small search spaces (< 100 combinations)

```python
from itertools import product

# Define grid
learning_rates = [0.001, 0.0005, 0.0001]
encodings = ['amplitude', 'angle', 'basis']
qubits = [4, 6, 8]

# Generate all combinations
for lr, enc, q in product(learning_rates, encodings, qubits):
    config = {
        'learning_rate': lr,
        'encoding_type': enc,
        'n_qubits': q
    }
    accuracy = train_and_evaluate(config)
    results.append((config, accuracy))

# Select best
best_config = max(results, key=lambda x: x[1])
```

**Pros**: Explores all combinations systematically  
**Cons**: Expensive (3×3×3 = 27 runs)

---

### Strategy 2: Random Search (Efficient)
**Best For**: Large search spaces (> 100 combinations)

```python
import random

# Define ranges
search_space = {
    'learning_rate': (0.0001, 0.01, 'log'),  # Log scale
    'n_qubits': (4, 8, 'int'),
    'n_layers': (2, 5, 'int'),
}

# Random sampling
n_trials = 50
for _ in range(n_trials):
    config = {
        'learning_rate': 10 ** random.uniform(-4, -2),  # Log scale
        'n_qubits': random.randint(4, 8),
        'n_layers': random.randint(2, 5),
    }
    accuracy = train_and_evaluate(config)
    results.append((config, accuracy))
```

**Pros**: More efficient than grid search  
**Cons**: May miss optimal combinations

---

### Strategy 3: Bayesian Optimization (Intelligent)
**Best For**: Expensive evaluations (quantum models)

```python
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

# Define search space
space = [
    Real(0.0001, 0.01, name='learning_rate', prior='log-uniform'),
    Integer(4, 8, name='n_qubits'),
    Categorical(['amplitude', 'angle', 'basis'], name='encoding'),
]

# Objective function
def objective(params):
    lr, qubits, encoding = params
    config = {'learning_rate': lr, 'n_qubits': qubits, 'encoding': encoding}
    accuracy = train_and_evaluate(config)
    return -accuracy  # Minimize (so negate)

# Optimize
result = gp_minimize(objective, space, n_calls=30, random_state=42)
best_params = result.x
```

**Pros**: Intelligent sampling, fewer evaluations needed  
**Cons**: Requires additional library (scikit-optimize)

---

### Strategy 4: Two-Stage Tuning (Recommended)
**Best For**: Our two-stage architecture

```python
# Stage 1: Tune feature extractors
for extractor in ['CNN_resnet18', 'QCNN_amplitude', 'QCNN_angle', 'QCNN_basis']:
    # Tune extraction hyperparameters
    best_extractor_config = tune_feature_extractor(extractor)
    
    # Extract features with best config
    features = extract_features(extractor, best_extractor_config)
    save_features(features, f'features/{extractor}.pt')

# Stage 2: Tune classifiers (fast, features already extracted)
for features_file in ['CNN_resnet18.pt', 'QCNN_amplitude.pt', ...]:
    features = load_features(features_file)
    
    for classifier in ['DNN', 'QNN']:
        # Quick grid search (features are pre-extracted, so fast)
        best_classifier_config = tune_classifier(classifier, features)
        
        # Train final model
        model = train_final(classifier, features, best_classifier_config)
```

**Advantages**:
1. **Separate concerns**: Tune extractors and classifiers independently
2. **Fast iteration**: Stage 2 is fast (no feature extraction)
3. **Comprehensive**: Test all combinations efficiently

---

## Summary Table

| Hyperparameter | Priority | Impact | Recommended Range | Default |
|---------------|----------|--------|-------------------|---------|
| Learning Rate | 🔴 Critical | Very High | 0.0001 - 0.01 | 0.001 |
| Encoding Type | 🔴 Critical | Very High | amplitude, angle, basis | amplitude |
| Number of Qubits | 🔴 Critical | Very High | 4 - 8 | 6 |
| Circuit Depth | 🟡 Important | High | 2 - 5 | 3 |
| Batch Size | 🟡 Important | Medium | 16 - 64 | 32 |
| CNN Backbone | 🟡 Important | Medium | resnet18, resnet34 | resnet18 |
| Entanglement | 🟡 Important | Medium | linear, circular | linear |
| Optimizer | 🟡 Important | Medium | adam, adamw | adam |
| Epochs | 🟡 Important | Medium | 20 - 100 | 50 |
| Augmentation | 🟡 Important | Medium | True/False | True (classical only) |
| DNN Layers | 🟢 Minor | Low | [128] to [512,256,128] | [256, 128] |
| Weight Decay | 🟢 Minor | Low | 0 - 0.01 | 0.0001 |
| Image Size | 🟢 Minor | Low | Fixed | (224,224) / (8,8) |

---

## Practical Tuning Workflow

### Step 1: Quick Baseline (1-2 hours)
Run with all defaults to establish baseline performance.

```bash
python main.py --config default
```

### Step 2: Critical Parameter Search (4-8 hours)
Grid search over learning rate, encoding, qubits.

```python
# 27 combinations × 30min = ~13.5 hours
python tune_critical_params.py
```

### Step 3: Refinement (2-4 hours)
Tune important parameters around best critical config.

```python
# Best from Step 2 + tune depth, batch size, etc.
python tune_refine.py --base_config best_critical.json
```

### Step 4: Final Optimization (1-2 hours)
Fine-tune minor parameters and train final model.

```python
python train_final.py --config best_refined.json --epochs 100
```

**Total Time**: ~20-25 hours for complete hyperparameter optimization

---

## Key Takeaways

1. **Start with learning rate** - Most important single parameter
2. **Test all three encodings** - Amplitude, angle, and basis behave differently
3. **Balance qubits and speed** - 6 qubits is sweet spot for most tasks
4. **Use two-stage approach** - Extract features once, tune classifiers quickly
5. **Don't augment quantum data** - Critical for quantum model performance
6. **Use early stopping** - Set high epoch limit, let model stop itself
7. **Monitor validation loss** - Overfitting is common with quantum models
8. **Start simple, add complexity** - Begin with shallow circuits, increase depth if needed

---

## Additional Resources

- `config.py`: Complete hyperparameter definitions
- `stage1_feature_extraction.py`: Feature extraction with timing analysis
- `stage2_classifier_training.py`: Classifier training framework
- `HPO_SEARCH_SPACE` in config.py: Ready-to-use search space dictionary

**Questions?** Check the code comments in `config.py` for detailed explanations of each parameter.
