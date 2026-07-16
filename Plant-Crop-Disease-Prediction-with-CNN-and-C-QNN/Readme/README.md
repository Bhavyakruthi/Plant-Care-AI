# Plant Disease Detection with Quantum-Classical Hybrid Models

A comprehensive implementation of four quantum-classical hybrid architectures for plant disease prediction using the PlantVillage dataset. This project compares quantum machine learning approaches with classical baselines.

## 🎯 Project Overview

This implementation explores four different architectures:

1. **Module 1: QCNN + DNN** - Quantum Convolutional Neural Network for feature extraction + Classical Dense Network for classification
2. **Module 2: CNN + QNN** - Classical CNN for feature extraction + Variational Quantum Classifier for classification
3. **Module 3: QCNN + QNN** - Fully quantum architecture (end-to-end quantum processing)
4. **Module 4: CNN + DNN** - Classical baseline using ResNet + Dense Network

## 📊 Dataset

The project uses the **PlantVillage dataset** containing 15 disease categories:
- Pepper Bell Bacterial Spot
- Pepper Bell Healthy
- Potato Early Blight
- Potato Healthy
- Potato Late Blight
- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Healthy
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Tomato Spider Mites
- Tomato Target Spot
- Tomato Mosaic Virus
- Tomato Yellow Leaf Curl Virus

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### Basic Usage

```bash
# Train all four modules
python main.py --module all

# Train a specific module
python main.py --module module1

# Run with reduced training data (for testing)
python main.py --module module4 --data-fraction 0.1

# Run ablation study
python main.py --ablation

# Use CPU instead of GPU
python main.py --module all --device cpu
```

## 📁 Project Structure

```
FINAL PROJECT/
├── PlantVillage/              # Dataset directory
│   ├── Pepper__bell___Bacterial_spot/
│   ├── Pepper__bell___healthy/
│   └── ... (other disease categories)
├── config.py                  # Configuration management
├── data_loader.py            # Data loading and preprocessing
├── quantum_layers.py         # Quantum circuit components
├── model_qcnn_dnn.py        # Module 1: QCNN + DNN
├── model_cnn_qnn.py         # Module 2: CNN + QNN
├── model_qcnn_qnn.py        # Module 3: QCNN + QNN (fully quantum)
├── model_cnn_dnn.py         # Module 4: CNN + DNN (classical baseline)
├── train.py                 # Unified training framework
├── evaluate.py              # Evaluation metrics and utilities
├── main.py                  # Main execution script
├── visualize.py             # Visualization tools
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## ⚙️ Configuration

### Key Configuration Parameters

Edit `config.py` to customize:

**Data Configuration:**
- `classical_image_size`: (224, 224) for CNN models
- `quantum_image_size`: (8, 8) for quantum models (must be power of 2!)
- `batch_size_classical`: 32
- `batch_size_quantum`: 8 (smaller for quantum simulators)

**QCNN Configuration:**
- `n_qubits`: 6 (for 8×8 images = 64 features)
- `conv_layers`: 2
- `encoding_type`: "amplitude" or "angle"
- `entanglement_pattern`: "ring", "full", or "nearest"

**QNN/VQC Configuration:**
- `n_qubits`: 5
- `n_layers`: 3
- `learning_rate`: 0.005

**CNN Configuration:**
- `architecture`: "resnet18", "resnet34", or "mobilenetv2"
- `pretrained`: True (use ImageNet weights)
- `learning_rate`: 0.001

**Training Configuration:**
- `num_epochs`: 100
- `early_stopping_patience`: 15

## 🔬 Key Implementation Details

### Critical Quantum Considerations

1. **No Data Augmentation for Quantum Models**: Based on research findings, data augmentation does NOT improve QCNN performance and may degrade it. Our implementation applies augmentation only to classical models.

2. **Power-of-2 Image Dimensions**: Quantum amplitude encoding requires image dimensions that are powers of 2 (8×8, 16×16, etc.)

3. **Amplitude Encoding Normalization**: Images must be L2-normalized for amplitude encoding:
   ```python
   normalized = tensor / torch.norm(tensor, p=2)
   ```

4. **Small Batch Sizes**: Quantum simulators require smaller batch sizes (4-8) due to memory constraints.

5. **Gradient Clipping**: Quantum circuits are susceptible to barren plateaus; we apply gradient clipping (max_norm=1.0).

### Data Pipeline

- **Classical Models** (Module 2, Module 4):
  - Input: 224×224 RGB images
  - Augmentation: rotation, flips, color jitter
  - Normalization: ImageNet mean/std

- **Quantum Models** (Module 1, Module 3):
  - Input: 8×8 grayscale images (64 features → 6 qubits)
  - NO augmentation
  - L2 normalization for amplitude encoding

### Optimizer Selection

- **Classical Models**: AdamW or SGD with momentum
- **Quantum Models**: Adam or RMSprop (SPSA/COBYLA for noisy settings)
- **Hybrid Models**: Separate parameter groups with different learning rates

## 📈 Expected Performance

Based on literature and preliminary results:

- **Module 4 (CNN+DNN)**: 92-96% accuracy (strongest baseline)
- **Module 2 (CNN+QNN)**: 88-94% accuracy
- **Module 1 (QCNN+DNN)**: 85-92% accuracy
- **Module 3 (QCNN+QNN)**: 75-85% accuracy (fully quantum, research-grade)

## 🔍 Evaluation Metrics

The evaluation framework computes:

1. **Classification Metrics:**
   - Overall accuracy
   - Per-class precision, recall, F1-score
   - Macro and weighted averages
   - Confusion matrices

2. **Efficiency Metrics:**
   - Inference time per sample
   - Training time per epoch
   - Parameter counts (total, trainable, quantum, classical)

3. **Robustness Tests:**
   - Performance with reduced training data (10%, 25%, 50%, 100%)
   - Variance across multiple seeds

## 📊 Results and Visualization

After training, results are saved to `experiments/` directory:

```
experiments/
├── module1/
│   ├── best_model.pth
│   ├── training_history.json
│   ├── complete_results.json
│   ├── module1_metrics.json
│   ├── module1_confusion_matrix.png
│   └── tensorboard/
├── module2/
├── module3/
├── module4/
├── model_comparison.csv
└── all_results.json
```

## 🧪 Ablation Studies

Run ablation studies to test:

```bash
# Data efficiency (10%, 25%, 50%, 100% of training data)
python main.py --ablation

# Custom data fraction
python main.py --module all --data-fraction 0.25
```

## 📚 Hyperparameter Tuning

The project includes comprehensive hyperparameter search spaces in `config.py`:

**Priority 1 (Highest Impact):**
- Learning rate (most critical)
- Circuit depth / number of layers
- Feature dimensions

**Priority 2:**
- Encoding type (amplitude vs angle)
- Entanglement pattern
- Batch size

**Priority 3:**
- Optimizer selection
- Weight decay
- Dropout rate

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory (OOM) with Quantum Models**
- Reduce batch size: `config.data.batch_size_quantum = 4`
- Reduce number of qubits: `config.qcnn.n_qubits = 6` (instead of 8)
- Use smaller image size: 8×8 instead of 16×16

**2. Slow Training**
- Quantum circuit evaluation is inherently slower
- Consider using subset of data for initial experiments
- Use CPU-based simulation for quick prototyping

**3. Barren Plateaus (Vanishing Gradients)**
- Reduce circuit depth
- Initialize parameters with smaller values (×0.1)
- Try different encoding strategies
- Enable gradient clipping (already implemented)

**4. Poor Quantum Model Performance**
- Ensure NO data augmentation for quantum-only models
- Verify L2 normalization for amplitude encoding
- Check that image dimensions are powers of 2
- Try angle encoding instead of amplitude

## 🔬 Research Insights

This implementation is based on key findings from quantum ML research:

1. **Augmentation Effects**: Data augmentation improves classical CNNs but NOT QCNNs
2. **Parameter Efficiency**: QCNNs use 10-100× fewer parameters but may have lower capacity
3. **Data Efficiency**: Quantum models sometimes excel with limited training data
4. **Hyperparameter Sensitivity**: Learning rate and circuit depth are most critical

## 📖 References

Key papers and resources:
- [Quantum Convolutional Neural Networks (PennyLane)](https://pennylane.ai/qml/demos/tutorial_quanvolution)
- [QCNN Tutorial (Qiskit)](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/11_quantum_convolutional_neural_networks.html)
- [Variational Quantum Classifier (PennyLane)](https://pennylane.ai/qml/demos/tutorial_variational_classifier/)
- PlantVillage Dataset

## 📧 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the configuration in `config.py`
3. Examine training logs in `experiments/*/tensorboard/`

## 📝 License

This project is for educational and research purposes.

---

**Note**: This is a research project exploring quantum machine learning. Quantum models are computationally expensive and experimental. For production plant disease detection, classical deep learning models (Module 4) are recommended.
