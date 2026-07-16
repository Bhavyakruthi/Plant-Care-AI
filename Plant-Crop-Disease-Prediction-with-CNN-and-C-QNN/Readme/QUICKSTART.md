# Quick Start Guide: Plant Disease Detection with Quantum-Classical Hybrid Models

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```powershell
# Navigate to project directory
cd "d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 5\Deep Learning\FINAL PROJECT"

# Install required packages
pip install -r requirements.txt
```

### Step 2: Verify Dataset

Your dataset structure should look like this:
```
PlantVillage/
├── Pepper__bell___Bacterial_spot/
├── Pepper__bell___healthy/
├── Potato___Early_blight/
├── ... (15 categories total)
```

### Step 3: Quick Test Run (5-10 minutes)

Test with a small fraction of data first:

```powershell
# Test classical baseline (fastest)
python main.py --module module4 --data-fraction 0.1

# Test one hybrid model
python main.py --module module2 --data-fraction 0.1
```

### Step 4: Full Training

Once you've verified everything works:

```powershell
# Train all four modules with full dataset
python main.py --module all

# Or train specific modules
python main.py --module module1  # QCNN + DNN
python main.py --module module2  # CNN + QNN
python main.py --module module3  # QCNN + QNN (fully quantum)
python main.py --module module4  # CNN + DNN (baseline)
```

## ⏱️ Expected Training Times

On a typical GPU (e.g., NVIDIA RTX 3060):

- **Module 4 (CNN+DNN)**: ~30-45 minutes
- **Module 2 (CNN+QNN)**: ~2-3 hours
- **Module 1 (QCNN+DNN)**: ~3-4 hours
- **Module 3 (QCNN+QNN)**: ~4-6 hours (slowest, fully quantum)

On CPU: Multiply times by 5-10×

## 📊 View Results

After training completes:

```powershell
# View comparison of all models
python visualize.py --results experiments/all_results.json --save-dir experiments

# Check individual model results
cd experiments/module1
# Open training_history.json, view confusion matrices, etc.

# View tensorboard logs
tensorboard --logdir experiments/module1/tensorboard
```

## 🎯 Recommended Workflow

### For Quick Experimentation (Testing):
```powershell
# Use 10% of data, classical baseline only
python main.py --module module4 --data-fraction 0.1
```

### For Research/Comparison:
```powershell
# Train all models with full data
python main.py --module all

# Then run ablation study
python main.py --ablation
```

### For Production (Plant Disease Detection):
```powershell
# Just use the classical baseline (best accuracy + speed)
python main.py --module module4
```

## 🔧 Common Commands

```powershell
# Use CPU instead of GPU
python main.py --module all --device cpu

# Train with custom config
python main.py --config my_config.yaml --module module1

# View help
python main.py --help
```

## 📁 Output Structure

After running, you'll find:

```
experiments/
├── module1/                    # QCNN + DNN results
│   ├── best_model.pth         # Best model checkpoint
│   ├── training_history.json  # Training curves data
│   ├── complete_results.json  # All metrics
│   ├── module1_confusion_matrix.png
│   └── tensorboard/           # TensorBoard logs
├── module2/                    # CNN + QNN results
├── module3/                    # QCNN + QNN results
├── module4/                    # CNN + DNN results
├── all_results.json           # Combined results
├── model_comparison.csv       # Comparison table
└── model_comparison.png       # Visualization
```

## ⚠️ Important Notes

1. **Quantum Models are Slow**: QCNN/QNN models take much longer to train than classical models. This is expected!

2. **Memory Requirements**: 
   - Quantum models: 8-16 GB RAM
   - Classical models: 4-8 GB RAM with GPU

3. **First Run**: The first run will:
   - Create data splits (saved for reproducibility)
   - Download pretrained weights (if using pretrained CNNs)
   - This is normal and only happens once

4. **Checkpoints**: Models are automatically saved:
   - `best_model.pth`: Best validation accuracy
   - `final_model.pth`: Final epoch
   - Checkpoints every 10 epochs

## 🐛 Troubleshooting

**Out of Memory?**
```python
# Edit config.py:
config.data.batch_size_quantum = 4  # Reduce from 8
config.data.batch_size_classical = 16  # Reduce from 32
```

**Too Slow?**
```powershell
# Start with smaller data fraction
python main.py --module module4 --data-fraction 0.25

# Or use fewer epochs (edit config.py)
config.training.num_epochs = 50  # Instead of 100
```

**Module Not Found?**
```powershell
# Make sure you're in the project directory
cd "d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 5\Deep Learning\FINAL PROJECT"

# Install dependencies again
pip install -r requirements.txt
```

## 📖 Next Steps

1. **Start with Module 4** (classical baseline) to verify setup
2. **Try Module 2** (CNN+QNN hybrid) - good balance of performance and novelty
3. **Experiment with Module 1** (QCNN+DNN) - interesting quantum features
4. **Test Module 3** (fully quantum) - research/comparison only

## 🎓 Understanding Results

- **Accuracy > 90%**: Excellent performance
- **Accuracy 80-90%**: Good performance
- **Accuracy < 80%**: May need hyperparameter tuning

Expected rankings (best to worst):
1. Module 4 (CNN+DNN) - Highest accuracy
2. Module 2 (CNN+QNN) - Close to classical
3. Module 1 (QCNN+DNN) - Moderate accuracy
4. Module 3 (QCNN+QNN) - Research-grade

## 💡 Tips for Success

1. **Always start with `--data-fraction 0.1`** for testing
2. **Monitor training** with TensorBoard: `tensorboard --logdir experiments`
3. **Compare models** only after all finish training
4. **Save your config** if you change hyperparameters
5. **Use GPU** for faster training (10× speedup)

---

**Ready to start?** Run this command:

```powershell
python main.py --module module4 --data-fraction 0.1
```

This will train the classical baseline on 10% of data in ~5 minutes and verify everything works!
