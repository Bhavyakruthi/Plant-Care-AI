# Plant Disease Prediction - Execution Guide

## 🎯 Complete Pipeline Execution

Follow these steps in order to execute the complete NLP pipeline:

---

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**What this does:**
- Installs all required Python packages
- Includes: pandas, scikit-learn, torch, transformers, etc.

**Expected time:** 5-10 minutes

---

## Step 2: Run Data Validation & EDA

```bash
python 01_data_validation_and_eda.py
```

**What this does:**
- Loads the dataset (16,125 samples)
- Checks for missing values
- Analyzes class distribution
- Parses dictionary fields (MORPHOLOGY, LESIONS)
- Creates 9 visualization types
- Extracts text statistics

**Outputs:**
- `processed_data_with_features.csv`
- `outputs/01_missing_values.png`
- `outputs/02_class_distribution.png`
- `outputs/03_severity_distribution.png`
- `outputs/04_text_statistics.png`
- `outputs/05_wordclouds.png`
- `outputs/06_top_keywords.png`
- `outputs/07_correlation_heatmap.png`

**Expected time:** 2-5 minutes

---

## Step 3: Run Preprocessing & Feature Engineering

```bash
python 02_preprocessing.py
```

**What this does:**
- **Checks CUDA availability** (GPU support)
- Cleans and normalizes text
- Extracts structured features (15 features)
- Creates TF-IDF features (5,000 features)
- Creates BERT tokenized encodings
- Performs stratified train/val/test split (70/15/15%)

**Outputs:**
- `preprocessed_data.pkl` (all features + splits)

**Expected time:** 3-5 minutes

**CUDA Check Output:**
If GPU available:
```
✅ CUDA is available!
   • GPU Device: NVIDIA GeForce RTX 3060
   🚀 Models will be trained on GPU (much faster!)
```

If GPU not available:
```
⚠️  CUDA is NOT available. Using CPU.
```

---

## Step 4: Train Baseline Models

```bash
python 03_train_baseline_models.py
```

**What this does:**
- Trains 3 baseline models:
  1. **SVM** (Support Vector Machine)
  2. **Random Forest** (200 trees)
  3. **Logistic Regression**
- Evaluates each on test set
- Creates confusion matrices
- Compares all baseline models

**Outputs:**
- `baseline_models.pkl`
- `outputs/confusion_matrix_svm.png`
- `outputs/confusion_matrix_random_forest.png`
- `outputs/confusion_matrix_logistic_regression.png`
- `outputs/baseline_models_comparison.png`
- `outputs/baseline_models_comparison.csv`

**Expected time:** 5-10 minutes

**Expected Results:**
- SVM: ~78-82% accuracy
- Random Forest: ~80-85% accuracy
- Logistic Regression: ~75-80% accuracy

---

## Step 5: Train BERT Transformer Model

```bash
python 04_train_bert_model.py
```

**What this does:**
- Loads pre-trained BERT (bert-base-uncased)
- Fine-tunes on plant disease data
- Trains for 4 epochs (adjustable)
- **Uses GPU if available** (10-15x faster!)
- Saves best model checkpoint
- Creates training history plots

**Outputs:**
- `best_bert_model.pth` (best model checkpoint)
- `bert_model_results.pkl`
- `outputs/confusion_matrix_bert.png`
- `outputs/bert_training_history.png`

**Expected time:**
- **With GPU:** 15-30 minutes
- **With CPU:** 3-5 hours ⚠️

**Expected Results:**
- Test Accuracy: ~88-95%
- Significant improvement over baselines

**Training Progress Example:**
```
Epoch 1/4
████████████████████ 100% | loss: 0.523 | acc: 82.5%
📊 Epoch 1 Summary:
   • Train Loss: 0.5234 | Train Acc: 82.50%
   • Val Loss: 0.4123 | Val Acc: 85.30%
   ✅ New best model saved!
```

---

## Step 6: Final Model Comparison

```bash
python 05_final_comparison.py
```

**What this does:**
- Loads all trained models
- Creates comprehensive comparison table
- Generates comparison visualizations
- Provides recommendations

**Outputs:**
- `outputs/final_model_comparison.csv`
- `outputs/final_comprehensive_comparison.png`

**Expected time:** 1-2 minutes

**Output Example:**
```
📊 Performance Comparison:
┌─────────────────────┬─────────────────────┬──────────────┬──────────────┬─────────────┬──────────────┬─────────────────────┐
│ Model               │ Type                │ Accuracy (%) │ Precision(%) │ Recall (%)  │ F1-Score (%) │ Training Time (min) │
├─────────────────────┼─────────────────────┼──────────────┼──────────────┼─────────────┼──────────────┼─────────────────────┤
│ SVM                 │ Baseline (TF-IDF)   │ 80.25        │ 79.84        │ 80.25       │ 79.92        │ 0.08                │
│ Random Forest       │ Baseline (TF-IDF)   │ 83.50        │ 83.12        │ 83.50       │ 83.20        │ 0.13                │
│ Logistic Regression │ Baseline (TF-IDF)   │ 77.80        │ 77.42        │ 77.80       │ 77.55        │ 0.03                │
│ BERT Transformer    │ Deep Learning       │ 92.15        │ 91.95        │ 92.15       │ 92.02        │ 25.50               │
└─────────────────────┴─────────────────────┴──────────────┴──────────────┴─────────────┴──────────────┴─────────────────────┘

🏆 BEST MODEL: BERT Transformer
   • Accuracy: 92.15%
   • F1-Score: 92.02%
```

---

## 📁 Complete File Structure (After Execution)

```
LANGUAGE_MODEL_PROJECT/
│
├── multimodal_plant_data.csv              # Original dataset
├── requirements.txt                        # Dependencies
├── README.md                               # Documentation
│
├── 01_data_validation_and_eda.py          # Script 1
├── 02_preprocessing.py                     # Script 2
├── 03_train_baseline_models.py            # Script 3
├── 04_train_bert_model.py                 # Script 4
├── 05_final_comparison.py                 # Script 5
│
├── outputs/                                # All visualizations (15+ files)
│   ├── 01_missing_values.png
│   ├── 02_class_distribution.png
│   ├── 03_severity_distribution.png
│   ├── 04_text_statistics.png
│   ├── 05_wordclouds.png
│   ├── 06_top_keywords.png
│   ├── 07_correlation_heatmap.png
│   ├── confusion_matrix_svm.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_bert.png
│   ├── baseline_models_comparison.png
│   ├── bert_training_history.png
│   ├── final_comprehensive_comparison.png
│   ├── baseline_models_comparison.csv
│   └── final_model_comparison.csv
│
├── processed_data_with_features.csv       # After step 1
├── preprocessed_data.pkl                  # After step 2 (large file)
├── baseline_models.pkl                    # After step 3
├── bert_model_results.pkl                 # After step 4
└── best_bert_model.pth                   # After step 4 (best BERT checkpoint)
```

---

## 💡 Tips & Recommendations

### For GPU Training (BERT):
- Ensure CUDA is installed: https://pytorch.org/get-started/locally/
- Monitor GPU usage: `nvidia-smi` (Windows/Linux)
- Reduce batch size if out of memory (edit line in `04_train_bert_model.py`)

### For CPU Training (BERT):
- Reduce epochs to 2-3 for faster training
- Reduce max_length to 128 (from 256)
- Or skip BERT and use baseline models

### Troubleshooting:
- **Import errors**: Run `pip install -r requirements.txt` again
- **Memory errors**: Close other applications, reduce batch size
- **Slow training**: Use GPU or reduce data size for testing

---

## 🎓 What You'll Learn

By running this complete pipeline, you'll understand:

1. ✅ **Data Validation**: How to check data quality and visualize distributions
2. ✅ **Text Preprocessing**: Cleaning, parsing structured fields, feature extraction
3. ✅ **TF-IDF**: Converting text to numerical features
4. ✅ **Classical ML**: SVM, Random Forest, Logistic Regression
5. ✅ **BERT Transformers**: Fine-tuning pre-trained models
6. ✅ **GPU Acceleration**: Using CUDA for faster training
7. ✅ **Model Evaluation**: Comprehensive metrics and visualizations
8. ✅ **Model Comparison**: Making informed decisions

---

## 📊 Expected Total Execution Time

| Step | Time (GPU) | Time (CPU) |
|------|------------|------------|
| 1. Install | 5-10 min | 5-10 min |
| 2. EDA | 2-5 min | 2-5 min |
| 3. Preprocessing | 3-5 min | 3-5 min |
| 4. Baseline Models | 5-10 min | 5-10 min |
| 5. BERT Model | **15-30 min** | **3-5 hours** |
| 6. Comparison | 1-2 min | 1-2 min |
| **TOTAL** | **~30-60 min** | **~3-6 hours** |

---

## 🚀 Quick Test Run (Without BERT)

If you want to test quickly without waiting for BERT:

```bash
# Run steps 1-4 only
python 01_data_validation_and_eda.py
python 02_preprocessing.py
python 03_train_baseline_models.py

# Skip step 5 (BERT) and results are already good!
```

This will give you baseline results in ~15-20 minutes.

---

## ✅ Success Indicators

You'll know everything worked when:

1. ✅ All scripts complete without errors
2. ✅ `outputs/` folder contains 15+ visualization files
3. ✅ Final comparison shows BERT > Baselines
4. ✅ Test accuracy > 90% for BERT
5. ✅ All `.pkl` and `.pth` files are created

---

🎉 **Ready to start? Run the first command and let the pipeline begin!**
