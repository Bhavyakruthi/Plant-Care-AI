import torch
import numpy as np
import pandas as pd
import pickle
import os
import sys
import shutil
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"
JOURNAL_DIR = os.path.join(BASE_DIR, "Journal_Results")
sys.path.append(os.path.join(BASE_DIR, "backend"))

from models.text import TextModel
from models.image import ImageModel

def main():
    print("🚀 GENERATING COMPREHENSIVE JOURNAL RESULTS...")
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    
    # Setup data
    data_path = os.path.join(BASE_DIR, "data", "preprocessed_data.pkl")
    image_map_path = os.path.join(BASE_DIR, "data", "image_mappings.pkl")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(image_map_path, 'rb') as f:
        image_data = pickle.load(f)
        
    X_test = data['X_test_text']
    y_test = data['y_test']
    image_paths_test = image_data['image_paths_test']
    label_encoder = data['label_encoder']
    classes = label_encoder.classes_
    
    # Load models
    text_model = TextModel(model_path=os.path.join(BASE_DIR, "models", "text", "best_hybrid_bert_model.pth"), data_path=data_path)
    image_model = ImageModel(model_path=os.path.join(BASE_DIR, "models", "image", "cnn_qnn_best.pt"))
    
    # 1. Gather Probabilities
    print("\n--- Model Inference ---")
    text_probs = np.array([text_model.predict(t)[0] for t in tqdm(X_test, desc="Text Model")])
    
    image_probs = []
    num_classes = len(classes)
    for img in tqdm(image_paths_test, desc="Image Model"):
        if img and os.path.exists(img):
            try:
                image_probs.append(image_model.predict(img)[0])
            except:
                image_probs.append(np.ones(num_classes) / num_classes)
        else:
            image_probs.append(np.ones(num_classes) / num_classes)
    image_probs = np.array(image_probs)
    
    # 2. Results per model and optimal ensemble (alpha=0.46)
    alpha = 0.46
    results = {}
    
    models_to_eval = {
        "Text_BioBERT": text_probs,
        "Image_CNN_QNN": image_probs,
        "Ensemble_Fused": alpha * image_probs + (1 - alpha) * text_probs
    }
    
    print("\n--- Computing Detailed Metrics ---")
    metrics_summary = []
    
    for name, probs in models_to_eval.items():
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_test, preds)
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted')
        
        results[name] = {
            "preds": preds,
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
            "report": classification_report(y_test, preds, target_names=classes, output_dict=True)
        }
        
        metrics_summary.append({
            "Model": name,
            "Accuracy (%)": f"{acc*100:.2f}",
            "Precision (%)": f"{p*100:.2f}",
            "Recall (%)": f"{r*100:.2f}",
            "F1 Score (%)": f"{f1*100:.2f}"
        })
        
        # Save Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix: {name}")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(JOURNAL_DIR, f"confusion_matrix_{name.lower()}.png"), dpi=300)
        plt.close()

    # 3. Save Summary Table
    df_metrics = pd.DataFrame(metrics_summary)
    df_metrics.to_csv(os.path.join(JOURNAL_DIR, "model_comparison_metrics.csv"), index=False)
    
    # 4. Generate Journal Markdown Report
    report_path = os.path.join(JOURNAL_DIR, "Journal_Data_Report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 📝 Multimodal Plant Disease Prediction: Final Metrics for Journal Writing\n\n")
        
        f.write("## 1. Global Performance Metrics\n")
        f.write(df_metrics.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 2. Detailed Ensemble Performance (α=0.46)\n")
        f.write("### Per-Class F1 Scores (Optimized Ensemble)\n")
        ens_report = results["Ensemble_Fused"]["report"]
        
        class_stats = []
        for cls in classes:
            stats = ens_report[cls]
            class_stats.append({
                "Class": cls,
                "Precision": f"{stats['precision']:.4f}",
                "Recall": f"{stats['recall']:.4f}",
                "F1-Score": f"{stats['f1-score']:.4f}"
            })
        
        f.write(pd.DataFrame(class_stats).to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 3. Comparison of F1 Scores Across Modalities\n")
        comp_f1 = []
        for cls in classes:
            comp_f1.append({
                "Class": cls,
                "BioBERT-F1": f"{results['Text_BioBERT']['report'][cls]['f1-score']:.4f}",
                "CNN-QNN-F1": f"{results['Image_CNN_QNN']['report'][cls]['f1-score']:.4f}",
                "Ensemble-F1": f"{results['Ensemble_Fused']['report'][cls]['f1-score']:.4f}"
            })
        f.write(pd.DataFrame(comp_f1).to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 4. Visualization Assets\n")
        f.write("- `confusion_matrix_ensemble_fused.png`: Final classification heatmap.\n")
        f.write("- `mle_vs_accuracy_optimized.png`: Parameter sensitivity plot (already in folder).\n")
        f.write("- `gradcam_sample.png`: Visual explainability proof (copying now).\n")
        
    # 5. Copy extra images from output_gradcam
    shutil.copy(os.path.join(BASE_DIR, "output_gradcam", "mle_vs_accuracy_optimized.png"), JOURNAL_DIR)
    
    # Copy a sample Grad-CAM if exists
    gradcam_sample = os.path.join(BASE_DIR, "output_gradcam", "gradcam_Tomato_healthy.png")
    if os.path.exists(gradcam_sample):
        shutil.copy(gradcam_sample, os.path.join(JOURNAL_DIR, "gradcam_sample.png"))

    print(f"✅ JOURNAL RESULTS GENERATED IN: {JOURNAL_DIR}")

if __name__ == "__main__":
    main()
