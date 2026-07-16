import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
import os
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Setup paths
BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"
sys.path.append(os.path.join(BASE_DIR, "backend"))

from models.text import TextModel
from models.image import ImageModel

def main():
    print("Generating MLE Parameter vs Accuracy Plot...")
    
    # Load data
    data_path = os.path.join(BASE_DIR, "data", "preprocessed_data.pkl")
    image_map_path = os.path.join(BASE_DIR, "data", "image_mappings.pkl")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(image_map_path, 'rb') as f:
        image_data = pickle.load(f)
        
    X_test = data['X_test_text']
    y_test = data['y_test']
    image_paths_test = image_data['image_paths_test']
    
    # Load models
    text_model = TextModel(model_path=os.path.join(BASE_DIR, "models", "text", "best_hybrid_bert_model.pth"), data_path=data_path)
    image_model = ImageModel(model_path=os.path.join(BASE_DIR, "models", "image", "cnn_qnn_best.pt"))
    
    # Get probabilities
    print("Gathering text model probabilities...")
    text_probs = []
    for text in tqdm(X_test, desc="Text Inference"):
        text_probs.append(text_model.predict(text)[0])
    text_probs = np.array(text_probs)
    
    print("Gathering image model probabilities...")
    image_probs = []
    num_classes = len(np.unique(y_test))
    for img_path in tqdm(image_paths_test, desc="Image Inference"):
        if img_path and os.path.exists(img_path):
            try:
                image_probs.append(image_model.predict(img_path)[0])
            except:
                image_probs.append(np.ones(num_classes) / num_classes)
        else:
            image_probs.append(np.ones(num_classes) / num_classes)
    image_probs = np.array(image_probs)
    
    # Sweep alpha
    alphas = np.linspace(0, 1, 101)
    accuracies = []
    
    print("Calculating accuracy sweep...")
    for alpha in tqdm(alphas, desc="Sweep"):
        fused = alpha * image_probs + (1 - alpha) * text_probs
        preds = np.argmax(fused, axis=1)
        accuracies.append(accuracy_score(y_test, preds))
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.plot(alphas, accuracies, color='#2ecc71', linewidth=3, label='Ensemble Accuracy')
    
    optimal_idx = np.argmax(accuracies)
    optimal_alpha = alphas[optimal_idx]
    max_acc = accuracies[optimal_idx]
    
    plt.axvline(optimal_alpha, color='#e74c3c', linestyle='--', alpha=0.8, 
                label=f'Optimal α = {optimal_alpha:.2f}')
    plt.scatter([optimal_alpha], [max_acc], color='#e74c3c', s=100, zorder=5)
    
    plt.annotate(f'Peak: {max_acc:.2%}', 
                 xy=(optimal_alpha, max_acc), 
                 xytext=(optimal_alpha-0.2, max_acc-0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=12, fontweight='bold')
    
    plt.xlabel('MLE Parameter α (Image Weight)', fontsize=14, fontweight='bold')
    plt.ylabel('Overall Accuracy', fontsize=14, fontweight='bold')
    plt.title('Optimization Landscape: MLE Parameter vs Ensemble Accuracy', fontsize=16, fontweight='bold', pad=20)
    plt.ylim(min(accuracies)-0.05, 1.0)
    plt.xlim(0, 1)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add stats box
    stats_text = (f"Image Accuracy: {accuracies[-1]:.1%}\n"
                  f"Text Accuracy: {accuracies[0]:.1%}\n"
                  f"Max Ensemble: {max_acc:.1%}")
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    output_path = os.path.join(BASE_DIR, 'output_gradcam', 'mle_vs_accuracy_optimized.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
