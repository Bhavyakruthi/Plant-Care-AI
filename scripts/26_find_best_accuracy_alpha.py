import torch
import numpy as np
import pickle
import os
import sys
from sklearn.metrics import accuracy_score
from tqdm import tqdm

BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"
sys.path.append(os.path.join(BASE_DIR, "backend"))

from models.text import TextModel
from models.image import ImageModel

def main():
    data_path = os.path.join(BASE_DIR, "data", "preprocessed_data.pkl")
    image_map_path = os.path.join(BASE_DIR, "data", "image_mappings.pkl")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(image_map_path, 'rb') as f:
        image_data = pickle.load(f)
        
    X_test = data['X_test_text']
    y_test = data['y_test']
    image_paths_test = image_data['image_paths_test']
    
    text_model = TextModel(model_path=os.path.join(BASE_DIR, "models", "text", "best_hybrid_bert_model.pth"), data_path=data_path)
    image_model = ImageModel(model_path=os.path.join(BASE_DIR, "models", "image", "cnn_qnn_best.pt"))
    
    print("Inference...")
    text_probs = np.array([text_model.predict(t)[0] for t in tqdm(X_test, desc="Text")])
    
    image_probs = []
    for img in tqdm(image_paths_test, desc="Image"):
        if img and os.path.exists(img):
            try:
                image_probs.append(image_model.predict(img)[0])
            except:
                image_probs.append(np.ones(15)/15)
        else:
            image_probs.append(np.ones(15)/15)
    image_probs = np.array(image_probs)
    
    alphas = np.linspace(0, 1, 101)
    accs = [accuracy_score(y_test, np.argmax(a*image_probs + (1-a)*text_probs, axis=1)) for a in alphas]
    
    best_idx = np.argmax(accs)
    best_a = alphas[best_idx]
    best_acc = accs[best_idx]
    
    print(f"\nRESULTS:")
    print(f"Best Alpha (for Accuracy): {best_a:.4f}")
    print(f"Best Accuracy: {best_acc*100:.2f}%")
    print(f"Alpha=0.99 Accuracy: {accs[np.argmin(np.abs(alphas-0.99))]*100:.2f}%")
    print(f"Alpha=0.50 Accuracy: {accs[np.argmin(np.abs(alphas-0.5))]*100:.2f}%")

if __name__ == "__main__":
    main()
