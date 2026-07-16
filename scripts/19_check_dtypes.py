import torch
import os
import sys

BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"
sys.path.append(os.path.join(BASE_DIR, "backend"))

from models.text import TextModel
from models.image import ImageModel

def check_dtypes():
    TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models", "text", "best_hybrid_bert_model.pth")
    DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed_data.pkl")
    IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "image", "cnn_qnn_best.pt")

    print("Checking Text Model...")
    text_model = TextModel(model_path=TEXT_MODEL_PATH, data_path=DATA_PATH)
    print(f"BERT Weight Dtype: {text_model.model.bert.embeddings.word_embeddings.weight.dtype}")
    print(f"FC1 Weight Dtype: {text_model.model.fc1.weight.dtype}")
    
    print("\nChecking Image Model...")
    image_model = ImageModel(model_path=IMAGE_MODEL_PATH)
    print(f"ResNet Weight Dtype: {next(image_model.feature_extractor.parameters()).dtype}")
    print(f"Quantum Reduction Weight Dtype: {image_model.classifier.reduction[0].weight.dtype}")
    print(f"Quantum Weight Dtype: {image_model.classifier.q_weights.dtype}")

if __name__ == "__main__":
    check_dtypes()
