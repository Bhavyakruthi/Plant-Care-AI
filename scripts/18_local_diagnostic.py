import torch
import os
import sys

# Paths
BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"
sys.path.append(os.path.join(BASE_DIR, "backend"))

from models.text import TextModel
from models.image import ImageModel
from services.explainability import XAIService
from services.inference import EnsembleInference

def run_local_diagnostic():
    print("🚀 Starting Local Diagnostic...")
    
    TEXT_MODEL_PATH = os.path.join(BASE_DIR, "models", "text", "best_hybrid_bert_model.pth")
    DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed_data.pkl")
    IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "image", "cnn_qnn_best.pt")
    TEST_IMAGE = os.path.join(BASE_DIR, "CNN+QNN codes", "Healthy (11).jpg")

    try:
        print("📦 Loading Models...")
        text_model = TextModel(model_path=TEXT_MODEL_PATH, data_path=DATA_PATH)
        text_model.model.float()
        
        image_model = ImageModel(model_path=IMAGE_MODEL_PATH)
        image_model.classifier.float()
        image_model.feature_extractor.float()
        
        ensemble = EnsembleInference(alpha=0.6)
        
        print(f"📷 Processing Image: {TEST_IMAGE}")
        # Image Prediction
        image_probs = image_model.predict(TEST_IMAGE)
        print("✅ Image Prediction Success")
        
        # Text Prediction (Mocking description since we aren't calling Gemini API for this local test)
        mock_desc = "The leaf appears healthy with no visible lesions or discoloration."
        text_probs = text_model.predict(mock_desc)
        print("✅ Text Prediction Success")
        
        # Fusion
        fusion_result = ensemble.fuse(image_probs, text_probs)
        print(f"🎯 Fusion Success: {fusion_result['predicted_idx']}")
        
        # XAI
        print("🔍 Generating XAI Explanations...")
        heatmap = XAIService.generate_gradcam(image_model, TEST_IMAGE, target_idx=fusion_result["predicted_idx"])
        if heatmap:
            print(f"✅ Grad-CAM Success ({len(heatmap)} bytes)")
        else:
            print("❌ Grad-CAM Failed (Returned None)")
            
        importance = XAIService.get_feature_importance(mock_desc)
        print(f"✅ Feature Importance Success: {importance}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Diagnostic Failed: {e}")

if __name__ == "__main__":
    run_local_diagnostic()
