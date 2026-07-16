"""
Test script to validate the trained CNN+QNN model weights.
"""
import torch
import sys
import os

BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"
sys.path.append(os.path.join(BASE_DIR, "backend"))

from models.image import ImageModel

def test_model_loading():
    print("\n" + "="*80)
    print("CNN+QNN MODEL WEIGHT VALIDATION")
    print("="*80)
    
    model_path = os.path.join(BASE_DIR, "models", "image", "cnn_qnn_best.pt")
    test_image = os.path.join(BASE_DIR, "CNN+QNN codes", "Healthy (11).jpg")
    
    print(f"\n📦 Loading model from: {model_path}")
    
    try:
        # Initialize model
        image_model = ImageModel(model_path=model_path)
        print("✅ Model initialized successfully")
        
        # Check model components
        print(f"\n🔍 Model Architecture:")
        print(f"   Feature Extractor: {type(image_model.feature_extractor).__name__}")
        print(f"   Classifier: {type(image_model.classifier).__name__}")
        print(f"   Device: {image_model.device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in image_model.feature_extractor.parameters())
        total_params += sum(p.numel() for p in image_model.classifier.parameters())
        print(f"   Total Parameters: {total_params:,}")
        
        # Test prediction
        print(f"\n🧪 Testing prediction on: {os.path.basename(test_image)}")
        probs = image_model.predict(test_image)
        
        print(f"✅ Prediction successful!")
        print(f"   Output shape: {probs.shape}")
        print(f"   Predicted class: {probs.argmax()}")
        print(f"   Max probability: {probs.max():.4f}")
        
        # Show top 3 predictions
        top3_idx = probs[0].argsort()[-3:][::-1]
        print(f"\n📊 Top 3 Predictions:")
        for i, idx in enumerate(top3_idx, 1):
            print(f"   {i}. Class {idx}: {probs[0][idx]:.4f}")
        
        print("\n" + "="*80)
        print("✅ MODEL VALIDATION SUCCESSFUL")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*80)
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
