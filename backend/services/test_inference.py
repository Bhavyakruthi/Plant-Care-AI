from services.inference import EnsembleInference
import numpy as np

def test_ensemble():
    image_probs = np.array([0.1, 0.8, 0.1])
    text_probs = np.array([0.1, 0.2, 0.7])
    
    # 60% Image, 40% Text
    ensemble = EnsembleInference(alpha=0.6)
    result = ensemble.fuse(image_probs, text_probs)
    
    print(f"Fused Probs: {result['probabilities']}")
    print(f"Predicted Class: {result['predicted_idx']}")
    
    # Combined Prob = 0.6*0.8 + 0.4*0.2 = 0.48 + 0.08 = 0.56
    assert result['predicted_idx'] == 1
    assert abs(result['confidence'] - 0.56) < 1e-5
    print("✅ Ensemble test passed!")

if __name__ == "__main__":
    test_ensemble()
