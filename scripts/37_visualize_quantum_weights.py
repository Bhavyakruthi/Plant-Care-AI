"""
Quantum Circuit Weight Visualization
====================================

Visualizes the learned weights/parameters of the Quantum Neural Network (QNN).
Shows how the quantum circuit parameters evolved to capture non-linear relationships.

What it shows: Grid of trained weights for each qubit and rotation gate
Why it matters: Explains the "Quantum" part of the CNN+QNN architecture

Author: NLP Project Team
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add backend path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from models.image import ImageModel

def visualize_quantum_weights(model, output_path):
    """Extract and visualize weights from the QuantumClassifier"""
    print("Extracting quantum circuit weights...")
    
    # In ImageModel, model.classifier is usually the QuantumClassifier
    classifier = model.classifier
    
    # Access the weight parameter
    # Assuming standard Pennylane/Torch interface where weights are a single parameter
    try:
        if hasattr(classifier, 'q_weights'):
            weights = classifier.q_weights.detach().cpu().numpy()
        elif hasattr(classifier, 'weights'):
            weights = classifier.weights.detach().cpu().numpy()
        else:
            # Try to find parameters containing 'weight'
            weights = None
            for name, param in classifier.named_parameters():
                if 'weight' in name.lower():
                    weights = param.detach().cpu().numpy()
                    break
        
        if weights is None:
            print("  ⚠️  Could not find quantum weights in classifier")
            return
            
        print(f"  ✅ Weights found with shape: {weights.shape}")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # If 3D (layers, qubits, rotations), flatten or show per layer
        if len(weights.shape) == 3:
            # Average across layers for a summary or show specific layer
            data = weights[0]  # First layer
            title = 'Quantum Circuit Weights (Layer 0)'
        elif len(weights.shape) == 2:
            data = weights
            title = 'Quantum Circuit Weights'
        else:
            data = weights.reshape(1, -1)
            title = 'Quantum Circuit Weights (Flattened)'
            
        sns.heatmap(data, annot=True, fmt=".2f", cmap="RdYlBu", ax=ax, center=0)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Rotation Gate / Parameter Index', fontweight='bold')
        ax.set_ylabel('Qubit / Feature Index', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved quantum weights visualization to {output_path}")
        
    except Exception as e:
        print(f"  ❌ Error visualizing quantum weights: {e}")

def main():
    # Configuration
    MODEL_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/models/image/cnn_qnn_best.pt'
    LABEL_MAPPING_PATH = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/models/image/label_mapping.pkl'
    OUTPUT_DIR = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/outputs/interpretability'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("QUANTUM WEIGHT VISUALIZATION")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Using device: {device}")
    
    print("\nLoading model...")
    model = ImageModel(model_path=MODEL_PATH, label_mapping_path=LABEL_MAPPING_PATH, device=device)
    print("✅ Model loaded")
    
    # Visualize quantum weights
    output_path = os.path.join(OUTPUT_DIR, 'quantum_weights.png')
    visualize_quantum_weights(model, output_path)
    
    print("\n" + "="*70)
    print("✅ QUANTUM WEIGHT VISUALIZATION COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()
