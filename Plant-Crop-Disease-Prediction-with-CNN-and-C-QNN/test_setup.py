"""
Quick test script to verify installation and basic functionality
Run this before starting full training
"""

import sys
import torch
import numpy as np
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'pennylane': 'PennyLane',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'tqdm',
        'PIL': 'Pillow'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!\n")
    return True


def check_dataset():
    """Check if PlantVillage dataset exists"""
    print("Checking dataset...")
    
    dataset_path = Path("PlantVillage")
    
    if not dataset_path.exists():
        print(f"✗ Dataset not found at {dataset_path.absolute()}")
        print("\nPlease ensure PlantVillage dataset is in the project directory")
        return False
    
    # Count classes
    classes = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if len(classes) == 0:
        print("✗ No class folders found in PlantVillage directory")
        return False
    
    print(f"✓ Found {len(classes)} disease classes:")
    for cls in sorted(classes)[:5]:
        n_images = len(list(cls.glob("*.jpg"))) + len(list(cls.glob("*.JPG"))) + len(list(cls.glob("*.png")))
        print(f"  - {cls.name}: {n_images} images")
    
    if len(classes) > 5:
        print(f"  ... and {len(classes) - 5} more classes")
    
    print()
    return True


def check_cuda():
    """Check CUDA availability"""
    print("Checking GPU/CUDA...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available!")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return "cuda"
    else:
        print("⚠ CUDA not available, will use CPU")
        print("  Training will be slower (~10x)")
        return "cpu"
    
    print()


def test_quantum_circuit():
    """Test basic quantum circuit functionality"""
    print("\nTesting quantum circuits...")
    
    try:
        import pennylane as qml
        
        # Create simple test circuit
        dev = qml.device("default.qubit", wires=2)
        
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        # Test execution
        x = np.array([0.5, 0.5])
        result = circuit(x)
        
        print(f"✓ Quantum circuit executed successfully!")
        print(f"  Test output: {result:.4f}")
        
    except Exception as e:
        print(f"✗ Quantum circuit test failed: {str(e)}")
        return False
    
    print()
    return True


def test_data_loading():
    """Test data loading"""
    print("Testing data loading...")
    
    try:
        from config import ExperimentConfig
        from data_loader import get_dataloaders
        
        config = ExperimentConfig()
        config.data.dataset_path = "PlantVillage"
        config.data.batch_size_classical = 4
        config.data.batch_size_quantum = 2
        config.data.num_workers = 0  # Avoid multiprocessing issues
        
        # Test classical dataloader
        train_loader, val_loader, test_loader, num_classes = get_dataloaders(
            config, mode="classical", data_fraction=0.01  # Only 1% for testing
        )
        
        # Get one batch
        for images, labels in train_loader:
            print(f"✓ Classical data loading works!")
            print(f"  Batch shape: {images.shape}")
            print(f"  Number of classes: {num_classes}")
            break
        
    except Exception as e:
        print(f"✗ Data loading test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_model_creation():
    """Test model creation (two-stage architecture)"""
    print("Testing feature extractors and classifiers...")
    
    try:
        # Test feature extractors
        from stage1_feature_extraction import CNNFeatureExtractor, QCNNFeatureExtractor
        from stage2_classifier_training import DNNClassifier, QNNClassifier
        from config import CNNConfig, QCNNConfig, DNNConfig, QNNConfig
        
        # Test CNN feature extractor
        cnn_config = CNNConfig()
        cnn_extractor = CNNFeatureExtractor(cnn_config)
        x_cnn = torch.randn(2, 3, 224, 224)
        features_cnn = cnn_extractor.extract_features_batch(x_cnn)
        
        print(f"✓ CNN Feature Extractor works!")
        print(f"  Input: {x_cnn.shape}, Features: {features_cnn.shape}")
        
        # Test quantum model (small)
        
        # Test DNN classifier
        dnn_config = DNNConfig()
        dnn_classifier = DNNClassifier(input_dim=512, n_classes=15, config=dnn_config)
        output_dnn = dnn_classifier(features_cnn)
        
        print(f"✓ DNN Classifier works!")
        print(f"  Features: {features_cnn.shape}, Output: {output_dnn.shape}")
        
        print("\n  Testing quantum components (this may take a moment)...")
        
        # Test QCNN feature extractor
        qcnn_config = QCNNConfig()
        qcnn_config.n_qubits = 4  # Small for testing
        qcnn_config.qcnn_depth = 1
        qcnn_extractor = QCNNFeatureExtractor(qcnn_config, encoding_type='amplitude')
        
        # Test single sample (quantum can't batch)
        x_quantum = torch.randn(1, 1, 8, 8)
        features_qcnn = qcnn_extractor.extract_features_single(x_quantum[0])
        
        print(f"✓ QCNN Feature Extractor works!")
        print(f"  Input: {x_quantum.shape}, Features: {features_qcnn.shape}")
        
        # Test QNN classifier
        qnn_config = QNNConfig()
        qnn_config.n_qubits = 4  # Small for testing
        qnn_classifier = QNNClassifier(input_dim=features_qcnn.shape[0], n_classes=15, config=qnn_config)
        output_qnn = qnn_classifier(features_qcnn.unsqueeze(0))
        
        print(f"✓ QNN Classifier works!")
        print(f"  Features: {features_qcnn.shape}, Output: {output_qnn.shape}")
        
    except Exception as e:
        print(f"✗ Feature extractor/classifier test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def main():
    """Run all tests"""
    print("="*70)
    print("QUANTUM-CLASSICAL HYBRID MODELS: SYSTEM CHECK")
    print("="*70)
    print()
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Dataset", check_dataset),
        ("GPU/CUDA", check_cuda),
        ("Quantum Circuits", test_quantum_circuit),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {str(e)}\n")
            results.append((test_name, False))
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, result in results:
        if result in [True, "cuda", "cpu"]:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_passed = False
        
        print(f"{test_name:<20} {status}")
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to start training.")
        print("\nNext steps:")
        print("1. Quick test:     python main.py --module module4 --data-fraction 0.1")
        print("2. Full training:  python main.py --module all")
        print("\nSee QUICKSTART.md for more details.")
    else:
        print("\n⚠ Some tests failed. Please fix the issues before training.")
        print("See error messages above for details.")
    
    print()


if __name__ == "__main__":
    main()
