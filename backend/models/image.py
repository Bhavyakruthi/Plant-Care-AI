import torch
import torch.nn as nn
import pennylane as qml
import os
import sys
import numpy as np
import pickle
from torchvision import models, transforms
from PIL import Image

# Default model path (PyTorch ZIP archive directory)
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                  "models", "image", "cnn_qnn_best.pt")

# 1. Define Quantum Device & QNode
n_qubits = 5  # Match checkpoint architecture
n_layers = 3  # Match checkpoint architecture
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev, interface='torch')
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        if n_qubits > 1:
            qml.CNOT(wires=[n_qubits-1, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumClassifier(nn.Module):
    def __init__(self, input_dim, n_qubits, n_layers, n_classes):
        super().__init__()
        self.reduction = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_qubits),
            nn.Tanh()
        )
        self.q_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)
        self.expansion = nn.Sequential(
            nn.Linear(n_qubits, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def forward(self, x):
        x = self.reduction(x.float())
        # Apply QNN to each item in batch
        q_out = []
        for i in range(x.size(0)):
            res = quantum_circuit(x[i], self.q_weights)
            q_out.append(torch.stack(res))
        q_out = torch.stack(q_out).float()
        return self.expansion(q_out)

class ImageModel:
    def __init__(self, model_path=None, label_mapping_path=None, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set default model path if not provided
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        
        # 1. Feature Extractor
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.feature_extractor.eval()
        
        # 2. Hybrid Classifier (using actual checkpoint dimensions)
        self.classifier = QuantumClassifier(input_dim=512, n_qubits=5, n_layers=3, n_classes=15).to(self.device)
        
        # 3. Transforms (Matching data_loader.py strategy: Resize(256) -> CenterCrop(224))
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 4. Load label mapping
        self.label_mapping = None
        if label_mapping_path is None:
            # Try default path
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            label_mapping_path = os.path.join(base_dir, "models", "image", "label_mapping.pkl")
        
        if os.path.exists(label_mapping_path):
            try:
                with open(label_mapping_path, 'rb') as f:
                    mapping_data = pickle.load(f)
                    self.label_mapping = mapping_data['image_to_text']
                print(f"Label mapping loaded from {label_mapping_path}")
            except Exception as e:
                print(f"Failed to load label mapping: {e}")
                print("Image predictions may not align with text labels!")
        else:
            print(f"Label mapping not found at {label_mapping_path}")
            print("Run scripts/24_create_label_mapping.py to generate it")
        
        # 5. Load Model Weights
        if model_path and os.path.exists(model_path):
            try:
                # Check if it's a directory (PyTorch ZIP archive) or file
                if os.path.isdir(model_path):
                    # Load from PyTorch ZIP archive structure
                    print(f"Loading from PyTorch ZIP archive: {model_path}")
                    import pickle as pkl
                    
                    class TorchUnpickler(pkl.Unpickler):
                        def __init__(self, file, model_dir, device):
                            super().__init__(file)
                            self.model_dir = model_dir
                            self.device = device
                            
                        def persistent_load(self, pid):
                            if isinstance(pid, tuple) and len(pid) > 0:
                                typename = pid[0]
                                if typename == 'storage':
                                    storage_type, key, location, size = pid[1:]
                                    dtype = storage_type.dtype
                                    data_file = os.path.join(self.model_dir, 'data', str(key))
                                    
                                    if os.path.exists(data_file):
                                        with open(data_file, 'rb') as f:
                                            # storage_type is like torch.FloatStorage
                                            # We need to load as raw bytes and convert to that storage type
                                            data = f.read()
                                            
                                        if hasattr(storage_type, 'from_buffer'):
                                            return storage_type.from_buffer(data, byte_order='native')
                                        else:
                                            # Fallback to ByteStorage/UntypedStorage and hope cast works
                                            if hasattr(torch, 'UntypedStorage'):
                                                storage = torch.UntypedStorage.from_buffer(data, dtype=torch.uint8)
                                            else:
                                                storage = torch.ByteStorage.from_buffer(data)
                                            return storage.cast(storage_type) if hasattr(storage, 'cast') else storage
                            return None
                    
                    data_pkl_path = os.path.join(model_path, 'data.pkl')
                    with open(data_pkl_path, 'rb') as f:
                        unpickler = TorchUnpickler(f, model_path, self.device)
                        checkpoint = unpickler.load()
                else:
                    # Regular file - try normal torch.load
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Handle both checkpoint dict and direct state_dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.classifier.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Image Model weights loaded from checkpoint (epoch {checkpoint.get('epoch', '?')})")
                else:
                    self.classifier.load_state_dict(checkpoint)
                    print(f"Image Model weights loaded from {model_path}")
                    
            except Exception as e:
                print(f"Failed to load Image Model weights: {e}")
                print("Model will use random initialization - accuracy will be very low!")
                import traceback
                traceback.print_exc()
        else:
            print(f"Model path not found: {model_path}")
            print("Model will use random initialization")
            
        self.classifier.eval()

    def predict(self, image_path: str):
        with Image.open(image_path) as img:
            img_tensor = self.transform(img.convert('RGB')).unsqueeze(0).to(self.device).float()
            
            with torch.no_grad():
                # Extract features from CNN
                features = self.feature_extractor(img_tensor).squeeze(-1).squeeze(-1)
                # NO NORMALIZATION - model was trained on raw CNN features
                logits = self.classifier(features.float())
                probs = torch.softmax(logits, dim=1)
            
            probs_np = probs.cpu().numpy()
            
            # Remap probabilities if label mapping exists
            if self.label_mapping is not None:
                # Create remapped probability array
                remapped_probs = np.zeros_like(probs_np)
                for img_idx, text_idx in enumerate(self.label_mapping):
                    remapped_probs[:, text_idx] = probs_np[:, img_idx]
                return remapped_probs
            
            return probs_np
    def predict_bytes(self, img_bytes: bytes):
        import io
        with Image.open(io.BytesIO(img_bytes)) as img:
            img_tensor = self.transform(img.convert('RGB')).unsqueeze(0).to(self.device).float()
            
            with torch.no_grad():
                # Extract features from CNN
                features = self.feature_extractor(img_tensor).squeeze(-1).squeeze(-1)
                # NO NORMALIZATION - model was trained on raw CNN features
                logits = self.classifier(features.float())
                probs = torch.softmax(logits, dim=1)
            
            probs_np = probs.cpu().numpy()
            
            # Remap probabilities if label mapping exists
            if self.label_mapping is not None:
                # Create remapped probability array
                remapped_probs = np.zeros_like(probs_np)
                for img_idx, text_idx in enumerate(self.label_mapping):
                    remapped_probs[:, text_idx] = probs_np[:, img_idx]
                return remapped_probs
            
            return probs_np
