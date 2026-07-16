import torch
import sys
import os
import pennylane as qml

BASE_DIR = r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT"
sys.path.append(os.path.join(BASE_DIR, "backend"))

from models.image import ImageModel

print("-" * 50)
print("INTERNAL MODEL INSPECTION")
print("-" * 50)

model = ImageModel()
device = model.device

print(f"Model Architecture: {model.classifier.n_qubits} qubits, 3 layers (expected)")

# Check weight stats
print("\nWeight Statistics:")
for name, param in model.classifier.named_parameters():
    print(f"  {name:30}: mean={param.data.mean():.4f}, std={param.data.std():.4f}, min={param.data.min():.4f}, max={param.data.max():.4f}")

# Load some features
feature_path = os.path.join(BASE_DIR, "dataset", "Image Dataset", "Features", "CNN_resnet18_test.pt")
data = torch.load(feature_path, map_location='cpu', weights_only=False)
features = data['features'][:5].to(device)
labels = data['labels'][:5]

print("\nTesting first 5 features...")
with torch.no_grad():
    # Step 1: Reduction
    red_out = model.classifier.reduction(features.float())
    print(f"Reduction output shape: {red_out.shape}")
    print(f"Reduction output min/max: {red_out.min().item():.4f} / {red_out.max().item():.4f}")
    
    # Step 2: Forward pass (includes Quantum)
    logits = model.classifier(features.float())
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)
    
    print("\nResults:")
    print(f"Predictions: {preds.cpu().numpy()}")
    print(f"True labels: {labels.cpu().numpy()}")
    
    print("\nLogits for first sample:")
    print(logits[0].cpu().numpy())
    
    print("\nProbabilities for first sample:")
    print(probs[0].cpu().numpy())

# check if any weights are NaN or suspiciously uniform
nan_found = False
for name, param in model.classifier.named_parameters():
    if torch.isnan(param).any():
        print(f"CRITICAL: NaN found in {name}")
        nan_found = True
    if param.data.std() < 1e-7:
        print(f"WARNING: Suspicously uniform weights in {name} (std={param.data.std().item()})")

if not nan_found:
    print("\nNo NaN weights found.")
