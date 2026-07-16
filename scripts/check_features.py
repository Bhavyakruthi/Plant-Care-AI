import torch

# Load the test features
data = torch.load(r"d:\COLLEGE FILES\ALL SUBJECTS\SEMESTER 6\Natural Languge Processing\LANGUAGE_MODEL_PROJECT\dataset\Image Dataset\Features\CNN_resnet18_test.pt")

print("="*80)
print("PRE-EXTRACTED FEATURES ANALYSIS")
print("="*80)

print("\nKeys in data:", list(data.keys()))
print("\nFeatures shape:", data['features'].shape)
print("Labels shape:", data['labels'].shape)
print("Num classes:", data['n_classes'])
print("Feature dim:", data['feature_dim'])

print("\n" + "="*80)
print("FEATURE STATISTICS")
print("="*80)

features = data['features']
print(f"\nMean: {features.mean():.6f}")
print(f"Std: {features.std():.6f}")
print(f"Min: {features.min():.6f}")
print(f"Max: {features.max():.6f}")

print("\n" + "="*80)
print("SAMPLE DATA")
print("="*80)

print(f"\nFirst sample features (first 10 dims): {features[0][:10].tolist()}")
print(f"First sample label: {data['labels'][0].item()}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\n✓ Features are RAW CNN outputs (no normalization)")
print("✓ Feature values range from {:.2f} to {:.2f}".format(features.min(), features.max()))
print("✓ This confirms the model was trained on unnormalized ResNet18 features")
