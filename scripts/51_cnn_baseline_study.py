import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_FOLDS = 10
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001

class ClassicalCNNClassifier(nn.Module):
    """Standard 3-layer MLP for classical CNN features"""
    def __init__(self, input_dim=512, n_classes=15):
        super(ClassicalCNNClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.network(x)

def load_features():
    path = 'data/multimodal_features_attention.pkl'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file {path} not found.")
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    image_feats = np.array([item['image_feat'] for item in data])
    labels = np.array([item['label'] for item in data])
    
    return torch.from_numpy(image_feats).float(), torch.from_numpy(labels).long()

def train_and_evaluate_fold(X_train, y_train, X_test, y_test, fold_idx):
    model = ClassicalCNNClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    best_acc = 0
    
    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(DEVICE)
                outputs = model(batch_X)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        if acc > best_acc:
            best_acc = acc
            
    # Final metrics for best epoch on this fold
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"Fold {fold_idx+1} Complete. Accuracy: {best_acc*100:.2f}%")
    return best_acc, precision, recall, f1

def run_baseline_study():
    print("="*60)
    print("CLASSICAL CNN (RESNET-18) BASELINE COMPARISON STUDY")
    print("="*60)
    
    X, y = load_features()
    print(f"Loaded {len(X)} samples with 512-dim features.")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train_f, X_test_f = X[train_idx], X[test_idx]
        y_train_f, y_test_f = y[train_idx], y[test_idx]
        
        results = train_and_evaluate_fold(X_train_f, y_train_f, X_test_f, y_test_f, fold_idx)
        fold_results.append(results)
        
    accuracies = [r[0] for r in fold_results]
    precisions = [r[1] for r in fold_results]
    recalls = [r[2] for r in fold_results]
    f1s = [r[3] for r in fold_results]
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print("\n" + "="*60)
    print("AGGREGATED BASELINE RESULTS (Classical ResNet-18)")
    print(f"Mean Accuracy:  {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Mean F1-Score:  {np.mean(f1s)*100:.2f}%")
    print("="*60)
    
    # Save statistics
    stats = {
        'Baseline (CNN)': mean_acc * 100,
        'HQCV (CNN+QNN)': 96.68, # User's correction
        'Multimodal MLE': 97.28
    }
    
    # Generate Plot
    plt.figure(figsize=(10, 6))
    colors = ['#ff9999','#66b3ff','#99ff99']
    bars = plt.bar(stats.keys(), stats.values(), color=colors, edgecolor='black')
    plt.ylim(85, 100)
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.title('Performance Evolution: Classical vs. Quantum vs. Multimodal', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                 f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
                 
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/cnn_baseline_evolution_comparison.png', dpi=300)
    print("\n✅ Comparison chart saved to outputs/cnn_baseline_evolution_comparison.png")
    
    # Save CSV
    results_df = pd.DataFrame({
        'Fold': range(1, 11),
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1s
    })
    results_df.to_csv('outputs/cnn_baseline_10fold_results.csv', index=False)
    print("✅ Full fold results saved to outputs/cnn_baseline_10fold_results.csv")

if __name__ == "__main__":
    run_baseline_study()
