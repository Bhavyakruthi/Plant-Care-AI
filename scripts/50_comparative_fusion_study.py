import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from tqdm import tqdm

# --- 0. Global Optimization ---
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("CUDA Optimization Enabled (cudnn.benchmark)")

# --- 1. Model Definitions ---

class ConcatenationFusionModel(nn.Module):
    def __init__(self, img_dim=512, txt_dim=768, num_classes=15):
        super(ConcatenationFusionModel, self).__init__()
        self.fusion_head = nn.Sequential(
            nn.Linear(img_dim + txt_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, img_feat, txt_feat):
        combined = torch.cat((img_feat, txt_feat), dim=1)
        return self.fusion_head(combined)

class WeightedAdditionFusionModel(nn.Module):
    def __init__(self, img_dim=512, txt_dim=768, num_classes=15, alpha=None):
        super(WeightedAdditionFusionModel, self).__init__()
        self.img_proj = nn.Linear(img_dim, 512)
        self.txt_proj = nn.Linear(txt_dim, 512)
        
        # If alpha is None, it is learnable. Otherwise it is fixed.
        if alpha is None:
            self.alpha_raw = nn.Parameter(torch.tensor(0.0)) # Starts at 0.5 (sigmoid(0))
            self.fixed_alpha = None
        else:
            self.alpha_raw = None
            self.fixed_alpha = alpha
            
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, img_feat, txt_feat):
        if self.fixed_alpha is not None:
            alpha = self.fixed_alpha
        else:
            alpha = torch.sigmoid(self.alpha_raw)
            
        p_img = self.img_proj(img_feat)
        p_txt = self.txt_proj(txt_feat)
        
        weighted_added = (alpha * p_img) + ((1 - alpha) * p_txt)
        return self.classifier(weighted_added)

# --- 2. Data Loading Utility ---

def load_multimodal_data(path):
    print(f"Loading features from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    img_feats = np.array([item['image_feat'] for item in data])
    txt_feats = np.array([item['text_feat'] for item in data])
    labels = np.array([item['label'] for item in data])
    
    return img_feats, txt_feats, labels

# --- 3. Training & Evaluation Logic per Fold ---

def train_and_eval_fold(exp_name, model_class, model_args, train_idx, test_idx, img_feats, txt_feats, labels, device, fold_num):
    # Prepare Fold Data
    X_img_train, X_img_test = img_feats[train_idx], img_feats[test_idx]
    X_txt_train, X_txt_test = txt_feats[train_idx], txt_feats[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_img_train), torch.FloatTensor(X_txt_train), torch.LongTensor(y_train)
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_img_test), torch.FloatTensor(X_txt_test), torch.LongTensor(y_test)
    )
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BatchSize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BatchSize, shuffle=False)
    
    # Initialize Model
    model = model_class(**model_args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_f1 = 0
    patience = 8
    trigger = 0
    
    for epoch in range(50): # Max epochs per fold
        model.train()
        for img, txt, lbl in train_loader:
            img, txt, lbl = img.to(device), txt.to(device), lbl.to(device)
            optimizer.zero_grad()
            outputs = model(img, txt)
            loss = criterion(outputs, lbl)
            loss.backward()
            optimizer.step()
            
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for img, txt, lbl in test_loader:
                img, txt, lbl = img.to(device), txt.to(device), lbl.to(device)
                outputs = model(img, txt)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbl.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                break
                
    return best_acc, best_f1

# --- 4. Main Script with 10-Fold CV ---

BatchSize = 64

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_PATH = "../data/multimodal_features_attention.pkl"
    OUTPUT_ROOT = "../outputs/journal_fusion_research"
    N_FOLDS = 10
    
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    print("="*60)
    print("PLANTCARE AI: MULTIMODAL FUSION RESEARCH (10-FOLD CV)")
    print("="*60)
    
    # Load Data
    img_feats, txt_feats, labels = load_multimodal_data(DATA_PATH)
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    # EXPERIMENT LIST
    experiments = [
        ("Concatenation", ConcatenationFusionModel, {"img_dim":512, "txt_dim":768}, "Combines image (512) and text (768) features into a 1280-dimensional vector."),
        ("Simple Addition", WeightedAdditionFusionModel, {"alpha":0.5}, "Projects both modalities to 512-dim and adds them equally (fixed alpha=0.5)."),
        ("Learned Weight", WeightedAdditionFusionModel, {"alpha":None}, "Uses an optimization-driven alpha parameter to weigh modalities dynamically."),
        ("Vision Bias", WeightedAdditionFusionModel, {"alpha":0.8}, "Manual weight bias prioritizing visual features (alpha=0.8)."),
        ("Text Bias", WeightedAdditionFusionModel, {"alpha":0.3}, "Manual weight bias prioritizing text symptoms (alpha=0.3).")
    ]
    
    final_results = {}

    for i, (name, model_class, model_args, desc) in enumerate(experiments):
        print(f"\n>>> EXPERIMENT {i}: {name}")
        print(f"    Description: {desc}")
        
        fold_accs = []
        fold_f1s = []
        
        folder_name = f"Exp_{i}_{name.replace(' ', '_').lower()}"
        exp_dir = os.path.join(OUTPUT_ROOT, folder_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(labels)), labels)):
            acc, f1 = train_and_eval_fold(name, model_class, model_args, train_idx, test_idx, img_feats, txt_feats, labels, DEVICE, fold_idx)
            fold_accs.append(acc)
            fold_f1s.append(f1)
            print(f"    Fold {fold_idx+1}/{N_FOLDS} | Acc: {acc:.4f} | F1: {f1:.4f}")
            
        final_results[name] = {
            'accuracy_mean': np.mean(fold_accs),
            'accuracy_std': np.std(fold_accs),
            'f1_mean': np.mean(fold_f1s),
            'f1_std': np.std(fold_f1s)
        }
        
        # Save per-experiment results
        results_df = pd.DataFrame({'Fold': range(1, 11), 'Accuracy': fold_accs, 'F1-Score': fold_f1s})
        results_df.to_csv(os.path.join(exp_dir, 'fold_results.csv'), index=False)
        
    # FINAL SUMMARY REPORT
    print("\n" + "="*60)
    print("GLOBAL PERFORMANCE SUMMARY (10-FOLD CV)")
    print("="*60)
    print(f"{'Strategy':<20} | {'Accuracy (Mean ± Std)':<25} | {'F1-Score (Mean ± Std)':<25}")
    print("-" * 75)
    
    with open(os.path.join(OUTPUT_ROOT, 'global_summary_10fold.txt'), 'w') as f:
        f.write("GLOBAL FUSION ANALYSIS REPORT (10-FOLD CV)\n")
        f.write("="*45 + "\n")
        f.write(f"{'Strategy':<20} | {'Accuracy (Mean ± Std)':<25} | {'F1-Score (Mean ± Std)':<25}\n")
        f.write("-" * 75 + "\n")
        
        for name, metrics in final_results.items():
            acc_str = f"{metrics['accuracy_mean']:.2%} ± {metrics['accuracy_std']:.2%}"
            f1_str = f"{metrics['f1_mean']:.2%} ± {metrics['f1_std']:.2%}"
            line = f"{name:<20} | {acc_str:<25} | {f1_str:<25}"
            print(line)
            f.write(line + "\n")

    # Generate Comparative Multi-Metric Plot
    strategies = list(final_results.keys())
    acc_means = [final_results[s]['accuracy_mean']*100 for s in strategies]
    f1_means = [final_results[s]['f1_mean']*100 for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, acc_means, width, label='Accuracy (%)', color='skyblue', edgecolor='black')
    rects2 = ax.bar(x + width/2, f1_means, width, label='F1-Score (%)', color='lightcoral', edgecolor='black')
    
    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_title('Comparative Performance of Fusion Strategies (10-Fold CV)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    ax.set_ylim(90, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_ROOT, 'fusion_comparison_10fold.png'), dpi=300)
    print(f"\n✅ Comparison chart saved to outputs/journal_fusion_research/fusion_comparison_10fold.png")

if __name__ == "__main__":
    main()
