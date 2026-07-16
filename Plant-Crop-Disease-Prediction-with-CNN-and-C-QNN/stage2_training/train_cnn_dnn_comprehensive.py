"""
Model 1: CNN + DNN (Classical-Classical Baseline)
==================================================
COMPREHENSIVE EVALUATION WITH ALL METRICS AND VISUALIZATIONS

Feature Extractor: ResNet18 CNN (pretrained)
Classifier: Deep Neural Network (3-layer MLP)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import json
from datetime import datetime
from tqdm import tqdm
import psutil
import gc
import time

# Add parent directory to path
sys.path.append('..')
from config import DNNConfig

print("=" * 100)
print("  MODEL 1: CNN + DNN (Classical-Classical Baseline)")
print("  COMPREHENSIVE EVALUATION WITH ALL METRICS")
print("=" * 100)
print()

# Create output directories
os.makedirs('results', exist_ok=True)
os.makedirs('results/cnn_dnn', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Track system resources
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
initial_memory = get_memory_usage()

print(f"Device: {device}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
print(f"Initial Memory Usage: {initial_memory:.2f} MB")
print()

# Load CNN features
print("Loading CNN features...")
load_start = time.time()

train_data = torch.load('../features/CNN_resnet18_train.pt')
val_data = torch.load('../features/CNN_resnet18_val.pt')
test_data = torch.load('../features/CNN_resnet18_test.pt')

X_train = train_data['features']
y_train = train_data['labels']
X_val = val_data['features']
y_val = val_data['labels']
X_test = test_data['features']
y_test = test_data['labels']

n_classes = train_data['n_classes']
feature_dim = train_data['feature_dim']

load_time = time.time() - load_start
print(f"✓ Data loaded in {load_time:.2f} seconds")
print(f"✓ Train: {X_train.shape}")
print(f"✓ Val:   {X_val.shape}")
print(f"✓ Test:  {X_test.shape}")
print(f"✓ Classes: {n_classes}, Feature dim: {feature_dim}")
print()

# Create data loaders
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# Define DNN Classifier
class DNNClassifier(nn.Module):
    """Deep Neural Network Classifier"""
    def __init__(self, input_dim, n_classes, hidden_dims=[256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, n_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# Initialize model
print("Initializing DNN classifier...")
dnn_config = DNNConfig()
model = DNNClassifier(
    input_dim=feature_dim,
    n_classes=n_classes,
    hidden_dims=dnn_config.hidden_dims
).to(device)

# Model information
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024

print(f"✓ Model Architecture: {dnn_config.hidden_dims}")
print(f"✓ Total Parameters: {total_params:,}")
print(f"✓ Trainable Parameters: {trainable_params:,}")
print(f"✓ Model Size: {model_size:.2f} MB")
print(f"✓ Memory After Model Load: {get_memory_usage():.2f} MB")
print()

# Print model architecture
print("Model Architecture:")
print(model)
print()

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# Training loop
n_epochs = 100  # Reduced for testing
best_val_acc = 0.0
best_epoch = 0
train_losses = []
train_accs = []
val_losses = []
val_accs = []
learning_rates = []

print("=" * 100)
print("  TRAINING")
print("=" * 100)
print()

training_start_time = datetime.now()

for epoch in range(n_epochs):
    epoch_start = time.time()
    
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = 100.0 * train_correct / train_total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100.0 * val_correct / val_total
    
    epoch_time = time.time() - epoch_start
    current_lr = optimizer.param_groups[0]['lr']
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    learning_rates.append(current_lr)
    
    print(f"Epoch {epoch+1:02d}/{n_epochs} | "
          f"Time: {epoch_time:.2f}s | "
          f"LR: {current_lr:.6f} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, 'models/cnn_dnn_best.pt')
        print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    scheduler.step(val_acc)

training_time = datetime.now() - training_start_time
print(f"\n✓ Training complete! Time: {training_time}")
print(f"✓ Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
print()

# Load best model for testing
print("Loading best model for testing...")
checkpoint = torch.load('models/cnn_dnn_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✓ Loaded model from epoch {checkpoint['epoch']+1}")
print()

# Test evaluation - COMPREHENSIVE
print("=" * 100)
print("  COMPREHENSIVE TEST EVALUATION")
print("=" * 100)
print()

model.eval()
all_predictions = []
all_labels = []
all_probabilities = []
test_correct = 0
test_total = 0
inference_times = []

test_start = time.time()

with torch.no_grad():
    for features, labels in tqdm(test_loader, desc="Testing"):
        batch_start = time.time()
        features, labels = features.to(device), labels.to(device)
        
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        batch_time = time.time() - batch_start
        inference_times.append(batch_time / features.size(0))
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
        
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_time = time.time() - test_start
avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
all_probabilities = np.array(all_probabilities)

# Calculate all metrics
test_acc = 100.0 * test_correct / test_total
precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

precision_weighted = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
recall_weighted = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

# Per-class metrics
precision_per_class = precision_score(all_labels, all_predictions, average=None, zero_division=0)
recall_per_class = recall_score(all_labels, all_predictions, average=None, zero_division=0)
f1_per_class = f1_score(all_labels, all_predictions, average=None, zero_division=0)

# ROC-AUC (one-vs-rest)
y_true_binarized = label_binarize(all_labels, classes=range(n_classes))
try:
    roc_auc_macro = roc_auc_score(y_true_binarized, all_probabilities, average='macro', multi_class='ovr')
    roc_auc_weighted = roc_auc_score(y_true_binarized, all_probabilities, average='weighted', multi_class='ovr')
except:
    roc_auc_macro = 0.0
    roc_auc_weighted = 0.0

print("Test Metrics Summary:")
print(f"  Accuracy: {test_acc:.2f}%")
print(f"  Precision (Macro): {precision_macro:.4f}")
print(f"  Recall (Macro): {recall_macro:.4f}")
print(f"  F1-Score (Macro): {f1_macro:.4f}")
print(f"  ROC-AUC (Macro): {roc_auc_macro:.4f}")
print()
print(f"  Precision (Weighted): {precision_weighted:.4f}")
print(f"  Recall (Weighted): {recall_weighted:.4f}")
print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
print(f"  ROC-AUC (Weighted): {roc_auc_weighted:.4f}")
print()
print(f"Performance:")
print(f"  Total Test Time: {test_time:.2f} seconds")
print(f"  Average Inference Time: {avg_inference_time:.2f} ms/sample")
print(f"  Throughput: {test_total/test_time:.2f} samples/sec")
print()

# ==================== VISUALIZATIONS ====================

plt.style.use('seaborn-v0_8-darkgrid')

# 1. Confusion Matrix
print("Creating visualizations...")
cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.title('CNN + DNN - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('results/cnn_dnn/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Confusion matrix saved")

# 2. Normalized Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(14, 12))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', cbar_kws={'label': 'Percentage'})
plt.title('CNN + DNN - Normalized Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('results/cnn_dnn/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Normalized confusion matrix saved")

# 3. Training Curves (Loss and Accuracy)
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Loss curves
ax = axes[0, 0]
ax.plot(train_losses, label='Train Loss', linewidth=2, color='#3498db')
ax.plot(val_losses, label='Val Loss', linewidth=2, color='#e74c3c')
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Accuracy curves
ax = axes[0, 1]
ax.plot(train_accs, label='Train Acc', linewidth=2, color='#3498db')
ax.plot(val_accs, label='Val Acc', linewidth=2, color='#e74c3c')
ax.axhline(y=best_val_acc, color='g', linestyle='--', label=f'Best Val: {best_val_acc:.2f}%', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Learning rate schedule
ax = axes[1, 0]
ax.plot(learning_rates, linewidth=2, color='#9b59b6')
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Loss vs Accuracy scatter
ax = axes[1, 1]
scatter = ax.scatter(train_losses, train_accs, alpha=0.6, s=80, c=range(len(train_losses)), 
                    cmap='viridis', edgecolors='black', linewidth=0.5, label='Train')
ax.scatter(val_losses, val_accs, alpha=0.6, s=80, c=range(len(val_losses)), 
          cmap='plasma', edgecolors='black', linewidth=0.5, marker='s', label='Val')
ax.set_xlabel('Loss', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Loss vs Accuracy Evolution', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Epoch')

plt.tight_layout()
plt.savefig('results/cnn_dnn/training_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Training curves saved")

# 4. ROC Curves (One-vs-Rest for each class)
fig, axes = plt.subplots(3, 5, figsize=(25, 15))
axes = axes.ravel()

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_binarized[:, i], all_probabilities[:, i])
    roc_auc = auc(fpr, tpr)
    
    axes[i].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[i].set_xlim([0.0, 1.0])
    axes[i].set_ylim([0.0, 1.05])
    axes[i].set_xlabel('False Positive Rate', fontsize=10)
    axes[i].set_ylabel('True Positive Rate', fontsize=10)
    axes[i].set_title(f'Class {i} ROC', fontsize=11, fontweight='bold')
    axes[i].legend(loc="lower right", fontsize=9)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('CNN + DNN - ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/cnn_dnn/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ ROC curves saved")

# 5. Per-Class Metrics Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

class_labels = [f'C{i}' for i in range(n_classes)]
x = np.arange(n_classes)
width = 0.6

# Precision
axes[0].bar(x, precision_per_class, width, color='#3498db', edgecolor='black', linewidth=1.2)
axes[0].axhline(y=precision_macro, color='r', linestyle='--', linewidth=2, label=f'Macro Avg: {precision_macro:.3f}')
axes[0].set_xlabel('Class', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Precision', fontsize=12, fontweight='bold')
axes[0].set_title('Per-Class Precision', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(class_labels, rotation=45)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim([0, 1.1])

# Recall
axes[1].bar(x, recall_per_class, width, color='#2ecc71', edgecolor='black', linewidth=1.2)
axes[1].axhline(y=recall_macro, color='r', linestyle='--', linewidth=2, label=f'Macro Avg: {recall_macro:.3f}')
axes[1].set_xlabel('Class', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Recall', fontsize=12, fontweight='bold')
axes[1].set_title('Per-Class Recall', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(class_labels, rotation=45)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim([0, 1.1])

# F1-Score
axes[2].bar(x, f1_per_class, width, color='#f39c12', edgecolor='black', linewidth=1.2)
axes[2].axhline(y=f1_macro, color='r', linestyle='--', linewidth=2, label=f'Macro Avg: {f1_macro:.3f}')
axes[2].set_xlabel('Class', fontsize=12, fontweight='bold')
axes[2].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
axes[2].set_title('Per-Class F1-Score', fontsize=14, fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(class_labels, rotation=45)
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].set_ylim([0, 1.1])

plt.suptitle('CNN + DNN - Per-Class Performance Metrics', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('results/cnn_dnn/per_class_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Per-class metrics saved")

# 6. Model Performance Summary
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Overall metrics
ax1 = fig.add_subplot(gs[0, :])
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metrics_values = [test_acc/100, precision_macro, recall_macro, f1_macro, roc_auc_macro]
colors_metric = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

bars = ax1.barh(metrics_names, metrics_values, color=colors_metric, edgecolor='black', linewidth=1.5)
ax1.set_xlim([0, 1.1])
ax1.set_xlabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Overall Test Performance Metrics', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

for i, (bar, val) in enumerate(zip(bars, metrics_values)):
    ax1.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=11, fontweight='bold')

# Training info
ax2 = fig.add_subplot(gs[1, 0])
ax2.axis('off')
training_info = [
    f"Total Epochs: {n_epochs}",
    f"Best Epoch: {best_epoch}",
    f"Training Time: {training_time}",
    f"Final Train Acc: {train_accs[-1]:.2f}%",
    f"Best Val Acc: {best_val_acc:.2f}%",
    f"Test Acc: {test_acc:.2f}%",
]
y_pos = 0.9
for info in training_info:
    ax2.text(0.1, y_pos, info, fontsize=11, family='monospace')
    y_pos -= 0.15
ax2.set_title('Training Information', fontsize=12, fontweight='bold', loc='left')

# Model info
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')
model_info = [
    f"Parameters: {total_params:,}",
    f"Model Size: {model_size:.2f} MB",
    f"Batch Size: {batch_size}",
    f"Architecture: {dnn_config.hidden_dims}",
    f"Optimizer: Adam",
    f"Initial LR: 0.001",
]
y_pos = 0.9
for info in model_info:
    ax3.text(0.1, y_pos, info, fontsize=11, family='monospace')
    y_pos -= 0.15
ax3.set_title('Model Information', fontsize=12, fontweight='bold', loc='left')

# Performance info
ax4 = fig.add_subplot(gs[1, 2])
ax4.axis('off')
perf_info = [
    f"Test Samples: {test_total}",
    f"Test Time: {test_time:.2f}s",
    f"Inference: {avg_inference_time:.2f} ms",
    f"Throughput: {test_total/test_time:.1f} samp/s",
    f"Memory: {get_memory_usage():.2f} MB",
    f"Device: {device.upper()}",
]
y_pos = 0.9
for info in perf_info:
    ax4.text(0.1, y_pos, info, fontsize=11, family='monospace')
    y_pos -= 0.15
ax4.set_title('Performance Information', fontsize=12, fontweight='bold', loc='left')

# Confusion matrix preview (small)
ax5 = fig.add_subplot(gs[2, :2])
sns.heatmap(cm_normalized, annot=False, cmap='Blues', cbar=True, ax=ax5, square=True)
ax5.set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
ax5.set_xlabel('Predicted', fontsize=10)
ax5.set_ylabel('True', fontsize=10)

# Class distribution
ax6 = fig.add_subplot(gs[2, 2])
class_counts = np.bincount(all_labels, minlength=n_classes)
ax6.bar(range(n_classes), class_counts, color='#3498db', edgecolor='black', linewidth=1)
ax6.set_xlabel('Class', fontsize=10, fontweight='bold')
ax6.set_ylabel('Count', fontsize=10, fontweight='bold')
ax6.set_title('Test Set Distribution', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.suptitle('CNN + DNN - Comprehensive Performance Summary', fontsize=16, fontweight='bold')
plt.savefig('results/cnn_dnn/performance_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Performance summary saved")

print("\n✓ All visualizations created!")
print()

# ==================== SAVE DETAILED RESULTS ====================

# Comprehensive results dictionary
results = {
    'model_name': 'CNN + DNN',
    'model_type': 'Classical-Classical Baseline',
    'feature_extractor': 'ResNet18 CNN (pretrained)',
    'classifier': 'Deep Neural Network',
    
    # Architecture
    'architecture': {
        'input_dim': feature_dim,
        'hidden_dims': dnn_config.hidden_dims,
        'output_dim': n_classes,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size,
    },
    
    # Training configuration
    'training_config': {
        'epochs': n_epochs,
        'batch_size': batch_size,
        'optimizer': 'Adam',
        'initial_lr': 0.001,
        'scheduler': 'ReduceLROnPlateau',
        'loss_function': 'CrossEntropyLoss',
    },
    
    # Dataset
    'dataset': {
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'n_classes': n_classes,
        'feature_dim': feature_dim,
    },
    
    # Training results
    'training_results': {
        'best_epoch': best_epoch,
        'best_val_acc': float(best_val_acc),
        'final_train_acc': float(train_accs[-1]),
        'final_val_acc': float(val_accs[-1]),
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
        'training_time': str(training_time),
        'training_time_seconds': training_time.total_seconds(),
    },
    
    # Test metrics
    'test_metrics': {
        'accuracy': float(test_acc),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'roc_auc_macro': float(roc_auc_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'roc_auc_weighted': float(roc_auc_weighted),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
    },
    
    # Performance
    'performance': {
        'test_time_seconds': float(test_time),
        'avg_inference_time_ms': float(avg_inference_time),
        'throughput_samples_per_sec': float(test_total / test_time),
        'memory_usage_mb': float(get_memory_usage()),
        'device': device,
    },
    
    # Training history
    'history': {
        'train_loss': [float(x) for x in train_losses],
        'train_acc': [float(x) for x in train_accs],
        'val_loss': [float(x) for x in val_losses],
        'val_acc': [float(x) for x in val_accs],
        'learning_rates': [float(x) for x in learning_rates],
    },
    
    # Confusion matrix
    'confusion_matrix': cm.tolist(),
    'confusion_matrix_normalized': cm_normalized.tolist(),
    
    # Timestamp
    'timestamp': datetime.now().isoformat(),
    'pytorch_version': torch.__version__,
}

# Save JSON
with open('results/cnn_dnn/results_detailed.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Detailed results saved to JSON")

# Save text report
with open('results/cnn_dnn/results_report.txt', 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("CNN + DNN - COMPREHENSIVE EVALUATION REPORT\n")
    f.write("=" * 100 + "\n\n")
    
    f.write("MODEL INFORMATION\n")
    f.write("-" * 100 + "\n")
    f.write(f"Model Type: Classical-Classical Baseline\n")
    f.write(f"Feature Extractor: ResNet18 CNN (pretrained)\n")
    f.write(f"Classifier: Deep Neural Network {dnn_config.hidden_dims}\n")
    f.write(f"Total Parameters: {total_params:,}\n")
    f.write(f"Model Size: {model_size:.2f} MB\n\n")
    
    f.write("TRAINING CONFIGURATION\n")
    f.write("-" * 100 + "\n")
    f.write(f"Epochs: {n_epochs}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Optimizer: Adam (lr=0.001)\n")
    f.write(f"Loss Function: CrossEntropyLoss\n")
    f.write(f"Scheduler: ReduceLROnPlateau\n\n")
    
    f.write("DATASET\n")
    f.write("-" * 100 + "\n")
    f.write(f"Train Samples: {len(X_train):,}\n")
    f.write(f"Val Samples: {len(X_val):,}\n")
    f.write(f"Test Samples: {len(X_test):,}\n")
    f.write(f"Classes: {n_classes}\n")
    f.write(f"Feature Dimension: {feature_dim}\n\n")
    
    f.write("TRAINING RESULTS\n")
    f.write("-" * 100 + "\n")
    f.write(f"Best Epoch: {best_epoch}/{n_epochs}\n")
    f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
    f.write(f"Final Train Accuracy: {train_accs[-1]:.2f}%\n")
    f.write(f"Final Val Accuracy: {val_accs[-1]:.2f}%\n")
    f.write(f"Training Time: {training_time}\n\n")
    
    f.write("TEST METRICS\n")
    f.write("-" * 100 + "\n")
    f.write(f"Accuracy: {test_acc:.2f}%\n")
    f.write(f"Precision (Macro): {precision_macro:.4f}\n")
    f.write(f"Recall (Macro): {recall_macro:.4f}\n")
    f.write(f"F1-Score (Macro): {f1_macro:.4f}\n")
    f.write(f"ROC-AUC (Macro): {roc_auc_macro:.4f}\n\n")
    f.write(f"Precision (Weighted): {precision_weighted:.4f}\n")
    f.write(f"Recall (Weighted): {recall_weighted:.4f}\n")
    f.write(f"F1-Score (Weighted): {f1_weighted:.4f}\n")
    f.write(f"ROC-AUC (Weighted): {roc_auc_weighted:.4f}\n\n")
    
    f.write("PER-CLASS METRICS\n")
    f.write("-" * 100 + "\n")
    f.write(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
    f.write("-" * 100 + "\n")
    for i in range(n_classes):
        f.write(f"{i:<8} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f}\n")
    f.write("\n")
    
    f.write("PERFORMANCE\n")
    f.write("-" * 100 + "\n")
    f.write(f"Test Time: {test_time:.2f} seconds\n")
    f.write(f"Average Inference Time: {avg_inference_time:.2f} ms/sample\n")
    f.write(f"Throughput: {test_total/test_time:.2f} samples/sec\n")
    f.write(f"Memory Usage: {get_memory_usage():.2f} MB\n")
    f.write(f"Device: {device.upper()}\n\n")
    
    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 100 + "\n")
    f.write(classification_report(all_labels, all_predictions, digits=4))
    f.write("\n")

print("✓ Text report saved")
print()

print("=" * 100)
print("  TRAINING AND EVALUATION COMPLETE!")
print("=" * 100)
print()
print(f"Model: CNN + DNN")
print(f"Best Val Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Test F1-Score: {f1_macro:.4f}")
print(f"Test ROC-AUC: {roc_auc_macro:.4f}")
print(f"Training Time: {training_time}")
print()
print("Results saved to:")
print("  📁 results/cnn_dnn/results_detailed.json")
print("  📁 results/cnn_dnn/results_report.txt")
print("  📊 results/cnn_dnn/*.png (6 visualization files)")
print("  💾 models/cnn_dnn_best.pt")
print()
