"""
Compare All 4 Models - Comprehensive Analysis
=============================================
Reads comprehensive results from all 4 models and generates
detailed comparison visualizations and reports.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime

print("=" * 100)
print("  MODEL COMPARISON - COMPREHENSIVE ANALYSIS")
print("=" * 100)
print()

# Model names and paths
models = {
    'CNN + DNN': 'results/cnn_dnn/results_detailed.json',
    'CNN + QNN': 'results/cnn_qnn/results_detailed.json',
}

# Load all results
print("Loading results from all models...")
results = {}
for model_name, path in models.items():
    if Path(path).exists():
        with open(path, 'r') as f:
            results[model_name] = json.load(f)
        print(f"  ✓ {model_name}: {path}")
    else:
        print(f"  ✗ {model_name}: NOT FOUND - {path}")

if len(results) == 0:
    print("\n❌ No results found! Please run the training scripts first.")
    exit(1)

print(f"\n✓ Loaded {len(results)}/4 models")
print()

# Create comparison output directory
Path('results/comparison').mkdir(parents=True, exist_ok=True)

# Extract metrics for comparison
comparison_data = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'ROC-AUC': [],
    'Training Time (s)': [],
    'Inference Time (ms)': [],
    'Throughput (samp/s)': [],
    'Parameters': [],
    'Model Size (MB)': [],
    'Memory (MB)': [],
}

for model_name, data in results.items():
    comparison_data['Model'].append(model_name)
    comparison_data['Accuracy'].append(data['test_metrics']['accuracy'])
    comparison_data['Precision'].append(data['test_metrics']['precision_macro'])
    comparison_data['Recall'].append(data['test_metrics']['recall_macro'])
    comparison_data['F1-Score'].append(data['test_metrics']['f1_macro'])
    comparison_data['ROC-AUC'].append(data['test_metrics']['roc_auc_macro'])
    comparison_data['Training Time (s)'].append(data['training_results']['training_time_seconds'])
    comparison_data['Inference Time (ms)'].append(data['performance']['avg_inference_time_ms'])
    comparison_data['Throughput (samp/s)'].append(data['performance']['throughput_samples_per_sec'])
    comparison_data['Parameters'].append(data['architecture']['total_parameters'])
    comparison_data['Model Size (MB)'].append(data['architecture']['model_size_mb'])
    comparison_data['Memory (MB)'].append(data['performance']['memory_usage_mb'])

df = pd.DataFrame(comparison_data)

# Print comparison table
print("=" * 100)
print("  COMPREHENSIVE COMPARISON TABLE")
print("=" * 100)
print()
print(df.to_string(index=False))
print()

# Save comparison table
df.to_csv('results/comparison/comparison_table.csv', index=False)
print("✓ Comparison table saved to CSV")
print()

# ==================== VISUALIZATIONS ====================

plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']  # Blue, Green, Purple, Red

print("Creating comparison visualizations...")

# 1. Overall Performance Metrics Comparison
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    values = comparison_data[metric]
    bars = ax.bar(comparison_data['Model'], values, color=colors[:len(results)], 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(values) * 1.15])
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}' if val < 1 else f'{val:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Combined metrics in last subplot
ax = axes[1, 2]
x = np.arange(len(comparison_data['Model']))
width = 0.15
for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']):
    values = [v if v <= 1 else v/100 for v in comparison_data[metric]]  # Normalize
    offset = width * (i - 2)
    ax.bar(x + offset, values, width, label=metric, alpha=0.8)

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('All Metrics Combined', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_data['Model'], rotation=45, ha='right')
ax.legend(fontsize=10)
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/comparison/performance_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Performance metrics comparison saved")

# 2. Training and Inference Performance
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Training Time
ax = axes[0, 0]
bars = ax.bar(comparison_data['Model'], comparison_data['Training Time (s)'], 
              color=colors[:len(results)], edgecolor='black', linewidth=1.5, alpha=0.8)
ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, comparison_data['Training Time (s)']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Inference Time
ax = axes[0, 1]
bars = ax.bar(comparison_data['Model'], comparison_data['Inference Time (ms)'], 
              color=colors[:len(results)], edgecolor='black', linewidth=1.5, alpha=0.8)
ax.set_ylabel('Inference Time (ms/sample)', fontsize=12, fontweight='bold')
ax.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, comparison_data['Inference Time (ms)']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.2f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Throughput
ax = axes[1, 0]
bars = ax.bar(comparison_data['Model'], comparison_data['Throughput (samp/s)'], 
              color=colors[:len(results)], edgecolor='black', linewidth=1.5, alpha=0.8)
ax.set_ylabel('Throughput (samples/sec)', fontsize=12, fontweight='bold')
ax.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, comparison_data['Throughput (samp/s)']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Memory Usage
ax = axes[1, 1]
bars = ax.bar(comparison_data['Model'], comparison_data['Memory (MB)'], 
              color=colors[:len(results)], edgecolor='black', linewidth=1.5, alpha=0.8)
ax.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
ax.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, comparison_data['Memory (MB)']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.1f}MB', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('Training and Inference Performance Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/comparison/performance_time.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Training/inference performance comparison saved")

# 3. Model Size and Parameters
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Parameters
ax = axes[0]
bars = ax.bar(comparison_data['Model'], comparison_data['Parameters'], 
              color=colors[:len(results)], edgecolor='black', linewidth=1.5, alpha=0.8)
ax.set_ylabel('Number of Parameters', fontsize=12, fontweight='bold')
ax.set_title('Model Parameters Comparison', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, comparison_data['Parameters']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Model Size
ax = axes[1]
bars = ax.bar(comparison_data['Model'], comparison_data['Model Size (MB)'], 
              color=colors[:len(results)], edgecolor='black', linewidth=1.5, alpha=0.8)
ax.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, comparison_data['Model Size (MB)']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.2f}MB', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('Model Complexity Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/comparison/model_complexity.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Model complexity comparison saved")

# 4. Radar Chart - Overall Performance
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

# Normalize metrics to 0-1 scale for radar chart
metrics_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for idx, model_name in enumerate(comparison_data['Model']):
    values = []
    for metric in metrics_radar:
        val = comparison_data[metric][idx]
        # Normalize accuracy to 0-1 if it's in percentage
        if val > 1:
            val = val / 100
        values.append(val)
    values += values[:1]  # Complete the circle
    
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_radar, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.set_title('Overall Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
ax.grid(True)

plt.tight_layout()
plt.savefig('results/comparison/radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Radar chart saved")

# 5. Comprehensive Dashboard
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)

# Best model identification
best_acc_idx = comparison_data['Accuracy'].index(max(comparison_data['Accuracy']))
best_f1_idx = comparison_data['F1-Score'].index(max(comparison_data['F1-Score']))
fastest_train_idx = comparison_data['Training Time (s)'].index(min(comparison_data['Training Time (s)']))
fastest_infer_idx = comparison_data['Inference Time (ms)'].index(min(comparison_data['Inference Time (ms)']))

# Title and summary
ax = fig.add_subplot(gs[0, :])
ax.axis('off')
summary_text = [
    "COMPREHENSIVE MODEL COMPARISON DASHBOARD",
    "",
    f"Models Compared: {len(results)}",
    f"Best Accuracy: {comparison_data['Model'][best_acc_idx]} ({comparison_data['Accuracy'][best_acc_idx]:.2f}%)",
    f"Best F1-Score: {comparison_data['Model'][best_f1_idx]} ({comparison_data['F1-Score'][best_f1_idx]:.4f})",
    f"Fastest Training: {comparison_data['Model'][fastest_train_idx]} ({comparison_data['Training Time (s)'][fastest_train_idx]:.1f}s)",
    f"Fastest Inference: {comparison_data['Model'][fastest_infer_idx]} ({comparison_data['Inference Time (ms)'][fastest_infer_idx]:.2f}ms)",
]
y_pos = 0.85
for line in summary_text:
    weight = 'bold' if line == summary_text[0] else 'normal'
    size = 16 if line == summary_text[0] else 12
    ax.text(0.5, y_pos, line, ha='center', fontsize=size, fontweight=weight, family='monospace')
    y_pos -= 0.12

# Accuracy comparison
ax1 = fig.add_subplot(gs[1, :2])
bars = ax1.bar(comparison_data['Model'], comparison_data['Accuracy'], 
               color=colors[:len(results)], edgecolor='black', linewidth=1.5, alpha=0.8)
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Test Accuracy', fontsize=13, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3, axis='y')

# F1-Score comparison
ax2 = fig.add_subplot(gs[1, 2:])
bars = ax2.bar(comparison_data['Model'], comparison_data['F1-Score'], 
               color=colors[:len(results)], edgecolor='black', linewidth=1.5, alpha=0.8)
ax2.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
ax2.set_title('F1-Score (Macro)', fontsize=13, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# Training time comparison
ax3 = fig.add_subplot(gs[2, :2])
bars = ax3.bar(comparison_data['Model'], comparison_data['Training Time (s)'], 
               color=colors[:len(results)], edgecolor='black', linewidth=1.5, alpha=0.8)
ax3.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
ax3.set_title('Training Time', fontsize=13, fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

# Inference time comparison
ax4 = fig.add_subplot(gs[2, 2:])
bars = ax4.bar(comparison_data['Model'], comparison_data['Inference Time (ms)'], 
               color=colors[:len(results)], edgecolor='black', linewidth=1.5, alpha=0.8)
ax4.set_ylabel('Time (ms/sample)', fontsize=11, fontweight='bold')
ax4.set_title('Inference Time', fontsize=13, fontweight='bold')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3, axis='y')

# Model parameters
ax5 = fig.add_subplot(gs[3, :2])
bars = ax5.bar(comparison_data['Model'], comparison_data['Parameters'], 
               color=colors[:len(results)], edgecolor='black', linewidth=1.5, alpha=0.8)
ax5.set_ylabel('Parameters', fontsize=11, fontweight='bold')
ax5.set_title('Model Parameters', fontsize=13, fontweight='bold')
ax5.tick_params(axis='x', rotation=45)
ax5.grid(True, alpha=0.3, axis='y')

# Model size
ax6 = fig.add_subplot(gs[3, 2:])
bars = ax6.bar(comparison_data['Model'], comparison_data['Model Size (MB)'], 
               color=colors[:len(results)], edgecolor='black', linewidth=1.5, alpha=0.8)
ax6.set_ylabel('Size (MB)', fontsize=11, fontweight='bold')
ax6.set_title('Model Size', fontsize=13, fontweight='bold')
ax6.tick_params(axis='x', rotation=45)
ax6.grid(True, alpha=0.3, axis='y')

plt.savefig('results/comparison/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Comprehensive dashboard saved")

print()
print("✓ All comparison visualizations created!")
print()

# ==================== SAVE COMPREHENSIVE REPORT ====================

print("Creating comparison report...")

with open('results/comparison/comparison_report.txt', 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("COMPREHENSIVE MODEL COMPARISON REPORT\n")
    f.write("=" * 100 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Models Compared: {len(results)}\n\n")
    
    f.write("COMPARISON TABLE\n")
    f.write("-" * 100 + "\n")
    f.write(df.to_string(index=False))
    f.write("\n\n")
    
    f.write("BEST PERFORMERS\n")
    f.write("-" * 100 + "\n")
    f.write(f"Best Accuracy:       {comparison_data['Model'][best_acc_idx]:<15} {comparison_data['Accuracy'][best_acc_idx]:.2f}%\n")
    f.write(f"Best Precision:      {comparison_data['Model'][comparison_data['Precision'].index(max(comparison_data['Precision']))]:<15} {max(comparison_data['Precision']):.4f}\n")
    f.write(f"Best Recall:         {comparison_data['Model'][comparison_data['Recall'].index(max(comparison_data['Recall']))]:<15} {max(comparison_data['Recall']):.4f}\n")
    f.write(f"Best F1-Score:       {comparison_data['Model'][best_f1_idx]:<15} {comparison_data['F1-Score'][best_f1_idx]:.4f}\n")
    f.write(f"Best ROC-AUC:        {comparison_data['Model'][comparison_data['ROC-AUC'].index(max(comparison_data['ROC-AUC']))]:<15} {max(comparison_data['ROC-AUC']):.4f}\n")
    f.write(f"Fastest Training:    {comparison_data['Model'][fastest_train_idx]:<15} {comparison_data['Training Time (s)'][fastest_train_idx]:.1f}s\n")
    f.write(f"Fastest Inference:   {comparison_data['Model'][fastest_infer_idx]:<15} {comparison_data['Inference Time (ms)'][fastest_infer_idx]:.2f}ms\n")
    f.write(f"Smallest Model:      {comparison_data['Model'][comparison_data['Parameters'].index(min(comparison_data['Parameters']))]:<15} {min(comparison_data['Parameters']):,} params\n")
    f.write("\n")
    
    f.write("QUANTUM ADVANTAGE ANALYSIS\n")
    f.write("-" * 100 + "\n")
    
    # Compare quantum vs classical
    if 'CNN + DNN' in results and 'QCNN + QNN' in results:
        classical_acc = comparison_data['Accuracy'][comparison_data['Model'].index('CNN + DNN')]
        quantum_acc = comparison_data['Accuracy'][comparison_data['Model'].index('QCNN + QNN')]
        improvement = quantum_acc - classical_acc
        f.write(f"Classical Baseline (CNN + DNN):  {classical_acc:.2f}%\n")
        f.write(f"Full Quantum (QCNN + QNN):       {quantum_acc:.2f}%\n")
        f.write(f"Improvement:                     {improvement:+.2f}% ({improvement/classical_acc*100:+.2f}%)\n\n")
    
    f.write("DETAILED MODEL INFORMATION\n")
    f.write("-" * 100 + "\n\n")
    
    for model_name, data in results.items():
        f.write(f"{model_name}\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Type: {data['model_type']}\n")
        f.write(f"  Feature Extractor: {data['feature_extractor']}\n")
        f.write(f"  Classifier: {data['classifier']}\n")
        f.write(f"  Parameters: {data['architecture']['total_parameters']:,}\n")
        f.write(f"  Model Size: {data['architecture']['model_size_mb']:.2f} MB\n")
        f.write(f"  Training Time: {data['training_results']['training_time']}\n")
        f.write(f"  Best Epoch: {data['training_results']['best_epoch']}\n")
        f.write(f"  Test Accuracy: {data['test_metrics']['accuracy']:.2f}%\n")
        f.write(f"  Test F1-Score: {data['test_metrics']['f1_macro']:.4f}\n")
        f.write(f"  Inference Time: {data['performance']['avg_inference_time_ms']:.2f} ms/sample\n")
        f.write("\n")

print("✓ Comparison report saved")
print()

# Save comprehensive JSON
comparison_summary = {
    'timestamp': datetime.now().isoformat(),
    'models_compared': len(results),
    'comparison_data': comparison_data,
    'best_performers': {
        'accuracy': {'model': comparison_data['Model'][best_acc_idx], 'value': comparison_data['Accuracy'][best_acc_idx]},
        'f1_score': {'model': comparison_data['Model'][best_f1_idx], 'value': comparison_data['F1-Score'][best_f1_idx]},
        'fastest_training': {'model': comparison_data['Model'][fastest_train_idx], 'value': comparison_data['Training Time (s)'][fastest_train_idx]},
        'fastest_inference': {'model': comparison_data['Model'][fastest_infer_idx], 'value': comparison_data['Inference Time (ms)'][fastest_infer_idx]},
    },
    'full_results': results
}

with open('results/comparison/comparison_summary.json', 'w') as f:
    json.dump(comparison_summary, f, indent=2)

print("✓ Comprehensive JSON summary saved")
print()

# ==================== FINAL SUMMARY ====================

print("=" * 100)
print("  COMPARISON COMPLETE!")
print("=" * 100)
print()
print(f"📊 Analyzed {len(results)} models")
print()
print("📁 Results saved to results/comparison/:")
print("  ✓ comparison_table.csv")
print("  ✓ comparison_report.txt")
print("  ✓ comparison_summary.json")
print("  ✓ performance_metrics.png")
print("  ✓ performance_time.png")
print("  ✓ model_complexity.png")
print("  ✓ radar_chart.png")
print("  ✓ comprehensive_dashboard.png")
print()
print("🏆 Best Model (Accuracy):", comparison_data['Model'][best_acc_idx], 
      f"({comparison_data['Accuracy'][best_acc_idx]:.2f}%)")
print("🏆 Best Model (F1-Score):", comparison_data['Model'][best_f1_idx], 
      f"({comparison_data['F1-Score'][best_f1_idx]:.4f})")
print()
