"""
10-Fold Cross-Validation - Comprehensive Report Generator
==========================================================

Consolidates results from all three modality tests (image-only, text-only, multimodal)
and generates comprehensive comparison tables and visualizations.

Author: NLP Project Team
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

def load_results(output_dir):
    """Load results from all three modality tests"""
    print("Loading 10-fold CV results...")
    
    results = {}
    
    # Load image-only results
    image_csv = os.path.join(output_dir, '10fold_cv_image_results.csv')
    if os.path.exists(image_csv):
        results['Image-Only (CNN+QNN)'] = pd.read_csv(image_csv)
        print("  ✅ Image-only results loaded")
    else:
        print("  ⚠️  Image-only results not found")
    
    # Load text-only results
    text_csv = os.path.join(output_dir, '10fold_cv_text_results.csv')
    if os.path.exists(text_csv):
        results['Text-Only (BioBERT)'] = pd.read_csv(text_csv)
        print("  ✅ Text-only results loaded")
    else:
        print("  ⚠️  Text-only results not found")
    
    # Load multimodal results
    multimodal_csv = os.path.join(output_dir, '10fold_cv_multimodal_results.csv')
    if os.path.exists(multimodal_csv):
        results['Multimodal (Ensemble)'] = pd.read_csv(multimodal_csv)
        print("  ✅ Multimodal results loaded")
    else:
        print("  ⚠️  Multimodal results not found")
    
    return results

def create_comparison_table(results):
    """Create comprehensive comparison table"""
    comparison_data = []
    
    for modality, df in results.items():
        comparison_data.append({
            'Modality': modality,
            'Accuracy (Mean ± Std)': f"{df['accuracy'].mean()*100:.2f}% ± {df['accuracy'].std()*100:.2f}%",
            'Accuracy (Min-Max)': f"[{df['accuracy'].min()*100:.2f}%, {df['accuracy'].max()*100:.2f}%]",
            'Precision (Mean ± Std)': f"{df['precision'].mean()*100:.2f}% ± {df['precision'].std()*100:.2f}%",
            'Recall (Mean ± Std)': f"{df['recall'].mean()*100:.2f}% ± {df['recall'].std()*100:.2f}%",
            'F1-Score (Mean ± Std)': f"{df['f1_score'].mean()*100:.2f}% ± {df['f1_score'].std()*100:.2f}%",
        })
    
    return pd.DataFrame(comparison_data)

def perform_statistical_tests(results):
    """Perform statistical significance tests between modalities"""
    print("\nPerforming statistical significance tests...")
    
    modalities = list(results.keys())
    test_results = []
    
    # Pairwise t-tests
    for i in range(len(modalities)):
        for j in range(i+1, len(modalities)):
            mod1 = modalities[i]
            mod2 = modalities[j]
            
            acc1 = results[mod1]['accuracy'].values
            acc2 = results[mod2]['accuracy'].values
            
            # Paired t-test
            t_stat, p_value = scipy_stats.ttest_rel(acc1, acc2)
            
            test_results.append({
                'Comparison': f"{mod1} vs {mod2}",
                't-statistic': f"{t_stat:.4f}",
                'p-value': f"{p_value:.4f}",
                'Significant (α=0.05)': 'Yes' if p_value < 0.05 else 'No'
            })
    
    return pd.DataFrame(test_results)

def create_visualizations(results, output_dir):
    """Create comprehensive visualizations"""
    print("\nGenerating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # 1. Bar chart with error bars
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        modalities = list(results.keys())
        means = [results[mod][metric].mean() * 100 for mod in modalities]
        stds = [results[mod][metric].std() * 100 for mod in modalities]
        
        x_pos = np.arange(len(modalities))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7,
                      color=['#3498db', '#e74c3c', '#2ecc71'])
        
        ax.set_xlabel('Modality', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric_name} (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'10-Fold CV {metric_name} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(modalities, rotation=15, ha='right')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                   f'{mean:.1f}%\n±{std:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_comparison_barplot.png'), dpi=300, bbox_inches='tight')
    print("  ✅ Bar chart saved")
    plt.close()
    
    # 2. Box plot distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        data_to_plot = []
        labels = []
        for modality in results.keys():
            data_to_plot.append(results[modality][metric].values * 100)
            labels.append(modality)
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        ax.set_ylabel(f'{metric_name} (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name} Distribution Across 10 Folds', fontsize=14, fontweight='bold')
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_boxplot_distribution.png'), dpi=300, bbox_inches='tight')
    print("  ✅ Box plot saved")
    plt.close()
    
    # 3. Per-fold comparison line plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    fold_numbers = np.arange(1, 11)
    for modality in results.keys():
        accuracies = results[modality]['accuracy'].values * 100
        ax.plot(fold_numbers, accuracies, marker='o', linewidth=2, markersize=8, label=modality)
    
    ax.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Across 10 Folds - All Modalities', fontsize=14, fontweight='bold')
    ax.set_xticks(fold_numbers)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_per_fold_comparison.png'), dpi=300, bbox_inches='tight')
    print("  ✅ Line plot saved")
    plt.close()
    
    # 4. Confusion matrices (if available)
    cm_files = [
        ('10fold_cv_image_confusion_matrix.npy', 'Image-Only (CNN+QNN)'),
        ('10fold_cv_text_confusion_matrix.npy', 'Text-Only (BioBERT)'),
        ('10fold_cv_multimodal_confusion_matrix.npy', 'Multimodal (Ensemble)')
    ]
    
    available_cms = [(f, name) for f, name in cm_files if os.path.exists(os.path.join(output_dir, f))]
    
    if available_cms:
        n_cms = len(available_cms)
        fig, axes = plt.subplots(1, n_cms, figsize=(8*n_cms, 6))
        if n_cms == 1:
            axes = [axes]
        
        for idx, (cm_file, name) in enumerate(available_cms):
            cm = np.load(os.path.join(output_dir, cm_file))
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            im = axes[idx].imshow(cm_normalized, cmap='Blues', aspect='auto')
            axes[idx].set_title(f'{name}\nAverage Confusion Matrix', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('True', fontsize=11, fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cv_confusion_heatmaps.png'), dpi=300, bbox_inches='tight')
        print("  ✅ Confusion matrix heatmaps saved")
        plt.close()

def main():
    # Configuration
    OUTPUT_DIR = 'd:/COLLEGE FILES/ALL SUBJECTS/SEMESTER 6/Natural Languge Processing/LANGUAGE_MODEL_PROJECT/outputs'
    
    print("="*70)
    print("10-FOLD CROSS-VALIDATION - COMPREHENSIVE REPORT GENERATION")
    print("="*70)
    
    # Load results
    results = load_results(OUTPUT_DIR)
    
    if not results:
        print("\n⚠️  No results found! Please run the individual CV scripts first.")
        return
    
    print(f"\nLoaded results for {len(results)} modalities")
    
    # Create comparison table
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*70)
    
    comparison_df = create_comparison_table(results)
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_csv = os.path.join(OUTPUT_DIR, '10fold_cv_comparison_table.csv')
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\n✅ Comparison table saved to: {comparison_csv}")
    
    # Statistical tests
    if len(results) > 1:
        print("\n" + "="*70)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*70)
        
        stat_tests_df = perform_statistical_tests(results)
        print("\n" + stat_tests_df.to_string(index=False))
        
        # Save statistical tests
        stat_tests_csv = os.path.join(OUTPUT_DIR, '10fold_cv_statistical_tests.csv')
        stat_tests_df.to_csv(stat_tests_csv, index=False)
        print(f"\n✅ Statistical tests saved to: {stat_tests_csv}")
    
    # Create visualizations
    create_visualizations(results, OUTPUT_DIR)
    
    # Generate final report
    report_path = os.path.join(OUTPUT_DIR, '10fold_cv_final_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("10-FOLD CROSS-VALIDATION - FINAL COMPREHENSIVE REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("COMPARISON TABLE:\n")
        f.write("-"*70 + "\n")
        f.write(comparison_df.to_string(index=False) + "\n\n")
        
        if len(results) > 1:
            f.write("STATISTICAL SIGNIFICANCE TESTS:\n")
            f.write("-"*70 + "\n")
            f.write(stat_tests_df.to_string(index=False) + "\n\n")
        
        f.write("="*70 + "\n")
        f.write("SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        # Find best performing modality
        best_modality = max(results.keys(), 
                           key=lambda m: results[m]['accuracy'].mean())
        best_acc = results[best_modality]['accuracy'].mean() * 100
        best_std = results[best_modality]['accuracy'].std() * 100
        
        f.write(f"Best Performing Modality: {best_modality}\n")
        f.write(f"  Mean Accuracy: {best_acc:.2f}% ± {best_std:.2f}%\n\n")
        
        # 95% confidence intervals for all modalities
        f.write("95% Confidence Intervals for Accuracy:\n")
        for modality in results.keys():
            mean_acc = results[modality]['accuracy'].mean() * 100
            std_acc = results[modality]['accuracy'].std() * 100
            ci_95 = 1.96 * std_acc / np.sqrt(10)
            f.write(f"  {modality}: {mean_acc:.2f}% ± {ci_95:.2f}%\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Generated Visualizations:\n")
        f.write("  - cv_comparison_barplot.png\n")
        f.write("  - cv_boxplot_distribution.png\n")
        f.write("  - cv_per_fold_comparison.png\n")
        f.write("  - cv_confusion_heatmaps.png\n")
        f.write("="*70 + "\n")
    
    print(f"\n✅ Final report saved to: {report_path}")
    
    print("\n" + "="*70)
    print("✅ COMPREHENSIVE REPORT GENERATION COMPLETED!")
    print("="*70)
    print("\nGenerated Files:")
    print("  - 10fold_cv_comparison_table.csv")
    print("  - 10fold_cv_statistical_tests.csv")
    print("  - 10fold_cv_final_report.txt")
    print("  - cv_comparison_barplot.png")
    print("  - cv_boxplot_distribution.png")
    print("  - cv_per_fold_comparison.png")
    print("  - cv_confusion_heatmaps.png")
    print("="*70)

if __name__ == "__main__":
    main()
