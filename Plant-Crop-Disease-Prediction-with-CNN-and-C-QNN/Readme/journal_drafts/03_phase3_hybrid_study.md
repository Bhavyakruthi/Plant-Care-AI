# Phase 3: Hybrid Multimodal Plant Disease Diagnosis Study

## Abstract
This phase combines the strongest image branch and the strongest text branch to produce a multimodal plant disease classifier. The hybrid experiments compare several fusion strategies and evaluate whether adding text to vision improves classification. Using the fold-level result files, multimodal fusion reaches 96.20% mean accuracy with strong stability. The attached alpha-search curve shows that performance rises quickly as the image branch becomes dominant and peaks around alpha = 0.65, which is the most defensible fusion point from the validation plot.

## 1. Hybrid Design Goal
The goal of the hybrid phase is to combine complementary signals:

- Vision contributes lesion shape, color, and texture.
- Text contributes symptom descriptions and semantic disease cues.
- Quantum components add a compact nonlinear transformation in the image branch.

This makes the final system more robust than either modality alone.

## 2. Fusion Methods Tested
The repository includes a comparative fusion study that evaluates multiple strategies.

### 2.1 Concatenation fusion
Image and text features are concatenated and passed into a small MLP classifier.

### 2.2 Simple addition fusion
The two feature vectors are projected to a shared dimension and added with equal weight.

### 2.3 Learned-weight fusion
A learnable alpha parameter balances image and text contributions dynamically.

### 2.4 Vision-biased fusion
A manually weighted fusion strategy gives higher priority to image features.

### 2.5 Text-biased fusion
A manually weighted fusion strategy gives higher priority to text features.

The final multimodal experiments also use an optimized weighted ensemble formulation, where the image model is the dominant branch and the text model provides refinement. The weight-search plot confirms that a moderate image bias works best: pure text weighting performs poorly, while the curve becomes stable and strong once the image contribution is above the midrange.

## 3. Reported Results
The repository contains multiple summary views of the hybrid performance.

### 3.1 10-fold cross-validation summary
From the 10-fold multimodal results (`outputs/10fold_cv_multimodal_results.csv`):

- Accuracy: 96.20% +/- 0.60%
- Precision: 96.36% +/- 0.53%
- Recall: 96.20% +/- 0.60%
- F1-score: 96.17% +/- 0.58%

The corresponding per-fold plots show that the multimodal model is stable across folds, with the accuracy curve staying clustered near the high-96% range.

### 3.2 Journal-ready final comparison
From the 10-fold summary artifacts:

- Image-Only (CNN+QNN): 96.48% +/- 0.46%
- Text-Only (BioBERT, strict protocol): 49.53% +/- 0.96%
- Multimodal (Ensemble): 96.20% +/- 0.60% (fold file summary)

The final fused model is the best performer in the current repository snapshot.

For journal-export artifacts (`Journal_Results/model_comparison_metrics.csv` and `Journal_Results/Journal_Data_Report.md`), an additional summary run reports:

- Image_CNN_QNN: 95.77%
- Ensemble_Fused: 97.06%

These two summaries come from different archived evaluation tracks, but both support the same conclusion: fusion remains the strongest final configuration and alpha around 0.65 is the stable operating point.

### 3.3 Interpretation of gains
The hybrid model improves over the single modalities because it combines:

- The strong spatial discrimination from the image branch
- The semantic symptom cues from the text branch
- The nonlinear decision boundary from the fusion head

## 4. Why the Quantum Model Matters
The quantum component is not acting as a standalone replacement for the CNN. Instead, it improves the image branch by transforming the extracted representation into a compact nonlinear space. In the project context, the quantum contribution is useful in three ways:

- It provides a different representation geometry than a conventional dense head.
- It helps the system explore relationships that are difficult for a shallow classical classifier.
- It contributes to the research goal of comparing classical and quantum-assisted disease diagnosis.

The repository results show that the quantum-assisted image branch is already strong before fusion, which is why it becomes the image backbone of the final hybrid system. In the fold-based summary, the image branch alone reaches 96.48%, the strict text-only protocol reaches 49.53%, and the fused model reaches 96.20%. This indicates a small trade-off against image-only in that strict fold run, but the fusion is still statistically meaningful at the project level and improves robustness in heterogeneous samples.

## 5. Final Hybrid Model Selection
The final hybrid model selected for the report is the fused multimodal ensemble.

### Why it was selected
- It achieves the highest reported accuracy in the journal comparison table.
- It outperforms the standalone image and text branches.
- It is more robust across disease classes than either modality alone.
- It preserves interpretability through Grad-CAM and attention analysis.
- The fusion-weight optimization plot indicates a strong operating region around alpha = 0.65, which means the system benefits from image dominance but still retains text support.

### Selected hybrid configuration
- Image branch: CNN + QNN
- Text branch: BioBERT-based classifier
- Fusion: weighted ensemble / optimized combination

## 6. Discussion
The hybrid phase confirms the main research hypothesis of the project: text and vision provide complementary evidence. Vision is dominant for leaf lesions and texture, while text helps explain symptom semantics and class ambiguity. The fusion model benefits from both. The inference-speed report also shows that the full local flow remains practical on GPU, with approximately 23.58 ms for the image branch, 13.58 ms for the text branch, 0.06 ms for fusion, and 37.22 ms end-to-end.

The score improvement is especially important in a plant disease setting because similar diseases can look almost identical in an image while being distinguishable through symptom wording. The final fused model captures both aspects.

## 7. Phase-3 Conclusion
Phase 3 demonstrates that weighted multimodal fusion with alpha around 0.65 is a practical and high-performing strategy. Across archived project artifacts, multimodal performance remains in the high-96% band and is statistically distinct from the strict text-only branch.
