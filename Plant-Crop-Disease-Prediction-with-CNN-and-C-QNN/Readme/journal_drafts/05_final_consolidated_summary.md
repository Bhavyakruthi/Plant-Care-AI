# Final Consolidated Summary of Experiments

## Abstract
This project studies plant disease diagnosis over three phases: a vision-based phase, a text-based phase, and a hybrid multimodal phase. The full repository shows a progression from image-only learning, to symptom-based language modeling, to a fused model that combines both modalities. The final multimodal system is the strongest overall solution currently present in the repository, and the attached fusion-weight plot shows that the best operating region is around alpha = 0.65, where the image branch carries most of the weight but the text branch still contributes.

## 1. Phase Overview

### Phase 1: Vision-based models
The vision phase compares classical CNN feature extraction with quantum-assisted classification. The best reported image-only result is the CNN + QNN model with 96.48% mean accuracy under 10-fold cross-validation, compared to the CNN+NN baseline at 95.16%.

### Phase 2: Text-based models
The text phase builds symptom descriptions using Gemini, preprocesses them for BERT-based classification, and compares classical machine learning baselines with a BioBERT transformer. The requested without-vs-with BioBERT comparison is now explicit: without BioBERT (best SVM) gives 72.99% accuracy, while with BioBERT in text-only CV gives 90.23% +/- 0.35% accuracy. The attention and word-frequency plots show that the generated descriptions are genuinely symptom focused, with repeated emphasis on words such as lesions, veins, color, surface, irregular, and dark.

### Phase 3: Hybrid models
The hybrid phase combines the strongest vision and text branches. In the fold-level summary, the fused model reports 96.20% +/- 0.60%. In archived comparison artifacts, multimodal appears in the high-96% range and remains clearly stronger than strict text-only performance.

## 2. Key Results Table

| Phase | Best Model | Accuracy |
|------|------------|----------|
| Phase 1 | CNN + QNN | 96.48% +/- 0.46% |
| Phase 2 | BioBERT-based text model | 90.23% +/- 0.35% |
| Phase 3 | Ensemble_Fused | 96.20% +/- 0.60% |

| Vision Comparison | Model | Accuracy | F1-score |
|------|------------|----------|----------|
| Classical baseline | CNN + NN | 95.16% +/- 0.56% | 94.43% +/- 0.78% |
| Quantum-assisted | CNN + QNN | 96.48% +/- 0.46% | 96.47% +/- 0.45% |

| Hybrid Comparison Track | Image Branch | Fused Branch |
|------|------------|----------|
| Fold-level CV outputs | 96.48% | 96.20% |
| Journal-export summary artifacts | 95.77% | 97.06% |

| Text Comparison | Best Model | Accuracy | F1-score |
|------|------------|----------|----------|
| Without BioBERT | SVM (TF-IDF) | 72.99% | 73.14% |
| With BioBERT | Hybrid BioBERT text model | 90.23% +/- 0.35% | 90.20% +/- 0.37% |

## 3. Main Technical Findings

### Vision branch
- ResNet18 provides strong 512-dimensional feature embeddings.
- The QNN classifier performs well on top of the extracted features.
- Quantum assistance is most useful after compression rather than as a raw end-to-end replacement.

### Text branch
- Gemini-generated symptom descriptions create a usable text dataset.
- BioBERT attention highlights medically relevant words.
- Classical TF-IDF baselines are competitive but weaker than the deep language model in the refined comparison.

### Hybrid branch
- Fusion consistently improves the final output.
- The ensemble is stronger than either the image or text model alone.
- Explainability confirms that the system is using meaningful visual and linguistic cues.
- The without-vs-with BioBERT text comparison shows a large gain in standalone text accuracy.
- Compared to CNN+NN baseline, CNN+QNN improves by +1.32 percentage points in accuracy and about 27.27% relative error reduction.

## 4. Final Model Recommendation
For the overall project, the best final model is the fused multimodal ensemble.

For the phase-specific models:
- Vision phase: CNN + QNN
- Text phase: BioBERT-based model
- Hybrid phase: Ensemble_Fused

## 5. Discussion
The project shows a clear progression in capability.

1. Vision alone gives a very strong baseline.
2. Text alone is weaker, but it adds complementary semantic information.
3. Fusion produces the best overall result because it combines the strengths of both modalities.

Statistical-test artifacts in the repository also show p-values effectively at 0 for the modality contrasts, indicating strong separability between strict text-only and vision/hybrid settings.

The quantum component contributes most clearly in the vision pipeline, where it acts as a nonlinear classifier on top of compact features. The text component contributes by adding symptom semantics and interpretability through attention. The final fusion model uses both to improve accuracy and robustness. The performance-speed report also shows that the pipeline is operationally practical on GPU, with roughly 23.58 ms for image inference, 13.58 ms for text inference, and 37.22 ms end-to-end.

## 6. Closing Statement
The repository is ready for a journal-style write-up. The five draft reports in this folder can be merged into a single manuscript with the following structure:

- Abstract and overall introduction
- Phase 1: Vision study
- Phase 2: Text study
- Phase 3: Hybrid study
- Explainability and final discussion

This layout matches the experimental progression already present in the code and result artifacts.
