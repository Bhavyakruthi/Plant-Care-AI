# Phase 1: Vision-Based Plant Disease Detection Study

## Abstract
This phase investigates vision-only plant disease recognition on the PlantVillage dataset using a two-stage hybrid pipeline. The first stage compares classical CNN-based feature extraction with quantum convolutional feature extraction. The second stage trains classifiers on the extracted features and evaluates whether a quantum classifier can improve performance over a purely classical classifier. The completed experiments in the repository show that the vision branch is highly effective, with the best reported image-only model reaching 96.48% mean accuracy under 10-fold cross-validation, while the journal comparison table for the final pipeline records the CNN+QNN branch at 95.77% accuracy.

## 1. Dataset
The vision experiments use the PlantVillage dataset, which contains 19,622 RGB images across 15 plant disease classes. The class set includes healthy and diseased leaves from pepper, potato, and tomato categories. The repository documents a 70/15/15 split into training, validation, and test sets:

- Train: 13,729 images
- Validation: 2,936 images
- Test: 2,957 images

The image-based pipeline is designed for both full-resolution CNN processing and reduced-resolution quantum feature extraction. The working project documentation also confirms that the same class labels are reused across the later text and multimodal phases, which makes the final comparison consistent across modalities. The repository plots also show a balanced class setup in the downstream text and multimodal work, which is important because the same 15 disease labels appear in every phase.

## 2. Data Preprocessing
The vision pipeline uses two preprocessing branches.

### 2.1 Classical branch
The classical branch keeps the standard 224 x 224 RGB image resolution required by pretrained CNN backbones such as ResNet18. The preprocessing flow includes:

- Image resize to 224 x 224
- RGB normalization using the pretrained backbone convention
- Optional augmentation for classical training only
- Stratified splitting to preserve class balance across folds

### 2.2 Quantum branch
The quantum branch reduces images to a compact square input suitable for amplitude encoding and circuit simulation. The project documentation shows the quantum path uses 8 x 8 grayscale representations in the earlier hybrid design and also supports a larger 224 x 224 to 1024-state compression path in the cleaned implementation. Across both versions, the core requirements are the same:

- A fixed-size input vector
- L2 normalization for amplitude encoding
- A small enough feature dimension for quantum simulation
- No random augmentation for quantum inputs

The repository notes that augmentation is intentionally avoided in the quantum path because it changes the encoded quantum state too aggressively and can reduce stability. This is consistent with the project design choice to keep the quantum branch deterministic and to preserve the meaning of each encoded sample.

## 3. Vision-Based Feature Extraction
The project compares two feature extraction strategies.

### 3.1 CNN feature extraction
The classical extractor is based on a pretrained ResNet18 backbone. The backbone outputs a 512-dimensional feature vector per image after global average pooling. The stored feature statistics for this extractor are:

- Feature dimension: 512
- Mean: 0.8701
- Standard deviation: 0.9142
- Minimum: 0.0
- Maximum: 12.4361
- Sparsity: 4.97%

The statistics indicate that the CNN produces dense, information-rich embeddings with only mild sparsity, which is appropriate for downstream classification.

### 3.2 QCNN feature extraction
The quantum feature extractor uses a QCNN-style circuit with amplitude encoding and variational layers. The earlier implementation documents the following design:

- Input compression from image space into a quantum state
- Amplitude encoding with L2 normalization
- 10-qubit circuit in the cleaned implementation path
- Two convolution-like quantum layers
- Pooling between layers
- Ring entanglement
- Output expansion back into a 512-dimensional feature vector

The QCNN branch was created so that the downstream classifier could be trained on a comparable embedding space while still testing whether quantum feature transformations improve separability.

## 4. Vision Classifier Training
Once the features are extracted, the project trains classifiers on top of them.

### 4.0 CNN + NN Baseline (Classical Reference)
The repository includes a 10-fold CNN baseline file (`outputs/cnn_baseline_10fold_results.csv`) that serves as the classical reference before introducing the quantum classifier head.

Baseline CNN+NN (10-fold) summary:

- Accuracy: 95.16% +/- 0.56%
- Precision: 94.60% +/- 0.71%
- Recall: 94.45% +/- 0.78%
- F1-score: 94.43% +/- 0.78%

### 4.1 CNN + QNN
This is the best reported vision-only configuration in the repository. CNN features are extracted once, then passed to a quantum classifier. The 10-fold cross-validation summary reports:

- Accuracy: 96.48% +/- 0.46%
- Precision: 96.62% +/- 0.41%
- Recall: 96.48% +/- 0.46%
- F1-score: 96.47% +/- 0.45%
- 95% confidence interval for accuracy: 96.48% +/- 0.28%

This result is the strongest image-based score currently available in the repository snapshot and serves as the selected vision model for the report. In the later journal-ready multimodal comparison, the same image branch is reported at 95.77% accuracy, which is still clearly stronger than the text branch and remains the image backbone of the final system.

Direct baseline comparison (same project family):

| Vision Model | Accuracy | Precision | Recall | F1-score |
|------|----------|----------|----------|----------|
| CNN + NN (baseline) | 95.16% +/- 0.56% | 94.60% +/- 0.71% | 94.45% +/- 0.78% | 94.43% +/- 0.78% |
| CNN + QNN | 96.48% +/- 0.46% | 96.62% +/- 0.41% | 96.48% +/- 0.46% | 96.47% +/- 0.45% |

This is a +1.32 percentage-point gain in accuracy and about 27.27% relative error reduction compared to the baseline CNN+NN model.

### 4.2 QCNN + DNN and QCNN + QNN
The training framework also supports the opposite pairing, where quantum features are fed into a classical DNN or another quantum classifier. These experiments are implemented in the cleaned project structure and are part of the intended fair-comparison setup.

In practice, these QCNN-feature pipelines performed substantially below expectation (roughly up to ~30% lower than the strongest CNN-feature branch in difficult settings). The main reason is feature quality: the QCNN branch did not preserve enough class-discriminative structure for all 15 disease categories.

Why QCNN feature variants underperformed:

- Information bottleneck during quantum encoding: aggressive reduction and encoding can discard fine lesion texture cues (tiny spots, edge halos, vein-localized color shifts) that are important for class separation.
- Shallow circuit expressivity limits: shallow quantum circuits are used intentionally to avoid barren plateaus, but this also limits representational capacity for high intra-class variability.
- Optimization sensitivity: quantum parameters are more sensitive to initialization and learning-rate mismatch, so convergence can stall before class boundaries are fully shaped.
- Noise-like variability in simulated quantum outputs: small perturbations in input normalization can produce unstable embeddings, which reduces downstream classifier consistency.
- Class imbalance amplifies the issue: low-support classes are affected first when feature quality is weak.

### 4.3 Why CNN+NN Baseline Was Weaker Than CNN+QNN
The classical CNN+NN baseline is strong overall, but it struggled to learn sharp boundaries for a few challenging classes. In the project outputs and error analysis, three recurring difficult classes are:

- Potato___healthy (about 150 samples; minority class)
- Tomato__Tomato_mosaic_virus (about 373 samples; minority class)
- Tomato_Early_blight (about 1000 samples; high visual overlap with related blight/spot classes)

These classes have either low support or high inter-class similarity, so a purely classical dense head tends to form smoother, less selective boundaries.

Why the QNN head helped relative to CNN+NN:

- Nonlinear embedding geometry: the variational quantum head maps CNN features into a different latent geometry, helping separate classes that overlap in classical dense space.
- Better margin shaping for ambiguous classes: entanglement-based transformations can increase separation where lesion morphology is subtle.
- Improved handling of fine-grained decision boundaries: QNN acts as a compact nonlinear classifier on top of strong CNN features, which improves class discrimination in the hard minority/overlap groups.

This is why the final selected vision model remains CNN feature extraction + QNN classification.

## 5. Final Model Selection
The final model selected for the vision phase is the CNN feature extractor combined with the QNN classifier.

### Why this model was selected
- It achieved the best reported vision-only accuracy in the available outputs.
- It keeps the strongest part of the classical pipeline, namely ResNet18 feature extraction.
- The quantum classifier adds a nonlinear decision layer that is well matched to the compact feature space.
- It provides a good balance between accuracy, inference speed, and experimental novelty.

### Selection summary
- Best feature extractor: CNN (ResNet18)
- Best training head: QNN
- Best reported vision accuracy: 96.48% in the vision-only CV run; 95.77% in the final journal comparison table
- Classical baseline reference: CNN+NN at 95.16% (10-fold mean)

## 6. Discussion
The phase-1 results show that the vision problem is already highly learnable with classical deep features. The quantum classifier does not replace the CNN backbone; instead, it works best as a classifier on top of the extracted representation. In practical terms, the project demonstrates that the quantum contribution is most useful when the image features are first compressed into a stable embedding space. The feature-space visualizations in the outputs also show strong cluster formation for the CNN+QNN branch, which supports the quantitative results and becomes the foundation for the later multimodal fusion stage.

## 7. Phase-1 Conclusion
Phase 1 establishes the strongest image pipeline in the project. The completed experiments show that CNN-based feature extraction is the most effective representation strategy, and the QNN classifier gives the best reported image-only performance. This result is the vision baseline used later when the text model and multimodal fusion are added.
