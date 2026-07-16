# Interpretability and Explainability Report

## Abstract
This report documents the explainability layer used across the project. It covers Grad-CAM for image-based diagnosis, CNN filter visualization, quantum weight inspection, BioBERT attention maps, and feature-space projections such as t-SNE. These tools are essential because the project is not only about accuracy; it is also about showing why the model made a prediction.

## 1. Why Explainability Was Needed
Plant disease diagnosis is a high-stakes classification problem. A model that predicts the correct class but focuses on the wrong image region or the wrong word pattern is not trustworthy. The explainability pipeline in the repository therefore checks whether the model is learning disease-relevant evidence rather than background noise or label shortcuts.

## 2. Image Explainability

### 2.1 Grad-CAM
Grad-CAM is used on the image branch to visualize the areas that contribute most to the final prediction. The attached gallery confirms that the heatmaps align with actual lesions and symptom regions rather than the background. In correct predictions, the hot region is centered on the diseased area; in errors, the heatmap often still tracks a plausible lesion region, which means the mistake is usually class confusion rather than attention failure.

What this means in practice:

- Hot regions correspond to diseased tissue, spots, and necrotic zones.
- Cooler regions correspond to healthy or irrelevant background pixels.
- Correct alignment increases trust in the classifier.

### 2.2 CNN filter maps
The CNN filter visualizations show what intermediate convolutional layers learn.

- Early filters detect edges, color gradients, and boundaries.
- Deeper filters detect texture patterns, lesion clusters, and disease-specific structures.
- The attached filter grids show that later layers become increasingly structured, while the feature maps highlight leaf shape, mottled texture, and lesion concentration.
- These maps support the claim that the CNN backbone learns biologically meaningful representations.

### 2.3 Quantum weight visualizations
The quantum model also exposes its learned weights. These weight maps are used as a compact diagnostic view of how the QNN transforms the extracted image features. Although they are not as intuitive as Grad-CAM, they help show that the quantum head is not random; it is learning a structured nonlinear mapping. The attached quantum weight plot shows a nontrivial signed pattern across qubits rather than near-zero noise.

## 3. Text Explainability

### 3.1 BioBERT attention maps
The text branch uses self-attention to identify the most important tokens in the generated disease description. The attached attention maps show that the model focuses on symptom terms and morphological descriptors such as:

- morphology
- color
- veins
- lesions
- dark
- irregular
- surface
- prominent

The token-importance plots show the same pattern repeatedly: the model assigns strong weight to lesion-related and morphology-related words, while filler words receive less importance.

This matters because these are exactly the words a human expert would consider relevant in a disease note.

### 3.2 Why attention is useful here
Attention maps make the text model auditable. They show whether the classifier is relying on symptom vocabulary, morphological descriptors, or irrelevant filler text. That is particularly important because the descriptions are generated using Gemini and then normalized through preprocessing.

## 4. Feature-Space Visualization
The project uses t-SNE and related projections to inspect the learned feature space.

### What the plots show
- Better-separated clusters mean the model has learned discriminative features.
- Overlapping clusters indicate confusion between visually or semantically similar diseases.
- Comparing the image and text feature spaces helps determine which modality is more separable.

The attached t-SNE figures show that both image and text feature spaces produce meaningful structure, but the image branch is more strongly clustered, which matches the higher image accuracy. The text feature space is more fragmented and elongated, which is consistent with the lower standalone text accuracy and with the reliance on attention to recover semantic signal.

## 5. Error Analysis
Misclassification analysis is used to inspect the failure modes.

Typical errors include:

- Confusion between different leaf spot diseases
- Confusion between healthy and mildly symptomatic classes
- Confusion among tomato diseases with similar visual symptoms

The attached Grad-CAM montage makes this concrete: some samples are correctly classified with the heatmap centered on the lesion, while others confuse Pepper bacterial spot with Potato early blight, Tomato healthy with Tomato spider mites, or Tomato late blight with Tomato early blight. These are the exact kinds of failures expected in a fine-grained plant pathology setting.

This error profile is expected in plant disease recognition and helps justify why the multimodal fusion model is valuable.

## 6. What We Infer from the Explainability Results
The explainability outputs support three conclusions:

1. The image branch focuses on real lesion regions, so the classifier is not learning spurious background cues.
2. The text branch attends to symptom keywords that are biologically meaningful.
3. The hybrid model benefits from both spatial and semantic evidence, which explains the accuracy gain in the final fusion stage.
4. The model errors are concentrated in visually similar disease pairs, which suggests the learned features are sensible rather than random.

## 7. Interpretability Conclusion
The interpretability pipeline strengthens the project substantially. It shows that the models are not only accurate but also understandable. Grad-CAM, attention maps, CNN filters, and feature projections provide complementary evidence that the system is learning the correct disease signatures.
