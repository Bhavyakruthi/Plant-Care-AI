# 📝 Multimodal Plant Disease Prediction: Final Metrics for Journal Writing

## 1. Global Performance Metrics
| Model          |   Accuracy (%) |   Precision (%) |   Recall (%) |   F1 Score (%) |
|:---------------|---------------:|----------------:|-------------:|---------------:|
| Text_BioBERT   |          66.72 |           67.75 |        66.72 |          66.19 |
| Image_CNN_QNN  |          95.77 |           95.87 |        95.77 |          95.77 |
| Ensemble_Fused |          97.06 |           97.08 |        97.06 |          97.06 |

## 2. Detailed Ensemble Performance (α=0.46)
### Per-Class F1 Scores (Optimized Ensemble)
| Class                                       |   Precision |   Recall |   F1-Score |
|:--------------------------------------------|------------:|---------:|-----------:|
| Pepper__bell___Bacterial_spot               |      0.9933 |   0.9933 |     0.9933 |
| Pepper__bell___healthy                      |      0.9955 |   1      |     0.9978 |
| Potato___Early_blight                       |      0.9933 |   1      |     0.9967 |
| Potato___Late_blight                        |      0.9932 |   0.9733 |     0.9832 |
| Potato___healthy                            |      0.9565 |   1      |     0.9778 |
| Tomato_Bacterial_spot                       |      0.978  |   0.9867 |     0.9823 |
| Tomato_Early_blight                         |      0.9507 |   0.9    |     0.9247 |
| Tomato_Late_blight                          |      0.9539 |   0.9718 |     0.9628 |
| Tomato_Leaf_Mold                            |      0.9783 |   0.9441 |     0.9609 |
| Tomato_Septoria_leaf_spot                   |      0.9458 |   0.9697 |     0.9576 |
| Tomato_Spider_mites_Two_spotted_spider_mite |      0.9557 |   0.97   |     0.9628 |
| Tomato__Target_Spot                         |      0.9194 |   0.9327 |     0.926  |
| Tomato__Tomato_YellowLeaf__Curl_Virus       |      0.9917 |   0.9876 |     0.9896 |
| Tomato__Tomato_mosaic_virus                 |      0.9123 |   0.9286 |     0.9204 |
| Tomato_healthy                              |      0.9955 |   0.9778 |     0.9865 |

## 3. Comparison of F1 Scores Across Modalities
| Class                                       |   BioBERT-F1 |   CNN-QNN-F1 |   Ensemble-F1 |
|:--------------------------------------------|-------------:|-------------:|--------------:|
| Pepper__bell___Bacterial_spot               |       0.8077 |       0.9967 |        0.9933 |
| Pepper__bell___healthy                      |       0.9021 |       1      |        0.9978 |
| Potato___Early_blight                       |       0.7692 |       0.9967 |        0.9967 |
| Potato___Late_blight                        |       0.7922 |       0.9763 |        0.9832 |
| Potato___healthy                            |       0.6842 |       0.9565 |        0.9778 |
| Tomato_Bacterial_spot                       |       0.5347 |       0.9669 |        0.9823 |
| Tomato_Early_blight                         |       0.5437 |       0.8776 |        0.9247 |
| Tomato_Late_blight                          |       0.7287 |       0.9647 |        0.9628 |
| Tomato_Leaf_Mold                            |       0.6834 |       0.9111 |        0.9609 |
| Tomato_Septoria_leaf_spot                   |       0.3259 |       0.9458 |        0.9576 |
| Tomato_Spider_mites_Two_spotted_spider_mite |       0.6056 |       0.9453 |        0.9628 |
| Tomato__Target_Spot                         |       0.4855 |       0.907  |        0.926  |
| Tomato__Tomato_YellowLeaf__Curl_Virus       |       0.837  |       0.979  |        0.9896 |
| Tomato__Tomato_mosaic_virus                 |       0.2553 |       0.8908 |        0.9204 |
| Tomato_healthy                              |       0.7196 |       0.982  |        0.9865 |

## 5. Explainable AI (XAI) Gallery Results
A comprehensive XAI gallery was generated for a representative subset of the test data (5 samples per class).

- **Gallery Size:** 75 samples (5 per class)
- **Gallery Accuracy:** 93.33% (Synchronized with 224x224 Two-Stage Pipeline)
- **XAI Confirmation:** Grad-CAM hot spots correctly align with leaf lesions across all categories.
- **Methodology:**
    - **Grad-CAM:** Applied to the CNN-QNN image backbone to visualize spatial attention.
    - **Symptom Importance:** Extracted from text descriptions using keyword weighting.
    - **Hybrid Descriptions:** Utilized pre-generated technical morphological features combined with live Gemini 2.5 Flash synthesis for error correction.

## 7. Interpreting Grad-CAM Visualizations
To understand the XAI output (e.g., `sample_3_xai.png`), follow this interpretation guide:

- **Heatmap Color Gradient:**
    - **Deep Red/Yellow:** High-importance regions. The model strongly associated these textures/patterns with the predicted disease. In Pepper Bacterial Spot, this should align with necrotic spots and chlorotic halos.
    - **Blue/Cyan:** Neutral or low-importance regions. These areas (typically healthy green tissue) did not contribute significantly to the decision.
- **Model Trustworthiness:**
    - If the "hot" spots overlap with actual lesions (as described in the metadata), the model is "looking" at the right features, confirming it has learned pathological patterns rather than background noise.
    - For `sample_3` (Pepper Bacterial Spot), observe how the red regions highlight the small irregular spots and the margins where chlorotic halos are concentrated.
- **Decision Correlation:**
    - High confidence (e.g., >80%) coupled with precise Grad-CAM localization indicates a robust, reliable prediction.
