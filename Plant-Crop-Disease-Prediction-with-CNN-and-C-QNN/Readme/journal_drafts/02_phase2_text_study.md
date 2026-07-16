# Phase 2: Text-Based Plant Disease Diagnosis Study

## Abstract
This phase studies the text-only branch of the project, where plant disease descriptions are generated, cleaned, and used for disease classification. The text pipeline combines synthetic description generation using Gemini, linguistic preprocessing, classical machine learning baselines, BioBERT-based deep learning, and attention-based explainability. The repository shows that text alone is weaker than vision, but it still contributes meaningful complementary information, especially when the final hybrid ensemble is built. The attached outputs also show that the text dataset is not arbitrary: it has a measurable severity distribution, with Medium cases dominating the corpus, followed by Low and High severity classes.

## 1. How the Text Dataset Was Created
The text dataset is built from image-derived symptom descriptions. The backend service uses Gemini 2.5 Flash to generate technical botanical descriptions from plant leaf images. The prompt instructs the model to describe:

- Leaf morphology
- Lesion color and shape
- Symptom distribution
- Curling, wilting, spotting, and mold patterns

A critical rule in the prompt is to avoid naming the plant directly. This forces the description to focus on symptoms rather than on a memorized class label. The result is a text corpus that acts as a symptom-oriented proxy for disease diagnosis. The word-frequency plots in the outputs confirm that the generated descriptions consistently emphasize biologically meaningful terms such as leaf, lesions, surface, veins, irregular, brown, chlorotic, and necrotic.

The project also stores structured metadata alongside the text so the final text model can combine language and tabular signals.

## 2. Preprocessing Pipeline
The text preprocessing flow is documented across the backend utilities and training scripts. The major steps are:

- Text cleaning and normalization
- Lowercasing and punctuation cleanup
- Tokenization with BioBERT / PubMedBERT tokenizer
- Attention mask creation
- Truncation and padding to a fixed sequence length
- Extraction of structured features from the metadata
- Label encoding for the disease classes

The exploratory plots in the outputs show a mean text length of about 581 characters and a mean word count of about 80 tokens, which is useful for reporting because it confirms that the descriptions are long enough to carry disease-specific information but still short enough for transformer-based processing.

The `TextModel` class loads the trained hybrid BioBERT classifier and applies the same preprocessing path during inference. This keeps training and evaluation consistent.

## 3. Classical Machine Learning Models
The first text experiments use TF-IDF style feature vectors with classical classifiers.

### Models tested
- Support Vector Machine
- Random Forest
- Logistic Regression

### Results from the stored comparison tables
The current repository outputs show the following baseline scores (without BioBERT):

- SVM: 72.99% accuracy
- Random Forest: 69.07% accuracy
- Logistic Regression: 72.29% accuracy

Among the classical baselines, SVM is the strongest single non-neural text model. The full comparison table also shows that a hard-voting ensemble slightly improves on the individual classical methods, which is a useful reference point for the final multimodal fusion stage.

## 4. BERT and BioBERT Models
The deep text branch uses a BioBERT-style transformer, specifically `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`. The model architecture in the repository is a hybrid classifier that concatenates:

- The BERT pooled output
- Structured features extracted from the sample metadata

This combined vector is passed through two fully connected layers and a final classifier. The model can optionally return attentions, which is important for later explainability.

### Transformer result summary
Using the 10-fold file outputs/text_cv/text_only_cv_metrics.csv (with BioBERT), the text branch reports:

- Accuracy: 90.23% +/- 0.35%
- Precision: 90.29% +/- 0.42%
- Recall: 90.23% +/- 0.35%
- F1-score: 90.20% +/- 0.37%

For completeness, the older cross-modal comparison table in outputs/10fold_cv_comparison_table.csv reports a stricter and lower text-only setting at 49.53% +/- 0.96%. Because the current request is to compare without BioBERT and with BioBERT, the primary text comparison for this report uses the explicit text-only CV file above.

### Without BioBERT vs With BioBERT (Text Branch)

| Setting | Best Model | Accuracy | F1-score |
|------|------------|----------|----------|
| Without BioBERT | SVM (TF-IDF) | 72.99% | 73.14% |
| With BioBERT | Hybrid BioBERT text model | 90.23% +/- 0.35% | 90.20% +/- 0.37% |

This indicates a clear gain of about +17.24 percentage points in accuracy when moving from the best non-BioBERT text baseline to the BioBERT-based model under the current text-only CV setup.

## 5. Attention-Based Model and Explainability
The text model is inherently attention-based because it relies on BioBERT self-attention. The repository includes explicit attention extraction logic that averages attention across heads and layers and then focuses on the [CLS] token attention over the input sequence.

The interpretability report notes that the attention maps highlight words such as:

- yellowing
- spots
- lesions

This is important because it shows that the model is attending to clinically meaningful symptom words rather than random tokens.

## 6. Final Text Model Selection
The best standalone text model in the repository is the BioBERT-based classifier. It is selected because:

- It is the strongest neural text model in the available comparisons.
- It supports attention-based interpretability.
- It is compatible with structured features.
- It becomes the text branch used in the multimodal fusion stage.

If the comparison is restricted to purely classical text baselines, SVM is the strongest non-transformer model. If the comparison includes deep language models, BioBERT is the preferred choice for the final pipeline because it generalizes better as a feature-rich language encoder and supports explainability.

## 7. Discussion
The text-only branch performs worse than the vision branch because the symptom descriptions are an indirect representation of the disease. The pipeline depends on the quality of the Gemini-generated text, the quality of the preprocessing, and the amount of discriminative language in each sample. Even so, the text branch still contributes useful complementary evidence, especially for classes with similar visual patterns but different symptom phrasing. This is visible in the attention plots, where the model focuses on symptom-bearing tokens rather than boilerplate words.

The attention maps strengthen the report because they provide a direct explanation of why the model made a prediction. This is especially useful in a journal-style study because it converts the text model from a black box into an interpretable diagnostic component.

## 8. Phase-2 Conclusion
Phase 2 shows that the text branch is viable, interpretable, and useful as a complementary modality. In the explicit without-vs-with BioBERT comparison, the BioBERT-based model is clearly stronger than the classical non-BioBERT baselines. BioBERT is therefore selected as the final text model for downstream fusion.
