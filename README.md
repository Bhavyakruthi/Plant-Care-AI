# 🌿 PlantCare AI — Multimodal Plant Disease Diagnosis
### Complete Technical Pipeline with Mathematical Derivations

---

## 📌 Project Overview

This project is a **multimodal AI system** for plant disease diagnosis. It combines:
- **Computer Vision** (CNN + Quantum Neural Network) to analyze leaf images
- **Natural Language Processing** (PubMedBERT + structured features) to analyze textual symptom descriptions
- **Generative AI** (Gemini 2.5 Flash) to auto-generate textual symptom descriptions from images
- **Probabilistic Ensemble Fusion** to combine both modal outputs into a final prediction
- **Explainability (XAI)** via Grad-CAM and BERT attention maps

The system classifies a plant leaf image into **15 disease/health categories**.

---

## 🏗️ System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INPUT: Leaf Image                            │
│                      Raw Image ∈ ℝ^{H×W×3}                          │
└───────────────────┬──────────────────────────┬───────────────────────┘
                    │                          │
                    ▼                          ▼
        ┌───────────────────┐     ┌────────────────────────┐
        │   BRANCH 1        │     │   BRANCH 2             │
        │  CNN + QNN        │     │  Gemini 2.5 Flash LLM  │
        │  (Image Branch)   │     │  (Description Gen.)    │
        └────────┬──────────┘     └──────────┬─────────────┘
                 │                           │
                 │                           ▼
                 │                ┌───────────────────────┐
                 │                │  NLP Pipeline         │
                 │                │  (Lemmatization etc.) │
                 │                └──────────┬────────────┘
                 │                           │
                 │                           ▼
                 │                ┌────────────────────────┐
                 │                │  Hybrid PubMedBERT     │
                 │                │  Classifier            │
                 │                │  (Text Branch)         │
                 │                └──────────┬─────────────┘
                 │                           │
                 ▼                           ▼
        p_image ∈ ℝ^15              p_text ∈ ℝ^15
                 │                           │
                 └──────────────┬────────────┘
                                ▼
                 ┌──────────────────────────┐
                 │  Ensemble Fusion (MLE)   │
                 │  P = 0.65·P_img +        │
                 │      0.35·P_text         │
                 └──────────────┬───────────┘
                                ▼
                 ┌──────────────────────────┐
                 │   Final Label + Grad-CAM  │
                 │   ŷ = argmax(P) ∈ {0..14}│
                 └──────────────────────────┘
```

---

## 📐 Stage-by-Stage Mathematical Pipeline

---

### STAGE 0 — Input Preprocessing

**Input:** A raw RGB leaf image.

```
I_raw ∈ ℝ^{H × W × 3}     (arbitrary resolution, e.g. 3024×4032×3)
```

**Preprocessing transforms applied:**

| Step | Operation | Output Shape |
|------|-----------|-------------|
| Resize | Bilinear interpolation to 224×224 | `ℝ^{224×224×3}` |
| ToTensor | Pixel values mapped to [0, 1] | `ℝ^{3×224×224}` |
| Normalize | Per-channel z-score | `ℝ^{3×224×224}` |

**Normalization Formula (ImageNet statistics):**

$$\hat{I}_{c,h,w} = \frac{I_{c,h,w} - \mu_c}{\sigma_c}$$

Where:
- $\mu = [0.485,\ 0.456,\ 0.406]$ (mean per RGB channel)
- $\sigma = [0.229,\ 0.224,\ 0.225]$ (std per RGB channel)    

These are standard ImageNet normalization values used for pretrained ResNet backbones.

---

### STAGE 1 — Gemini Visual Language Model (Description Generation)

**Model:** `gemini-2.5-flash` (Google Generative AI)

The raw image bytes are passed directly to the Gemini multimodal model. Gemini processes the image using its internal Vision Transformer encoder and generates a **100–150 word plain-text description** of the visible plant symptoms.

**Key prompt constraints:**
- Must describe morphology, lesions, and symptom distribution
- Must NOT mention the plant species name (prevents label leakage)

**Output:**

$$D \in \Sigma^* \quad \text{(natural language string)}$$

Example output: *"The leaf exhibits irregular, necrotic brown lesions with chlorotic yellow halos distributed along interveinal regions. Marginal curling and wilting are visible..."*


Gemini uses an internal multimodal transformer architecture and does not expose all intermediate tensor dimensions through the API, so only input-output behavior is documented here.
---

### STAGE 2 — NLP Preprocessing Pipeline

Before feeding the description $D$ to the BERT model, it goes through a **5-stage NLP pipeline**:

| Stage | Operation | Implementation |
|-------|-----------|----------------|
| 1 | JSON Structure Stripping | `re.sub` on braces/quotes |
| 2 | Noise Removal | Remove `[NA]`, `None`, `null` tokens |
| 3 | Lowercasing + Char Normalization | `text.lower()`, keep only `[a-z\s]` |
| 4 | Stop Word Removal | NLTK English stopword list |
| 5 | Lemmatization | SpaCy `en_core_web_sm` model |

**Lemmatization formula (conceptual):**

$$\text{lemma}(w) = \underset{l \in \mathcal{L}}{\arg\min}\ \text{edit}\_\text{dist}(w, l)$$

Where $\mathcal{L}$ is the lexicon of base forms (e.g., "lesions" → "lesion", "yellowing" → "yellow").

**Output:** Cleaned text string


---

### STAGE 3 — PubMedBERT Tokenization

**Model:** `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`  
**Tokenizer:** WordPiece (BERT-style)

The cleaned description is tokenized with the following encoding:

```python
encoding = tokenizer(D_clean,
    add_special_tokens=True,    # [CLS] and [SEP]
    max_length=128,
    padding='max_length',
    truncation=True,
    return_attention_mask=True)
```

**Output tensors:**

| Tensor | Shape | Description |
|--------|-------|-------------|
| `input_ids` | $\mathbb{Z}^{128}$ | WordPiece vocabulary token IDs |
| `attention_mask` | $\{0,1\}^{128}$ | 1 for real tokens, 0 for padding |

The token sequence is:

$$[CLS],\ t_1,\ t_2,\ \ldots,\ t_n,\ [SEP],\ [PAD],\ \ldots,\ [PAD]$$

Where $n \leq 126$ meaningful tokens.

---

### STAGE 4 — Hybrid PubMedBERT Text Classifier (Text Branch)

**Architecture: `HybridBioBERTClassifier`**

#### 4a. BERT Transformer Encoding

PubMedBERT is a 12-layer Transformer encoder pretrained on PubMed abstracts.

**Self-Attention (per head, per layer):**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q = XW^Q,\quad K = XW^K,\quad V = XW^V$
- $X \in \mathbb{R}^{128 \times 768}$ — input sequence embeddings
- $W^Q, W^K, W^V \in \mathbb{R}^{768 \times 64}$ — per-head projection matrices
- $d_k = 64$ — head dimension
- BERT has **12 heads** and **12 layers** (BERT-base configuration)

**Multi-Head Concatenation:**

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_{12}) W^O$$

$$\text{Output} \in \mathbb{R}^{128 \times 768}$$

#### 4b. Pooler Output ([CLS] token)

BERT extracts the `[CLS]` token's final hidden state and passes it through a dense layer:

$$\mathbf{h}_{CLS} = \tanh(W_{pool} \cdot \mathbf{z}^{(12)}_0 + b_{pool})$$

$$\mathbf{h}_{CLS} \in \mathbb{R}^{768}$$

This 768-dimensional vector is the **semantic representation of the entire symptom description**.

#### 4c. Structured Feature Engineering

In addition to text, hand-crafted structured features are extracted from dataset columns (`MORPHOLOGY`, `LESIONS`, `DISTRIBUTION`, `SEVERITY`):

$$\mathbf{s} \in \mathbb{R}^{F_s} \quad \text{(structured feature vector)}$$

Where $F_s$ is the number of structured features extracted via `feature_engineering.py`.

#### 4d. Fusion and Classification Head

The BERT pooler output and structured features are concatenated:

$$\mathbf{x}_{fused} = [\mathbf{h}_{CLS}\ \|\ \mathbf{s}] \in \mathbb{R}^{768 + F_s}$$

Then passed through a 3-layer MLP classification head:

| Layer | Operation | Input Dim | Output Dim |
|-------|-----------|-----------|------------|
| FC1 | Linear + ReLU + Dropout(0.3) | $768 + F_s$ | 512 |
| FC2 | Linear + ReLU + Dropout(0.3) | 512 | 256 |
| Classifier | Linear (logits) | 256 | 15 |

**Mathematically:**

$$\mathbf{x}_1 = \text{Dropout}(\text{ReLU}(W_1 \mathbf{x}_{fused} + b_1)) \in \mathbb{R}^{512}$$

$$\mathbf{x}_2 = \text{Dropout}(\text{ReLU}(W_2 \mathbf{x}_1 + b_2)) \in \mathbb{R}^{256}$$

$$\mathbf{z}_{text} = W_3 \mathbf{x}_2 + b_3 \in \mathbb{R}^{15} \quad \text{(logits)}$$

$$\mathbf{p}_{text} = \text{softmax}(\mathbf{z}_{text}) \in \mathbb{R}^{15} \quad \text{where}\ \sum_{c=1}^{15} p_c = 1$$

$$\text{softmax}(z_c) = \frac{e^{z_c}}{\sum_{j=1}^{15} e^{z_j}}$$

**Training Details:**

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| BERT encoder LR | 1e-6 |
| Classification head LR | 1e-4 |
| Weight decay | 0.01 |
| Scheduler | Linear warmup (10% steps) |
| Max epochs | 20 |
| Batch size | 16 |
| Max sequence length | 128 tokens |
| Early stopping patience | 3 epochs |
| Loss function | Cross-Entropy |

**Cross-Entropy Loss:**

$$\mathcal{L}_{CE} = -\sum_{c=1}^{15} y_c \log(\hat{p}_c)$$

Where $y_c$ is the one-hot ground truth label.

---

### STAGE 5 — CNN Feature Extraction (Image Branch)

**Feature Extractor:** ResNet-18 (pretrained on ImageNet, last FC layer removed)

**Architecture Flow:**

```
Input:  ℝ^{3×224×224}
  → Conv1 (7×7, stride 2)  → ℝ^{64×112×112}
  → MaxPool (3×3, stride 2) → ℝ^{64×56×56}
  → Layer1 (2× BasicBlock)  → ℝ^{64×56×56}
  → Layer2 (2× BasicBlock)  → ℝ^{128×28×28}
  → Layer3 (2× BasicBlock)  → ℝ^{256×14×14}
  → Layer4 (2× BasicBlock)  → ℝ^{512×7×7}
  → AdaptiveAvgPool(1×1)    → ℝ^{512×1×1}
  → Flatten                 → ℝ^{512}
```

**Residual Block Formula (BasicBlock):**

$$\mathbf{y} = \mathcal{F}(\mathbf{x},\ \{W_i\}) + \mathbf{x}$$

Where $\mathcal{F}(\mathbf{x}) = W_2 \cdot \text{BN}(\text{ReLU}(W_1 \cdot \text{BN}(\mathbf{x})))$

**Output:** CNN feature vector $\mathbf{f}_{CNN} \in \mathbb{R}^{512}$

---

### STAGE 6 — Quantum Neural Network (QNN) Classifier

**Architecture: `QuantumClassifier`**

The QNN processes the 512-dim CNN features to produce disease class logits.

#### 6a. Classical Reduction (CNN → Qubit Angles)

```
Input:  f_CNN ∈ ℝ^512
  → Linear(512→256) + ReLU + Dropout(0.3)  → ℝ^256
  → Linear(256→256) + ReLU + Dropout(0.3)  → ℝ^256
  → Linear(256→5)   + Tanh                 → ℝ^5
Output: θ_in ∈ [-1, 1]^5                   (qubit rotation angles)
```

The `Tanh` activation ensures angles remain in $[-1, 1]$, suitable for $R_Y$ gates.

#### 6b. Quantum Circuit (PennyLane)

**Device:** `default.qubit` with $n = 5$ qubits, $L = 3$ variational layers.

**Step 1 — Data Encoding (Angle Embedding):**

$$R_Y(\theta_{in,i}) |0\rangle \quad \forall\ i \in \{0,1,2,3,4\}$$

$$R_Y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

This encodes the 5-dimensional CNN feature vector into the quantum state as rotation angles.

**Step 2 — Variational Layers (3 layers):**

For each layer $l \in \{0,1,2\}$:

1. **Parameterized Rotations per qubit $i$:**
$$R_Y(w_{l,i,0}) \cdot R_Z(w_{l,i,1}) \quad \forall i$$

   Where $w_{l,i,\cdot}$ are **learnable parameters** $\in \mathbf{W} \in \mathbb{R}^{3 \times 5 \times 2}$

   $$R_Z(\phi) = \begin{pmatrix} e^{-i\phi/2} & 0 \\ 0 & e^{i\phi/2} \end{pmatrix}$$

2. **Entanglement (Linear CNOT chain):**
$$\text{CNOT}_{0 \to 1},\ \text{CNOT}_{1 \to 2},\ \text{CNOT}_{2 \to 3},\ \text{CNOT}_{3 \to 4}$$
$$\text{CNOT}_{4 \to 0} \quad \text{(ring closure)}$$

**Step 3 — Measurement:**

$$q_i = \langle \psi | Z_i | \psi \rangle \in [-1, +1] \quad \forall\ i \in \{0,\ldots,4\}$$

**Quantum circuit output:**

$$\mathbf{q} = [q_0,\ q_1,\ q_2,\ q_3,\ q_4] \in \mathbb{R}^5$$

**Total trainable quantum parameters:** $3 \times 5 \times 2 = 30$ parameters.

#### 6c. Classical Expansion (Qubit Measurements → Logits)

```
Input:  q ∈ ℝ^5
  → Linear(5→128) + ReLU + Dropout(0.3)  → ℝ^128
  → Linear(128→15)                        → ℝ^15
Output: z_image ∈ ℝ^15                    (logits)
```

$$\mathbf{p}_{image} = \text{softmax}(\mathbf{z}_{image}) \in \mathbb{R}^{15}$$

**Complete Dimension Trace — Image Branch:**

```
ℝ^{3×224×224}  →[ResNet18]→  ℝ^{512}
ℝ^{512}        →[MLP]→       ℝ^{5}       (qubit angles)
ℝ^{5}          →[QCircuit]→  ℝ^{5}       (Pauli-Z expectations)
ℝ^{5}          →[MLP]→       ℝ^{15}      (logits)
ℝ^{15}         →[Softmax]→   ℝ^{15}      (probabilities p_image)
```

---

### STAGE 7 — Ensemble Fusion (MLE-Optimized Weighted Average)

Both branches output probability vectors over 15 disease classes:
- $\mathbf{p}_{image} \in \mathbb{R}^{15}$ (from CNN+QNN)
- $\mathbf{p}_{text} \in \mathbb{R}^{15}$ (from PubMedBERT)

**Fusion Formula:**

$$\mathbf{P}_{fused} = \alpha \cdot \mathbf{p}_{image} + (1 - \alpha) \cdot \mathbf{p}_{text}$$

**Optimized weight:** $\alpha = 0.65$ (determined via Maximum Likelihood Estimation over the validation set)

$$\mathbf{P}_{fused} = 0.65 \cdot \mathbf{p}_{image} + 0.35 \cdot \mathbf{p}_{text}$$

$$\mathbf{P}_{fused} \in \mathbb{R}^{15}, \quad \sum_{c=1}^{15} P_c = 1$$

**Final Prediction:**

$$\hat{y} = \underset{c \in \{0,\ldots,14\}}{\arg\max}\ P_{fused,c}$$

**Confidence:**

$$\text{confidence} = \max_c\ P_{fused,c}$$

**Alpha Optimization (MLE):** The optimal $\alpha^*$ was found by grid search over $[0, 1]$ evaluating:

$$\alpha^* = \underset{\alpha}{\arg\max}\ \mathcal{L}(\alpha) = \underset{\alpha}{\arg\max} \sum_{i=1}^{N} \log P_{fused,\hat{y}_i}^{(\alpha)}$$

This yields $\alpha^* = 0.65$. In fold-level result files, the multimodal ensemble reports **96.20% +/- 0.60%** mean accuracy. Across archived comparison artifacts, multimodal remains in the high-96% range.

---

### STAGE 8 — Explainability (XAI)

#### 8a. Grad-CAM (Image Explanation)

Grad-CAM generates a spatial heatmap highlighting **which regions of the leaf influenced the prediction**.

**Target Layer:** ResNet18's `layer4` (last convolutional block)

$$\text{Feature maps}: A^k \in \mathbb{R}^{7 \times 7}, \quad k \in \{1, \ldots, 512\}$$

**Step 1 — Compute gradient of class score w.r.t. feature maps:**

$$\frac{\partial y^c}{\partial A^k_{ij}} \quad \text{via backpropagation}$$

**Step 2 — Global Average Pool gradients (channel importance weights):**

$$\alpha^c_k = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A^k_{ij}}$$

Where $Z = 7 \times 7 = 49$.

**Step 3 — Weighted combination + ReLU:**

$$\text{CAM}^c = \text{ReLU}\!\left(\sum_k \alpha^c_k A^k\right) \in \mathbb{R}^{7 \times 7}$$

**Step 4 — Upsample to 224×224:**

$$\text{Heatmap} = \text{Resize}_{224}(\text{CAM}^c) \in \mathbb{R}^{224 \times 224}$$

Normalized to $[0, 1]$ and overlaid on the leaf image using `cv2.addWeighted(0.6, 0.4)`.

#### 8b. Leaf Segmentation (Pre-Grad-CAM)

Before Grad-CAM, the leaf is segmented from background using **HSV thresholding**:

```
HSV Ranges:
  Green (healthy tissue):  H=[35,90], S=[30,255], V=[30,255]
  Brown/Yellow (lesions):  H=[5,35],  S=[30,255], V=[30,255]
```

Morphological operations (Opening + Closing with 3×3 kernel) refine the binary mask. The Grad-CAM heatmap is **masked to the leaf region only**, suppressing background noise.

#### 8c. BERT Attention Analysis (Text Explanation)

For text, the CLS token's attention from all 12 heads across all 12 layers is averaged:

$$\bar{A}_{0,j} = \frac{1}{L \cdot H} \sum_{l=1}^{L} \sum_{h=1}^{H} A^{(l,h)}_{0,j}$$

Where:
- $L = 12$ layers, $H = 12$ heads
- $A^{(l,h)}_{0,j}$ = attention from `[CLS]` to token $j$ in layer $l$, head $h$

This gives per-token importance weights for explanation.

---

## 🔬 Fusion Strategy Comparison (10-Fold Cross-Validation)

Five fusion strategies were compared using **10-Fold Stratified Cross-Validation** on pre-extracted feature vectors:

| Strategy | Image Dim | Text Dim | Fused Dim | Formula |
|----------|-----------|----------|-----------|---------|
| **Concatenation** | 512 | 768 | **1280** | $[\mathbf{f}_{img} \| \mathbf{f}_{txt}]$ |
| **Simple Addition** | 512→512 | 768→512 | **512** | $0.5\mathbf{f}_{img} + 0.5\mathbf{f}_{txt}$ |
| **Learned Weight** | 512→512 | 768→512 | **512** | $\sigma(\alpha_{raw})\mathbf{f}_{img} + (1-\sigma)\mathbf{f}_{txt}$ |
| **Vision Bias** | 512→512 | 768→512 | **512** | $0.8\mathbf{f}_{img} + 0.2\mathbf{f}_{txt}$ |
| **Text Bias** | 512→512 | 768→512 | **512** | $0.3\mathbf{f}_{img} + 0.7\mathbf{f}_{txt}$ |

**Concatenation Fusion Head (MLP):**

```
ℝ^{1280} → Linear(1280→512) + BN + ReLU + Dropout(0.3)
         → Linear(512→256)  + BN + ReLU + Dropout(0.2)
         → Linear(256→15)
         → ℝ^15 logits
```

**Weighted Addition Fusion:**

ProjectionL_img: $W_{img} \in \mathbb{R}^{512 \times 512}$, proj_txt: $W_{txt} \in \mathbb{R}^{768 \times 512}$

$$\mathbf{f}_{fused} = \alpha \cdot (W_{img} \mathbf{f}_{img}) + (1-\alpha) \cdot (W_{txt} \mathbf{f}_{txt})$$

For **Learned Weight**, $\alpha = \sigma(w_{raw}) = \frac{1}{1+e^{-w_{raw}}}$ is an optimized scalar parameter.

**Cross-modal features extractor** (30_extract_multimodal_features.py) extracts:
- Image features: $\mathbf{f}_{img} \in \mathbb{R}^{512}$ — from ResNet18 AdaptiveAvgPool output
- Text features: $\mathbf{f}_{txt} \in \mathbb{R}^{768}$ — from PubMedBERT CLS pooler output

---

## 📊 System Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| CNN + NN baseline (10-fold) | 95.16% +/- 0.56% | outputs/cnn_baseline_10fold_results.csv |
| Image-Only (CNN+QNN, 10-fold) | 96.48% +/- 0.46% | outputs/10fold_cv_image_results.csv |
| Text-Only (BioBERT, strict protocol, 10-fold) | 49.53% +/- 0.96% | outputs/10fold_cv_comparison_table.csv |
| **Multimodal (Ensemble, 10-fold)** | **96.20% +/- 0.60%** | α=0.65 weighted fusion |

### Vision Branch Comparison: Baseline vs Quantum-Assisted

| Setting | Model | Accuracy | F1-score |
|-------|------------|----------|----------|
| Classical baseline | CNN + NN | 95.16% +/- 0.56% | 94.43% +/- 0.78% |
| Quantum-assisted | CNN + QNN | 96.48% +/- 0.46% | 96.47% +/- 0.45% |

This is +1.32 percentage points absolute accuracy gain and about 27.27% relative error reduction for CNN+QNN over the baseline CNN+NN.

### Dual Result Tracks (Repository Archive Consistency)

| Track | Image Branch | Fused Branch |
|-------|--------------|--------------|
| Fold-level CV result files | 96.48% | 96.20% |
| Journal-export summary files | 95.77% | 97.06% |

These values are preserved from different archived runs. Both tracks consistently show that the multimodal system is competitive at the high-96% level and that alpha around 0.65 is a robust fusion weight.

### Text Branch Comparison: Without BioBERT vs With BioBERT

| Setting | Best Model | Accuracy | F1-score |
|-------|------------|----------|----------|
| Without BioBERT | SVM (TF-IDF) | 72.99% | 73.14% |
| With BioBERT | Hybrid BioBERT text model (10-fold) | 90.23% +/- 0.35% | 90.20% +/- 0.37% |

The without-vs-with BioBERT comparison is reported separately because the strict protocol in the multimodal comparison table and the dedicated text-only CV file represent different evaluation settings.

---

## 🗂️ Complete Vector Dimension Summary

| Component | Input Shape | Output Shape | Key Operation |
|-----------|-------------|--------------|---------------|
| Image Preprocessing | `H×W×3` | `3×224×224` | Resize, Normalize |
| ResNet-18 | `3×224×224` | `512` | Conv blocks + AvgPool |
| QNN Reduction | `512` | `5` | MLP + Tanh |
| Quantum Circuit | `5` (angles) | `5` (Pauli-Z) | RY, RZ, CNOT |
| QNN Expansion | `5` | `15` | MLP |
| Image Softmax | `15` (logits) | `15` (probs) | Softmax |
| Gemini Output | Image pixels | String (100-150w) | Multimodal LLM |
| NLP Pipeline | Raw string | Cleaned string | Lemmatize+Stopwords |
| BERT Tokenizer | String | `2×128` (ids+mask) | WordPiece |
| BERT Encoder | `128×768` | `768` | 12× Transformer |
| Structured Features | CSV columns | `F_s` | Feature Engineering |
| BERT Fusion Input | `768 + F_s` | `768 + F_s` | Concatenation |
| Text MLP | `768+F_s` | `15` (logits) | 3-layer MLP |
| Text Softmax | `15` (logits) | `15` (probs) | Softmax |
| Ensemble Fusion | `15 + 15` | `15` (probs) | Weighted Average |
| Final Prediction | `15` (probs) | scalar | ArgMax |
| Grad-CAM | `512×7×7` | `224×224` | Grad Pooling |

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Image Feature Extractor | ResNet-18 (pretrained, torchvision) |
| Quantum Layer | PennyLane `default.qubit` |
| Text Encoder | PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract) |
| Description Generation | Google Gemini 2.5 Flash |
| NLP Preprocessing | SpaCy `en_core_web_sm` + NLTK stopwords |
| Backend API | FastAPI + Uvicorn |
| Frontend | React.js |
| Explainability (Image) | Grad-CAM (custom implementation) |
| Explainability (Text) | BERT Attention Maps |
| Leaf Segmentation | OpenCV HSV thresholding |
| Model Validation | 10-Fold Stratified Cross-Validation |
| Optimization | AdamW + Linear LR Scheduling |
| Framework | PyTorch + HuggingFace Transformers |

---

## 🚀 Running the Project

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Key Environment Variables

```
GEMINI_API_KEY=your_google_api_key_here
```

---

## 📁 Key Model File Paths

| Model | Path |
|-------|------|
| CNN+QNN weights | `models/image/cnn_qnn_best.pt` |
| PubMedBERT weights | `models/text/overhaul/best_biobert_overhaul.pth` |
| Label encoder & metadata | `models/text/overhaul/overhaul_metadata.pkl` |
| Label mapping (image↔text) | `models/image/label_mapping.pkl` |

---

## 📋 Disease Classes (15 Categories)

The system classifies across **15 plant disease/health states** spanning tomato, corn, grape, and other crops including bacterial spots, blights, molds, rusts, and healthy categories, encoded via `sklearn.LabelEncoder` during training.

---

*This README is generated to support mathematical presentation and journal paper writing for the PlantCare AI multimodal fusion project.*
