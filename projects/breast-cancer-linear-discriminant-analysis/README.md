# CS 722 — Assignment #1: Linear Discriminant Analysis
**Thomas Rucinski | Spring 2026**

---

## Overview

This assignment implements **Linear Discriminant Analysis (LDA)** from scratch in Python for binary classification. LDA is a generative, probabilistic classifier that models each class as a multivariate Gaussian with a **shared covariance structure**. Under this assumption, the decision boundary between classes is linear, and class membership is determined via Bayes' theorem.

---

## Theory

LDA estimates three per-class quantities from the training data:

- **Class priors** — π̂_c = N_c / N  `[numpy]`
- **Class means** — μ̂_c = mean of all training samples in class c  `[numpy.mean]`
- **Pooled covariance** — a shared covariance matrix Σ estimated as the weighted average of per-class unbiased covariance matrices  `[numpy.cov, ddof=1]`:

```
Σ_pooled = Σ_c (N_c − 1) Σ̂_c  /  Σ_c (N_c − 1)
```

For binary classification, the log-posterior ratio reduces to a **linear discriminant function**:

```
log[p(y=1|x) / p(y=0|x)] = wᵀx + w₀
```

where:
- `w  = Σ⁻¹(μ̂₁ − μ̂₀)` — the discriminant weight vector, solved via `numpy.linalg.solve` (avoids explicit inversion)
- `w₀ = −½(μ̂₁ᵀΣ⁻¹μ̂₁ − μ̂₀ᵀΣ⁻¹μ̂₀) + log(π̂₁/π̂₀)` — the bias term, computed via `numpy.einsum`

The posterior probability is obtained via the sigmoid (logistic) function `[from scipy.special import expit]`:

```
p(y=1|x, θ) = σ(wᵀx + w₀)
```

A test instance is assigned to class 1 if p ≥ 0.5, else class 0.

---

## Dataset

**Wisconsin Diagnostic Breast Cancer (WDBC)**
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 continuous (nucleus geometry: radius, texture, perimeter, area, etc.) |
| Classes | Malignant (M) = 1 → 212 samples; Benign (B) = 0 → 357 samples |
| Missing values | None |

Features are derived from digitized fine needle aspirate (FNA) images of breast masses, capturing 10 geometric characteristics each computed as mean, standard error, and worst value.

---

## Methods

### Preprocessing
- Stratified 80/20 train/test split `[sklearn.model_selection.train_test_split, stratify=y, random_state=100]`
- **Standardization** (zero mean, unit variance) fit on training set only, applied to both splits to prevent information leakage `[sklearn.preprocessing.StandardScaler]`

### Implementation
- Per-class covariance estimated with `numpy.cov` (`ddof=1`, unbiased)
- LDA parameters solved via `numpy.linalg.solve` — numerically more stable than explicit matrix inversion
- Bias term collapsed across classes via `numpy.einsum`
- Sigmoid applied via `scipy.special.expit` (numerically stable)
- No built-in LDA library functions used

### Evaluation
- Confusion matrix → sensitivity & specificity derived manually `[sklearn.metrics.confusion_matrix]`
- Accuracy `[sklearn.metrics.accuracy_score]`, F1 `[sklearn.metrics.f1_score]`, Log Loss `[sklearn.metrics.log_loss]`

---

## Results

| Metric | Training | Test |
|---|---|---|
| Accuracy    | 0.9648 | 0.9649 |
| Sensitivity | 0.9176 | 0.9048 |
| Specificity | 0.9930 | 1.0000 |
| F1-Score    | 0.9512 | 0.9500 |
| Log Loss    | 0.0981 | 0.0857 |

Train and test performance are nearly identical, indicating no overfitting. The model achieves perfect specificity on the test set (zero false positives — no benign samples misclassified as malignant). Sensitivity of ~90% reflects the harder challenge of detecting all true malignant cases.

---

## Project Structure

```
cs722-homework1/
├── src/
│   └── lda.py       # Full LDA implementation
├── data/
│   └── breast+cancer+wisconsin+diagnostic.zip
├── documents/
│   ├── CS722_Spring2026_Assignment1.pdf
│   ├── Rucinski_Assignment1.docx
│   └── Rucinski_Assignment1.pdf
├── requirements.txt
└── README.md
```

## Dependencies

```
numpy==2.4.2
scipy==1.17.1
scikit-learn==1.8.0
joblib==1.5.3
threadpoolctl==3.6.0
```

Install: `pip install -r requirements.txt`

## Running

```bash
python src/lda.py
```
