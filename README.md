# Machine Learning Projects Portfolio

This repository is a curated collection of **machine learning and statistical modeling projects** developed in Python, with an emphasis on **reproducibility, interpretability, and applied decision-making** in real-world contexts (healthcare, finance, and diagnostics).

Each project is self-contained, documented, and reproducible, with clearly defined data requirements, environments, and evaluation metrics.

---

## Repository Structure

Projects are organized as independent subdirectories under `projects/`.

Each project typically contains:
- `src/` — source code
- `data/` — datasets required to run the model (or instructions to obtain them)
- `requirements.txt` — Python dependencies
- `README.md` — project-specific documentation
- `docs/` — optional reports or PDFs (when applicable)

---

## Projects

### 1. Breast Cancer Diagnostic Classification (MLP, scikit-learn)

**Folder:** `projects/breast-cancer-mlp-sklearn/`

- Multi-Layer Perceptron (MLP) models trained on the **Wisconsin Breast Cancer Diagnostic dataset**
- Grid search across:
  - Hidden layer architectures
  - Activation functions
  - Solvers (Adam, SGD)
- Preprocessing pipeline with:
  - Label encoding
  - Standard scaling
  - Stratified train/test splitting
- Evaluation prioritizes **recall (sensitivity)** and **log loss**, reflecting clinical risk considerations

**Tech stack:** Python, NumPy, pandas, scikit-learn

---

### 2. Naive Bayes Loan Approval Classifier (From Scratch)

**Folder:** `projects/naivebayes-implementation/`

- Fully manual implementation of a **Naive Bayes classifier** (no scikit-learn)
- Predicts loan approval using:
  - Income
  - Credit score
  - Loan amount
  - Years employed
- Numeric features discretized using quantile binning
- Reproducible stratified train/test split
- Reports:
  - Accuracy
  - Sensitivity (Recall)
  - Specificity
  - F1 score
  - Log loss
  - Confusion matrix

**Tech stack:** Python, NumPy, pandas

---

## Environment & Reproducibility

- Each project includes its own `requirements.txt`
- Virtual environments (`.venv/`) are used locally and excluded from version control
- All scripts are designed to be run from the project root

Typical setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py