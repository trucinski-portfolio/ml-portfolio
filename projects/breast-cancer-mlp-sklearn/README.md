# Breast Cancer Diagnostic Classification with MLP (scikit-learn)

This project evaluates multiple **Multi-Layer Perceptron (MLP)** neural network architectures for classifying breast tumors as **Malignant** or **Benign** using the **Wisconsin Breast Cancer Diagnostic Dataset**.

Originally developed for CS-622, the project has been refactored for reproducibility, clear evaluation, and portfolio presentation.

---

## Project Overview

- **Task:** Binary classification (Malignant vs Benign)
- **Model:** scikit-learn `MLPClassifier`
- **Focus:** Architecture comparison, optimization behavior, and clinically relevant metrics
- **Clinical emphasis:** Recall (sensitivity) and log loss to minimize false negatives

The script systematically evaluates **12 distinct MLP configurations** to identify models that balance accuracy with reliable probabilistic predictions.

---

## Dataset

- **Source:** Wisconsin Breast Cancer Diagnostic Dataset  
  https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- **Samples:** 569
- **Features:** 30 real-valued, image-derived nuclear features
- **Target Encoding:**
  - Malignant (`M`) → 1
  - Benign (`B`) → 0

### Preprocessing
- Label encoding of diagnosis
- Feature standardization via `StandardScaler`
- Stratified 80/20 train–test split

---

## Experimental Design

### Hyperparameters Evaluated

| Component | Values |
|---------|--------|
| Hidden layers | `(100,)`, `(50, 50)`, `(100, 50, 25)` |
| Activations | ReLU, Tanh |
| Solvers | Adam, SGD |
| Max iterations | 2000 |

Each configuration is trained and evaluated independently to assess generalization performance.

---

## Evaluation Metrics

Models are evaluated on both training and test sets using:

- **Accuracy**
- **Sensitivity (Recall)** – critical for minimizing missed malignancies
- **Specificity**
- **F1 Score**
- **Log Loss** – evaluates probabilistic confidence

Results are ranked by **test accuracy** to highlight models that generalize well.

---

## Repository Structure

- `src/` — MLP training and evaluation script  
- `data/` — Breast cancer dataset  
- `env/` — Environment and dependency specifications  
  - `requirements.txt` — Python dependencies  
- `README.md` — Project documentation

---

## Setup and Execution

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate

### 2. Install Dependencies
pip install -r env/requirements.txt

### 3. Run the experiment
python src/main.py