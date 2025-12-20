# Naive Bayes Loan Approval Classifier (CS 622 Homework 2)

A **from-scratch Naive Bayes classifier** (no scikit-learn) that predicts **loan approval** using four numeric input features:

- income  
- credit_score  
- loan_amount  
- years_employed  

Numeric features are discretized into bins, a Naive Bayes model is trained on the binned features, and performance is evaluated using a **reproducible stratified train/test split**.

---

## What this project does

1. Loads `data.csv`
2. Drops non-feature columns (`name`, `points`, `city`) if present
3. Ensures the target column (`loan_approved`) is numeric (0/1)
4. Creates binned categorical features:
   - Uses quantile bins (`pd.qcut`) when possible  
   - Falls back to fixed bins (`pd.cut`) if necessary
5. Performs a reproducible **stratified 80/20 train/test split**
6. Trains a Naive Bayes classifier:
   - Computes class priors P(y)
   - Computes likelihoods P(x | y) using frequency tables
7. Predicts class probabilities and labels
8. Reports:
   - Class ratios (train/test)
   - Accuracy
   - Sensitivity (Recall)
   - Specificity
   - F1 score
   - Log loss
   - Confusion matrix (TP, TN, FP, FN)

---

## Files

- `main.py` — Naive Bayes implementation  
- `data.csv` — Input dataset (must be in the same directory)  
- `requirements.txt` — Python dependencies  

---

## Data requirements

`data.csv` must include the following columns:

- `loan_approved` (target; 0/1 or True/False)
- `income`
- `credit_score`
- `loan_amount`
- `years_employed`

Optional columns (ignored if present):
- `name`
- `points`
- `city`

---

## Environment

- **Python:** 3.13.x  
- Dependencies are recorded in `requirements.txt`

To install dependencies:

```bash
python -m pip install -r env/requirements.txt

## How to run

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt
python src/main.py