"""
CS 722 Assignment #1 Linear Discriminant Analysis
Dataset: Wisconsin Diagnostic Breast Cancer (WDBC), UCI ML Repository (https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
Author: Thomas Rucinski
"""

import numpy as np
import zipfile
import os
from scipy.special import expit                          # sigmoid implementation
from sklearn.preprocessing import StandardScaler         # standardization 
from sklearn.model_selection import train_test_split     # train/test splitting 
from sklearn.metrics import (                            # evaluation metrics
    accuracy_score, f1_score, log_loss,
    confusion_matrix
)

# Load dataset 

DATA_ZIP = os.path.join(os.path.dirname(__file__), "..", "data",
                        "breast+cancer+wisconsin+diagnostic.zip")

with zipfile.ZipFile(DATA_ZIP) as z:
    with z.open("wdbc.data") as f:
        raw = np.genfromtxt(f, delimiter=",", dtype=str)

# Column layout: [ID, diagnosis, feat_1 feat_2 ...]
y_raw = raw[:, 1]                    # 'M' or 'B'
X     = raw[:, 2:].astype(float)    # 30 continuous features

# Encode Output Feature to Boolean: Malignant = 1, Benign = 0
y = (y_raw == "M").astype(int)

# Train / test split (80% / 20%, stratified by class)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)

# Standardize (fit on training set only preventing information leakage)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit + transform on train
X_test_s  = scaler.transform(X_test)        # transform only on test, using train parameters

# LDA fitting

def fit_lda(X, y):
    """
    Estimate LDA parameters from standardized training data.

    For each class c (Malignant=1, Benign=0):
        μ̂_c  = np.mean(X_c, axis=0)
        Σ̂_c  = np.cov(X_cᵀ, ddof=1)
        π̂_c  = N_c / N

    Pooled covariance (shared across classes):
        Σ = Σ_{c} (N_c - 1) Σ̂_c  /  Σ_{c} (N_c - 1)

    LDA parameters (solved jointly via np.linalg.solve):
        [β_0 | β_1] = Σ⁻¹ [μ̂_0 | μ̂_1]   →  np.linalg.solve(Σ, M)
        γ_c = -½ μ̂_cᵀ β_c + log(π̂_c)     →  np.einsum for all c at once

    Returns:
        w  : (d,)  decision weight vector  β_1 - β_0
        w0 : float decision bias           γ_1 - γ_0
    """
    classes = np.unique(y)
    N       = len(y)

    # Per-class statistics
    masks = [y == c for c in classes]
    N_c   = np.array([m.sum() for m in masks])               # Output size = (C,)
    pi_c  = N_c / N                                          # Output size = (C,) priors
    mu_c  = np.array([X[m].mean(axis=0) for m in masks])    # Output size = (C, d)

    # np.cov with ddof=1 → unbiased covariance Σ̂_c = (1/(N_c-1)) Σ (x-μ)(x-μ)ᵀ
    Sig_c = np.array([np.cov(X[m].T, ddof=1) for m in masks])  # Output size = (C, d, d)

    # Pooled covariance: weighted sum by (N_c - 1), then normalize
    weights      = (N_c - 1)[:, None, None]
    Sigma_pooled = np.sum(weights * Sig_c, axis=0) / weights.sum()  # Output size = (d, d)

    # Solve Σ B = Mᵀ for all classes simultaneously
    B = np.linalg.solve(Sigma_pooled, mu_c.T)               # Output size = (d, C)

    # γ_c = -½ μ̂_cᵀ β_c + log(π̂_c)  — all classes at once via einsum
    gamma_c = -0.5 * np.einsum("dc,dc->c", mu_c.T, B) + np.log(pi_c)

    # Collapse to single weight vector and bias for binary classification
    w  = B[:, 1] - B[:, 0]       # (β_1 - β_0)
    w0 = gamma_c[1] - gamma_c[0] # (γ_1 - γ_0)
    return w, w0

w, w0 = fit_lda(X_train_s, y_train)

# LDA prediction

def predict_lda(X, w, w0):
    """
    p(y=1 | x, θ) = sigmoid( wᵀx + w_0 )
    Classify as 1 if p ≥ 0.5 (equivalent to wᵀx + w_0 ≥ 0).
    scipy.special.expit used as a built-in sigmoid function
    """
    prob_class1 = expit(X @ w + w0)
    y_pred      = (prob_class1 >= 0.5).astype(int)
    return y_pred, prob_class1

y_pred_train, prob_train = predict_lda(X_train_s, w, w0)
y_pred_test,  prob_test  = predict_lda(X_test_s,  w, w0)

# Evaluation metrics (sklearn)

def compute_metrics(y_true, y_pred, y_prob, split_name):
    """
    Accuracy, Sensitivity (TPR), Specificity (TNR), F1-Score, Log Loss, and Confusion Matrix
    via sklearn.metrics built-in functions.
    """
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()

    accuracy    = accuracy_score(y_true, y_pred)
    sensitivity = TP / (TP + FN)   
    specificity = TN / (TN + FP)   
    f1          = f1_score(y_true, y_pred)
    ll          = log_loss(y_true, y_prob)

    print(f"\n  #### {split_name}: ####\n")
    print(f"  Accuracy     : {accuracy:.4f}  ({TP+TN}/{len(y_true)} correct)")
    print(f"  Sensitivity  : {sensitivity:.4f}  (TP={TP}, FN={FN})")
    print(f"  Specificity  : {specificity:.4f}  (TN={TN}, FP={FP})")
    print(f"  F1-Score     : {f1:.4f}")
    print(f"  Log Loss     : {ll:.4f}\n")

print("\n  #### Linear Discriminant Analysis Results ####")
print(f"  Samples : {len(y)}  (Train={len(y_train)} (80%), Test={len(y_test)} (20%))")
print(f"  Features: {X.shape[1]}")
print(f"  Classes : Malignant=1 ({(y==1).sum()}), Benign=0 ({(y==0).sum()})")

compute_metrics(y_train, y_pred_train, prob_train, "Training Set")
compute_metrics(y_test,  y_pred_test,  prob_test,  "Test Set    ")
