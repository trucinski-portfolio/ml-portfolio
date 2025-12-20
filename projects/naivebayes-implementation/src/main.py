#CS622 Homework 2 - Thomas Rucinski - 10/24/25
#This program will create a Naive Bayes Classifier to predict the outcome of loan approval with input features: income, credit score, requested loan amount, and years employed.
import pandas as pd
import numpy as np

#Inititialize file path, variables, input features, reproducible seed, and training ratio
CSV_PATH = "data/data.csv"
TARGET = "loan_approved"
INPUT_COLS = ["income", "credit_score", "loan_amount", "years_employed"]
N_BINS = 4
SEED = 4
TRAIN_FRAC = 0.80

# Load file and drop unnecessary columns
df = pd.read_csv(CSV_PATH)
df = df.drop(columns=["name", "points", "city"], errors="ignore")

# Converts boolean feature output True/False to Integer 1/0
if df[TARGET].dtype == bool:
    df[TARGET] = df[TARGET].astype(int)

# Use bins to categorize input parameters
for c in INPUT_COLS:
    # Use quantile bins, fallback to regular bins if not enough unique values within feature
    try:
        df[c + "_bin"] = pd.qcut(df[c], q=min(N_BINS, df[c].nunique()),
                                 duplicates="drop")
    except ValueError:
        uniq = df[c].nunique()
        bins = min(max(2, uniq), N_BINS)
        df[c + "_bin"] = pd.cut(df[c], bins=bins, include_lowest=True)

FEATURES = [c + "_bin" for c in INPUT_COLS if (c + "_bin") in df.columns]

# initialize train/test arrays, stratify sets to keep 1/0 ratios similar within both arrays using reproducible randomization of dataset
rng = np.random.default_rng(SEED)
train_index = []
test_index = []
for cls in [0, 1]:
    index = np.where(df[TARGET].values == cls)[0]
    rng.shuffle(index)
    n_train = int(len(index) * TRAIN_FRAC)
    train_index.extend(index[:n_train])
    test_index.extend(index[n_train:])

#convert train/test sets to NumPy arrays
train_index = np.array(train_index)
test_index = np.array(test_index)

train = df.iloc[train_index].reset_index(drop=True)
test = df.iloc[test_index].reset_index(drop=True)

# Train Naive Bayes
# Priors
prior_counts = train[TARGET].value_counts(normalize=True)
#Calculate p(y=0) and p(y=1)
priors = {0: float(prior_counts.get(0, 0.0)), 1: float(prior_counts.get(1, 0.0))}
if priors[0] == 0.0 or priors[1] == 0.0:
    raise ValueError("Stratified split failed to include both classes in train.")

# Likelihoods: P(feature=value | class), i.e. P(income_bin = “(30k,50k]” | loan_approved = 1)
likelihoods = {}
for f in FEATURES:
    likelihoods[f] = {}
    for cls in [0, 1]:
        vc = train.loc[train[TARGET] == cls, f].value_counts(normalize=True, dropna=False)
        likelihoods[f][cls] = vc.to_dict()

def predict_row_get_probs(row):
    # Unnormalized class likelihoods (Posterior = Prior x Joint Likelihoods) for 1 row at a time
    p = {0: priors[0], 1: priors[1]}
    for cls in (0, 1):
        for f in FEATURES:
            val = row[f]
            p[cls] *= likelihoods[f][cls].get(val, 0)
    p0, p1 = p[0], p[1]
    tot = p0 + p1
    if tot == 0.0:
        # backoff to priors if both zero
        return priors[0], priors[1]
    return p0 / tot, p1 / tot

# Predicting probabilities for entire dataframe
def predict(df_in):
    probs1 = []
    preds = []
    for _, r in df_in.iterrows():
        p0, p1 = predict_row_get_probs(r)
        probs1.append(p1)
        preds.append(1 if p1 >= 0.5 else 0)
    return np.array(preds, dtype=int), np.array(probs1, dtype=float)

# Metrics calculation
# Calculation of True Positive, True Negative, False Positive, and False Negative
def confusion(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn

# Computation of accuracy, sensitivity, specificity, precision, f1 score, and log loss for model
def compute_metrics(y_true, y_pred, p1):
    tp, tn, fp, fn = confusion(y_true, y_pred)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    # Log loss
    eps = 1e-15
    p = np.clip(p1, eps, 1 - eps)
    y = np.asarray(y_true, dtype=float)
    logloss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return acc, sens, spec, f1, logloss, (tp, tn, fp, fn)

# Run and report
train_pred, train_p1 = predict(train)
test_pred, test_p1 = predict(test)
acc_tr, sens_tr, spec_tr, f1_tr, ll_tr, conf_tr = compute_metrics(train[TARGET], train_pred, train_p1)
acc_te, sens_te, spec_te, f1_te, ll_te, conf_te = compute_metrics(test[TARGET], test_pred, test_p1)

def fmt(x): return f"{x:.4f}"

# Printing training size & class ratio, test size & class ratio
print(f"Train size: {len(train)}  |  Test size: {len(test)}  |  Output class ratios (train):")
print(train[TARGET].value_counts(normalize=True))
print("Output class ratios (test):")
print(test[TARGET].value_counts(normalize=True))

## Printing training set features used, as well as accuracy, sensitivity, specificity, F1 score, and Log loss.
print("\nInput features:", FEATURES)
print("\nTraining Metrics:")
print(f"N={len(train)} | Accuracy={fmt(acc_tr)}  Sensitivity={fmt(sens_tr)}  Specificity={fmt(spec_tr)}  F1={fmt(f1_tr)}  Log Loss={fmt(ll_tr)}")
print("Confusion Matrix: (TP, TN, FP, FN):", conf_tr)

## Printing test set features used, as well as accuracy, sensitivity, specificity, F1 score, and Log loss.
print("\nTest Metrics:")
print(f"N={len(test)}  | Accuracy={fmt(acc_te)}  Sensitivity={fmt(sens_te)}  Specificity={fmt(spec_te)}  F1={fmt(f1_te)}  Log Loss={fmt(ll_te)}")
print("Confusion Matrix: (TP, TN, FP, FN):", conf_te)