# CS-622 Homework 4 | December 6th, 2025
# Thomas Rucinski
# ML model variation comparison for breast cancer tumor diagnosis performance
# dataset from: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data

# Importing necessary libraries and functions from pandas, numpy, and sklearn
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix

filename = "data.csv"
df = pd.read_csv(filename, header=0) # header=0 since first row is feature name

X = df.iloc[:, 2:32].values # Using input features from col 2-32 (30 total)
y_raw = df.iloc[:, 1].values # Col 2 is the output feature col (M/B for Malignant or Benign)

label_encoder = LabelEncoder() # Convert 'M'/'B' to 1/0
y = label_encoder.fit_transform(y_raw)

print(f"Data Loaded Successfully.")
print(f"Input Features: {X.shape[1]}")  # Validating that there are 30 input features
print(f"Target Classes: {label_encoder.classes_}")

# Preprocessing (dataset splitting, randomization, and standardization)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Defines the grid search (models to be used for the performance evaluation)
# 3 x 2 x 2 x 1 = 12 unique models for comparison
param_grid = {
    'hidden_layer_sizes': [(100,), (50, 50), (100, 50, 25)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'max_iter': [2000]
}

# Generate the list of 12 experiments automatically
grid_list = list(ParameterGrid(param_grid))

# evaluation metric computation function
def get_metrics(model, X, y, prefix):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    return {
        f"{prefix} Accuracy": accuracy_score(y, y_pred),
        f"{prefix} Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        f"{prefix} Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        f"{prefix} F1 Score": f1_score(y, y_pred),
        f"{prefix} Log Loss": log_loss(y, y_prob)
    }

# Run experimental model variations; calculate metrics, rank by test dataset accuracy (descending), display results

results = []
print(f"\nRunning {len(grid_list)} MLP Classifier Variations:")

for i, params in enumerate(grid_list):
    # Create a model experiment ID (ex. "Model 1: (100,) / relu /adam")
    exp_name = f"MLP V{i + 1}: {params['hidden_layer_sizes']} / {params['activation']} / {params['solver']}"
    print(f"Training {exp_name}")

    # Initialize and Train
    mlp = MLPClassifier(**params, random_state=24)
    mlp.fit(X_train_scaled, y_train)

    # Calculate metrics for both training and test datasets
    train_metrics = get_metrics(mlp, X_train_scaled, y_train, prefix="Train")
    test_metrics = get_metrics(mlp, X_test_scaled, y_test, prefix="Test")

    # Store parameters + metrics
    entry = {'Model Name': exp_name}
    entry.update(train_metrics)
    entry.update(test_metrics)
    results.append(entry)

# Display evaluation metrics for top 5 most accurate MLP classifiers out of the 12 unique combinations
df_results = pd.DataFrame(results)
df_sorted = df_results.sort_values(by="Test Accuracy", ascending=False) # Sort by test accuracy (Descending) so both tables share the same ranking
df_sorted['Rank'] = df_sorted['Test Accuracy'].rank(method='min', ascending=False).astype(int)

# Define column lists for the two separate tables
cols_common = ['Rank', 'Model Name']
cols_train = [c for c in df_sorted.columns if "Train" in c]
cols_test = [c for c in df_sorted.columns if "Test" in c]

print("\nTABLE 1: Training Data Metrics (top 5 models sorted by test rank)")
print(df_sorted[cols_common + cols_train].head(5).to_string(index=False))
print("\nTABLE 2: Test Data Metrics (top 5 models sorted by test rank)")
print(df_sorted[cols_common + cols_test].head(5).to_string(index=False))