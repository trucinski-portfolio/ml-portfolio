# README: Breast Cancer Diagnosis MLP Performance Comparison
**Course:** CS-622  
**Author:** Thomas Rucinski  
**Date:** December 6, 2025  

## Project Overview
This project evaluates the performance of various **Multi-Layer Perceptron (MLP)** neural network configurations in diagnosing breast cancer tumors as either **Malignant** or **Benign**. Using the *Wisconsin Breast Cancer Diagnostic* dataset, the script automates the training and testing of 12 unique model variations to determine which architectural and optimization hyperparameters yield the highest diagnostic accuracy and clinical reliability.



---

## Dataset Specification
* **Source:** [UCI Machine Learning Repository / Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
* **Input Features:** 30 numerical features (computed from digitized images of fine needle aspirates of breast masses).
* **Target Classes:** 2 (Malignant 'M' = 1, Benign 'B' = 0).
* **Pre-processing:** * **Label Encoding:** Converts categorical diagnosis to binary integers.
    * **Standardization:** Utilizes `StandardScaler` to ensure features have a mean of 0 and a variance of 1, which is critical for MLP convergence.
    * **Data Split:** 80% Training / 20% Testing.

---

## Experimental Design
The script implements a grid search to compare model performance across different hidden layer depths and optimization functions.

### Hyperparameters Explored
| Parameter | Values Evaluated |
| :--- | :--- |
| **Hidden Layer Sizes** | `(100,)`, `(50, 50)`, `(100, 50, 25)` |
| **Activation Functions** | `ReLU`, `Tanh` |
| **Solvers (Optimizers)** | `Adam`, `SGD` |
| **Max Iterations** | `2000` |

### Evaluation Metrics
To provide a comprehensive view of model performance, the following metrics are calculated for both training and test sets:
* **Accuracy:** Overall percentage of correct predictions.
* **Sensitivity (Recall):** The ability to identify all malignant tumors (minimizing False Negatives).
* **Specificity:** The ability to identify all benign tumors (minimizing False Positives).
* **F1 Score:** The balance between precision and sensitivity.
* **Log Loss:** Evaluates the certainty of the model's probabilistic predictions.



---

## Installation & Execution
1.  Ensure `data.csv` is located in the same directory as the script.
2.  **Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn
    ```
3.  **Run Analysis:**
    ```bash
    python main.py
    ```

## Summary of Results
The script outputs two distinct tables ranking the **Top 5 Models** based on **Test Accuracy**. This ranking allows for a quick comparison between training success and real-world generalization (test) performance, helping to identify models that may be overfitting.
