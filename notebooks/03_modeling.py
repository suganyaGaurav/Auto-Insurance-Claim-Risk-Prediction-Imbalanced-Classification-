"""
03_modeling.py
---------------------------------
Model Training & Comparison for
Auto Insurance Claim Risk Prediction

Purpose:
- Train multiple ML models
- Evaluate models using business-relevant metrics
- Compare performance on imbalanced classification
- Select best-performing model

Author: Suganya P
"""

# ===============================
# 1. Imports
# ===============================

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)

# ===============================
# 2. Load Preprocessed Data
# ===============================

X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# ===============================
# 3. Evaluation Utility Function
# ===============================

def evaluate_model(model_name, y_true, y_pred):
    print(f"\n================ {model_name} ================\n")

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    print("Accuracy :", round(accuracy_score(y_true, y_pred), 3))
    print("Precision:", round(precision_score(y_true, y_pred), 3))
    print("Recall   :", round(recall_score(y_true, y_pred), 3))
    print("F1-score :", round(f1_score(y_true, y_pred), 3))
    print("ROC-AUC  :", round(roc_auc_score(y_true, y_pred), 3))


# ===============================
# 4. Logistic Regression
# ===============================

log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)
evaluate_model("Logistic Regression", y_test, y_pred_log)

# ===============================
# 5. Linear Support Vector Machine
# ===============================

svm_model = LinearSVC(dual=False, max_iter=2000)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
evaluate_model("Linear SVM", y_test, y_pred_svm)

# ===============================
# 6. AdaBoost Classifier
# ===============================

ada_model = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)
ada_model.fit(X_train, y_train)

y_pred_ada = ada_model.predict(X_test)
evaluate_model("AdaBoost", y_test, y_pred_ada)

# ===============================
# 7. XGBoost Classifier
# ===============================

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
evaluate_model("XGBoost", y_test, y_pred_xgb)

# ===============================
# 8. Model Selection Summary
# ===============================

print("""
Model Selection Summary:
- Logistic Regression: High accuracy, weak minority recall
- Linear SVM: Limited improvement on imbalance
- AdaBoost: No significant gain
- XGBoost: Best balance of recall, precision, and F1-score

Selected Model: XGBoost
Reason: Superior detection of minority (claim) class
""")
