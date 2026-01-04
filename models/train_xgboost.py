"""
train_xgboost.py
---------------------------------
Train & Save Final XGBoost Model
for Auto Insurance Claim Prediction

This script:
- Loads preprocessed data
- Trains XGBoost classifier
- Evaluates performance
- Saves model as .pkl

Author: Suganya P
"""

# ===============================
# 1. Imports
# ===============================

import numpy as np
import pickle

from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ===============================
# 2. Load Preprocessed Data
# ===============================

X_train = np.load("data/X_train.npy")
X_test  = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test  = np.load("data/y_test.npy")

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ===============================
# 3. Train XGBoost Model
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

# ===============================
# 4. Evaluate Model
# ===============================

y_pred = xgb_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Accuracy :", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall   :", round(recall_score(y_test, y_pred), 3))
print("F1-score :", round(f1_score(y_test, y_pred), 3))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_pred), 3))

# ===============================
# 5. Save Model
# ===============================

with open("models/xgboost_claim_model.pkl", "wb") as file:
    pickle.dump(xgb_model, file)

print("\nModel saved successfully as:")
print("models/xgboost_claim_model.pkl")
