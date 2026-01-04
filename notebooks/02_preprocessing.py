"""
02_preprocessing.py
---------------------------------
Data Preprocessing for
Auto Insurance Claim Risk Prediction

Purpose:
- Handle missing values
- Encode categorical features
- Scale interval features
- Split data into train/test
- Apply SMOTE correctly (train data only)

Author: Suganya P
"""

# ===============================
# 1. Imports
# ===============================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

# ===============================
# 2. Load Dataset
# ===============================

data = pd.read_csv("data/train.csv")

# Drop ID column (non-informative)
if "id" in data.columns:
    data.drop("id", axis=1, inplace=True)

# ===============================
# 3. Handle Missing Values
# ===============================

# Replace -1 with NaN
data.replace(-1, np.nan, inplace=True)

# Identify feature types
columns = data.columns.tolist()

categorical_features = [c for c in columns if "cat" in c]
binary_features = [c for c in columns if "bin" in c]
interval_features = [
    c for c in columns
    if c.startswith(("ps_reg", "ps_car", "ps_calc"))
    and c not in categorical_features
    and c not in binary_features
    and c != "target"
]

# Fill categorical with mode
for col in categorical_features:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Fill interval features with mean
for col in interval_features:
    data[col].fillna(data[col].mean(), inplace=True)

# Binary features (0/1) → fill with 0
for col in binary_features:
    data[col].fillna(0, inplace=True)

# ===============================
# 4. Split Features & Target
# ===============================

X = data.drop("target", axis=1)
y = data["target"]

# Train-test split BEFORE SMOTE (important!)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ===============================
# 5. Encoding & Scaling Pipeline
# ===============================

numeric_features = interval_features
categorical_features = categorical_features

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
    ],
    remainder="passthrough"  # keeps binary features
)

# Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Processed Train shape:", X_train_processed.shape)
print("Processed Test shape:", X_test_processed.shape)

# ===============================
# 6. Handle Class Imbalance (SMOTE)
# ===============================

smote = SMOTE(
    sampling_strategy=0.12,   # ~12% minority class as per original logic
    random_state=42,
    k_neighbors=2
)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_processed, y_train
)

print("Before SMOTE:", y_train.value_counts(normalize=True))
print("After SMOTE:", y_train_resampled.value_counts(normalize=True))

# ===============================
# 7. Save Processed Data (Optional)
# ===============================

np.save("data/X_train.npy", X_train_resampled)
np.save("data/X_test.npy", X_test_processed)
np.save("data/y_train.npy", y_train_resampled)
np.save("data/y_test.npy", y_test)

print("\nPreprocessing completed successfully.")
