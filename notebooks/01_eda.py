"""
01_eda.py
---------------------------------
Exploratory Data Analysis (EDA) for
Auto Insurance Claim Risk Prediction

Purpose:
- Understand data structure
- Identify imbalance in target variable
- Analyze feature correlations
- Categorize features (binary, categorical, ordinal, interval)
- Generate key insights for modeling

Author: Suganya P
"""

# ===============================
# 1. Imports
# ===============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization settings
plt.style.use("seaborn-whitegrid")
sns.set_context("notebook")

# ===============================
# 2. Load Dataset
# ===============================

# NOTE:
# Replace path if required.
# Original dataset is not committed to GitHub.
data = pd.read_csv("data/train.csv")

print("Dataset Shape:", data.shape)
print("\nColumns:\n", data.columns.tolist())
print("\nSample Records:\n", data.head())

# ===============================
# 3. Target Variable Analysis
# ===============================

target_counts = data["target"].value_counts()
target_percent = (target_counts / len(data)) * 100

print("\nTarget Distribution:")
print(target_counts)
print("\nTarget Percentage:")
print(target_percent)

# Bar plot for imbalance visualization
target_percent.plot(
    kind="bar",
    title="Target Variable Distribution (%)",
    ylabel="Percentage",
    xlabel="Target Class",
    rot=0
)
plt.show()

# Inference:
# - Data is highly imbalanced
# - Minority class (~3.6%) represents claim cases

# ===============================
# 4. Feature Categorization
# ===============================

columns = data.columns.tolist()

categorical_features = [c for c in columns if "cat" in c]
binary_features = [c for c in columns if "bin" in c]
interval_features = [c for c in columns if c.startswith(("ps_reg", "ps_car", "ps_calc")) and c not in categorical_features and c not in binary_features]
ordinal_features = [c for c in columns if c not in categorical_features + binary_features + interval_features + ["id", "target"]]

print("\nFeature Counts:")
print("Categorical:", len(categorical_features))
print("Binary:", len(binary_features))
print("Interval:", len(interval_features))
print("Ordinal:", len(ordinal_features))

# ===============================
# 5. Correlation Analysis
# ===============================

# Overall correlation heatmap (numerical features)
plt.figure(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr, cmap="YlGnBu", linewidths=0.1)
plt.title("Correlation Heatmap (All Numerical Features)")
plt.show()

# ===============================
# 6. Interval Feature Analysis
# ===============================

interval_subset = data[interval_features]

plt.figure(figsize=(10, 8))
sns.heatmap(interval_subset.corr(), cmap="YlGnBu", linewidths=0.1)
plt.title("Correlation Heatmap - Interval Features")
plt.show()

# Distribution plots
for col in interval_subset.columns[:10]:
    sns.histplot(interval_subset[col], kde=True)
    plt.title(f"Distribution: {col}")
    plt.show()

# ===============================
# 7. Ordinal Feature Analysis
# ===============================

ordinal_subset = data[ordinal_features]

plt.figure(figsize=(10, 8))
sns.heatmap(ordinal_subset.corr(), cmap="YlGnBu", linewidths=0.1)
plt.title("Correlation Heatmap - Ordinal Features")
plt.show()

# ===============================
# 8. Binary Feature Analysis
# ===============================

binary_subset = data[binary_features]

plt.figure(figsize=(10, 8))
sns.heatmap(binary_subset.corr(), cmap="YlGnBu", linewidths=0.1)
plt.title("Correlation Heatmap - Binary Features")
plt.show()

# ===============================
# 9. Missing Value Analysis
# ===============================

# Missing values encoded as -1
missing_features = [col for col in data.columns if (data[col] == -1).any()]

print("\nFeatures with Missing Values (-1 encoding):")
print(missing_features)
print("\nTotal features with missing values:", len(missing_features))

# ===============================
# 10. Key EDA Inferences (Summary)
# ===============================

print("""
Key EDA Insights:
1. Target variable is severely imbalanced (~3.6% positive class).
2. Interval variables show relatively higher correlation with target.
3. Binary and ordinal features are largely independent.
4. Missing values are present and encoded as -1.
5. Accuracy alone is not a reliable metric for this dataset.
""")
