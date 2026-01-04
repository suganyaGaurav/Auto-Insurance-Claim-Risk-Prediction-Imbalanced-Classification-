# Auto Insurance Claim Risk Prediction (Imbalanced Classification)

> Note: This project intentionally avoids deployment to focus on modeling rigor. 
> Production deployment examples are demonstrated in other repositories.


## Overview

This project focuses on building a classical machine learning model to predict whether an auto insurance policyholder is likely to initiate an insurance claim in the next year.  
The key challenge addressed in this project is **severe class imbalance**, which makes naive accuracy-based models unreliable for real-world decision-making.

The project demonstrates strong **EDA discipline, imbalance handling, model evaluation rigor, and business-aligned reasoning**, making it suitable as a portfolio case study for traditional ML roles.

## Business Problem

Auto insurance companies incur significant financial risk when high-risk customers are misclassified as low risk.  
Missing a potential claim (false negative) can lead to unexpected losses, improper pricing, and poor underwriting decisions.

The goal of this project is to **identify claim-prone customers early**, enabling:
- Better risk assessment
- Improved pricing strategies
- Proactive customer engagement
- Reduced financial exposure

## Key Challenge: Class Imbalance

- Target variable distribution:
  - **~96.4%** → No Claim
  - **~3.6%** → Claim Initiated

Due to this imbalance:
- Accuracy alone is misleading
- Models predicting only the majority class appear “good” but fail in practice
- Metrics like **Recall, Precision, and F1-score** are more meaningful

## Project Objectives

- Perform detailed **Exploratory Data Analysis (EDA)**
- Categorize features into binary, ordinal, interval, and categorical
- Handle missing values and data quality issues
- Address class imbalance using resampling techniques
- Compare multiple ML models using business-relevant metrics
- Select a model that minimizes **false negatives**

## Dataset Description

- Large tabular dataset with mixed feature types
- Features include:
  - Customer demographics
  - Vehicle characteristics
  - Regional and calculated attributes
- Missing values are encoded as `-1`
- Target variable:
  - `0` → No insurance claim
  - `1` → Insurance claim initiated

### ⚠️ Data Availability
Due to confidentiality and licensing constraints, the original dataset is **not included** in this repository.

A small dummy dataset is provided only to understand:
- Feature naming
- Data structure
- Target variable format

## Exploratory Data Analysis (EDA)

Key insights obtained:
- Most features have weak individual correlation with the target
- Interval features related to vehicle and regional attributes show relatively higher correlation
- Binary and ordinal features are largely independent
- Strong multicollinearity observed among certain interval variables
- Severe class imbalance confirmed through distribution analysis

EDA involved:
- Correlation heatmaps
- Bar plots
- Distribution plots
- Box plots
- Target proportion analysis

## Data Preprocessing

- **Missing Value Handling**
  - Categorical features → Mode
  - Continuous features → Mean
- **Encoding**
  - One-Hot Encoding applied to categorical variables
- **Scaling**
  - Applied where suitable for interval features
- **Imbalance Handling**
  - SMOTE used **only on training data** to avoid data leakage

## Models Evaluated

The following models were trained and evaluated:

- Logistic Regression
- Support Vector Machine (Linear SVC)
- AdaBoost
- XGBoost
- Multi-Layer Perceptron (MLP)

## Evaluation Metrics

Given the insurance business context, the following metrics were prioritized:

- **Recall (Positive Class)** – minimize missed claims
- **F1-score** – balance between precision and recall
- Confusion Matrix
- ROC-AUC (secondary)

Accuracy was used only as a **supporting metric**, not as the primary decision factor.

## Model Comparison Summary

| Model | Observation |
|------|-------------|
| Logistic Regression | High accuracy but failed to detect minority class |
| SVM | Limited improvement on recall |
| AdaBoost | No significant performance gain |
| MLP | Sensitive to hyperparameters |
| **XGBoost** | Best balance of recall, precision, and stability |

✅ **XGBoost** was selected as the final model due to superior minority-class detection and robustness on tabular data.

## Business Impact

The selected model can assist insurers in:
- Risk-based customer segmentation
- Improved underwriting decisions
- Better pricing strategies
- Proactive risk mitigation

⚠️ This model is intended as a **decision-support tool**, not a fully automated approval or rejection system.

## Limitations & Future Improvements

- Introduce PR-AUC as a primary metric
- Apply explicit cost-sensitive learning
- Optimize decision thresholds based on business cost
- Add explainability using SHAP
- Integrate real-time policy data
- Improve pipeline reproducibility


## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)

## Project Status

✔ Completed  
✔ Resume-ready  
✔ GitHub portfolio suitable  

This project complements modern GenAI systems by demonstrating strong foundations in classical machine learning and evaluation rigor.
