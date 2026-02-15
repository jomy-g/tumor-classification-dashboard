# Breast Cancer Classification ML App

## Problem Statement
The goal of this project is to build multiple machine learning classification models to predict whether a tumor is malignant or benign using the Breast Cancer dataset. The models are evaluated using multiple performance metrics and deployed using Streamlit.

---

## Dataset Description
The Breast Cancer Wisconsin dataset contains features computed from digitized images of breast mass samples.

- Total Instances: 569
- Total Features: 30 numeric features
- Target Variable: Diagnosis (Malignant = 1, Benign = 0)

---

## Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Naive Bayes (Gaussian)
5. Random Forest
6. XGBoost

---

## Performance Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------|----------|-----|----------|-------|---------|-----|
| Logistic Regression | 0.98 | 0.99 | 0.98 | 0.99 | 0.98 | 0.96 |
| Decision Tree | 0.94 | 0.94 | 0.93 | 0.93 | 0.93 | 0.88 |
| KNN | 0.97 | 0.98 | 0.97 | 0.98 | 0.97 | 0.94 |
| Naive Bayes | 0.95 | 0.97 | 0.94 | 0.95 | 0.94 | 0.90 |
| Random Forest | 0.97 | 0.99 | 0.97 | 0.97 | 0.97 | 0.94 |
| XGBoost | 0.98 | 0.99 | 0.98 | 0.98 | 0.98 | 0.96 |

---

## Observations

| Model | Observation |
|------|------------|
| Logistic Regression | Performs very well due to linear separability of dataset |
| Decision Tree | Slight overfitting observed |
| KNN | Good accuracy but sensitive to scaling |
| Naive Bayes | Fast but assumes feature independence |
| Random Forest | Robust and stable performance |
| XGBoost | Best overall performance |

---

## How to Run Locally

1. Install dependencies
pip install -r requirements.txt

2. Train models
python model_training.py

3. Run Streamlit app
streamlit run app.py

---

## Deployment

The application is deployed on Streamlit Community Cloud.

