# Breast Cancer Classification ML App


**🔗 Quick Links:**
- **Live App:** [https://tumor-classification-dashboard-d6kxf5xxmzk8gza3ujfuae.streamlit.app/](https://tumor-classification-dashboard-d6kxf5xxmzk8gza3ujfuae.streamlit.app/)
- **GitHub:** [https://github.com/jomy-g/tumor-classification-dashboard](https://github.com/jomy-g/tumor-classification-dashboard)

---

## Problem Statement
The goal of this project is to build multiple machine learning classification models to predict whether a breast tumor is malignant or benign using the Breast Cancer Wisconsin dataset. The models are evaluated using comprehensive performance metrics including accuracy, AUC, precision, recall, F1 score, and Matthews Correlation Coefficient (MCC). The project demonstrates a complete end-to-end ML workflow including data preprocessing, model training, evaluation, and deployment via an interactive Streamlit web application.

---

## Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset from sklearn.datasets

**Source:** UCI Machine Learning Repository

**Description:** The dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast mass. The features describe characteristics of cell nuclei present in the image.

- **Total Instances:** 569 samples
- **Total Features:** 30 numeric, real-valued features
- **Target Variable:** Diagnosis
  - Class 0: Malignant (212 samples)
  - Class 1: Benign (357 samples)
- **Feature Types:** All features are computed from digitized images:
  - radius (mean of distances from center to points on the perimeter)
  - texture (standard deviation of gray-scale values)
  - perimeter
  - area
  - smoothness (local variation in radius lengths)
  - compactness (perimeter^2 / area - 1.0)
  - concavity (severity of concave portions of the contour)
  - concave points (number of concave portions of the contour)
  - symmetry
  - fractal dimension ("coastline approximation" - 1)

For each of the 10 characteristics above, three values are computed: mean, standard error, and "worst" (mean of the three largest values), resulting in 30 features total.

**Data Split:** 80% training (455 samples), 20% testing (114 samples)

---

## Models Used

The following six classification models were implemented and evaluated:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.9737 | 0.9974 | 0.9722 | 0.9859 | 0.9790 | 0.9439 |
| Decision Tree | 0.9298 | 0.9253 | 0.9437 | 0.9437 | 0.9437 | 0.8506 |
| KNN | 0.9474 | 0.9820 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| Naive Bayes | 0.9737 | 0.9984 | 0.9595 | 1.0000 | 0.9793 | 0.9447 |
| Random Forest | 0.9561 | 0.9966 | 0.9583 | 0.9718 | 0.9650 | 0.9064 |
| XGBoost | 0.9649 | 0.9954 | 0.9722 | 0.9718 | 0.9720 | 0.9246 |

---

## Observations about Model Performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieved excellent performance (97.37% accuracy, 99.74% AUC) demonstrating the dataset has strong linear separability. The high recall (98.59%) indicates minimal false negatives, which is crucial for medical diagnosis. Feature scaling significantly improved convergence and performance. |
| Decision Tree | Showed good but comparatively lower performance (92.98% accuracy) with balanced precision and recall. The model is prone to overfitting on training data and exhibits higher variance. May require pruning or ensemble methods for better generalization. |
| KNN | Delivered solid performance (94.74% accuracy, 98.20% AUC) with balanced classification across both classes. Highly sensitive to feature scaling, which was critical for achieving these results. Performance could be further optimized through hyperparameter tuning of k value. |
| Naive Bayes | Surprisingly achieved top-tier accuracy (97.37%) and the highest AUC (99.84%) despite its independence assumption. Perfect recall (100%) means zero false negatives - highly desirable for cancer screening. Fast training and prediction make it suitable for real-time applications. |
| Random Forest | Demonstrated robust and stable performance (95.61% accuracy, 99.66% AUC) with high precision-recall balance. The ensemble approach provides good generalization and is resistant to overfitting. Feature importance analysis capability is valuable for understanding prediction drivers. |
| XGBoost | Achieved strong overall performance (96.49% accuracy, 99.54% AUC) with excellent precision-recall balance (both ~97%). The gradient boosting approach effectively captures complex patterns. Provides best trade-off between performance and interpretability through feature importance rankings. |

---

## Key Insights

1. **Best Overall Models:** Logistic Regression and Naive Bayes tied for highest accuracy (97.37%), with Naive Bayes achieving perfect recall.

2. **Best AUC Score:** Naive Bayes (99.84%) edges out Logistic Regression (99.74%), indicating superior probability calibration.

3. **Most Balanced:** XGBoost and Logistic Regression show the best balance across all metrics.

4. **Clinical Relevance:** Naive Bayes' perfect recall (zero false negatives) makes it particularly suitable for cancer screening where missing a malignant case has severe consequences.

5. **Feature Importance:** The high performance across all models suggests that the engineered features from FNA images are highly predictive of malignancy.

---

## Technical Implementation

### Preprocessing
- StandardScaler applied for Logistic Regression and KNN
- No scaling required for tree-based models (Decision Tree, Random Forest, XGBoost)
- Train-test split: 80-20 with random_state=42 for reproducibility

### Model Training
- All models trained on same train-test split for fair comparison
- Hyperparameters:
  - Logistic Regression: max_iter=1000, default regularization
  - Decision Tree: default parameters (Gini impurity)
  - KNN: default k=5 neighbors
  - Naive Bayes: Gaussian distribution assumption
  - Random Forest: default 100 estimators
  - XGBoost: logloss evaluation metric

### Evaluation Metrics
Six comprehensive metrics used for holistic model assessment:
1. **Accuracy:** Overall correctness
2. **AUC-ROC:** Probability ranking quality
3. **Precision:** Positive prediction accuracy
4. **Recall:** True positive detection rate
5. **F1 Score:** Harmonic mean of precision and recall
6. **MCC:** Correlation between predictions and actual values (best for imbalanced data)

---

## Project Structure

```
tumor-classification-dashboard/
├── app.py                          # Streamlit web application
├── model_training.py               # Model training script
├── model_training.ipynb            # Jupyter notebook version (alternative)
├── model/                          # Saved model files
│   ├── Logistic Regression.pkl
│   ├── Decision Tree.pkl
│   ├── KNN.pkl
│   ├── Naive Bayes.pkl
│   ├── Random Forest.pkl
│   ├── XGBoost.pkl
│   └── scaler.pkl                  # StandardScaler for preprocessing
├── breast_cancer_test_data.csv     # Test dataset with labels
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```


---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/jomy-g/tumor-classification-dashboard.git
cd tumor-classification-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train models (optional, models are already trained)
```bash
python model_training.py
```

### 4. Run Streamlit app
```bash
streamlit run app.py
```

Local Access: The app will open in your browser at `http://localhost:8501`

OR use the deployed version: https://tumor-classification-dashboard-d6kxf5xxmzk8gza3ujfuae.streamlit.app/

---

## Streamlit App Features

The deployed web application provides:

1. ✅ **Test Data Download:** Quick download button for test dataset CSV file
2. ✅ **Dataset Upload:** Upload CSV files containing breast cancer features
3. ✅ **Model Selection:** Choose from 6 trained classification models via dropdown
4. ✅ **Real-time Predictions:** Get instant predictions on uploaded data
5. ✅ **Comprehensive Metrics:** View all 6 evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
6. ✅ **Confusion Matrix:** Interactive heatmap visualization of classification results
7. ✅ **Classification Report:** Detailed per-class performance statistics
8. ✅ **Probability Distribution:** Histogram of prediction probabilities
9. ✅ **Prediction vs Actual Comparison:** Side-by-side comparison of predictions and ground truth

---

## Deployment

**Platform:** Streamlit Community Cloud (FREE)

**Live Application:** [Breast Cancer Classification App](https://tumor-classification-dashboard-d6kxf5xxmzk8gza3ujfuae.streamlit.app/)

**GitHub Repository:** [https://github.com/jomy-g/tumor-classification-dashboard](https://github.com/jomy-g/tumor-classification-dashboard)

**Deployment Steps:**
1. Push code to GitHub repository
2. Sign in to [share.streamlit.io](https://share.streamlit.io) with GitHub
3. Select repository and branch
4. Specify `app.py` as main file
5. Deploy

**Status:** ✅ Successfully Deployed

---

## Requirements

```
streamlit==1.32.0
scikit-learn==1.4.0
pandas==2.2.0
numpy==1.26.3
matplotlib==3.8.2
seaborn==0.13.1
xgboost==2.0.3
```

---

## Dataset Usage

For testing the app, use the provided `breast_cancer_test_data.csv` file which contains:
- 114 test samples
- 30 feature columns
- 1 diagnosis column (0=Malignant, 1=Benign)

---
