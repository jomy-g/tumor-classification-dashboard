import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Classification App", layout="wide")

st.title("🧬 Breast Cancer Classification ML App")

st.markdown("""
This application demonstrates multiple machine learning models for breast cancer classification.
Upload test dataset (CSV with 'diagnosis' column) and select a model to see predictions and evaluation metrics.
""")

# Add download button for test data
st.info("📥 **Download Test Data:** Use the button below to download the test dataset for evaluation")

try:
    # Load test data for download
    test_data = pd.read_csv("breast_cancer_test_data.csv")
    csv_data = test_data.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="⬇️ Download Test Data CSV",
        data=csv_data,
        file_name="breast_cancer_test_data.csv",
        mime="text/csv",
        help="Download the test dataset to upload and test the models"
    )
except Exception as e:
    st.warning(f"Test data file not found. Please ensure 'breast_cancer_test_data.csv' is in the app directory.")

st.markdown("---")

# Model selection
model_name = st.selectbox("Select Model", [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
])

# File upload
uploaded_file = st.file_uploader("Upload CSV file (must include 'diagnosis' column for evaluation)", type=['csv'])

if uploaded_file:
    # Read data
    data = pd.read_csv(uploaded_file)
    
    st.success(f"✅ File uploaded successfully! Shape: {data.shape}")
    
    # Check if diagnosis column exists
    if 'diagnosis' not in data.columns:
        st.error("❌ Error: 'diagnosis' column not found in uploaded file. Please upload a file with the target column.")
    else:
        # Separate features and target
        X = data.drop('diagnosis', axis=1)
        y_true = data['diagnosis']
        
        st.info(f"📊 Features: {X.shape[1]} | Samples: {X.shape[0]}")
        
        try:
            # Load model
            model = pickle.load(open(f"model/{model_name}.pkl", "rb"))
            
            # Load scaler for models that need scaling
            if model_name in ["Logistic Regression", "KNN"]:
                scaler = pickle.load(open("model/scaler.pkl", "rb"))
                X_processed = scaler.transform(X)
            else:
                X_processed = X
            
            # Make predictions
            predictions = model.predict(X_processed)
            pred_proba = model.predict_proba(X_processed)[:, 1]
            
            # Display predictions
            st.subheader("🎯 Predictions")
            pred_df = pd.DataFrame({
                'Actual': y_true,
                'Predicted': predictions,
                'Probability (Class 1)': pred_proba
            })
            st.dataframe(pred_df.head(20), use_container_width=True)
            
            # Calculate metrics
            st.subheader("📈 Model Evaluation Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                acc = accuracy_score(y_true, predictions)
                st.metric("Accuracy", f"{acc:.4f}")
                
            with col2:
                auc = roc_auc_score(y_true, pred_proba)
                st.metric("AUC Score", f"{auc:.4f}")
                
            with col3:
                precision = precision_score(y_true, predictions)
                st.metric("Precision", f"{precision:.4f}")
            
            col4, col5, col6 = st.columns(3)
            
            with col4:
                recall = recall_score(y_true, predictions)
                st.metric("Recall", f"{recall:.4f}")
                
            with col5:
                f1 = f1_score(y_true, predictions)
                st.metric("F1 Score", f"{f1:.4f}")
                
            with col6:
                mcc = matthews_corrcoef(y_true, predictions)
                st.metric("MCC Score", f"{mcc:.4f}")
            
            # Confusion Matrix
            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_true, predictions)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Malignant (0)', 'Benign (1)'],
                       yticklabels=['Malignant (0)', 'Benign (1)'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - {model_name}')
            st.pyplot(fig)
            
            # Classification Report
            with st.expander("📋 Detailed Classification Report"):
                report = classification_report(y_true, predictions, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
            
            # Distribution of predictions
            st.subheader("📊 Prediction Distribution")
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Actual vs Predicted counts
            actual_counts = pd.Series(y_true).value_counts().sort_index()
            pred_counts = pd.Series(predictions).value_counts().sort_index()
            
            x = ['Malignant (0)', 'Benign (1)']
            ax1.bar(x, actual_counts.values, alpha=0.7, label='Actual', color='blue')
            ax1.bar(x, pred_counts.values, alpha=0.7, label='Predicted', color='orange')
            ax1.set_ylabel('Count')
            ax1.set_title('Actual vs Predicted Distribution')
            ax1.legend()
            
            # Probability distribution
            ax2.hist(pred_proba, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Predicted Probability (Class 1)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Predicted Probabilities')
            
            st.pyplot(fig2)
            
        except FileNotFoundError:
            st.error(f"❌ Model file '{model_name}.pkl' not found. Please ensure models are trained and saved in the 'model' folder.")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)
else:
    st.info("👆 Please upload a CSV file to begin classification.")
    st.markdown("""
    ### Expected CSV Format:
    - Must contain all 30 breast cancer feature columns
    - Must include a 'diagnosis' column (0 = Malignant, 1 = Benign)
    - Use the provided `breast_cancer_test_data.csv` for testing
    """)
