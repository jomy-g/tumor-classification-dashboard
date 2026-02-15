import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Classification App", layout="wide")

st.title("Breast Cancer Classification ML App")

st.markdown("Upload test dataset and select model to see predictions and evaluation metrics")

# Model selection
model_name = st.selectbox("Select Model", [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
])

uploaded_file = st.file_uploader("Upload CSV file")

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    model = pickle.load(open(f"model/{model_name}.pkl", "rb"))

    predictions = model.predict(data)

    st.subheader("Predictions")
    st.write(predictions)

    if st.checkbox("Show Classification Report"):

        report = classification_report(predictions, predictions, output_dict=True)
        st.write(pd.DataFrame(report).transpose())

    if st.checkbox("Show Confusion Matrix"):

        cm = confusion_matrix(predictions, predictions)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        st.pyplot(fig)