import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocess import preprocess_data
from src.train_models import train_and_evaluate

st.title("Auto ML Model Selector")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Run AutoML"):

        X, y = preprocess_data(df, target)

        results = train_and_evaluate(X, y)

        st.subheader("Model Performance")

        for model, score in results.items():
            st.write(f"{model}: {score:.3f}")

        best_model = max(results, key=results.get)
        st.success(f"Best Model: {best_model}")

        models = list(results.keys())
        scores = list(results.values())

        fig, ax = plt.subplots()
        ax.bar(models, scores)
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Comparison")

        st.pyplot(fig)