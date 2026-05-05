import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Placement Prediction", page_icon="🎓", layout="centered")

st.title("🎓 Placement Prediction App")
st.write("Predict expected placement package based on CGPA using Linear Regression.")

@st.cache_data
def load_data():
    return pd.read_csv("placement.csv")

try:
    df = load_data()

    X = df[["cgpa"]]
    y = df["package"]

    model = LinearRegression()
    model.fit(X, y)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    cgpa = st.number_input(
        "Enter CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1,
        format="%.1f",
    )

    if st.button("Predict Package"):
        pred = model.predict(np.array([[cgpa]]))[0]
        st.success(f"Estimated Package: {pred:.2f} LPA")

    st.subheader("Model Information")
    col1, col2 = st.columns(2)
    col1.metric("Slope", f"{model.coef_[0]:.4f}")
    col2.metric("Intercept", f"{model.intercept_:.4f}")

except Exception as e:
    st.error("Error: Make sure placement.csv exists in the same folder as app.py")
    st.exception(e)

