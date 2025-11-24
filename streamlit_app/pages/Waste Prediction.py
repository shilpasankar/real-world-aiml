import streamlit as st
import pandas as pd

st.title("🗑️ Waste Prediction")

uploaded_file = st.file_uploader("Upload daily demand CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Input Data Preview")
    st.write(df.head())

    st.info("Model integration coming next — data successfully loaded!")
else:
    st.info("Upload a CSV file to continue.")
