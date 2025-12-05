import streamlit as st
import pandas as pd
import io

from price_volatility.volatility_pred import (
    read_prices,
    read_exog,
    run_for_chemical,
)

st.set_page_config(page_title="📈 Price Volatility", layout="wide")

st.title("📈 Price Volatility – Oil & Gas Chemicals")
st.write("Upload datasets to run SARIMAX + ML hybrid forecasts.")


# ---------------------------
# File Upload Section
# ---------------------------
st.header("📁 Upload Your Data")

price_file = st.file_uploader(
    "Upload chemical_prices.csv",
    type=["csv"],
    help="Columns required: date, chemical, price"
)

exog_file = st.file_uploader(
    "Upload macro_factors.csv",
    type=["csv"],
    help="Must include date + one or more exogenous factor columns."
)

if price_file and exog_file:
    try:
        prices = pd.read_csv(price_file)
        exog = pd.read_csv(exog_file)

        prices["date"] = pd.to_datetime(prices["date"])
        exog["date"] = pd.to_datetime(exog["date"])

        chemicals = sorted(prices["chemical"].unique())

        st.success("Files uploaded successfully!")

    except Exception as e:
        st.error(f"Error reading files: {e}")
        st.stop()

else:
    st.info("Please upload both CSV files to continue.")
    st.stop()



# ---------------------------
# User Controls
# ---------------------------
st.header("⚙️ Configuration")

col1, col2 = st.columns(2)

selected_chemical = col1.selectbox(
    "Choose chemical to forecast",
    chemicals,
)

horizon = col2.number_input(
    "Forecast horizon (periods)",
    min_value=3,
    max_value=36,
    value=12,
    step=1
)

freq = st.selectbox(
    "Data frequency (leave 'Auto' to infer)",
    ["Auto", "D", "W", "M"],
    index=0
)

use_freq = None if freq == "Auto" else freq



# ---------------------------
# Run Forecast
# ---------------------------
st.header("🚀 Run Model")

run_btn = st.button("Run Forecast")

if run_btn:
    with st.spinner("Running SARIMAX + Tree model..."):

        try:
            result = run_for_chemical(
                chem=selected_chemical,
                prices=prices,
                exog=exog,
                freq=use_freq,
                horizon=horizon,
                output_dir=None  # Streamlit mode: no file saving
            )
        except Exception as e:
            st.error(f"Model error: {e}")
            st.stop()

        metrics = result["metrics"]
        forecast_df = result["forecast_df"]
        fig = result["fig"]

        # --- Show Metrics ---
        st.subheader("📊 Forecast Metrics")

        st.json(metrics)

        # --- Chart ---
        st.subheader("📈 Forecast Chart")
        st.pyplot(fig)

        # --- Download CSV ---
        st.subheader("⬇️ Download Forecast CSV")
        csv_bytes = forecast_df.to_csv().encode("utf-8")

        st.download_button(
            label="Download forecast CSV",
            data=csv_bytes,
            file_name=f"forecast_{selected_chemical}.csv",
            mime="text/csv"
        )
