import streamlit as st
import pandas as pd

from price_volatility.volatility_pred import run_for_chemical

st.set_page_config(page_title="📈 Price Volatility", layout="wide")

st.title("📈 Price Volatility – Oil & Gas Chemicals")
st.caption("Short-term forecasting using SARIMAX + Gradient Boosting with macroeconomic drivers.")


# ---------------------------
# File Uploads
# ---------------------------
st.subheader("📁 Upload Your Data")

price_file = st.file_uploader("chemical_prices.csv", type=["csv"])
exog_file = st.file_uploader("macro_factors.csv", type=["csv"])

if not price_file or not exog_file:
    st.info("Upload both CSV files to proceed.")
    st.stop()

try:
    prices = pd.read_csv(price_file)
    exog = pd.read_csv(exog_file)

    prices["date"] = pd.to_datetime(prices["date"])
    exog["date"] = pd.to_datetime(exog["date"])

    chemicals = sorted(prices["chemical"].unique())

except Exception as e:
    st.error(f"Failed to read uploaded CSVs: {e}")
    st.stop()


# ---------------------------
# Controls
# ---------------------------
st.subheader("⚙️ Forecast Configuration")

col1, col2 = st.columns(2)

chemical = col1.selectbox("Chemical", chemicals)

horizon = col2.number_input(
    "Forecast horizon",
    min_value=3,
    max_value=24,
    value=12,
    step=1,
)

freq_option = st.selectbox(
    "Frequency Override",
    ["Auto", "D", "W", "M"],
    index=0
)
freq = None if freq_option == "Auto" else freq_option


# ---------------------------
# Run
# ---------------------------
st.subheader("🚀 Run Forecast")

if st.button("Run Model"):
    with st.spinner("Training models & computing forecasts..."):
        try:
            result = run_for_chemical(
                chem=chemical,
                prices=prices,
                exog=exog,
                freq=freq,
                horizon=horizon,
                output_dir=None,
            )
        except Exception as e:
            st.error(f"Model error: {e}")
            st.stop()

        metrics = result["metrics"]
        forecast_df = result["forecast_df"]
        fig = result["fig"]

        # Display metrics
        st.markdown("### 📊 Model Performance")
        st.json(metrics)

        # Display plot (small)
        st.markdown("### 📈 Forecast Plot")
        st.pyplot(fig, clear_figure=True)

        # Download CSV
        st.download_button(
            "⬇️ Download Forecast CSV",
            forecast_df.to_csv().encode("utf-8"),
            file_name=f"forecast_{chemical}.csv",
            mime="text/csv",
        )
