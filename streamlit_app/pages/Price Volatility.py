import streamlit as st
import pandas as pd

from price_volatility.volatility_pred import run_for_chemical

st.set_page_config(page_title="📈 Price Volatility", layout="wide")

st.title("📈 Price Volatility – Oil & Gas Chemicals")
st.caption("SARIMAX + Tree-based baseline with rich diagnostics.")


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

    if not {"date", "chemical", "price"}.issubset(prices.columns):
        st.error("chemical_prices.csv must contain: date, chemical, price")
        st.stop()

    if "date" not in exog.columns:
        st.error("macro_factors.csv must contain a 'date' column.")
        st.stop()

    chemicals = sorted(prices["chemical"].unique())

except Exception as e:
    st.error(f"Failed to read uploaded CSVs: {e}")
    st.stop()


# ---------------------------
# Controls
# ---------------------------
st.subheader("⚙️ Forecast Configuration")

col1, col2, col3 = st.columns(3)

chemical = col1.selectbox("Chemical", chemicals)

horizon = col2.number_input(
    "Forecast horizon (periods)",
    min_value=3,
    max_value=24,
    value=12,
    step=1,
)

freq_option = col3.selectbox(
    "Frequency Override",
    ["Auto", "D", "W", "M"],
    index=0,
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

        fig_main = result["fig_main"]
        fig_resid = result["fig_resid"]
        fig_hist = result["fig_hist"]
        fig_imp = result["fig_importance"]

        # ---------- Metrics: pretty comparison ----------
        st.markdown("### 📊 Model Performance")

        m = metrics
        sar = m["sarimax"]
        tree = m["tree"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "SARIMAX MAPE (%)",
                f"{sar['mape']:.2f}",
                help=f"MAE: {sar['mae']:.2f}"
            )
        with col_b:
            st.metric(
                "Tree MAPE (%)",
                f"{tree['mape']:.2f}",
                help=f"MAE: {tree['mae']:.2f}"
            )

        st.write("Raw metrics JSON (for debugging):")
        st.json(metrics)

        # ---------- Main forecast plot ----------
        st.markdown("### 📈 Forecast vs Actuals")
        st.pyplot(fig_main, clear_figure=True)

        # ---------- Residual diagnostics ----------
        st.markdown("### 📉 Residual Diagnostics")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.caption("Residuals over time")
            st.pyplot(fig_resid, clear_figure=True)
        with col_r2:
            st.caption("Residual distribution (best model)")
            st.pyplot(fig_hist, clear_figure=True)

        # ---------- Feature importance ----------
        st.markdown("### 🌳 Feature Importance (Tree Model)")
        st.pyplot(fig_imp, clear_figure=True)

        # ---------- Download CSV ----------
        st.markdown("### ⬇️ Download Forecast CSV")
        st.download_button(
            "Download CSV",
            forecast_df.to_csv().encode("utf-8"),
            file_name=f"forecast_{chemical}.csv",
            mime="text/csv",
        )
