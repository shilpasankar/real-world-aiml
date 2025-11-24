# streamlit_app/airport_recommender/forecast.py
# Streamlit Forecast Demo (Prophet) — compact visuals, uploads, metrics, downloads
# - Upload your own time series (columns: ds, y) or simulate data
# - Optional separate VALIDATION file; otherwise uses a test split from TRAIN
# - Fixed-size Matplotlib visuals, MAE/RMSE/MAPE, and a split summary table
# - Download forecast CSV

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- Page ----------
st.set_page_config(page_title="✈️ Airport Passenger Forecast", layout="wide")
st.title("✈️ Airport Passenger Forecast (Prophet)")

st.markdown(
    "Upload daily passenger data (`ds` date, `y` count). "
    "Optionally provide a separate **VALIDATION** CSV. "
    "If no validation is uploaded, a TEST window is carved from the end of TRAIN."
)

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("Controls")
    horizon = st.slider("Forecast horizon (days)", min_value=30, max_value=180, value=90, step=15)
    test_days = st.slider("Test window from TRAIN (days)", min_value=30, max_value=180, value=90, step=15)
    yearly = st.checkbox("Yearly seasonality", value=True)
    weekly = st.checkbox("Weekly seasonality", value=True)
    daily  = st.checkbox("Daily seasonality", value=False)
    cps = st.slider("Changepoint prior scale", 0.01, 0.5, 0.1, 0.01)
    st.caption("Plots use fixed figsize + tight layout for clean screenshots.")

# ---------- Uploaders ----------
train_file = st.file_uploader("Upload TRAIN time-series CSV (columns: ds, y)", type=["csv"])
val_file   = st.file_uploader("Optional: Upload VALIDATION CSV (columns: ds, y)", type=["csv"])

# ---------- Helpers ----------
def _compact_show(fig, width=6.5, height=3.6):
    fig.set_size_inches(width, height)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False, clear_figure=True)

def _read_series(upload) -> pd.DataFrame:
    df = pd.read_csv(upload)
    if "ds" not in df.columns or "y" not in df.columns:
        raise ValueError("CSV must have columns: ds (date), y (value)")
    df = df[["ds", "y"]].copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds")
    return df.reset_index(drop=True)

def _mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, y_true)
    return np.nanmean(np.abs((y_true - y_pred) / denom))

@st.cache_resource
def _fit_prophet(train_df: pd.DataFrame, yearly: bool, weekly: bool, daily: bool, cps: float) -> Prophet:
    m = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=daily,
        changepoint_prior_scale=cps,
    )
    m.fit(train_df)
    return m

# ---------- Main ----------
if train_file is None:
    st.info("Upload the TRAIN CSV to begin.")
    st.stop()

try:
    train_df_full = _read_series(train_file)
    st.success(f"TRAIN loaded: {len(train_df_full):,} rows ({train_df_full['ds'].min().date()} → {train_df_full['ds'].max().date()})")
except Exception as e:
    st.error(f"Failed to read TRAIN CSV: {e}")
    st.stop()

# If validation provided: we'll use TRAIN file as-is for fitting, VALIDATION for evaluation
# Else: carve TEST from the tail of TRAIN file
if val_file is not None:
    try:
        val_df = _read_series(val_file)
        st.success(f"VALIDATION loaded: {len(val_df):,} rows ({val_df['ds'].min().date()} → {val_df['ds'].max().date()})")
    except Exception as e:
        st.error(f"Failed to read VALIDATION CSV: {e}")
        st.stop()
    # Train on all of TRAIN
    train_df = train_df_full.copy()
    test_df = None
else:
    if len(train_df_full) < test_days + 30:
        st.error("Not enough TRAIN rows for the selected TEST window. Reduce TEST window or add more data.")
        st.stop()
    train_df = train_df_full.iloc[:-test_days].copy()
    test_df  = train_df_full.iloc[-test_days:].copy()

# ---------- Fit ----------
model = _fit_prophet(train_df, yearly=yearly, weekly=weekly, daily=daily, cps=cps)

future = model.make_future_dataframe(periods=horizon, freq="D")
forecast = model.predict(future)
fc_idx = forecast.set_index("ds")

# ---------- Evaluate ----------
metrics_source = []
if val_file is not None:
    common = val_df["ds"]
    if not set(common).issubset(set(fc_idx.index)):
        st.warning("Validation dates extend beyond forecast range; metrics computed on overlapping dates only.")
        common = common[common.isin(fc_idx.index)]
    if len(common) > 0:
        yhat_val = fc_idx.loc[common, "yhat"].values
        ytrue_val = val_df.set_index("ds").loc[common, "y"].values
        mae_val = mean_absolute_error(ytrue_val, yhat_val)
        rmse_val = mean_squared_error(ytrue_val, yhat_val, squared=False)
        mape_val = _mape(ytrue_val, yhat_val)
        metrics_source.append(("Validation", mae_val, rmse_val, mape_val))
else:
    # Evaluate on the held-out TEST slice from TRAIN
    common = test_df["ds"]
    yhat_test = fc_idx.loc[common, "yhat"].values
    ytrue_test = test_df["y"].values
    mae_test = mean_absolute_error(ytrue_test, yhat_test)
    rmse_test = mean_squared_error(ytrue_test, yhat_test, squared=False)
    mape_test = _mape(ytrue_test, yhat_test)
    metrics_source.append(("Test", mae_test, rmse_test, mape_test))

# ---------- Metrics Row ----------
if metrics_source:
    label, m_mae, m_rmse, m_mape = metrics_source[0]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"MAE ({label})", f"{m_mae:,.0f}")
    c2.metric(f"RMSE ({label})", f"{m_rmse:,.0f}")
    c3.metric(f"MAPE ({label})", f"{m_mape*100:,.1f}%")

# ---------- Split Overview ----------
st.subheader("Train / Test / Forecast Overview")
rows = [("Train", train_df["ds"].min().date(), train_df["ds"].max().date(), len(train_df))]
if test_df is not None:
    rows.append(("Test", test_df["ds"].min().date(), test_df["ds"].max().date(), len(test_df)))
else:
    rows.append(("Validation", val_df["ds"].min().date(), val_df["ds"].max().date(), len(val_df)))
rows.append(("Forecast (Future)", forecast["ds"].iloc[-horizon].date(), forecast["ds"].iloc[-1].date(), horizon))
split_info = pd.DataFrame(rows, columns=["Split", "Start Date", "End Date", "Rows"])
st.dataframe(split_info, use_container_width=True, hide_index=True)

# ---------- Plot: Forecast vs Actuals ----------
st.subheader("Forecast vs Actuals")
fig1 = model.plot(forecast, xlabel="Date", ylabel="Passengers")
fig1.axes[0].set_title("Passenger Count Forecast", fontsize=12)
ax = fig1.axes[0]

# Overlay actuals: TRAIN and (TEST or VALIDATION)
ax.plot(train_df["ds"], train_df["y"], linestyle="none", marker="o", ms=2.5, color="black", label="Actual (Train)")
if test_df is not None:
    ax.plot(test_df["ds"], test_df["y"], linestyle="none", marker="o", ms=2.5, color="red", label="Actual (Test)")
elif val_file is not None:
    ax.plot(val_df["ds"], val_df["y"], linestyle="none", marker="o", ms=2.5, color="red", label="Actual (Validation)")

ax.legend(loc="upper left", fontsize=8, frameon=False)
_compact_show(fig1, width=7.0, height=3.8)

# ---------- Plot: Components ----------
st.subheader("Decomposition (Trend & Seasonality)")
fig2 = model.plot_components(forecast)
_compact_show(fig2, width=7.0, height=4.8)

# ---------- Preview & Download ----------
with st.expander("Preview Forecast Data"):
    tail = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).copy()
    st.dataframe(tail, use_container_width=True)

csv_buf = io.StringIO()
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(csv_buf, index=False)
st.download_button(
    "Download Forecast CSV",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name="airport_forecast.csv",
    mime="text/csv",
)

st.caption("Tip: Your CSV must have daily frequency with columns: ds (date), y (count).")
