# streamlit_app/airport_recommender/forecast.py
# Streamlit Forecast Demo (Prophet) — uploads, validation, fixed-size annotated visuals

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

st.markdown("""
**What this page does**  
- Upload daily passenger data (`ds` = date, `y` = count), optionally a separate **Validation** CSV.  
- Train a **Prophet** model with configurable seasonality and changepoint prior.  
- Get **MAE / RMSE / MAPE** on Test or Validation, plus **interpretable visuals**.  
- On-chart annotations explain *what to look for* (bands, seasonality, confidence).  
""")

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("Controls")
    horizon = st.slider("Forecast horizon (days)", min_value=30, max_value=180, value=90, step=15)
    test_days = st.slider("Test window from TRAIN (days)", min_value=30, max_value=180, value=90, step=15)
    yearly = st.checkbox("Yearly seasonality", value=True)
    weekly = st.checkbox("Weekly seasonality", value=True)
    daily  = st.checkbox("Daily seasonality", value=False)
    cps = st.slider("Changepoint prior scale", 0.01, 0.5, 0.1, 0.01)
    st.caption("All plots use fixed figsize + tight layout for clean screenshots.")

# ---------- Uploaders ----------
train_file = st.file_uploader("Upload TRAIN time-series CSV (columns: ds, y)", type=["csv"])
val_file   = st.file_uploader("Optional: Upload VALIDATION CSV (columns: ds, y)", type=["csv"])

# ---------- Helpers ----------
def _compact_show(fig, width=6.8, height=3.8):
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

# Validation or internal test split
if val_file is not None:
    try:
        val_df = _read_series(val_file)
        st.success(f"VALIDATION loaded: {len(val_df):,} rows ({val_df['ds'].min().date()} → {val_df['ds'].max().date()})")
    except Exception as e:
        st.error(f"Failed to read VALIDATION CSV: {e}")
        st.stop()
    train_df = train_df_full.copy()
    test_df = None
else:
    if len(train_df_full) < test_days + 30:
        st.error("Not enough TRAIN rows for the selected TEST window. Reduce TEST window or add more data.")
        st.stop()
    train_df = train_df_full.iloc[:-test_days].copy()
    test_df  = train_df_full.iloc[-test_days:].copy()

# ---------- Fit & Forecast ----------
model = _fit_prophet(train_df, yearly=yearly, weekly=weekly, daily=daily, cps=cps)
future = model.make_future_dataframe(periods=horizon, freq="D")
forecast = model.predict(future)
fc_idx = forecast.set_index("ds")

# ---------- Evaluate ----------
metrics_label = "Validation" if val_file is not None else "Test"
if val_file is not None:
    common = val_df["ds"]
    if not set(common).issubset(set(fc_idx.index)):
        st.warning("Validation dates extend beyond forecast range; metrics computed on overlapping dates only.")
        common = common[common.isin(fc_idx.index)]
    yhat = fc_idx.loc[common, "yhat"].values
    ytrue = val_df.set_index("ds").loc[common, "y"].values
else:
    common = test_df["ds"]
    yhat = fc_idx.loc[common, "yhat"].values
    ytrue = test_df["y"].values

mae = mean_absolute_error(ytrue, yhat)
rmse = mean_squared_error(ytrue, yhat, squared=False)
mape = _mape(ytrue, yhat)

# ---------- Metrics Row ----------
c1, c2, c3 = st.columns(3)
c1.metric(f"MAE ({metrics_label})", f"{mae:,.0f}")
c2.metric(f"RMSE ({metrics_label})", f"{rmse:,.0f}")
c3.metric(f"MAPE ({metrics_label})", f"{mape*100:,.1f}%")

# ---------- Split Overview ----------
st.subheader("Train / Test / Forecast Overview")
rows = [("Train", train_df["ds"].min().date(), train_df["ds"].max().date(), len(train_df))]
if val_file is not None:
    rows.append(("Validation", val_df["ds"].min().date(), val_df["ds"].max().date(), len(val_df)))
else:
    rows.append(("Test", test_df["ds"].min().date(), test_df["ds"].max().date(), len(test_df)))
rows.append(("Forecast (Future)", forecast["ds"].iloc[-horizon].date(), forecast["ds"].iloc[-1].date(), horizon))
split_info = pd.DataFrame(rows, columns=["Split", "Start Date", "End Date", "Rows"])
st.dataframe(split_info, use_container_width=True, hide_index=True)

# ---------- Plot: Forecast vs Actuals (ANNOTATED) ----------
st.subheader("Forecast vs Actuals")
fig1 = model.plot(forecast, xlabel="Date", ylabel="Passengers")
fig1.axes[0].set_title("Passenger Count Forecast", fontsize=12)
ax = fig1.axes[0]

# Overlay actuals
ax.plot(train_df["ds"], train_df["y"], linestyle="none", marker="o", ms=2.5, color="black", label="Actual (Train)")
if val_file is not None:
    ax.plot(val_df["ds"], val_df["y"], linestyle="none", marker="o", ms=2.5, color="red", label="Actual (Validation)")
else:
    ax.plot(test_df["ds"], test_df["y"], linestyle="none", marker="o", ms=2.5, color="red", label="Actual (Test)")

# Annotation: CI bands & interpretation
ax.annotate(
    "Blue line = forecast (yhat)\nShaded band = confidence interval\nRed points = held-out actuals",
    xy=(0.01, 0.97), xycoords="axes fraction",
    va="top", ha="left",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", alpha=0.9)
)
ax.legend(loc="upper left", fontsize=8, frameon=False)
# Small note near end of forecast
ax.annotate("Forecast horizon", xy=(forecast["ds"].iloc[-horizon], forecast["yhat"].iloc[-horizon]),
            xytext=(15, 15), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)

def _compact_show(fig, width=6.8, height=3.8):
    fig.set_size_inches(width, height)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False, clear_figure=True)

_compact_show(fig1)

# ---------- Plot: Components (ANNOTATED) ----------
st.subheader("Decomposition (Trend & Seasonality)")
fig2 = model.plot_components(forecast)
# Add small on-figure notes
for axi in fig2.axes:
    axi.annotate(
        "Interpretation:\n• Trend shows long-term movement\n• Weekly/Yearly show periodic patterns",
        xy=(0.01, 0.97), xycoords="axes fraction",
        va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9)
    )
_compact_show(fig2, width=6.8, height=4.6)

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
