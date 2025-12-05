import json
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

plt.switch_backend("Agg")


# ---------------------------
# Utilities
# ---------------------------

def infer_frequency(index: pd.DatetimeIndex) -> str:
    """Infer D/W/M frequency from a DatetimeIndex."""
    guess = pd.infer_freq(index)
    if guess is None:
        deltas = index.to_series().diff().dt.days.dropna()
        if len(deltas) == 0:
            return "M"
        med = deltas.median()
        if med <= 2:
            return "D"
        if med <= 10:
            return "W"
        return "M"

    if guess.startswith("D"):
        return "D"
    if guess.startswith("W"):
        return "W"
    return "M"   # treat everything else as monthly-ish


def align_and_merge(
    prices: pd.DataFrame,
    exog: pd.DataFrame,
    chem: str,
    freq: Optional[str]
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Filter price series for one chemical, resample to chosen frequency,
    align exogenous data to same frequency, forward-fill.
    Returns merged df, exogenous column names, and final frequency.
    """
    df = prices[prices["chemical"] == chem].copy()
    if df.empty:
        raise ValueError(f"No rows found for chemical '{chem}'.")

    df = df.set_index("date")

    final_freq = freq or infer_frequency(df.index)

    # --- Resample price series (numeric-only) ---
    if final_freq == "D":
        df = df.resample("D").interpolate()
    elif final_freq == "W":
        # mean over week but only on numeric cols
        df = df.resample("W").mean(numeric_only=True)
    else:
        final_freq = "M"
        df = df.resample("M").mean(numeric_only=True)

    # Ensure price is numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # --- Exogenous: resample + ffill ---
    ex = exog.copy()
    ex = ex.set_index("date")
    ex = ex.resample(final_freq).ffill()

    merged = df.join(ex, how="left")
    merged = merged.ffill().bfill()

    # Exogenous columns = everything except 'price'
    exog_cols = [c for c in merged.columns if c != "price"]

    return merged, exog_cols, final_freq


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features & simple moving averages for tree model.
    """
    df = df.copy()

    # Moving averages
    for w in [3, 6, 12]:
        df[f"sma_{w}"] = df["price"].rolling(w, min_periods=1).mean()

    # Lags
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df["price"].shift(lag)

    df["month"] = df.index.month
    df["quarter"] = df.index.quarter

    # Drop rows where we have NA from lagging
    df = df.dropna()

    return df


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, 1e-9, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


# ---------------------------
# Models
# ---------------------------

def sarimax_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    exog_cols: List[str]
) -> pd.Series:
    """
    Fit SARIMAX on train (price + exog_cols) and forecast for test index.
    If all candidate models fail, fall back to naïve (last-value) forecast.
    """
    y = train["price"]

    if exog_cols:
        exog_train = train[exog_cols]
        exog_test = test[exog_cols]
    else:
        exog_train = None
        exog_test = None

    best_model = None
    best_aic = np.inf

    pdq_candidates = [(1, 1, 1), (2, 1, 1)]
    seasonal_candidates = [(0, 0, 0, 0), (1, 0, 1, 12)]

    for (p, d, q) in pdq_candidates:
        for (P, D, Q, s) in seasonal_candidates:
            try:
                model = SARIMAX(
                    y,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    exog=exog_train,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)

                if model.aic < best_aic:
                    best_aic = model.aic
                    best_model = model
            except Exception:
                # just skip bad combos
                continue

    if best_model is None:
        # --- Fallback: naïve forecast (repeat last observed value) ---
        last_val = float(y.iloc[-1])
        return pd.Series(last_val, index=test.index, name="sarimax_fallback")

    forecast_res = best_model.get_forecast(
        steps=len(test),
        exog=exog_test
    )
    return pd.Series(
        forecast_res.predicted_mean,
        index=test.index,
        name="sarimax"
    )


def tree_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Gradient Boosting on lag + exog features."""
    X_train = train.drop(columns=["price"])
    y_train = train["price"]

    X_test = test.drop(columns=["price"])

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return pd.Series(preds, index=test.index, name="tree")


# ---------------------------
# Main pipeline
# ---------------------------

def run_for_chemical(
    chem: str,
    prices: pd.DataFrame,
    exog: pd.DataFrame,
    freq: Optional[str],
    horizon: int,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    End-to-end pipeline for a single chemical:
    - Align data & exogenous drivers
    - Add features
    - Train/test split with final-horizon holdout
    - Train SARIMAX + tree models
    - Return metrics, forecast df, and a small matplotlib figure
    """
    merged, exog_cols, final_freq = align_and_merge(prices, exog, chem, freq)
    df = add_features(merged)

    if len(df) <= horizon + 10:
        raise ValueError("Not enough history relative to forecast horizon.")

    train = df.iloc[:-horizon]
    test = df.iloc[-horizon:]

    # SARIMAX uses price + exog only
    sar = sarimax_forecast(
        train[["price"] + exog_cols],
        test[["price"] + exog_cols],
        exog_cols,
    )

    # Tree uses full feature set
    tree = tree_forecast(train, test)

    y_true = test["price"]

    metrics = {
        "chemical": chem,
        "frequency_used": final_freq,
        "horizon": horizon,
        "sarimax": {
            "mae": float(mean_absolute_error(y_true, sar)),
            "mape": mape(y_true, sar),
        },
        "tree": {
            "mae": float(mean_absolute_error(y_true, tree)),
            "mape": mape(y_true, tree),
        },
    }

    forecast_df = pd.DataFrame(
        {
            "actual": y_true,
            "sarimax_forecast": sar,
            "tree_forecast": tree,
        }
    )

    # --- Small, pretty plot ---
    fig, ax = plt.subplots(figsize=(6, 3))  # not huge
    ax.plot(df.index, df["price"], label="Actual", linewidth=1)
    ax.plot(sar.index, sar.values, label="SARIMAX", linewidth=1)
    ax.plot(tree.index, tree.values, label="Tree", linewidth=1)
    ax.set_title(f"{chem} – Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8)
    fig.tight_layout()

    return {
        "metrics": metrics,
        "forecast_df": forecast_df,
        "fig": fig,
    }
