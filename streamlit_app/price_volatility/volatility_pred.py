import json
import os
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
    guess = pd.infer_freq(index)
    if guess is None:
        deltas = index.to_series().diff().dt.days.dropna()
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
    return "M"


def align_and_merge(prices, exog, chem, freq):
    df = prices[prices["chemical"] == chem].copy()
    df = df.set_index("date")

    final_freq = freq or infer_frequency(df.index)

    # Resample prices
    if final_freq == "D":
        df = df.resample("D").interpolate()
    elif final_freq == "W":
        df = df.resample("W").mean()
    else:
        df = df.resample("M").mean(numeric_only=True)

    ex = exog.set_index("date")
    ex = ex.resample(final_freq).ffill()

    merged = df.join(ex, how="left")
    merged = merged.ffill().bfill()

    exog_cols = [c for c in merged.columns if c not in ["price", "chemical"]]
    return merged, exog_cols, final_freq


def add_features(df):
    df = df.copy()

    for w in [3, 6, 12]:
        df[f"sma_{w}"] = df["price"].rolling(w, min_periods=1).mean()

    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df["price"].shift(lag)

    df["month"] = df.index.month
    df["quarter"] = df.index.quarter

    return df.dropna()


def mape(y, yhat):
    denom = np.where(y == 0, 1e-9, np.abs(y))
    return np.mean(np.abs((y - yhat) / denom)) * 100


# ---------------------------
# Models
# ---------------------------
def sarimax_forecast(train, test, exog_cols):
    y = train["price"]
    exog_train = train[exog_cols]

    best_model = None
    best_aic = np.inf
    pdq = [(1, 1, 1), (2, 1, 1)]
    seasonal = [(0, 0, 0, 0), (1, 0, 1, 12)]

    for p, d, q in pdq:
        for P, D, Q, s in seasonal:
            try:
                m = SARIMAX(
                    y,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    exog=exog_train,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                if m.aic < best_aic:
                    best_model = m
                    best_aic = m.aic
            except:
                pass

    forecast = best_model.get_forecast(
        steps=len(test),
        exog=test[exog_cols]
    )
    return pd.Series(forecast.predicted_mean, index=test.index)


def tree_forecast(train, test):
    X_train = train.drop(columns=["price"])
    y_train = train["price"]
    X_test = test.drop(columns=["price"])

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    return pd.Series(model.predict(X_test), index=test.index)


# ---------------------------
# Pipeline
# ---------------------------
def run_for_chemical(chem, prices, exog, freq, horizon, output_dir=None):
    merged, exog_cols, f = align_and_merge(prices, exog, chem, freq)
    df = add_features(merged)

    train = df.iloc[:-horizon]
    test = df.iloc[-horizon:]

    sar = sarimax_forecast(train[["price"] + exog_cols], test[["price"] + exog_cols], exog_cols)
    tree = tree_forecast(train, test)

    y = test["price"]
    metrics = {
        "chemical": chem,
        "sarimax": {"mae": float(mean_absolute_error(y, sar)), "mape": float(mape(y, sar))},
        "tree": {"mae": float(mean_absolute_error(y, tree)), "mape": float(mape(y, tree))},
        "frequency_used": f
    }

    # Compact plot
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df.index, df["price"], label="Actual", linewidth=1)
    ax.plot(sar.index, sar.values, label="SARIMAX", linewidth=1)
    ax.plot(tree.index, tree.values, label="Tree", linewidth=1)
    ax.set_title(f"{chem} Forecast")
    ax.legend()
    fig.tight_layout()

    forecast_df = pd.DataFrame({
        "actual": y,
        "sarimax_forecast": sar,
        "tree_forecast": tree,
    })

    return {
        "metrics": metrics,
        "forecast_df": forecast_df,
        "fig": fig,
    }
