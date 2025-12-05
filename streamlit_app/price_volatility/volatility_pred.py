import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

warnings.simplefilter("ignore", ConvergenceWarning)
plt.switch_backend("Agg")  # allow plotting without a GUI


# ---------------------------
# Utilities
# ---------------------------

def read_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {"date", "chemical", "price"}.issubset(df.columns), \
        "chemical_prices.csv must have columns: date, chemical, price"
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["chemical", "date"])
    return df


def read_exog(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "date" in df.columns, "macro_factors.csv must have a date column"
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    exog_cols = [c for c in df.columns if c != "date"]
    if not exog_cols:
        raise ValueError("macro_factors.csv needs at least one exogenous column.")
    return df


def infer_frequency(index: pd.DatetimeIndex) -> str:
    """
    Try to infer frequency from dates; fallback to monthly.
    """
    freq = pd.infer_freq(index)
    if freq is None:
        # Try to guess by median diff
        diffs = index.to_series().diff().dropna().dt.days
        if len(diffs) > 0:
            md = diffs.median()
            if md <= 2:
                return "D"
            elif md <= 10:
                return "W"
        return "M"

    # Normalize alias
    if freq.startswith("M"):
        return "M"
    if freq.startswith("W"):
        return "W"
    if freq.startswith("D"):
        return "D"
    return "M"


def align_and_merge(
    prices: pd.DataFrame,
    exog: pd.DataFrame,
    chem: str,
    freq: Optional[str]
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Filter to a single chemical, resample price and exogenous to same frequency,
    forward-fill exogenous, interpolate/aggregate prices.
    Returns merged dataframe, exog column names, and final frequency.
    """
    df_c = prices[prices["chemical"] == chem].copy()
    if df_c.empty:
        raise ValueError(f"No rows for chemical: {chem}")

    df_c = df_c.set_index("date")

    # Determine target frequency
    final_freq = freq or infer_frequency(df_c.index)

    # Resample prices
    if final_freq == "D":
        df_c = df_c.resample("D").interpolate()
    elif final_freq == "W":
        df_c = df_c.resample("W").mean()
    else:
        final_freq = "M"
        df_c = df_c.resample("M").mean()

    ex = exog.set_index("date")
    # Resample exogenous to same freq, forward-fill (macro often monthly/weekly)
    if final_freq == "D":
        ex = ex.resample("D").ffill()
    elif final_freq == "W":
        ex = ex.resample("W").ffill()
    else:
        ex = ex.resample("M").ffill()

    merged = df_c.join(ex, how="left")
    merged = merged.ffill().bfill()  # handle initial NaNs
    exog_cols = [c for c in merged.columns if c not in ["price", "chemical"]]
    return merged, exog_cols, final_freq


def add_time_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    X = df.copy()
    X.index = pd.to_datetime(X.index)
    X["month"] = X.index.month
    X["quarter"] = X.index.quarter

    # Simple moving averages and lags
    for w in [3, 6, 12]:
        col = f"sma_{w}"
        X[col] = X["price"].rolling(window=w, min_periods=1).mean()
    for lag in [1, 2, 3, 6, 12]:
        X[f"lag_{lag}"] = X["price"].shift(lag)

    return X


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.where(y_true == 0, 1e-8, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


@dataclass
class ModelResult:
    mae: float
    mape: float
    preds: pd.Series


# ---------------------------
# Models
# ---------------------------

def fit_sarimax(train_df: pd.DataFrame, test_df: pd.DataFrame, exog_cols: List[str]) -> pd.Series:
    """
    Fit a SARIMAX on train data and forecast over test horizon.
    train_df / test_df must have columns: price + exog_cols.
    """
    y = train_df["price"]
    exog_train = train_df[exog_cols] if exog_cols else None

    seasonal_period = 12  # monthly default; harmless for W/D (can be tuned)
    pdq_candidates = [(1, 1, 1), (2, 1, 1), (1, 1, 2)]
    seasonal_candidates = [(0, 0, 0, 0), (1, 0, 1, seasonal_period), (1, 1, 1, seasonal_period)]

    best_model = None
    best_aic = np.inf

    for (p, d, q) in pdq_candidates:
        for (P, D, Q, s) in seasonal_candidates:
            try:
                mod = SARIMAX(
                    y,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    exog=exog_train,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
                if mod.aic < best_aic:
                    best_aic = mod.aic
                    best_model = mod
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError("SARIMAX failed to converge for all candidate orders.")

    # Forecast over test horizon with exog
    exog_test = test_df[exog_cols] if exog_cols else None
    forecast = best_model.get_forecast(steps=len(test_df), exog=exog_test)
    mean_forecast = pd.Series(forecast.predicted_mean, index=test_df.index, name="sarimax")
    return mean_forecast


def fit_tree_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    """
    GradientBoosting over lag & exog features (no heavy deps).
    Expects train_df / test_df with 'price' and feature columns.
    """
    feat_cols = [c for c in train_df.columns if c != "price"]
    X_train, y_train = train_df[feat_cols].fillna(0), train_df["price"]
    X_test = test_df[feat_cols].fillna(0)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = pd.Series(model.predict(X_test), index=test_df.index, name="tree_baseline")
    return preds


# ---------------------------
# Pipeline helpers
# ---------------------------

def evaluate(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mp = float(mape(y_true, y_pred))
    return mae, mp


def expanding_split(df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Final split: train up to last N, test last N.
    """
    if len(df) <= horizon:
        raise ValueError("Not enough observations to create a holdout set with given horizon.")
    train = df.iloc[:-horizon].copy()
    test
