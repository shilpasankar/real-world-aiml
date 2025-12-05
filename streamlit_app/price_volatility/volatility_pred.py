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
    test = df.iloc[-horizon:].copy()
    if len(train) < 24:
        raise ValueError("Not enough history. Provide at least 24 periods before the forecast horizon.")
    return train, test


def build_forecast_plot(
    chem: str,
    df: pd.DataFrame,
    sarimax_pred: pd.Series,
    tree_pred: pd.Series
):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["price"], label="actual")
    ax.plot(sarimax_pred.index, sarimax_pred.values, label="sarimax forecast")
    ax.plot(tree_pred.index, tree_pred.values, label="tree baseline")
    ax.set_title(f"{chem} — Actuals vs Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    return fig


def run_for_chemical(
    chem: str,
    prices: pd.DataFrame,
    exog: pd.DataFrame,
    freq: Optional[str],
    horizon: int,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Core pipeline for a single chemical.
    Can be used from CLI (with output_dir) or Streamlit (output_dir=None).

    Returns a dict with:
        - metrics (dict)
        - forecast_df (DataFrame)
        - fig (matplotlib Figure)
    """
    merged, exog_cols, final_freq = align_and_merge(prices, exog, chem, freq)

    with_features = add_time_features(merged, final_freq)
    # Drop rows with NA from lagging
    with_features = with_features.dropna()

    # Train/test split
    train, test = expanding_split(with_features, horizon)

    # SARIMAX uses only price + exog (not lag features)
    sarimax_train = train[["price"] + exog_cols]
    sarimax_test = test[["price"] + exog_cols]

    sarimax_pred = fit_sarimax(sarimax_train, sarimax_test, exog_cols)

    # Tree baseline uses all features
    tree_pred = fit_tree_baseline(train, test)

    # Evaluate
    y_true = test["price"]
    sarimax_mae, sarimax_mape = evaluate(y_true, sarimax_pred)
    tree_mae, tree_mape = evaluate(y_true, tree_pred)

    metrics = {
        "chemical": chem,
        "horizon": horizon,
        "frequency": final_freq,
        "sarimax": {"mae": sarimax_mae, "mape": sarimax_mape},
        "tree_baseline": {"mae": tree_mae, "mape": tree_mape},
    }

    forecast_df = pd.DataFrame({
        "date": y_true.index,
        "actual": y_true.values,
        "sarimax_forecast": sarimax_pred.values,
        "tree_baseline_forecast": tree_pred.values
    }).set_index("date")

    fig = build_forecast_plot(chem, with_features, sarimax_pred, tree_pred)

    # Optional: write artifacts to disk (for CLI usage)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"metrics_{chem}.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        forecast_df.to_csv(os.path.join(output_dir, f"forecast_{chem}.csv"))

        plot_path = os.path.join(output_dir, f"plot_{chem}.png")
        fig.savefig(plot_path)

    print(f"[{chem}] SARIMAX MAPE={sarimax_mape:.2f}% | Tree MAPE={tree_mape:.2f}%")

    return {
        "metrics": metrics,
        "forecast_df": forecast_df,
        "fig": fig,
    }


# ---------------------------
# CLI entrypoint
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="KSA Chemicals Price Volatility Forecasting")
    parser.add_argument("--prices", required=True, help="Path to chemical_prices.csv")
    parser.add_argument("--exog", required=True, help="Path to macro_factors.csv")
    parser.add_argument("--chemicals", nargs="*", default=None, help="Chemicals to model; default=all")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon (periods)")
    parser.add_argument("--freq", type=str, default=None, help="Data frequency alias: D/W/M")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    prices = read_prices(args.prices)
    exog = read_exog(args.exog)

    chems = args.chemicals or sorted(prices["chemical"].unique())

    for chem in chems:
        try:
            _ = run_for_chemical(
                chem=chem,
                prices=prices,
                exog=exog,
                freq=args.freq,
                horizon=args.horizon,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"[{chem}] Skipped due to error: {e}")


if __name__ == "__main__":
    main()
