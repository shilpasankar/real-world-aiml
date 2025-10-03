
---

# price_volatility_ksa.py

```python
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
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

def infer_frequency(series: pd.Series) -> str:
    # Try to infer frequency from dates; fallback to monthly
    freq = pd.infer_freq(series.index)
    if freq is None:
        # Try to guess by median diff
        diffs = series.index.to_series().diff().dropna().dt.days
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
) -> Tuple[pd.DataFrame, List[str]]:
    df_c = prices[prices["chemical"] == chem].copy()
    if df_c.empty:
        raise ValueError(f"No rows for chemical: {chem}")
    df_c = df_c.set_index("date").asfreq(freq or None)  # None keeps original index
    if freq is None:
        f = infer_frequency(df_c["price"])
    else:
        f = freq
    # Resample to target freq with last observation carry-forward for price (or mean)
    if f == "D":
        df_c = df_c.resample("D").interpolate()
    elif f == "W":
        df_c = df_c.resample("W").mean()
    else:
        df_c = df_c.resample("M").mean()

    ex = exog.set_index("date")
    # Resample exogenous to same freq, forward-fill (macro often monthly/weekly)
    if f == "D":
        ex = ex.resample("D").ffill()
    elif f == "W":
        ex = ex.resample("W").ffill()
    else:
        ex = ex.resample("M").ffill()

    merged = df_c.join(ex, how="left")
    merged = merged.ffill().bfill()  # handle initial NaNs
    exog_cols = [c for c in merged.columns if c not in ["price", "chemical"]]
    return merged, exog_cols

def add_time_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    X = df.copy()
    X["month"] = X.index.month
    X["quarter"] = X.index.quarter
    # Simple moving averages and lags
    for w in [3, 6, 12]:
        col = f"sma_{w}"
        X[col] = X["price"].rolling(window=w, min_periods=1).mean()
    for lag in [1, 2, 3, 6, 12]:
        X[f"lag_{lag}"] = X["price"].shift(lag)
    # Drop initial rows with NA lags
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
    # Keep model simple/robust: auto-ish parameters by small search
    # We'll try a few (p,d,q) and (P,D,Q,s) combos and pick best AIC on train
    y = train_df["price"]
    exog_train = train_df[exog_cols] if exog_cols else None

    seasonal_period = 12  # monthly default; harmless for W/D (can be tuned)
    pdq_candidates = [(1,1,1), (2,1,1), (1,1,2)]
    seasonal_candidates = [(0,0,0,0), (1,0,1,seasonal_period), (1,1,1,seasonal_period)]

    best_model = None
    best_aic = np.inf
    for (p,d,q) in pdq_candidates:
        for (P,D,Q,s) in seasonal_candidates:
            try:
                mod = SARIMAX(
                    y,
                    order=(p,d,q),
                    seasonal_order=(P,D,Q,s),
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

def fit_tree_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, exog_cols: List[str]) -> pd.Series:
    # GradientBoosting over lag & exog features (no heavy deps)
    feat_cols = [c for c in train_df.columns if c != "price"]
    X_train, y_train = train_df[feat_cols].fillna(0), train_df["price"]
    X_test = test_df[feat_cols].fillna(0)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = pd.Series(model.predict(X_test), index=test_df.index, name="tree_baseline")
    return preds

# ---------------------------
# Pipeline
# ---------------------------

def evaluate(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mp = float(mape(y_true, y_pred))
    return mae, mp

def expanding_split(df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Final split: train up to last N, test last N
    train = df.iloc[:-horizon].copy()
    test = df.iloc[-horizon:].copy()
    if len(train) < 24:
        raise ValueError("Not enough history. Provide at least 24 periods before the forecast horizon.")
    return train, test

def plot_forecast(chem: str, df: pd.DataFrame, sarimax_pred: pd.Series, tree_pred: pd.Series, output_dir: str):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["price"], label="actual")
    # Align predictions with their test index
    plt.plot(sarimax_pred.index, sarimax_pred.values, label="sarimax forecast")
    plt.plot(tree_pred.index, tree_pred.values, label="tree baseline")
    plt.title(f"{chem} — Actuals vs Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    out_path = os.path.join(output_dir, f"plot_{chem}.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def run_for_chemical(
    chem: str,
    prices: pd.DataFrame,
    exog: pd.DataFrame,
    freq: Optional[str],
    horizon: int,
    output_dir: str
):
    merged, exog_cols = align_and_merge(prices, exog, chem, freq)
    # Keep a clean feature set for tree model (lags & SMA on price + exog)
    with_features = add_time_features(merged, freq or infer_frequency(merged["price"]))
    # Drop rows with NA from lagging
    with_features = with_features.dropna()

    # Prepare split
    train, test = expanding_split(with_features, horizon)
    # SARIMAX uses raw y + exog (not the lag features)
    sarimax_train = train[["price"] + exog_cols]
    sarimax_test = test[["price"] + exog_cols]

    # Fit models
    sarimax_pred = fit_sarimax(sarimax_train, sarimax_test, exog_cols)
    tree_pred = fit_tree_baseline(train.drop(columns=["chemical"]) if "chemical" in train.columns else train,
                                  test.drop(columns=["chemical"]) if "chemical" in test.columns else test,
                                  [c for c in with_features.columns if c not in ["price"]])

    # Evaluate
    y_true = test["price"]
    sarimax_mae, sarimax_mape = evaluate(y_true, sarimax_pred)
    tree_mae, tree_mape = evaluate(y_true, tree_pred)

    # Save metrics
    metrics = {
        "chemical": chem,
        "horizon": horizon,
        "sarimax": {"mae": sarimax_mae, "mape": sarimax_mape},
        "tree_baseline": {"mae": tree_mae, "mape": tree_mape},
    }
    with open(os.path.join(output_dir, f"metrics_{chem}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save forecasts
    out = pd.DataFrame({
        "date": y_true.index,
        "actual": y_true.values,
        "sarimax_forecast": sarimax_pred.values,
        "tree_baseline_forecast": tree_pred.values
    }).set_index("date")
    out.to_csv(os.path.join(output_dir, f"forecast_{chem}.csv"))

    # Plot
    plot_forecast(chem, with_features, sarimax_pred, tree_pred, output_dir)

    print(f"[{chem}] SARIMAX MAPE={sarimax_mape:.2f}% | Tree MAPE={tree_mape:.2f}%")

# ---------------------------
# CLI
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
            run_for_chemical(
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
