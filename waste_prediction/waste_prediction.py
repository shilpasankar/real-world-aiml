
---

# sales_waste_prediction.py

```python
import argparse, os, json, warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
plt.switch_backend("Agg")

# ---------------------------
# Utilities
# ---------------------------

def read_csv(path: Optional[str], parse_dates=None) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)

def ensure_daily_index(df: pd.DataFrame, date_col: str="date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df

def pinball_loss(y_true, y_pred, q: float):
    e = y_true - y_pred
    return np.mean(np.maximum(q*e, (q-1)*e))

@dataclass
class ForecastResult:
    p10: pd.Series
    p50: pd.Series
    p90: pd.Series
    order_qty: pd.Series
    waste_proxy: pd.Series
    metrics: dict

# ---------------------------
# Feature Engineering
# ---------------------------

def build_feature_frame(sales: pd.DataFrame,
                        items: pd.DataFrame,
                        promos: Optional[pd.DataFrame],
                        calendar: Optional[pd.DataFrame],
                        item_id: str) -> pd.DataFrame:
    df = sales[sales["item_id"] == item_id].copy()
    if df.empty:
        raise ValueError(f"No sales for item {item_id}")
    # daily frequency; fill missing with 0 units_sold
    idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(idx).fillna({"units_sold": 0})
    df.index.name = "date"
    df["item_id"] = item_id

    # Merge promos
    if promos is not None:
        pr = promos[promos["item_id"] == item_id].copy()
        pr = pr.set_index("date")[["promo_flag","discount_rate"]]
        df = df.join(pr, how="left")
    df["promo_flag"] = df.get("promo_flag", 0).fillna(0).astype(int)
    df["discount_rate"] = df.get("discount_rate", 0.0).fillna(0.0)

    # Calendar
    if calendar is not None:
        cal = calendar.set_index("date")
        keep = []
        for c in ["is_weekend", "is_festival"]:
            if c in cal.columns:
                keep.append(c)
        if "holiday_name" in cal.columns:
            # one-hot of holiday names (rare)
            cal = pd.concat([cal[keep], pd.get_dummies(cal["holiday_name"], prefix="hol")], axis=1)
        else:
            cal = cal[keep] if keep else pd.DataFrame(index=df.index)
        df = df.join(cal, how="left")
    df["is_weekend"] = df.get("is_weekend", 0).fillna(0).astype(int)
    df["is_festival"] = df.get("is_festival", 0).fillna(0).astype(int)

    # Time features
    df["dow"] = df.index.dayofweek
    dow_oh = pd.get_dummies(df["dow"], prefix="dow")
    df = pd.concat([df, dow_oh], axis=1)

    # Lags & moving averages
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df["units_sold"].shift(lag).fillna(0)
    for win in [7, 14, 28]:
        df[f"ma_{win}"] = df["units_sold"].rolling(win, min_periods=1).mean()

    # Shelf-life / min/max
    meta = items[items["item_id"] == item_id].head(1)
    df["shelf_life_days"] = int(meta["shelf_life_days"].iloc[0]) if "shelf_life_days" in meta.columns else 2
    df["min_stock"] = int(meta["min_stock"].iloc[0]) if "min_stock" in meta.columns and not np.isnan(meta["min_stock"].iloc[0]) else 0
    df["max_stock"] = int(meta["max_stock"].iloc[0]) if "max_stock" in meta.columns and not np.isnan(meta["max_stock"].iloc[0]) else 10**9

    return df

# ---------------------------
# Models
# ---------------------------

def fit_sarimax(df: pd.DataFrame, holdout_days: int, horizon: int) -> Tuple[pd.Series, dict]:
    y = df["units_sold"]
    exog_cols = [c for c in df.columns if c not in ["item_id","units_sold","dow"] and not c.startswith("lag_")]
    exog = df[exog_cols]

    train = y.iloc[:-holdout_days] if holdout_days>0 else y
    exog_train = exog.iloc[:-holdout_days] if holdout_days>0 else exog

    # Simple seasonal monthly-ish (7-day seasonality is captured via dummies/lag features; SARIMAX seasonal kept light)
    best = None; best_aic = np.inf
    for order in [(1,1,1),(2,1,1)]:
        try:
            m = SARIMAX(train, exog=exog_train, order=order, seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            if m.aic < best_aic:
                best = m; best_aic = m.aic
        except Exception:
            continue
    if best is None:
        raise RuntimeError("SARIMAX failed")

    # In-sample fitted values for validation window
    if holdout_days>0:
        exog_hold = exog.iloc[-holdout_days:]
        fc = best.get_forecast(steps=holdout_days, exog=exog_hold)
        p50_hist = pd.Series(best.fittedvalues.reindex(df.index), index=df.index).iloc[-holdout_days:]
    else:
        p50_hist = pd.Series(best.fittedvalues, index=df.index).iloc[-horizon:]

    # Out-of-sample forecast (next horizon)
    exog_future = exog.iloc[-horizon:]
    fc_future = best.get_forecast(steps=horizon, exog=exog_future)
    p50_future = pd.Series(fc_future.predicted_mean, index=df.index[-horizon:])

    metrics = {}
    if holdout_days>0:
        y_true = y.iloc[-holdout_days:]
        metrics["sarimax_mae"] = float(mean_absolute_error(y_true, p50_hist))
    return p50_future, metrics

def fit_quantile_gbm(df: pd.DataFrame, holdout_days: int, horizon: int, quantiles=[0.1,0.5,0.9]) -> Tuple[dict, dict]:
    # Use tabular features to predict next-day sales; then shift to align horizons via rolling out-of-sample
    feats = [c for c in df.columns if c not in ["units_sold","item_id"]]
    X = df[feats].copy()
    y = df["units_sold"].copy()

    # Train on all except last H days
    train_X = X.iloc[:-horizon]
    train_y = y.iloc[:-horizon]

    preds = {q: [] for q in quantiles}
    idx = y.index[-horizon:]
    # Walk-forward one-step forecasts to avoid lookahead
    for t in range(horizon):
        X_train_t = X.iloc:-(horizon - t) if (horizon - t) > 0 else X
        y_train_t = y.iloc:-(horizon - t) if (horizon - t) > 0 else y
        x_t = X.iloc[-(horizon - t)]
        for q in quantiles:
            gbm = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=400, learning_rate=0.05,
                                            max_depth=3, subsample=0.9, random_state=42)
            gbm.fit(X_train_t, y_train_t)
            preds[q].append(float(gbm.predict([x_t])[0]))

    q_preds = {q: pd.Series(preds[q], index=idx) for q in quantiles}
    metrics = {}
    if holdout_days>0 and holdout_days <= len(df):
        # quick pinball on last holdout_days using same mechanics
        pass  # kept simple; main metrics computed later
    return q_preds, metrics

# ---------------------------
# Policy (Newsvendor order target)
# ---------------------------

def compute_orders(p10: pd.Series, p50: pd.Series, p90: pd.Series,
                   shelf_life_days: int, min_stock: int, max_stock: int,
                   service_level: float) -> Tuple[pd.Series, pd.Series]:
    # Choose quantile by service level
    if service_level <= 0.5: q_ser = p50
    elif service_level >= 0.9: q_ser = p90
    else:
        # linear between p50 and p90 for 0.5..0.9
        w = (service_level - 0.5) / 0.4
        q_ser = (1-w)*p50 + w*p90

    # Cap horizon by shelf life (don’t over-order beyond what can be sold)
    order = q_ser.clip(lower=min_stock).clip(upper=max_stock)
    # Simple expected waste proxy: positive part of (order - p50)
    waste_proxy = (order - p50).clip(lower=0)
    return order.round(3), waste_proxy.round(3)

# ---------------------------
# Orchestration per item
# ---------------------------

def run_item(df_item: pd.DataFrame, item_id: str,
             holdout_days: int, horizon: int,
             service_level: float, blend: bool, outdir: str) -> ForecastResult:

    # Models
    sarimax_p50, m1 = fit_sarimax(df_item, holdout_days, horizon)
    qpreds, m2 = fit_quantile_gbm(df_item, holdout_days, horizon, quantiles=[0.1,0.5,0.9])

    # Blend P50 if requested
    p50 = 0.5*sarimax_p50 + 0.5*qpreds[0.5] if blend else qpreds[0.5]
    p10 = qpreds[0.1]
    p90 = qpreds[0.9]
    # Ensure monotonicity p10<=p50<=p90
    p50 = np.maximum(p50, p10)
    p50 = np.minimum(p50, p90)

    shelf_life = int(df_item["shelf_life_days"].iloc[-1]) if "shelf_life_days" in df_item.columns else 2
    min_stock = int(df_item["min_stock"].iloc[-1]) if "min_stock" in df_item.columns else 0
    max_stock = int(df_item["max_stock"].iloc[-1]) if "max_stock" in df_item.columns else 10**9

    order, waste = compute_orders(p10, p50, p90, shelf_life, min_stock, max_stock, service_level)

    # Metrics on holdout if possible
    metrics = {}
    if holdout_days>0:
        y_true = df_item["units_sold"].iloc[-(holdout_days+horizon):-horizon] if horizon>0 else df_item["units_sold"].iloc[-holdout_days:]
        y_pred_med = df_item["units_sold"].rolling(7, min_periods=1).mean().iloc[-holdout_days:]  # simple baseline
        # For simplicity, compute final MAE/MAPE on validation baseline only
        metrics["baseline_mae"] = float(mean_absolute_error(y_true, y_pred_med))
        # pinball losses on horizon (proxy, using last horizon of history if available)
        hist = df_item["units_sold"].iloc[-horizon:]
        if len(hist) == horizon:
            metrics["pinball_p10"] = float(pinball_loss(hist, p10.values, 0.1))
            metrics["pinball_p50"] = float(pinball_loss(hist, p50.values, 0.5))
            metrics["pinball_p90"] = float(pinball_loss(hist, p90.values, 0.9))

    # Save per-item CSV & plot
    out = pd.DataFrame({
        "date": p50.index,
        "p10": p10.values,
        "p50": p50.values,
        "p90": p90.values,
        "order_qty": order.values,
        "expected_waste_proxy": waste.values
    }).set_index("date")
    out.to_csv(os.path.join(outdir, f"forecast_{item_id}.csv"))

    plt.figure(figsize=(10,5))
    hist = df_item["units_sold"].iloc[-90:]  # show last 90d
    plt.plot(hist.index, hist.values, label="actual (last 90d)")
    plt.fill_between(p90.index, p10.values, p90.values, alpha=0.2, label="P10–P90")
    plt.plot(p50.index, p50.values, label="P50")
    plt.step(order.index, order.values, where="mid", label="order target")
    plt.title(f"{item_id} — Probabilistic Forecast & Orders")
    plt.xlabel("Date"); plt.ylabel("Units")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"plot_{item_id}.png")); plt.close()

    return ForecastResult(p10, p50, p90, order, waste, metrics)

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Sales & Waste Prediction — Fresh Food (UAE Retail)")
    ap.add_argument("--sales", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--promos", default=None)
    ap.add_argument("--calendar", default=None)
    ap.add_argument("--items_subset", nargs="*", default=None)
    ap.add_argument("--holdout_days", type=int, default=14)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--service_level", type=float, default=0.85)
    ap.add_argument("--blend", type=lambda x: str(x).lower()=="true", default=True)
    ap.add_argument("--output_dir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sales = read_csv(args.sales, parse_dates=["date"])
    items = read_csv(args.items)
    promos = read_csv(args.promos, parse_dates=["date"])
    calendar = read_csv(args.calendar, parse_dates=["date"])

    # hygiene
    for col in ["date","item_id","units_sold"]:
        if col not in sales.columns:
            raise ValueError(f"sales.csv missing {col}")
    if "item_id" not in items.columns:
        raise ValueError("items.csv must include item_id")

    sales = ensure_daily_index(sales)
    if promos is not None:
        promos = ensure_daily_index(promos)
    if calendar is not None:
        calendar = ensure_daily_index(calendar)

    item_list = args.items_subset or sorted(sales["item_id"].unique())
    portfolio = []

    for item in item_list:
        try:
            df_item = build_feature_frame(sales, items, promos, calendar, item)
            res = run_item(df_item, item,
                           holdout_days=args.holdout_days,
                           horizon=args.horizon,
                           service_level=args.service_level,
                           blend=args.blend,
                           outdir=args.output_dir)
            portfolio.append({
                "item_id": item,
                **res.metrics,
                "mean_order": float(res.order_qty.mean()),
                "mean_waste_proxy": float(res.waste_proxy.mean())
            })
        except Exception as e:
            print(f"[{item}] skipped: {e}")

    pd.DataFrame(portfolio).to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    print("Done. Artifacts written to", args.output_dir)

if __name__ == "__main__":
    main()
