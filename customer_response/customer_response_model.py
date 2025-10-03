
---

# customer_response_model.py

```python
import argparse
import json
import os
from datetime import timedelta
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

plt.switch_backend("Agg")


# ---------------------------
# IO
# ---------------------------

def read_csv(path, parse_dates=None):
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)


# ---------------------------
# Feature engineering helpers
# ---------------------------

def build_rfm(txn: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    g = txn.groupby("customer_id")
    last_dt = g["date"].max().rename("last_txn")
    recency = (asof - last_dt).dt.days.rename("R")
    freq = g.size().rename("F")
    monetary = g["amount"].sum().rename("M")
    rfm = pd.concat([recency, freq, monetary], axis=1).reset_index()
    # Clip for stability
    rfm["R"] = rfm["R"].clip(lower=0, upper=rfm["R"].quantile(0.99))
    rfm["F"] = rfm["F"].clip(upper=rfm["F"].quantile(0.99))
    rfm["M"] = rfm["M"].clip(upper=rfm["M"].quantile(0.99))
    return rfm


def build_promo_behaviour(txn: pd.DataFrame) -> pd.DataFrame:
    df = txn.copy()
    df["promo_applied"] = df.get("promo_applied", 0)
    df["discount_rate"] = df.get("discount_rate", 0.0)

    g = df.groupby("customer_id")
    orders = g.size().rename("orders")
    spend = g["amount"].sum().rename("spend")

    pg = df[df["promo_applied"] == 1].groupby("customer_id")
    promo_orders = pg.size().rename("promo_orders")
    promo_spend = pg["amount"].sum().rename("promo_spend")
    avg_discount = pg["discount_rate"].mean().rename("avg_discount")

    og = df[df["promo_applied"] == 0].groupby("customer_id")
    off_orders = og.size().rename("off_orders")
    off_spend = og["amount"].sum().rename("off_spend")

    feats = pd.DataFrame(orders).join([spend, promo_orders, promo_spend, avg_discount, off_orders, off_spend])
    feats = feats.fillna({"promo_orders": 0, "promo_spend": 0.0, "avg_discount": 0.0,
                          "off_orders": 0, "off_spend": 0.0})

    feats["redemption_rate"] = np.where(feats["orders"] > 0, feats["promo_orders"] / feats["orders"], 0.0)
    feats["promo_spend_rate"] = feats["promo_spend"] / (feats["promo_orders"] + 1)
    feats["off_spend_rate"] = feats["off_spend"] / (feats["off_orders"] + 1)
    feats["promo_lift"] = np.where(feats["off_spend_rate"] > 0, feats["promo_spend_rate"] / feats["off_spend_rate"], 1.0)
    feats["basket_on_promo"] = np.where(feats["promo_orders"] > 0, feats["promo_spend"] / feats["promo_orders"], 0.0)
    feats["basket_off_promo"] = np.where(feats["off_orders"] > 0, feats["off_spend"] / feats["off_orders"], 0.0)
    feats["basket_delta"] = feats["basket_on_promo"] - feats["basket_off_promo"]
    feats.reset_index(inplace=True)
    return feats


def build_category_shares(txn: pd.DataFrame, topk: int = 8) -> pd.DataFrame:
    spend = txn.groupby(["customer_id", "category"])["amount"].sum().reset_index()
    totals = spend.groupby("customer_id")["amount"].sum().rename("total_spend")
    df = spend.merge(totals, on="customer_id")
    df["share"] = np.where(df["total_spend"] > 0, df["amount"] / df["total_spend"], 0.0)

    # Keep top-K categories by global spend
    topcats = spend.groupby("category")["amount"].sum().sort_values(ascending=False).head(topk).index.tolist()
    pivot = df[df["category"].isin(topcats)].pivot(index="customer_id", columns="category", values="share").fillna(0.0)
    pivot.columns = [f"share_cat_{c}" for c in pivot.columns]
    pivot = pivot.reset_index()
    return pivot, topcats


def attach_region_dummies(txn: pd.DataFrame, region_cols: List[str]) -> pd.DataFrame:
    # one-hot by last region seen
    last_region = txn.sort_values("date").groupby("customer_id")["region"].last().rename("region")
    oh = pd.get_dummies(last_region, prefix="region")
    # filter to requested regions if provided
    if region_cols:
        keep = [f"region_{r}" for r in region_cols]
        for col in keep:
            if col not in oh.columns:
                oh[col] = 0
        oh = oh[keep]
    oh = oh.reset_index()
    return oh


def merge_sensitivity(feats: pd.DataFrame, sens: Optional[pd.DataFrame]) -> pd.DataFrame:
    if sens is None:
        feats["promo_sensitivity"] = "Medium"
        return feats
    s = sens[["customer_id", "final_label"]].rename(columns={"final_label": "promo_sensitivity"})
    return feats.merge(s, on="customer_id", how="left").fillna({"promo_sensitivity": "Medium"})


# ---------------------------
# Label construction
# ---------------------------

def build_labels(campaigns: Optional[pd.DataFrame],
                 txn: pd.DataFrame,
                 window_days: int = 7) -> pd.DataFrame:
    if campaigns is not None and "responded" in campaigns.columns:
        labs = campaigns.groupby("customer_id")["responded"].max().rename("responded").reset_index()
        return labs

    # weak labels: purchase within window after any campaign send_date per customer
    if campaigns is None:
        # fabricate a pseudo send date: use last transaction date minus 14 days
        last_tx = txn.groupby("customer_id")["date"].max().rename("pseudo_send").reset_index()
        campaigns = last_tx
        campaigns["category"] = "NA"
        campaigns.rename(columns={"pseudo_send": "send_date"}, inplace=True)

    txn = txn.copy()
    campaigns = campaigns.copy()
    responded = []
    txn.sort_values("date", inplace=True)
    cmap = campaigns[["customer_id", "send_date"]].dropna()
    cmap["send_date"] = pd.to_datetime(cmap["send_date"])

    txn["date"] = pd.to_datetime(txn["date"])

    # for each customer, check any txn within window
    send_by_cust = cmap.groupby("customer_id")["send_date"].max()  # use latest to keep it fast
    for cust, send_dt in send_by_cust.items():
        end_dt = send_dt + timedelta(days=window_days)
        has_resp = ((txn["customer_id"] == cust) & (txn["date"] > send_dt) & (txn["date"] <= end_dt)).any()
        responded.append((cust, int(has_resp)))

    labs = pd.DataFrame(responded, columns=["customer_id", "responded"])
    return labs


# ---------------------------
# Modeling
# ---------------------------

def select_and_train_lr(X: pd.DataFrame, y: pd.Series, rfe_keep: int = 25):
    # Tree to rank features
    tree = GradientBoostingClassifier(random_state=42)
    tree.fit(X, y)

    base_lr = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=200
    )
    rfe = RFE(estimator=base_lr, n_features_to_select=min(rfe_keep, X.shape[1]), step=1)
    rfe.fit(X, y)

    kept_cols = X.columns[rfe.support_].tolist()

    lr = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=500
    )
    # Calibrate for probabilities
    clf = CalibratedClassifierCV(lr, method="isotonic", cv=3)
    clf.fit(X[kept_cols], y)
    return clf, kept_cols, tree


def evaluate(y_true, p_pred, threshold=0.5) -> dict:
    y_hat = (p_pred >= threshold).astype(int)
    roc = roc_auc_score(y_true, p_pred)
    pr = average_precision_score(y_true, p_pred)
    f1 = f1_score(y_true, y_hat)
    cm = confusion_matrix(y_true, y_hat).tolist()
    return {"roc_auc": float(roc), "pr_auc": float(pr), "f1": float(f1), "confusion_matrix": cm, "threshold": threshold}


def deciles(scores: pd.Series) -> pd.Series:
    # 1 (top) .. 10 (bottom)
    ranks = scores.rank(method="first", ascending=False)
    q = pd.qcut(ranks, 10, labels=False, duplicates="drop")
    return (q + 1).astype(int)


# ---------------------------
# Category interest (One-vs-Rest LR)
# ---------------------------

def train_category_models(df: pd.DataFrame, topcats: List[str]) -> pd.DataFrame:
    """Train a simple LR per category using category share + core features."""
    base_feats = [c for c in df.columns if c.startswith("share_cat_")] + ["R", "F", "M", "redemption_rate", "promo_lift"]
    results = {}
    X = df[base_feats].fillna(0.0)
    for cat in topcats:
        y = (df.get(f"share_cat_{cat}", 0.0) > 0.15).astype(int)  # crude interest threshold
        lr = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=300)
        clf = CalibratedClassifierCV(lr, method="isotonic", cv=3)
        clf.fit(X, y)
        proba = clf.predict_proba(X)[:, 1]
        results[cat] = proba
    return pd.DataFrame(results, index=df.index)


# ---------------------------
# Plots
# ---------------------------

def plot_roc_pr(y_true, p_pred, outdir):
    fpr, tpr, _ = roc_curve(y_true, p_pred)
    prec, rec, _ = precision_recall_curve(y_true, p_pred)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curve.png"))
    plt.close()

    plt.figure(figsize=(6,5))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pr_curve.png"))
    plt.close()

def plot_calibration(y_true, p_pred, outdir):
    prob_true, prob_pred = calibration_curve(y_true, p_pred, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6,5))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "calibration_curve.png"))
    plt.close()

def plot_confusion(y_true, y_hat, outdir):
    cm = confusion_matrix(y_true, y_hat)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"))
    plt.close()

def plot_deciles(scores, outdir):
    d = deciles(scores)
    counts = d.value_counts().sort_index()
    plt.figure(figsize=(6,4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Score Deciles (1=Top)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "deciles.png"))
    plt.close()


# ---------------------------
# Main pipeline
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Customer Response Model (RFE + Class weights + Logistic Regression)")
    ap.add_argument("--txns", required=True)
    ap.add_argument("--customers", required=True)
    ap.add_argument("--campaigns", default=None)
    ap.add_argument("--sensitivity", default=None)
    ap.add_argument("--region_cols", default="UAE,Qatar,KSA")
    ap.add_argument("--weak_label_window", type=int, default=7)
    ap.add_argument("--topk_categories", type=int, default=8)
    ap.add_argument("--rfe_features", type=int, default=25)
    ap.add_argument("--output_dir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    txns = read_csv(args.txns, parse_dates=["date"])
    customers = read_csv(args.customers, parse_dates=["join_date"])
    campaigns = read_csv(args.campaigns, parse_dates=["send_date"]) if args.campaigns else None
    sensitivity = read_csv(args.sensitivity) if args.sensitivity else None

    # Basic hygiene
    for col in ["customer_id", "date", "amount", "category", "region"]:
        if col not in txns.columns:
            raise ValueError(f"transactions.csv missing column: {col}")

    txns["date"] = pd.to_datetime(txns["date"])
    asof = txns["date"].max()

    # Features
    rfm = build_rfm(txns, asof)
    promo = build_promo_behaviour(txns)
    cat_shares, topcats = build_category_shares(txns, topk=args.topk_categories)

    regions = [c.strip() for c in (args.region_cols or "").split(",") if c.strip()]
    region_oh = attach_region_dummies(txns, regions)

    feats = rfm.merge(promo, on="customer_id", how="left") \
               .merge(cat_shares, on="customer_id", how="left") \
               .merge(region_oh, on="customer_id", how="left") \
               .fillna(0.0)

    feats = merge_sensitivity(feats, sensitivity)
    # One-hot the sensitivity label
    feats = pd.get_dummies(feats, columns=["promo_sensitivity"], prefix="sens")

    # Labels (supervised or weak)
    labels = build_labels(campaigns, txns, window_days=args.weak_label_window)  # customer_id, responded
    data = feats.merge(labels, on="customer_id", how="left").fillna({"responded": 0}).copy()
    data["responded"] = data["responded"].astype(int)

    # Time-aware split: last 30 days for validation
    split_date = asof - pd.Timedelta(days=30)
    last_tx = txns.groupby("customer_id")["date"].max().rename("last_txn").reset_index()
    data = data.merge(last_tx, on="customer_id", how="left")
    data["is_valid"] = (data["last_txn"] > split_date).astype(int)

    # Feature list
    drop_cols = ["customer_id", "last_txn", "responded", "orders", "off_orders", "promo_orders", "off_spend", "promo_spend", "spend", "basket_on_promo", "basket_off_promo"]
    feature_cols = [c for c in data.columns if c not in drop_cols and not c.startswith("Unnamed")]

    # Scale continuous features (robustly, but keep binaries as-is)
    X = data[feature_cols].copy()
    num_cols = [c for c in X.columns if X[c].nunique() > 2]
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    y = data["responded"].values
    train_idx = data["is_valid"] == 0
    valid_idx = data["is_valid"] == 1

    X_train, y_train = X.loc[train_idx], y[train_idx]
    X_valid, y_valid = X.loc[valid_idx], y[valid_idx]

    # Model (RFE + LR calibrated)
    clf, kept_cols, tree = select_and_train_lr(X_train, y_train, rfe_keep=args.rfe_features)

    # Validation
    p_valid = clf.predict_proba(X_valid[kept_cols])[:, 1]
    metrics = evaluate(y_valid, p_valid, threshold=0.5)

    # Plots
    plot_roc_pr(y_valid, p_valid, args.output_dir)
    plot_calibration(y_valid, p_valid, args.output_dir)
    yhat_v = (p_valid >= metrics["threshold"]).astype(int)
    plot_confusion(y_valid, yhat_v, args.output_dir)
    plot_deciles(pd.Series(p_valid, index=X_valid.index), args.output_dir)

    # Score all customers
    p_all = clf.predict_proba(X[kept_cols])[:, 1]
    scores = pd.DataFrame({
        "customer_id": data["customer_id"],
        "p_response": p_all
    })
    scores["score_decile"] = deciles(scores["p_response"])
    scores = scores.sort_values("p_response", ascending=False)

    # Category interest
    cat_probas = train_category_models(data, topcats)
    cat_probas["customer_id"] = data["customer_id"].values
    # Rank top 3 categories per customer
    cat_cols = [c for c in cat_probas.columns if c != "customer_id"]
    topn = []
    for _, row in cat_probas.iterrows():
        cust = row["customer_id"]
        probs = row[cat_cols].to_dict()
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        for cat, p in ranked:
            topn.append((cust, cat.replace("share_cat_", ""), float(p)))
    cat_out = pd.DataFrame(topn, columns=["customer_id", "category", "p_interest"])

    # Save artifacts
    data.to_csv(os.path.join(args.output_dir, "customer_features.csv"), index=False)
    scores.to_csv(os.path.join(args.output_dir, "scores_customers.csv"), index=False)
    cat_out.to_csv(os.path.join(args.output_dir, "category_interest.csv"), index=False)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({
            "kept_features": kept_cols,
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "f1": metrics["f1"],
            "threshold": metrics["threshold"]
        }, f, indent=2)

    print("Saved artifacts to", args.output_dir)
    print("Validation ROC-AUC:", round(metrics["roc_auc"], 4))

if __name__ == "__main__":
    main()
