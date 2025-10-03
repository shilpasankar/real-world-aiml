
---

# promo_sensitivity.py

```python
import argparse
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

plt.switch_backend("Agg")


# ---------------------------
# IO helpers
# ---------------------------

def read_csv(path, parse_dates=None):
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)


# ---------------------------
# Feature engineering
# ---------------------------

def build_features(txns: pd.DataFrame,
                   customers: pd.DataFrame,
                   returns: pd.DataFrame = None,
                   margins: pd.DataFrame = None) -> pd.DataFrame:
    df = txns.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ["promo_applied", "discount_rate"]:
        if col not in df.columns:
            df[col] = 0 if col == "promo_applied" else 0.0
    if "amount" not in df.columns:
        raise ValueError("transactions.csv must include 'amount'")

    # Aggregate core stats
    g = df.groupby("customer_id")
    orders = g.size().rename("orders")
    spend = g["amount"].sum().rename("spend")
    last_dt = g["date"].max().rename("last_txn")

    promo_g = df[df["promo_applied"] == 1].groupby("customer_id")
    promo_orders = promo_g.size().rename("promo_orders")
    promo_spend = promo_g["amount"].sum().rename("promo_spend")
    avg_discount = promo_g["discount_rate"].mean().rename("avg_discount")

    offpromo_g = df[df["promo_applied"] == 0].groupby("customer_id")
    off_orders = offpromo_g.size().rename("off_orders")
    off_spend = offpromo_g["amount"].sum().rename("off_spend")

    # Join features
    feats = pd.DataFrame(orders).join([spend, last_dt, promo_orders, promo_spend, avg_discount, off_orders, off_spend])
    feats = feats.fillna({"promo_orders": 0, "promo_spend": 0.0, "avg_discount": 0.0,
                          "off_orders": 0, "off_spend": 0.0})

    # Derived
    feats["redemption_rate"] = np.where(feats["orders"] > 0, feats["promo_orders"] / feats["orders"], 0.0)

    # Spend per day during promo vs off-promo (approximate rates)
    # Use counts to avoid divide-by-zero; +1 smoothing
    feats["promo_spend_rate"] = feats["promo_spend"] / (feats["promo_orders"] + 1)
    feats["off_spend_rate"] = feats["off_spend"] / (feats["off_orders"] + 1)
    feats["promo_lift"] = np.where(feats["off_spend_rate"] > 0,
                                   feats["promo_spend_rate"] / feats["off_spend_rate"], 1.0)

    # Basket delta on promo (proxy using spend/order)
    feats["basket_on_promo"] = np.where(feats["promo_orders"] > 0, feats["promo_spend"] / feats["promo_orders"], 0.0)
    feats["basket_off_promo"] = np.where(feats["off_orders"] > 0, feats["off_spend"] / feats["off_orders"], 0.0)
    feats["basket_delta"] = feats["basket_on_promo"] - feats["basket_off_promo"]

    # Recency (R)
    asof = df["date"].max()
    feats["days_since_last_txn"] = (asof - feats["last_txn"]).dt.days

    # Optional: returns
    if returns is not None:
        r = returns.copy()
        r["date"] = pd.to_datetime(r["date"])
        gr = r.groupby("customer_id")["return_amount"].sum().rename("return_amount")
        feats = feats.join(gr, how="left").fillna({"return_amount": 0.0})
        feats["return_rate"] = np.where(feats["spend"] > 0, feats["return_amount"] / feats["spend"], 0.0)
    else:
        feats["return_rate"] = 0.0

    # Optional: margins
    if margins is not None:
        m = margins.copy()
        m["date"] = pd.to_datetime(m["date"])
        gm = m.groupby("customer_id")["gross_margin"].mean().rename("avg_margin")
        feats = feats.join(gm, how="left").fillna({"avg_margin": 0.3})
    else:
        feats["avg_margin"] = 0.3

    # Minimal RFM-lite
    feats["F"] = feats["orders"]
    feats["M"] = feats["spend"]
    feats["R"] = feats["days_since_last_txn"]

    # Merge customer meta (optional; not used for clustering directly)
    if customers is not None and "customer_id" in customers.columns:
        meta = customers[["customer_id"]].drop_duplicates().set_index("customer_id")
        feats = meta.join(feats, how="left")

    feats = feats.fillna(0)
    feats.reset_index(inplace=True)
    return feats


# ---------------------------
# Clustering + Labeling
# ---------------------------

def cluster_and_label(feats: pd.DataFrame, k: int = 3):
    feature_cols = [
        "redemption_rate",
        "avg_discount",
        "promo_lift",
        "basket_delta",
        "R", "F", "M",
        "return_rate",
        "avg_margin"
    ]
    X = feats[feature_cols].astype(float).values
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(Xz)

    # Interpret cluster → label by average redemption_rate & promo_lift
    tmp = feats.copy()
    tmp["cluster"] = clusters
    prof = tmp.groupby("cluster")[["redemption_rate", "promo_lift", "avg_discount", "return_rate"]].mean()
    # Rank clusters by redemption_rate * promo_lift
    prof["score"] = prof["redemption_rate"] * prof["promo_lift"]
    order = prof["score"].rank(method="first").astype(int)
    # lowest score → Low, middle → Medium, highest → High
    label_map = {}
    for c in prof.index:
        if order.loc[c] == 1:
            label_map[c] = "Low"
        elif order.loc[c] == k:
            label_map[c] = "High"
        else:
            label_map[c] = "Medium"

    feats["cluster"] = clusters
    feats["cluster_label"] = feats["cluster"].map(label_map)
    return feats, kmeans, scaler, feature_cols, prof.reset_index().rename(columns={"score": "rank_score"})


# ---------------------------
# Supervised model (Decision Tree)
# ---------------------------

def train_tree(feats_labeled: pd.DataFrame, feature_cols):
    X = feats_labeled[feature_cols].astype(float).values
    y = feats_labeled["cluster_label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=50, random_state=42
    )
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_hat))
    return model, acc


# ---------------------------
# Rule-based overrides
# ---------------------------

def apply_rules(df: pd.DataFrame,
                margin_floor: float = 0.05,
                return_ceiling: float = 0.25,
                cooldown_days: int = 21):
    final = df.copy()
    final["tree_label"] = final.get("tree_label", final["cluster_label"])

    # Rule 1: Dominance to High
    high_mask = (final["redemption_rate"] >= 0.70) & (final["avg_discount"] >= 0.15)
    final.loc[high_mask, "final_label"] = "High"

    # Rule 2: Dominance to Low
    low_mask = (final["promo_lift"] <= 1.05) & (final["redemption_rate"] <= 0.20)
    final.loc[low_mask, "final_label"] = "Low"

    # Rule 3: Margin/Returns safeguard (downgrade one level)
    def downgrade(lbl):
        return {"High": "Medium", "Medium": "Low", "Low": "Low"}.get(lbl, lbl)

    guard_mask = (final["avg_margin"] <= margin_floor) | (final["return_rate"] >= return_ceiling)
    final.loc[guard_mask, "final_label"] = final.loc[guard_mask, "final_label"].fillna(final.loc[guard_mask, "tree_label"]).map(downgrade)

    # Rule 4: Cooldown — if recently redeemed a promo, keep their current (tree) label
    cooldown_mask = final["days_since_last_txn"] <= cooldown_days
    final.loc[cooldown_mask, "final_label"] = final.loc[cooldown_mask, "final_label"].fillna(final.loc[cooldown_mask, "tree_label"])

    # Fill any remaining nulls with tree label
    final["final_label"] = final["final_label"].fillna(final["tree_label"])
    return final


# ---------------------------
# Plots
# ---------------------------

def plot_cluster_scatter(feats: pd.DataFrame, outpath: str):
    # 2D proxy: redemption_rate vs promo_lift, colored by label
    colors = {"Low": 0, "Medium": 1, "High": 2}
    cvals = feats["cluster_label"].map(colors).fillna(1).values
    plt.figure(figsize=(7,5))
    plt.scatter(feats["redemption_rate"], feats["promo_lift"], c=cvals, alpha=0.6)
    plt.xlabel("Redemption rate")
    plt.ylabel("Promo lift")
    plt.title("Cluster scatter (behavioral view)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_importance(tree: DecisionTreeClassifier, feature_cols, outpath: str):
    imp = tree.feature_importances_
    idx = np.argsort(imp)[::-1]
    plt.figure(figsize=(8,6))
    plt.barh(range(len(idx)), imp[idx][::-1])
    plt.yticks(range(len(idx)), [feature_cols[i] for i in idx][::-1])
    plt.title("Decision Tree Feature Importance")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_label_breakdown(labels: pd.Series, outpath: str):
    counts = labels.value_counts().sort_index()
    plt.figure(figsize=(5,4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Final Label Breakdown")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Promotion Sensitivity Model (KMeans + Decision Tree + Rules)")
    ap.add_argument("--txns", required=True, help="transactions.csv")
    ap.add_argument("--customers", required=True, help="customers.csv")
    ap.add_argument("--returns", default=None, help="returns.csv (optional)")
    ap.add_argument("--margins", default=None, help="margins.csv (optional)")
    ap.add_argument("--clusters", type=int, default=3)
    ap.add_argument("--margin_floor", type=float, default=0.05)
    ap.add_argument("--return_ceiling", type=float, default=0.25)
    ap.add_argument("--cooldown_days", type=int, default=21)
    ap.add_argument("--output_dir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    txns = read_csv(args.txns, parse_dates=["date"])
    customers = read_csv(args.customers, parse_dates=["join_date"])
    returns = read_csv(args.returns, parse_dates=["date"]) if args.returns else None
    margins = read_csv(args.margins, parse_dates=["date"]) if args.margins else None

    feats = build_features(txns, customers, returns, margins)

    # Clustering & interpretation
    feats_labeled, kmeans, scaler, feature_cols, profiles = cluster_and_label(feats, k=args.clusters)

    # Train Decision Tree on cluster labels
    tree, acc = train_tree(feats_labeled, feature_cols)
    feats_labeled["tree_label"] = tree.predict(feats_labeled[feature_cols].astype(float).values)

    # Apply business rules
    final = apply_rules(
        feats_labeled,
        margin_floor=args.margin_floor,
        return_ceiling=args.return_ceiling,
        cooldown_days=args.cooldown_days
    )

    # Save artifacts
    feats.to_csv(os.path.join(args.output_dir, "customer_features.csv"), index=False)
    profiles.to_csv(os.path.join(args.output_dir, "cluster_profiles.csv"), index=False)

    preds = final[["customer_id", "cluster_label", "tree_label", "final_label"]].copy()
    preds.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    metrics = {
        "tree_accuracy_vs_clusters": acc,
        "clusters": int(args.clusters),
        "feature_cols": feature_cols
    }
    with open(os.path.join(args.output_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_cluster_scatter(feats_labeled, os.path.join(args.output_dir, "cluster_scatter.png"))
    plot_importance(tree, feature_cols, os.path.join(args.output_dir, "feature_importance.png"))
    plot_label_breakdown(final["final_label"], os.path.join(args.output_dir, "label_breakdown.png"))

    print(f"Saved outputs to {args.output_dir}")
    print(f"Decision tree accuracy vs cluster labels (proxy): {acc:.3f}")

if __name__ == "__main__":
    main()
