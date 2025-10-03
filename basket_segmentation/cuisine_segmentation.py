
---

# cuisine_segmentation.py

```python
import argparse
import json
import os
from datetime import datetime
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

plt.switch_backend("Agg")

# -----------------------
# IO
# -----------------------

def read_csv(path: str, parse_dates=None) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)

# -----------------------
# Feature Engineering
# -----------------------

def build_rfm(txn: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    g = txn.groupby("customer_id")
    last_date = g["date"].max().rename("last_txn")
    recency_days = (asof - last_date).dt.days.rename("R")
    freq = g.size().rename("F")
    monetary = g["amount"].sum().rename("M")
    rfm = pd.concat([recency_days, freq, monetary], axis=1).reset_index()
    # Clip outliers for stability
    rfm["R"] = rfm["R"].clip(lower=0, upper=rfm["R"].quantile(0.99))
    rfm["F"] = rfm["F"].clip(upper=rfm["F"].quantile(0.99))
    rfm["M"] = rfm["M"].clip(upper=rfm["M"].quantile(0.99))
    return rfm

def build_cuisine_shares(txn: pd.DataFrame, sku_map: pd.DataFrame) -> pd.DataFrame:
    df = txn.merge(sku_map, on="sku", how="left")
    df["cuisine_tag"] = df["cuisine_tag"].fillna("unknown")
    spend = df.groupby(["customer_id", "cuisine_tag"])["amount"].sum().reset_index()
    pivot = spend.pivot(index="customer_id", columns="cuisine_tag", values="amount").fillna(0.0)
    # Normalize to shares
    totals = pivot.sum(axis=1).replace(0.0, np.nan)
    shares = pivot.div(totals, axis=0).fillna(0.0)
    shares.columns = [f"share_{c}" for c in shares.columns]
    shares = shares.reset_index()
    # Diversity (Herfindahl-Hirschman Index)
    share_cols = [c for c in shares.columns if c.startswith("share_")]
    hhi = shares[share_cols].pow(2).sum(axis=1).rename("hhi")
    shares["diversity"] = 1 - hhi  # higher = more diverse
    return shares

def auto_labels_from_dominance(shares_df: pd.DataFrame, dominance_threshold: float = 0.6) -> pd.Series:
    share_cols = [c for c in shares_df.columns if c.startswith("share_")]
    share_subset = shares_df[share_cols].copy()
    # Choose top cuisine column and value
    top_idx = np.argmax(share_subset.values, axis=1)
    top_val = share_subset.values[np.arange(len(share_subset)), top_idx]
    # Map top index to segments A/B/C/D by stable order of columns
    # (You can later remap columns→friendly names externally)
    seg_map = {}
    unique_indices = sorted(np.unique(top_idx))
    alphabet = list("ABCD")
    for i, idx in enumerate(unique_indices):
        seg_map[idx] = alphabet[i % 4]
    seg = pd.Series([seg_map[i] for i in top_idx], index=shares_df.index, name="pref_segment")
    # If no dominant cuisine (below threshold), mark as Unknown
    seg = seg.where(top_val >= dominance_threshold, other="U")
    return seg

def prepare_dataset(
    txns: pd.DataFrame,
    sku_map: pd.DataFrame,
    labels: Optional[pd.DataFrame],
    cutoff_date: Optional[pd.Timestamp],
    dominance_threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    # Basic hygiene
    txns = txns.copy()
    txns["date"] = pd.to_datetime(txns["date"])
    if "region" not in txns.columns:
        txns["region"] = "NA"

    asof = pd.to_datetime(cutoff_date) if cutoff_date else txns["date"].max()
    rfm = build_rfm(txns, asof)
    shares = build_cuisine_shares(txns, sku_map)

    feats = rfm.merge(shares, on="customer_id", how="left").fillna(0.0)

    # Labels: use provided or auto-generate
    if labels is not None:
        y = labels[["customer_id", "pref_segment"]].copy()
    else:
        auto = auto_labels_from_dominance(shares, dominance_threshold)
        y = pd.DataFrame({"customer_id": shares["customer_id"], "pref_segment": auto})

    # Keep only labeled (exclude Unknown 'U' for training; still score later)
    labeled = feats.merge(y, on="customer_id", how="left")
    train_mask = labeled["pref_segment"].isin(list("ABCD"))
    labeled_train = labeled[train_mask].copy()
    unlabeled = labeled[~train_mask].copy()

    # Train/valid split by time: customers with last_txn <= cutoff-30d → train, else valid
    split_date = asof - pd.Timedelta(days=30)
    labeled_train["split"] = np.where(labeled_train["R"] > 30, "train", "valid")

    feature_cols = [c for c in labeled_train.columns if c not in ["customer_id", "pref_segment", "split", "last_txn"]]
    return labeled_train, unlabeled, feats, feature_cols

# -----------------------
# Modeling
# -----------------------

def train_xgb(train_df: pd.DataFrame, feature_cols: list) -> Tuple[XGBClassifier, Dict[str, float]]:
    train = train_df[train_df["split"] == "train"]
    valid = train_df[train_df["split"] == "valid"]

    X_train = train[feature_cols].values
    y_train = train["pref_segment"].values
    X_valid = valid[feature_cols].values
    y_valid = valid["pref_segment"].values

    classes = np.array(sorted(pd.unique(y_train)))
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw = {cls: w for cls, w in zip(classes, class_weights)}

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42
    )
    # Map class weights to sample weights
    w_train = np.array([cw[c] for c in y_train])
    model.fit(X_train, y_train, sample_weight=w_train)

    # Validation metrics
    y_hat = model.predict(X_valid)
    acc = float(accuracy_score(y_valid, y_hat)) if len(y_valid) else None
    f1 = float(f1_score(y_valid, y_hat, average="macro")) if len(y_valid) else None
    metrics = {"val_accuracy": acc, "val_macro_f1": f1, "classes": classes.tolist()}
    return model, metrics

# -----------------------
# Rules & Inference
# -----------------------

def apply_rules(
    feats: pd.DataFrame,
    preds_proba: pd.DataFrame,
    dominance_threshold: float,
    override_conf: float
) -> pd.Series:
    # Determine dominant cuisine in basket
    share_cols = [c for c in feats.columns if c.startswith("share_")]
    share_vals = feats[share_cols].values
    top_idx = np.argmax(share_vals, axis=1)
    top_val = share_vals[np.arange(len(feats)), top_idx]

    # Map top cuisine index -> A/B/C/D consistently
    unique_indices = sorted(np.unique(top_idx))
    seg_map = {}
    alphabet = list("ABCD")
    for i, idx in enumerate(unique_indices):
        seg_map[idx] = alphabet[i % 4]
    dominant_seg = pd.Series([seg_map[i] for i in top_idx], index=feats.index)

    model_seg = preds_proba["pred_segment_model"]
    model_conf = preds_proba["pred_proba_max"]

    final = model_seg.copy()

    # Rule 1: dominance override when model confidence is low
    override_mask = (top_val >= dominance_threshold) & (model_conf < override_conf)
    final[override_mask] = dominant_seg[override_mask]

    return final

# -----------------------
# Plots & Reporting
# -----------------------

def plot_confusion(y_true, y_pred, out_path):
    labels = sorted(pd.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_importance(model: XGBClassifier, feature_cols: list, out_path: str):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:25]
    plt.figure(figsize=(8,8))
    plt.barh(range(len(idx)), imp[idx][::-1])
    plt.yticks(range(len(idx)), [feature_cols[i] for i in idx][::-1])
    plt.title("Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Cuisine Preference Segmentation (RFM + XGBoost + Rules)")
    ap.add_argument("--txns", required=True, help="transactions.csv")
    ap.add_argument("--sku_map", required=True, help="sku_cuisine_map.csv")
    ap.add_argument("--labels", default=None, help="labels.csv (optional)")
    ap.add_argument("--cutoff_date", default=None, help="YYYY-MM-DD; for time split")
    ap.add_argument("--dominance_threshold", type=float, default=0.6)
    ap.add_argument("--override_confidence", type=float, default=0.55)
    ap.add_argument("--output_dir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    txns = read_csv(args.txns, parse_dates=["date"])
    sku_map = read_csv(args.sku_map)

    labels = None
    if args.labels:
        labels = read_csv(args.labels)

    labeled_train, unlabeled, feats_all, feature_cols = prepare_dataset(
        txns=txns,
        sku_map=sku_map,
        labels=labels,
        cutoff_date=args.cutoff_date,
        dominance_threshold=args.dominance_threshold
    )

    # Train model
    model, metrics = train_xgb(labeled_train, feature_cols)

    # Validation diagnostics
    valid = labeled_train[labeled_train["split"] == "valid"]
    if len(valid):
        y_true = valid["pref_segment"].values
        y_pred = model.predict(valid[feature_cols].values)
        plot_confusion(y_true, y_pred, os.path.join(args.output_dir, "confusion_matrix.png"))

    # Feature importance
    plot_importance(model, feature_cols, os.path.join(args.output_dir, "feature_importance.png"))

    # Score everyone (labeled + unlabeled)
    feats_all = feats_all.copy()
    proba = model.predict_proba(feats_all[feature_cols].values)
    pred_labels = model.classes_[np.argmax(proba, axis=1)]
    pred_conf = np.max(proba, axis=1)
    preds_df = pd.DataFrame({
        "customer_id": feats_all["customer_id"],
        "pred_segment_model": pred_labels,
        "pred_proba_max": pred_conf
    })

    # Apply rule overrides
    final_seg = apply_rules(
        feats=feats_all,
        preds_proba=preds_df,
        dominance_threshold=args.dominance_threshold,
        override_conf=args.override_confidence
    )
    preds_df["final_segment"] = final_seg.values

    # Save artifacts
    feats_all.to_csv(os.path.join(args.output_dir, "customer_features.csv"), index=False)
    preds_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
    with open(os.path.join(args.output_dir, "train_eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Segment summary
    seg_summary = preds_df.groupby("final_segment").size().rename("customers").reset_index()
    seg_summary.to_csv(os.path.join(args.output_dir, "segment_summary.csv"), index=False)

    print("Done. Artifacts written to:", args.output_dir)

if __name__ == "__main__":
    main()
