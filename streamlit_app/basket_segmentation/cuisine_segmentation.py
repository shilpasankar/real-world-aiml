# cuisine_segmentation.py
# Cleaned version for Streamlit Cloud

import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

# -----------------------
# IO Helpers
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
    # Clip outliers
    rfm["R"] = rfm["R"].clip(lower=0, upper=rfm["R"].quantile(0.99))
    rfm["F"] = rfm["F"].clip(upper=rfm["F"].quantile(0.99))
    rfm["M"] = rfm["M"].clip(upper=rfm["M"].quantile(0.99))
    return rfm

def build_cuisine_shares(txn: pd.DataFrame, sku_map: pd.DataFrame) -> pd.DataFrame:
    df = txn.merge(sku_map, on="sku", how="left")
    df["cuisine_tag"] = df["cuisine_tag"].fillna("unknown")
    spend = df.groupby(["customer_id", "cuisine_tag"])["amount"].sum().reset_index()
    pivot = spend.pivot(index="customer_id", columns="cuisine_tag", values="amount").fillna(0.0)
    totals = pivot.sum(axis=1).replace(0.0, np.nan)
    shares = pivot.div(totals, axis=0).fillna(0.0)
    shares.columns = [f"share_{c}" for c in shares.columns]
    shares = shares.reset_index()
    return shares

def auto_labels_from_dominance(shares_df: pd.DataFrame, dominance_threshold: float = 0.6) -> pd.Series:
    share_cols = [c for c in shares_df.columns if c.startswith("share_")]
    share_vals = shares_df[share_cols].values
    top_idx = np.argmax(share_vals, axis=1)
    top_val = share_vals[np.arange(len(shares_df)), top_idx]
    unique_indices = sorted(np.unique(top_idx))
    seg_map = {idx: ch for idx, ch in zip(unique_indices, list("ABCD"))}
    seg = pd.Series([seg_map[i] for i in top_idx], index=shares_df.index, name="pref_segment")
    seg = seg.where(top_val >= dominance_threshold, other="U")
    return seg

def prepare_dataset(
    txns: pd.DataFrame,
    sku_map: pd.DataFrame,
    labels: Optional[pd.DataFrame],
    cutoff_date: Optional[pd.Timestamp],
    dominance_threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:

    txns = txns.copy()
    txns["date"] = pd.to_datetime(txns["date"])
    if "region" not in txns.columns:
        txns["region"] = "NA"

    asof = pd.to_datetime(cutoff_date) if cutoff_date else txns["date"].max()
    rfm = build_rfm(txns, asof)
    shares = build_cuisine_shares(txns, sku_map)

    feats = rfm.merge(shares, on="customer_id", how="left").fillna(0.0)

    if labels is not None:
        y = labels[["customer_id", "pref_segment"]].copy()
    else:
        auto = auto_labels_from_dominance(shares, dominance_threshold)
        y = pd.DataFrame({"customer_id": shares["customer_id"], "pref_segment": auto})

    labeled = feats.merge(y, on="customer_id", how="left")
    train_mask = labeled["pref_segment"].isin(list("ABCD"))
    labeled_train = labeled[train_mask].copy()
    unlabeled = labeled[~train_mask].copy()

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
    w_train = np.array([cw[c] for c in y_train])
    model.fit(X_train, y_train, sample_weight=w_train)

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

    share_cols = [c for c in feats.columns if c.startswith("share_")]
    share_vals = feats[share_cols].values
    top_idx = np.argmax(share_vals, axis=1)
    top_val = share_vals[np.arange(len(feats)), top_idx]

    unique_indices = sorted(np.unique(top_idx))
    seg_map = {idx: ch for idx, ch in zip(unique_indices, list("ABCD"))}
    dominant_seg = pd.Series([seg_map[i] for i in top_idx], index=feats.index)

    final = preds_proba["pred_segment_model"].copy()
    override_mask = (top_val >= dominance_threshold) & (preds_proba["pred_proba_max"] < override_conf)
    final[override_mask] = dominant_seg[override_mask]

    return final
