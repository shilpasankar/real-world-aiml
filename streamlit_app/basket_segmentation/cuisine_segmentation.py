# cuisine_segmentation.py
# Robust, Streamlit-safe version:
# - dtype-safe feature engineering (float32 for XGB)
# - stratified split fallback if data is small
# - fixed label space + dense class ids for XGB
# - safe rules for >4 cuisines (collapse extras to 'D')

from typing import Optional, Tuple, Dict
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


# -----------------------
# IO (optional helper)
# -----------------------
def read_csv(path: str, parse_dates=None) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)


# -----------------------
# Feature Engineering
# -----------------------
def build_rfm(txn: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    g = txn.groupby("customer_id", dropna=False)
    last_date = g["date"].max().rename("last_txn")
    recency_days = (asof - last_date).dt.days.rename("R")
    freq = g.size().rename("F")
    monetary = g["amount"].sum().rename("M")
    rfm = pd.concat([recency_days, freq, monetary], axis=1).reset_index()

    # ensure numeric
    rfm["R"] = pd.to_numeric(rfm["R"], errors="coerce").astype(float)
    rfm["F"] = pd.to_numeric(rfm["F"], errors="coerce").astype(float)
    rfm["M"] = pd.to_numeric(rfm["M"], errors="coerce").astype(float)
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
    """Auto-label a pref_segment by the dominant cuisine share.
       Any extra cuisines beyond four map to 'D'; low dominance -> 'U' (unlabeled)."""
    share_cols = [c for c in shares_df.columns if c.startswith("share_")]
    if not share_cols:
        raise ValueError("No share_* columns found. Check sku_map join and cuisine_tag values.")

    # stable order
    share_cols = sorted(share_cols)
    share_vals = shares_df[share_cols].to_numpy(copy=False)
    top_idx = np.argmax(share_vals, axis=1)
    top_val = share_vals[np.arange(len(shares_df)), top_idx]

    # map cuisine indices -> segments; extras collapse to 'D'
    unique_indices = sorted(np.unique(top_idx))
    label_space = list("ABCD")
    seg_lookup = {idx: (label_space[j] if j < len(label_space) else "D") for j, idx in enumerate(unique_indices)}

    seg = pd.Series([seg_lookup.get(i, "D") for i in top_idx], index=shares_df.index, name="pref_segment")
    seg = seg.where(top_val >= dominance_threshold, other="U")
    return seg


def _coerce_numeric_features(df: pd.DataFrame, exclude: list) -> pd.DataFrame:
    """Coerce non-numeric to numeric; cast numerics to float32 for XGB."""
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype("int8")
        elif not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    num_cols = [c for c in out.columns if c not in exclude and pd.api.types.is_numeric_dtype(out[c])]
    out[num_cols] = out[num_cols].fillna(0.0).astype("float32")
    return out


def prepare_dataset(
    txns: pd.DataFrame,
    sku_map: pd.DataFrame,
    labels: Optional[pd.DataFrame],
    cutoff_date: Optional[pd.Timestamp],
    dominance_threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """Return (labeled_train_df_with_split, unlabeled_df, feats_all, feature_cols)."""
    txns = txns.copy()
    txns["date"] = pd.to_datetime(txns["date"], errors="coerce")
    if "region" not in txns.columns:
        txns["region"] = "NA"

    asof = pd.to_datetime(cutoff_date) if cutoff_date is not None else txns["date"].max()

    rfm = build_rfm(txns, asof)
    shares = build_cuisine_shares(txns, sku_map)

    feats = rfm.merge(shares, on="customer_id", how="left").fillna(0.0)

    # ensure numeric features
    exclude_cols = ["customer_id", "pref_segment", "split", "last_txn", "region"]
    feats = _coerce_numeric_features(feats, exclude=exclude_cols)

    # labels
    if labels is not None:
        y = labels[["customer_id", "pref_segment"]].copy()
    else:
        auto = auto_labels_from_dominance(shares, dominance_threshold)
        y = pd.DataFrame({"customer_id": shares["customer_id"], "pref_segment": auto})

    merged = feats.merge(y, on="customer_id", how="left")

    # only A-D for modeling; keep others (U/NaN) as unlabeled
    model_mask = merged["pref_segment"].isin(list("ABCD"))
    labeled = merged[model_mask].copy()
    unlabeled = merged[~model_mask].copy()

    # stratified split where possible; else all train
    labeled["pref_segment"] = labeled["pref_segment"].astype(str)
    if len(labeled) >= 5 and labeled["pref_segment"].nunique() >= 2:
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            idx = np.arange(len(labeled))
            y_strat = labeled["pref_segment"].to_numpy()
            train_idx, valid_idx = next(sss.split(idx, y_strat))
            labeled.loc[:, "split"] = "train"
            labeled.loc[labeled.index[valid_idx], "split"] = "valid"
        except Exception:
            labeled.loc[:, "split"] = "train"
    else:
        labeled.loc[:, "split"] = "train"

    # numeric feature columns only
    feature_cols = [
        c for c in labeled.columns
        if c not in ["customer_id", "pref_segment", "split", "last_txn", "region"]
        and pd.api.types.is_numeric_dtype(labeled[c])
    ]

    return labeled, unlabeled, feats, feature_cols


# -----------------------
# Modeling (dense class ids + safe fallbacks)
# -----------------------
def train_xgb(train_df: pd.DataFrame, feature_cols: list) -> Tuple[object, Dict[str, float]]:
    """
    Trains XGB on dense class ids (0..k-1) for classes present in training.
    Stores mapping back to full labels for safe inference.
    Returns (model, metrics).
    """
    # Splits (fallback if train empty)
    train = train_df[train_df["split"] == "train"].copy()
    valid = train_df[train_df["split"] == "valid"].copy()
    if len(train) == 0 and len(train_df) > 0:
        train = train_df.copy()
        valid = train_df.iloc[0:0].copy()

    # Features → float32
    X_train = train[feature_cols].to_numpy(dtype=np.float32, copy=False)
    X_valid = valid[feature_cols].to_numpy(dtype=np.float32, copy=False)

    # Fixed full label space (keeps transforms consistent)
    full_label_space = np.array(list("ABCD"))
    le_full = LabelEncoder().fit(full_label_space)

    y_train_full = le_full.transform(train["pref_segment"].astype(str).to_numpy())
    y_valid_full = le_full.transform(valid["pref_segment"].astype(str).to_numpy()) if len(valid) else np.array([], dtype=np.int64)

    # Dense mapping for PRESENT training classes only (0..k-1)
    present_full_ids = np.unique(y_train_full)
    k = len(present_full_ids)

    if k == 0:
        raise ValueError("No trainable classes found after preprocessing (0 classes in train split).")

    dense_ids = {full_id: i for i, full_id in enumerate(present_full_ids)}
    y_train_dense = np.array([dense_ids[fid] for fid in y_train_full], dtype=np.int64)
    y_valid_dense = np.array([dense_ids[fid] for fid in y_valid_full if fid in dense_ids], dtype=np.int64) if len(y_valid_full) else np.array([], dtype=np.int64)

    # Class weights over dense ids
    if len(y_train_dense):
        present_dense = np.unique(y_train_dense)
        cw_vals = compute_class_weight(class_weight="balanced", classes=present_dense, y=y_train_dense)
        cw = {cls: w for cls, w in zip(present_dense, cw_vals)}
        w_train = np.array([cw[c] for c in y_train_dense], dtype=np.float32)
    else:
        w_train = None

    # Single-class fallback: deterministic "model"
    if k == 1:
        class _DummySingleClassModel:
            def __init__(self, present_full_ids_, le_full_):
                self._present_full_ids_ = np.array(present_full_ids_, dtype=int)   # e.g., array([2])
                self._present_labels_ = le_full_.inverse_transform(self._present_full_ids_)  # e.g., ['C']
                self._full_labels_ = le_full_.classes_

            def predict_proba(self, X):
                # Only one column, prob = 1
                return np.ones((X.shape[0], 1), dtype=float)

            def predict(self, X):
                return np.repeat(self._present_labels_[0], X.shape[0])

        model = _DummySingleClassModel(present_full_ids, le_full)
        metrics = {"val_accuracy": None, "val_macro_f1": None, "classes": le_full.classes_.tolist()}
        # Store mapping for consistency (already set in dummy __init__)
        return model, metrics

    # Multi-class XGB
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
    model.fit(X_train, y_train_dense, sample_weight=w_train)

    # Validation (only if any overlap)
    if len(y_valid_dense) and X_valid.shape[0]:
        y_hat_dense = model.predict(X_valid)
        acc = float(accuracy_score(y_valid_dense, y_hat_dense))
        f1 = float(f1_score(y_valid_dense, y_hat_dense, average="macro"))
    else:
        acc, f1 = None, None

    # Store mappings for inference:
    # _present_full_ids_: ids in FULL label space corresponding to predict_proba columns (order matters)
    # _present_labels_: human-readable labels aligned to predict_proba columns
    model._present_full_ids_ = present_full_ids
    model._present_labels_ = le_full.inverse_transform(present_full_ids)
    model._full_labels_ = le_full.classes_  # ['A','B','C','D']

    metrics = {"val_accuracy": acc, "val_macro_f1": f1, "classes": le_full.classes_.tolist()}
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
    """Override low-confidence model predictions with dominant cuisine where strong."""
    share_cols = [c for c in feats.columns if c.startswith("share_")]
    if not share_cols:
        # If no shares exist, nothing to override — return model labels as final
        return preds_proba["pred_segment_model"]

    share_cols = sorted(share_cols)  # stable order
    share_vals = feats[share_cols].to_numpy(copy=False)
    top_idx = np.argmax(share_vals, axis=1)
    top_val = share_vals[np.arange(len(feats)), top_idx]

    # Robust mapping: collapse any extra cuisines beyond first 4 to 'D'
    unique_indices = sorted(np.unique(top_idx))
    label_space = list("ABCD")
    seg_lookup = {idx: (label_space[j] if j < len(label_space) else "D") for j, idx in enumerate(unique_indices)}
    dominant_seg = pd.Series([seg_lookup.get(i, "D") for i in top_idx], index=feats.index)

    final = preds_proba["pred_segment_model"].copy()
    override_mask = (top_val >= dominance_threshold) & (preds_proba["pred_proba_max"] < override_conf)
    final.loc[override_mask] = dominant_seg.loc[override_mask]
    return final
