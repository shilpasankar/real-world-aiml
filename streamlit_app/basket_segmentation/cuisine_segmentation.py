# cuisine_segmentation.py
# Clean, Streamlit-safe version

import os
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


# -----------------------
# IO Helpers (optional)
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

    # Ensure numeric dtype
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
    share_cols = [c for c in shares_df.columns if c.startswith("share_")]
    share_vals = shares_df[share_cols].values
    top_idx = np.argmax(share_vals, axis=1)
    top_val = share_vals[np.arange(len(shares_df)), top_idx]

    unique_indices = sorted(np.unique(top_idx))
    seg_map = {idx: ch for idx, ch in zip(unique_indices, list("ABCD"))}
    seg = pd.Series([seg_map[i] for i in top_idx], index=shares_df.index, name="pref_segment")
    seg = seg.where(top_val >= dominance_threshold, other="U")
    return seg


def _coerce_numeric_features(df: pd.DataFrame, exclude: list) -> pd.DataFrame:
    """Coerce non-numeric columns to numeric and cast numerics to float32 for XGB."""
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

    txns = txns.copy()
    txns["date"] = pd.to_datetime(txns["date"], errors="coerce")
    if "region" not in txns.columns:
        txns["region"] = "NA"

    asof = pd.to_datetime(cutoff_date) if cutoff_date is not None else txns["date"].max()

    rfm = build_rfm(txns, asof)
    shares = build_cuisine_shares(txns, sku_map)

    feats = rfm.merge(shares, on="customer_id", how="left").fillna(0.0)

    # Ensure numeric features (leave identifiers/non-features out)
    exclude_cols = ["customer_id", "pref_segment", "split", "last_txn", "region"]
    feats = _coerce_numeric_features(feats, exclude=exclude_cols)

    # Labels: use provided, else auto from dominance
    if labels is not None:
        y = labels[["customer_id", "pref_segment"]].copy()
    else:
        auto = auto_labels_from_dominance(shares, dominance_threshold)
        y = pd.DataFrame({"customer_id": shares["customer_id"], "pref_segment": auto})

    labeled = feats.merge(y, on="customer_id", how="left")
    train_mask = labeled["pref_segment"].isin(list("ABCD"))
    labeled_train = labeled[train_mask].copy()
    unlabeled = labeled[~train_mask].copy()

    # Simple split: recency > 30 days → train; else valid
    labeled_train["split"] = np.where(labeled_train["R"] > 30, "train", "valid")

    # Feature columns must be numeric
    feature_cols = [
        c for c in labeled_train.columns
        if c not in ["customer_id", "pref_segment", "split", "last_txn", "region"]
        and pd.api.types.is_numeric_dtype(labeled_train[c])
    ]
    return labeled_train, unlabeled, feats, feature_cols


# -----------------------
# Modeling
# -----------------------
def train_xgb(train_df: pd.DataFrame, feature_cols: list) -> Tuple[XGBClassifier, Dict[str, float]]:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.utils.class_weight import compute_class_weight
    from xgboost import XGBClassifier
    import numpy as np
    import pandas as pd

    # Split
    train = train_df[train_df["split"] == "train"]
    valid = train_df[train_df["split"] == "valid"]

    # X as float32
    X_train = train[feature_cols].to_numpy(dtype=np.float32, copy=False)
    X_valid = valid[feature_cols].to_numpy(dtype=np.float32, copy=False)

    # ---- LABEL ENCODING FIX ----
    # Fit encoder on the *full* intended label space, not just what's in the train split
    full_label_space = np.array(list("ABCD"))   # your intended segments
    le = LabelEncoder()
    le.fit(full_label_space)

    y_train_raw = train["pref_segment"].astype(str).to_numpy()
    y_valid_raw = valid["pref_segment"].astype(str).to_numpy()

    # Transform with the fixed encoder (no "previously unseen labels" now)
    y_train = le.transform(y_train_raw)
    y_valid = le.transform(y_valid_raw) if len(y_valid_raw) else np.array([], dtype=np.int64)

    # Class weights computed only on present classes (safe if some classes missing in train)
    if len(y_train):
        present_classes = np.unique(y_train)
        cw_vals = compute_class_weight(class_weight="balanced", classes=present_classes, y=y_train)
        cw = {cls: w for cls, w in zip(present_classes, cw_vals)}
        w_train = np.array([cw[c] for c in y_train], dtype=np.float32)
    else:
        w_train = None

    # Model
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
    model.fit(X_train, y_train, sample_weight=w_train)

    # Validation metrics (handle empty valid)
    if len(y_valid):
        y_hat = model.predict(X_valid)
        acc = float(accuracy_score(y_valid, y_hat))
        f1 = float(f1_score(y_valid, y_hat, average="macro"))
    else:
        acc, f1 = None, None

    # Keep original class names for downstream mapping.
    # NOTE: XGB's predict_proba columns will correspond to the classes *present in y_train*,
    # so we'll store both: the global label space and the actually-trained class indices.
    model.classes_ = le.classes_                     # ['A','B','C','D']
    model._label_encoder = le
    model._trained_class_indices_ = np.unique(y_train)  # e.g., array([0,1]) if only A,B were in train

    metrics = {"val_accuracy": acc, "val_macro_f1": f1, "classes": le.classes_.tolist()}
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
