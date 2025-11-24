import streamlit as st
import pandas as pd
import os

from basket_segmentation.cuisine_segmentation import (
    read_csv,
    prepare_dataset,
    train_xgb,
    apply_rules
)

st.title("🍽️ Basket Segmentation (Demo)")

txns_file = st.file_uploader("Upload transactions CSV", type=["csv"])
sku_file = st.file_uploader("Upload SKU map CSV", type=["csv"])
dominance_threshold = st.slider("Dominance Threshold", 0.0, 1.0, 0.6)
override_conf = st.slider("Override Confidence", 0.0, 1.0, 0.55)

if txns_file and sku_file:
    st.info("Running model...")

    txns_df = pd.read_csv(txns_file)
    sku_df = pd.read_csv(sku_file)

    # Prepare dataset
    labeled_train, unlabeled, feats_all, feature_cols = prepare_dataset(
        txns=txns_df,
        sku_map=sku_df,
        labels=None,
        cutoff_date=None,
        dominance_threshold=dominance_threshold
    )

    # Train XGBoost model
    model, metrics = train_xgb(labeled_train, feature_cols)

    # Score all customers
    proba = model.predict_proba(feats_all[feature_cols].values)
    pred_labels = model.classes_[proba.argmax(axis=1)]
    pred_conf = proba.max(axis=1)
    preds_df = pd.DataFrame({
        "customer_id": feats_all["customer_id"],
        "pred_segment_model": pred_labels,
        "pred_proba_max": pred_conf
    })

    # Apply rules
    final_seg = apply_rules(
        feats=feats_all,
        preds_proba=preds_df,
        dominance_threshold=dominance_threshold,
        override_conf=override_conf
    )
    preds_df["final_segment"] = final_seg.values

    st.success("Model run completed!")
    st.subheader("Sample Predictions")
    st.dataframe(preds_df.head())

    # Segment summary
    summary = preds_df.groupby("final_segment").size().rename("customers").reset_index()
    st.subheader("Segment Summary")
    st.dataframe(summary)
