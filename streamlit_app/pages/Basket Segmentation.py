import streamlit as st
import pandas as pd
import numpy as np

from basket_segmentation.cuisine_segmentation import (
    prepare_dataset,
    train_xgb,
    apply_rules,
)

st.set_page_config(page_title="Basket Segmentation", layout="wide")
st.title("🍽️ Basket Segmentation (Demo)")

st.markdown("""
Upload your **transactions CSV** and **SKU map CSV** to run the Basket Segmentation model.
This demo uses your existing RFM + XGBoost + Rules pipeline.
""")

# File uploads
txns_file = st.file_uploader("Upload transactions CSV", type=["csv"])
sku_file = st.file_uploader("Upload SKU map CSV", type=["csv"])

dominance_threshold = st.slider("Dominance Threshold", 0.0, 1.0, 0.6)
override_conf = st.slider("Override Confidence", 0.0, 1.0, 0.55)

if txns_file and sku_file:
    st.info("Running Basket Segmentation model...")

    # ---- Robust CSV reads (dtype-safe) ----
    txns_df = pd.read_csv(
        txns_file,
        dtype={"customer_id": "string", "sku": "string"},
        parse_dates=["date"],
        keep_default_na=True,
    )
    txns_df["amount"] = pd.to_numeric(txns_df["amount"], errors="coerce").fillna(0.0).astype("float32")

    sku_df = pd.read_csv(
        sku_file,
        dtype={"sku": "string", "cuisine_tag": "string"},
    )

    try:
        labeled_train, unlabeled, feats_all, feature_cols = prepare_dataset(
            txns=txns_df,
            sku_map=sku_df,
            labels=None,
            cutoff_date=None,
            dominance_threshold=dominance_threshold,
        )

        # Optional debug panel
        with st.expander("Debug: Feature dtypes"):
            st.write(feats_all[feature_cols].dtypes)

        model, metrics = train_xgb(labeled_train, feature_cols)

        # XGBoost expects float32; ensure dtype before predict_proba
        X_all = feats_all[feature_cols].to_numpy(dtype=np.float32, copy=False)

        proba = model.predict_proba(X_all)
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = model.classes_[pred_idx]
        pred_conf = proba.max(axis=1)

        preds_df = pd.DataFrame({
            "customer_id": feats_all["customer_id"].astype(str),
            "pred_segment_model": pred_labels,
            "pred_proba_max": pred_conf
        })

        final_seg = apply_rules(
            feats=feats_all,
            preds_proba=preds_df,
            dominance_threshold=dominance_threshold,
            override_conf=override_conf
        )
        preds_df["final_segment"] = final_seg.values

        st.success("Model run completed!")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sample Predictions")
            st.dataframe(preds_df.head(25), use_container_width=True)

        with col2:
            st.subheader("Segment Summary")
            summary = preds_df.groupby("final_segment").size().rename("customers").reset_index()
            st.dataframe(summary, use_container_width=True)

        # Optional: show validation metrics if available
        if metrics.get("val_accuracy") is not None:
            st.caption(f"Validation Accuracy: {metrics['val_accuracy']:.3f} | Macro F1: {metrics['val_macro_f1']:.3f}")

        # Download results
        st.download_button(
            label="Download predictions as CSV",
            data=preds_df.to_csv(index=False).encode("utf-8"),
            file_name="basket_segmentation_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Model run failed: {e}")

else:
    st.info("Upload both transactions CSV and SKU map CSV to run the model.")
