import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from basket_segmentation.cuisine_segmentation import (
    prepare_dataset,
    train_xgb,
    apply_rules,
)

st.set_page_config(page_title="Basket Segmentation", layout="wide")
st.title("🍽️ Basket Segmentation (Demo)")

st.markdown("""
Upload your **transactions CSV** and **SKU map CSV** to run the Basket Segmentation model.
This demo uses your RFM + XGBoost + Rules pipeline, **now with validation & visuals**.
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
        # Prepare features & split
        labeled_train, unlabeled, feats_all, feature_cols = prepare_dataset(
            txns=txns_df,
            sku_map=sku_df,
            labels=None,
            cutoff_date=None,
            dominance_threshold=dominance_threshold,
        )

        # Optional debug
        with st.expander("Debug: Feature dtypes"):
            st.write(feats_all[feature_cols].dtypes)

        # Train
        model, metrics = train_xgb(labeled_train, feature_cols)

        # ---------- Inference on ALL customers ----------
        X_all = feats_all[feature_cols].to_numpy(dtype=np.float32, copy=False)
        proba_all = model.predict_proba(X_all)
        idx_all = np.argmax(proba_all, axis=1)
        conf_all = proba_all.max(axis=1)

        present_labels = getattr(model, "_present_labels_", None)
        if present_labels is None:
            present_labels = getattr(model, "classes_", np.array(list("ABCD")))
        label_all = np.asarray(present_labels)[idx_all]

        preds_df = pd.DataFrame({
            "customer_id": feats_all["customer_id"].astype(str),
            "pred_segment_model": label_all,       # human-readable A/B/C/D
            "pred_proba_max": conf_all
        })

        final_seg = apply_rules(
            feats=feats_all,
            preds_proba=preds_df,
            dominance_threshold=dominance_threshold,
            override_conf=override_conf
        )
        preds_df["final_segment"] = final_seg.values

        # ---------- VALIDATION (on split == 'valid') ----------
        valid_mask = labeled_train["split"] == "valid"
        has_valid = bool(valid_mask.any())

        val_acc = None
        val_f1 = None
        cm = None
        class_order = np.array(list("ABCD"))

        if has_valid:
            valid_df = labeled_train.loc[valid_mask]
            X_valid = valid_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
            y_valid = valid_df["pref_segment"].astype(str).to_numpy()

            proba_v = model.predict_proba(X_valid)
            idx_v = np.argmax(proba_v, axis=1)
            y_pred = np.asarray(present_labels)[idx_v]

            # Compute metrics
            val_acc = accuracy_score(y_valid, y_pred)
            val_f1 = f1_score(y_valid, y_pred, average="macro")

            # Confusion matrix in fixed A/B/C/D order (missing classes appear as 0 rows/cols)
            cm = confusion_matrix(y_valid, y_pred, labels=class_order)

        # ---------- UI: Results ----------
        st.success("Model run completed!")

        # Top metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Validation Accuracy", f"{val_acc:.3f}" if val_acc is not None else "—")
        with m2:
            st.metric("Validation Macro F1", f"{val_f1:.3f}" if val_f1 is not None else "—")
        with m3:
            st.metric("Classes Trained", ", ".join(getattr(model, "_present_labels_", getattr(model, "_full_labels_", []))))

        # Confusion matrix
        if cm is not None:
            st.subheader("Confusion Matrix (Valid)")
            fig = plt.figure()
            plt.imshow(cm, interpolation="nearest")
            plt.title("Confusion Matrix (A/B/C/D)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(range(len(class_order)), class_order)
            plt.yticks(range(len(class_order)), class_order)
            # annotate cells
            for i in range(len(class_order)):
                for j in range(len(class_order)):
                    plt.text(j, i, int(cm[i, j]), ha="center", va="center")
            st.pyplot(fig, clear_figure=True)

            # Classification report (text)
            st.caption("Classification Report (Valid)")
            st.text(classification_report(y_valid, y_pred, labels=class_order, zero_division=0))

        # Predictions & summary
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sample Predictions")
            st.dataframe(preds_df.head(25), use_container_width=True)
        with col2:
            st.subheader("Segment Summary (Final)")
            summary = preds_df.groupby("final_segment").size().rename("customers").reset_index()
            st.dataframe(summary, use_container_width=True)

        # Quick bar chart of final segments
        st.subheader("Distribution of Final Segments")
        seg_counts = preds_df["final_segment"].value_counts().sort_index()
        fig2 = plt.figure()
        plt.bar(seg_counts.index.astype(str), seg_counts.values)
        plt.xlabel("Final Segment")
        plt.ylabel("Customers")
        plt.title("Final Segment Distribution")
        st.pyplot(fig2, clear_figure=True)

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
