import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

from basket_segmentation.cuisine_segmentation import (
    prepare_dataset,
    train_xgb,
    apply_rules,
)

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="🍽️ Basket Segmentation", layout="wide")
st.title("🍽️ Basket Segmentation (Demo)")

st.markdown("""
**What this page does**  
- Upload **TRAIN** transactions (CSV), optional **VALIDATION** transactions, and **SKU map**.  
- Trains your RFM + XGBoost model and applies the rule override layer.  
- Shows **Validation Accuracy / Macro F1**, **Confusion Matrix**, **Segment Mix**, **Confidence violins**, and **Feature Importance**.  
- Each chart has a 🛈 **How to read this chart** toggle for quick interpretation.  
""")

# -----------------------
# Uploaders
# -----------------------
txns_file = st.file_uploader("Upload TRAIN transactions CSV", type=["csv"])
val_file  = st.file_uploader("Optional: Upload VALIDATION transactions CSV", type=["csv"])
sku_file  = st.file_uploader("Upload SKU map CSV", type=["csv"])

with st.sidebar:
    st.header("Controls")
    dominance_threshold = st.slider("Dominance Threshold", 0.0, 1.0, 0.6, 0.01)
    override_conf = st.slider("Override Confidence", 0.0, 1.0, 0.55, 0.01)
    normalize_cm = st.checkbox("Normalize Confusion Matrix", value=True)
    topk_features = st.slider("Top-K Features to Plot", 5, 30, 15, 1)
    st.caption("Plotly visuals use fixed heights and a clean template for screenshots.")

# -----------------------
# Helpers
# -----------------------
def _read_txn_csv(file):
    df = pd.read_csv(
        file,
        dtype={"customer_id": "string", "sku": "string"},
        parse_dates=["date"],
        keep_default_na=True,
    )
    df["amount"] = pd.to_numeric(df.get("amount", 0.0), errors="coerce").fillna(0.0).astype("float32")
    if "region" not in df.columns:
        df["region"] = "NA"
    return df

def _plot_cm(cm, class_order, normalized: bool):
    # Heatmap with annotations
    z = cm.astype(float)
    z_text = np.empty_like(z).astype(object)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            z_text[i, j] = f"{z[i, j]:.0%}" if normalized else f"{int(z[i, j])}"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=class_order,
            y=class_order,
            colorscale="Blues" if normalized else "Oranges",
            colorbar=dict(title="%" if normalized else "Count"),
            zmin=0,
            zmax=1 if normalized else None,
            hovertemplate="True %{y}<br>Pred %{x}<br><b>%{z}</b><extra></extra>"
        )
    )
    # Add text annotations
    for i, ylab in enumerate(class_order):
        for j, xlab in enumerate(class_order):
            fig.add_annotation(
                x=xlab, y=ylab, text=z_text[i, j],
                showarrow=False, font=dict(size=12, color="black")
            )
    fig.update_layout(
        template="plotly_white",
        title="Confusion Matrix (A/B/C/D)",
        xaxis=dict(title="Predicted"),
        yaxis=dict(title="True", autorange="reversed"),
        height=420,
        margin=dict(t=60, b=40, l=60, r=30),
    )
    return fig

# -----------------------
# Main
# -----------------------
if txns_file and sku_file:
    st.info("Running Basket Segmentation model...")

    # Read inputs
    txns_df = _read_txn_csv(txns_file)
    sku_df = pd.read_csv(
        sku_file,
        dtype={"sku": "string", "cuisine_tag": "string"},
    )
    val_df = _read_txn_csv(val_file) if val_file is not None else None

    try:
        # --------- Prepare datasets ---------
        if val_df is not None:
            st.info("Using uploaded VALIDATION dataset (no internal split).")

            labeled_train_only, _, feats_all_train, feature_cols = prepare_dataset(
                txns=txns_df,
                sku_map=sku_df,
                labels=None,
                cutoff_date=None,
                dominance_threshold=dominance_threshold,
            )
            labeled_valid_only, _, feats_all_valid, _ = prepare_dataset(
                txns=val_df,
                sku_map=sku_df,
                labels=None,
                cutoff_date=None,
                dominance_threshold=dominance_threshold,
            )

            labeled_train_only["split"] = "train"
            labeled_valid_only["split"] = "valid"

            labeled_train = pd.concat([labeled_train_only, labeled_valid_only], ignore_index=True)
            feats_all = pd.concat([feats_all_train, feats_all_valid], ignore_index=True)

            # If validation introduced new features, pad them
            for col in feature_cols:
                if col not in feats_all.columns:
                    feats_all[col] = 0.0
        else:
            labeled_train, _, feats_all, feature_cols = prepare_dataset(
                txns=txns_df,
                sku_map=sku_df,
                labels=None,
                cutoff_date=None,
                dominance_threshold=dominance_threshold,
            )

        # --------- Train ---------
        if len(labeled_train) == 0:
            raise ValueError("No labeled rows for training (segments A-D). Check your inputs.")

        model, metrics = train_xgb(labeled_train, feature_cols)

        # --------- Inference (ALL customers) ---------
        X_all = feats_all[feature_cols].to_numpy(dtype=np.float32, copy=False)
        proba_all = model.predict_proba(X_all)
        if proba_all.ndim != 2:
            raise ValueError("Model returned unexpected probability shape.")
        idx_all = np.argmax(proba_all, axis=1)
        conf_all = proba_all.max(axis=1)

        # ---- FIX: avoid ambiguous truth value with NumPy arrays
        present_labels = getattr(model, "_present_labels_", None)
        if present_labels is None:
            present_labels = getattr(model, "classes_", np.array(list("ABCD")))
        present_labels = np.asarray(present_labels)

        label_all = present_labels[idx_all]

        preds_df = pd.DataFrame({
            "customer_id": feats_all["customer_id"].astype(str),
            "pred_segment_model": label_all,
            "pred_proba_max": conf_all
        })

        final_seg = apply_rules(
            feats=feats_all,
            preds_proba=preds_df,
            dominance_threshold=dominance_threshold,
            override_conf=override_conf
        )
        preds_df["final_segment"] = final_seg.values

        # --------- VALIDATION (split == 'valid') ---------
        valid_mask = labeled_train["split"] == "valid"
        has_valid = bool(valid_mask.any())

        val_acc = None
        val_f1 = None
        cm = None
        class_order = np.array(list("ABCD"))

        if has_valid:
            valid_df = labeled_train.loc[valid_mask].copy()
            X_valid = valid_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
            y_valid = valid_df["pref_segment"].astype(str).to_numpy()

            proba_v = model.predict_proba(X_valid)
            idx_v = np.argmax(proba_v, axis=1)
            y_pred = present_labels[idx_v]

            # Metrics
            val_acc = accuracy_score(y_valid, y_pred)
            val_f1 = f1_score(y_valid, y_pred, average="macro")

            # Confusion matrix
            cm_counts = confusion_matrix(y_valid, y_pred, labels=class_order)
            if normalize_cm:
                with np.errstate(divide="ignore", invalid="ignore"):
                    row_sums = cm_counts.sum(axis=1, keepdims=True)
                    cm = np.divide(cm_counts, row_sums, out=np.zeros_like(cm_counts, dtype=float), where=row_sums != 0)
            else:
                cm = cm_counts

        # --------- UI: Metrics ---------
        st.success("Model run completed!")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Validation Accuracy", f"{val_acc:.3f}" if val_acc is not None else "—")
        with m2:
            st.metric("Validation Macro F1", f"{val_f1:.3f}" if val_f1 is not None else "—")
        with m3:
            trained_labels = getattr(model, "_present_labels_", None)
            if trained_labels is None:
                trained_labels = getattr(model, "_full_labels_", [])
            trained_labels_str = ", ".join(map(str, trained_labels)) if isinstance(trained_labels, (list, tuple, np.ndarray)) else str(trained_labels)
            st.metric("Classes Trained", trained_labels_str if trained_labels_str else "—")

        # --------- Confusion Matrix (Plotly) ---------
        if cm is not None:
            st.subheader("Confusion Matrix")
            show_help_cm = st.checkbox("🛈 How to read this chart — Confusion Matrix", value=True)
            fig_cm = _plot_cm(np.array(cm, dtype=float), class_order, normalized=normalize_cm)
            st.plotly_chart(fig_cm, use_container_width=False)
            if show_help_cm:
                st.caption(
                    "**Read it like this:** Rows are *true* segments, columns are *predicted*. "
                    "Darker diagonal = better. If normalized, each row sums to 100%."
                )
            # Text report
            with st.expander("Classification Report (Valid)"):
                st.text(classification_report(y_valid, y_pred, labels=class_order, zero_division=0))

        # --------- Segment Distribution (Pie/Donut) ---------
        st.subheader("Distribution of Final Segments")
        show_help_mix = st.checkbox("🛈 How to read this chart — Segment Mix", value=True)
        seg_counts = preds_df["final_segment"].value_counts().sort_index()
        mix_df = seg_counts.rename_axis("segment").reset_index(name="customers")
        fig_pie = px.pie(
            mix_df, names="segment", values="customers",
            hole=0.6, title="Final Segment Mix",
            color="segment",
        )
        fig_pie.update_layout(template="plotly_white", height=360, margin=dict(t=60, b=40, l=40, r=40))
        st.plotly_chart(fig_pie, use_container_width=False)
        if show_help_mix:
            st.caption("**Read it like this:** The donut shows how customers are distributed across final segments after the rule overrides.")

        # --------- Confidence by Final Segment (Violin) ---------
        st.subheader("Prediction Confidence by Final Segment")
        show_help_violin = st.checkbox("🛈 How to read this chart — Confidence", value=True)
        vio_df = preds_df[["final_segment", "pred_proba_max"]].rename(columns={"final_segment": "segment", "pred_proba_max": "confidence"})
        if len(vio_df) and vio_df["segment"].nunique() > 0:
            fig_violin = px.violin(
                vio_df, x="segment", y="confidence", points=False, box=True,
                title="Confidence (max predicted probability) by Segment"
            )
            fig_violin.update_layout(template="plotly_white", yaxis=dict(range=[0,1]), height=380, margin=dict(t=60, b=40, l=60, r=30))
            st.plotly_chart(fig_violin, use_container_width=False)
            if show_help_violin:
                st.caption("**Read it like this:** Taller/shifted violins indicate higher or more variable confidence. Boxes show medians and quartiles.")
        else:
            st.info("Not enough data per segment to draw violins.")

        # --------- Feature Importance (Bar) ---------
        st.subheader("Feature Importance")
        show_help_imp = st.checkbox("🛈 How to read this chart — Feature Importance", value=True)

        imp_df = pd.DataFrame(columns=["feature", "importance"])

        booster_getter = getattr(model, "get_booster", None)
        if callable(booster_getter):
            try:
                booster = model.get_booster()
                raw_score = booster.get_score(importance_type="gain")  # {'f0': 0.12, ...}
                fmap = {f"f{i}": name for i, name in enumerate(feature_cols)}
                rows = [{"feature": fmap.get(k, k), "importance": float(v)} for k, v in raw_score.items()]
                imp_df = pd.DataFrame(rows)
            except Exception:
                pass

        if imp_df.empty and hasattr(model, "feature_importances_"):
            try:
                arr = np.asarray(model.feature_importances_, dtype=float)[: len(feature_cols)]
                imp_df = pd.DataFrame({"feature": feature_cols, "importance": arr})
            except Exception:
                pass

        if imp_df.empty:
            st.info("Feature importances unavailable (e.g., single-class fallback or tiny dataset).")
        else:
            imp_df = (
                imp_df.groupby("feature", as_index=False)["importance"].sum()
                      .sort_values("importance", ascending=False)
            )
            imp_df["importance_norm"] = imp_df["importance"] / imp_df["importance"].sum()
            top_imp = imp_df.head(topk_features)

            fig_imp = px.bar(
                top_imp.sort_values("importance_norm", ascending=True),
                x="importance_norm", y="feature", orientation="h",
                title="Top Feature Importances (normalized gain)",
                labels={"importance_norm": "Relative Importance", "feature": "Feature"},
                text="importance_norm",
            )
            fig_imp.update_traces(texttemplate="%{text:.0%}", textposition="outside", cliponaxis=False)
            fig_imp.update_layout(template="plotly_white", height=420, margin=dict(t=60, b=40, l=80, r=40), xaxis_tickformat=".0%")
            st.plotly_chart(fig_imp, use_container_width=False)

            if show_help_imp:
                st.caption("**Read it like this:** Higher bars = stronger influence on model decisions. "
                           "Use this to explain which behaviors/cuisines drive segment predictions.")

        # --------- Predictions & Summary ---------
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sample Predictions")
            st.dataframe(preds_df.head(25), use_container_width=True)
        with col2:
            st.subheader("Segment Summary (Final)")
            summary = preds_df.groupby("final_segment").size().rename("customers").reset_index()
            st.dataframe(summary, use_container_width=True)

        # --------- Download ---------
        st.download_button(
            label="Download predictions as CSV",
            data=preds_df.to_csv(index=False).encode("utf-8"),
            file_name="basket_segmentation_predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Model run failed: {e}")

else:
    st.info("Upload TRAIN transactions CSV, optional VALIDATION transactions CSV, and the SKU map CSV to run the model.")
