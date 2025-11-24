import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick

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
st.set_page_config(page_title="Basket Segmentation", layout="wide")
st.title("🍽️ Basket Segmentation (Demo)")

st.markdown("""
Upload your **TRAIN transactions CSV**, optional **VALIDATION transactions CSV**, and **SKU map CSV**.
This version includes **validation metrics** and **compact visuals** (confusion matrix with normalization, donut chart,
violin plots for confidence, and feature-importance charts).
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
    topk_features = st.slider("Top-K Features to Plot", 5, 30, 12, 1)

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
    # Robust numeric
    df["amount"] = pd.to_numeric(df.get("amount", 0.0), errors="coerce").fillna(0.0).astype("float32")
    # Optional region default
    if "region" not in df.columns:
        df["region"] = "NA"
    return df

# Small wrapper to keep all plots tight & compact
def _show(fig):
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False, clear_figure=True)

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

    # Optional validation
    val_df = _read_txn_csv(val_file) if val_file is not None else None

    try:
        # --------- Prepare datasets ---------
        if val_df is not None:
            st.info("Using uploaded validation dataset (no internal split).")

            # Prepare separately for train & valid, then tag splits
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
            # For whole-population predictions/visuals, use train+val population together:
            feats_all = pd.concat([feats_all_train, feats_all_valid], ignore_index=True)

            # If validation introduced new numeric features (e.g., new cuisines), ensure columns exist in feats_all
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

        # Optional debug
        with st.expander("Debug: Feature dtypes"):
            try:
                st.write(feats_all[feature_cols].dtypes)
            except Exception:
                st.write("Feature dtypes unavailable (no features).")

        # --------- Train ---------
        if len(labeled_train) == 0:
            raise ValueError("No labeled rows available for training (segments A-D). Check your inputs.")

        model, metrics = train_xgb(labeled_train, feature_cols)

        # --------- Inference on ALL customers (for outputs & visuals) ---------
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
            y_pred = np.asarray(present_labels)[idx_v]

            # Metrics
            val_acc = accuracy_score(y_valid, y_pred)
            val_f1 = f1_score(y_valid, y_pred, average="macro")

            # Confusion matrix in fixed A/B/C/D order
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
            trained_labels = ", ".join(map(str, getattr(model, "_present_labels_", getattr(model, "_full_labels_", []))))
            st.metric("Classes Trained", trained_labels if trained_labels else "—")

        # --------- Confusion Matrix (compact) ---------
        if cm is not None:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues" if normalize_cm else "Oranges")
            ax.set_title("Confusion Matrix (A/B/C/D)", fontsize=12)
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("True", fontsize=10)
            ax.set_xticks(range(len(class_order)), class_order)
            ax.set_yticks(range(len(class_order)), class_order)

            # Colorbar with % when normalized
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if normalize_cm:
                cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

            # annotate cells
            for i in range(len(class_order)):
                for j in range(len(class_order)):
                    val = cm[i, j]
                    text = f"{val:.0%}" if normalize_cm else f"{int(val)}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=9, weight="bold")

            _show(fig)

            # Classification report (compact text)
            st.caption("Classification Report (Valid)")
            st.text(classification_report(y_valid, y_pred, labels=class_order, zero_division=0))

        # --------- Segment Distribution (compact donut) ---------
        st.subheader("Distribution of Final Segments")
        seg_counts = preds_df["final_segment"].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(4.2, 4.2))
        wedges, _ = ax2.pie(seg_counts.values, startangle=90)
        # Donut hole
        centre_circle = plt.Circle((0, 0), 0.65, fc="white")
        ax2.add_artist(centre_circle)
        ax2.axis("equal")
        ax2.set_title("Final Segment Mix", fontsize=12)
        # compact legend
        ax2.legend(
            wedges,
            [f"{k}: {v}" for k, v in zip(seg_counts.index, seg_counts.values)],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=9,
            frameon=False
        )
        _show(fig2)

        # --------- Confidence by Final Segment (compact violins) ---------
        st.subheader("Prediction Confidence by Final Segment")
        order = [c for c in "ABCD" if c in seg_counts.index.tolist()]
        grouped = [preds_df.loc[preds_df["final_segment"] == seg, "pred_proba_max"].values for seg in order]

        if any(len(g) > 0 for g in grouped):
            fig3, ax3 = plt.subplots(figsize=(5.5, 3.2))
            parts = ax3.violinplot(grouped, showmeans=True, showextrema=False)
            ax3.set_xticks(range(1, len(order) + 1))
            ax3.set_xticklabels(order)
            ax3.set_ylim(0, 1)
            ax3.set_ylabel("Max Predicted Probability")
            ax3.set_title("Confidence by Final Segment", fontsize=12)
            _show(fig3)
        else:
            st.info("Not enough data per segment to draw violins.")

        # --------- Feature Importance (compact bar + polar) ---------
        st.subheader("Feature Importance")

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
                arr = np.asarray(model.feature_importances_, dtype=float)
                arr = arr[: len(feature_cols)]
                imp_df = pd.DataFrame({"feature": feature_cols, "importance": arr})
            except Exception:
                pass

        if imp_df.empty:
            st.info("Feature importances unavailable (e.g., single-class fallback or tiny dataset).")
        else:
            imp_df = imp_df.groupby("feature", as_index=False)["importance"].sum()
            imp_df = imp_df.sort_values("importance", ascending=False)
            imp_df["importance_norm"] = imp_df["importance"] / imp_df["importance"].sum()

            # Table (top-k, compact)
            st.dataframe(
                imp_df[["feature", "importance", "importance_norm"]].head(topk_features),
                use_container_width=True,
            )

            # Ranked horizontal bar (top-k, compact)
            top_imp = imp_df.head(topk_features).iloc[::-1]  # reverse for barh top-down
            fig_imp, ax_imp = plt.subplots(figsize=(6.2, 4.0))
            ax_imp.barh(top_imp["feature"], top_imp["importance_norm"])
            ax_imp.set_xlabel("Relative Importance (gain)")
            ax_imp.set_title("Top Feature Importances", fontsize=12)
            ax_imp.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            _show(fig_imp)

            # Polar "rose" chart (top-k, compact)
            fig_polar, ax_pol = plt.subplots(figsize=(5.0, 5.0), subplot_kw={'polar': True})
            theta = np.linspace(0.0, 2 * np.pi, len(top_imp), endpoint=False)
            radii = top_imp["importance_norm"].values
            width = (2 * np.pi) / max(len(top_imp), 1)
            ax_pol.bar(theta, radii, width=width, bottom=0.0, align="edge")
            ax_pol.set_yticklabels([])
            ax_pol.set_xticks(theta + width / 2)
            ax_pol.set_xticklabels(top_imp["feature"])
            ax_pol.set_title("Top Features (Polar)", va="bottom", fontsize=12)
            _show(fig_polar)

        # --------- Predictions & Summary ---------
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sample Predictions")
            st.dataframe(preds_df.head(20), use_container_width=True)
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
