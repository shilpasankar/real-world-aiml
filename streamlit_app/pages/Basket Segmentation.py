import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick

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
This version adds **validation metrics** and **rich visuals** (confusion matrix, donut chart, violins, and feature importance).
""")

# File uploads
txns_file = st.file_uploader("Upload TRAIN transactions CSV", type=["csv"])
val_file = st.file_uploader("Optional: Upload VALIDATION transactions CSV", type=["csv"])
sku_file = st.file_uploader("Upload SKU map CSV", type=["csv"])


with st.sidebar:
    st.header("Controls")
    dominance_threshold = st.slider("Dominance Threshold", 0.0, 1.0, 0.6, 0.01)
    override_conf = st.slider("Override Confidence", 0.0, 1.0, 0.55, 0.01)
    normalize_cm = st.checkbox("Normalize Confusion Matrix", value=True)
    topk_features = st.slider("Top-K Features to Plot", 5, 30, 15, 1)

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

            # Confusion matrix in fixed A/B/C/D order (missing classes appear as 0)
            cm_counts = confusion_matrix(y_valid, y_pred, labels=class_order)
            if normalize_cm:
                with np.errstate(divide="ignore", invalid="ignore"):
                    row_sums = cm_counts.sum(axis=1, keepdims=True)
                    cm = np.divide(cm_counts, row_sums, out=np.zeros_like(cm_counts, dtype=float), where=row_sums != 0)
            else:
                cm = cm_counts

        # ---------- UI: Metrics ----------
        st.success("Model run completed!")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Validation Accuracy", f"{val_acc:.3f}" if val_acc is not None else "—")
        with m2:
            st.metric("Validation Macro F1", f"{val_f1:.3f}" if val_f1 is not None else "—")
        with m3:
            trained_labels = ", ".join(map(str, getattr(model, "_present_labels_", getattr(model, "_full_labels_", []))))
            st.metric("Classes Trained", trained_labels if trained_labels else "—")

        # ---------- Confusion Matrix (rich) ----------
        if cm is not None:
            st.subheader("Confusion Matrix")
            fig = plt.figure()
            ax = plt.gca()
            im = ax.imshow(cm, interpolation="nearest")
            plt.title("Confusion Matrix (A/B/C/D)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(range(len(class_order)), class_order)
            plt.yticks(range(len(class_order)), class_order)
            # Colorbar with % when normalized
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            if normalize_cm:
                cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
                im.set_cmap("Blues")
            else:
                im.set_cmap("Oranges")

            # annotate cells
            for i in range(len(class_order)):
                for j in range(len(class_order)):
                    val = cm[i, j]
                    text = f"{val:.0%}" if normalize_cm else f"{int(val)}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=10, weight="bold")
            st.pyplot(fig, clear_figure=True)

            # Classification report (text)
            st.caption("Classification Report (Valid)")
            st.text(classification_report(y_valid, y_pred, labels=class_order, zero_division=0))

        # ---------- Segment Distribution (donut) ----------
        st.subheader("Distribution of Final Segments")
        seg_counts = preds_df["final_segment"].value_counts().sort_index()
        fig2 = plt.figure()
        ax2 = plt.gca()
        wedges, _ = ax2.pie(seg_counts.values, startangle=90)
        # Donut hole
        centre_circle = plt.Circle((0, 0), 0.65, fc="white")
        fig2.gca().add_artist(centre_circle)
        ax2.axis("equal")
        ax2.set_title("Final Segment Distribution (Donut)")
        ax2.legend(wedges, [f"{k}: {v}" for k, v in zip(seg_counts.index, seg_counts.values)], loc="center left", bbox_to_anchor=(1, 0.5))
        st.pyplot(fig2, clear_figure=True)

        # ---------- Confidence by Final Segment (violins) ----------
        st.subheader("Prediction Confidence by Final Segment")
        order = [c for c in "ABCD" if c in seg_counts.index.tolist()]
        grouped = [preds_df.loc[preds_df["final_segment"] == seg, "pred_proba_max"].values for seg in order]

        if any(len(g) > 0 for g in grouped):
            fig3 = plt.figure()
            ax3 = plt.gca()
            parts = ax3.violinplot(grouped, showmeans=True, showextrema=False)
            ax3.set_xticks(range(1, len(order) + 1))
            ax3.set_xticklabels(order)
            ax3.set_ylim(0, 1)
            ax3.set_ylabel("Predicted Probability (max)")
            ax3.set_title("Confidence by Final Segment (Violin)")
            st.pyplot(fig3, clear_figure=True)
        else:
            st.info("Not enough data per segment to draw violins.")

        # ---------- Feature Importance ----------
        st.subheader("Feature Importance")

        # Build importance DataFrame
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

            # Table (top-k)
            st.dataframe(imp_df[["feature", "importance", "importance_norm"]].head(topk_features), use_container_width=True)

            # Ranked horizontal bar (top-k)
            fig_imp = plt.figure()
            ax_imp = plt.gca()
            top_imp = imp_df.head(topk_features).iloc[::-1]  # reverse for barh top-down
            ax_imp.barh(top_imp["feature"], top_imp["importance_norm"])
            ax_imp.set_xlabel("Relative Importance (gain)")
            ax_imp.set_title("Top Feature Importances (Bar)")
            ax_imp.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            st.pyplot(fig_imp, clear_figure=True)

            # Polar "rose" chart (top-k) for a more expressive visual
            fig_polar = plt.figure()
            ax_pol = plt.subplot(111, polar=True)
            theta = np.linspace(0.0, 2 * np.pi, len(top_imp), endpoint=False)
            radii = top_imp["importance_norm"].values
            width = (2 * np.pi) / max(len(top_imp), 1)
            bars = ax_pol.bar(theta, radii, width=width, bottom=0.0, align="edge")
            ax_pol.set_yticklabels([])
            ax_pol.set_xticks(theta + width / 2)
            ax_pol.set_xticklabels(top_imp["feature"])
            ax_pol.set_title("Top Feature Importances (Polar Rose)", va="bottom")
            st.pyplot(fig_polar, clear_figure=True)

        # ---------- Predictions & Summary ----------
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sample Predictions")
            st.dataframe(preds_df.head(25), use_container_width=True)
        with col2:
            st.subheader("Segment Summary (Final)")
            summary = preds_df.groupby("final_segment").size().rename("customers").reset_index()
            st.dataframe(summary, use_container_width=True)

        # ---------- Download ----------
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
