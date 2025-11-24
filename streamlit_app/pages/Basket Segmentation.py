import streamlit as st
import pandas as pd
import os
import subprocess

st.title("🍽️ Basket Segmentation (Demo)")

st.markdown("""
Upload your transactions and SKU map CSVs to run the Basket Segmentation model.
This demo runs your existing `cuisine_segmentation.py` pipeline and shows the results.
""")

txns_file = st.file_uploader("Upload transactions CSV", type=["csv"])
sku_file = st.file_uploader("Upload SKU map CSV", type=["csv"])

dominance_threshold = st.slider("Dominance Threshold", min_value=0.0, max_value=1.0, value=0.6)
override_conf = st.slider("Override Confidence", min_value=0.0, max_value=1.0, value=0.55)

if txns_file and sku_file:
    st.info("Running the model...")

    # Save uploaded files to a temporary folder
    os.makedirs("temp_uploads", exist_ok=True)
    txns_path = os.path.join("temp_uploads", "txns.csv")
    sku_path = os.path.join("temp_uploads", "sku_map.csv")
    txns_file.seek(0)
    sku_file.seek(0)
    with open(txns_path, "wb") as f:
        f.write(txns_file.read())
    with open(sku_path, "wb") as f:
        f.write(sku_file.read())

    # Run the existing cuisine_segmentation.py script
    cmd = [
        "python", "../basket_segmentation/cuisine_segmentation.py",
        "--txns", txns_path,
        "--sku_map", sku_path,
        "--dominance_threshold", str(dominance_threshold),
        "--override_confidence", str(override_conf),
        "--output_dir", "temp_uploads"
    ]

    try:
        subprocess.run(cmd, check=True)
        st.success("Model run completed!")

        # Load predictions
        preds = pd.read_csv("temp_uploads/predictions.csv")
        st.subheader("Sample Predictions")
        st.dataframe(preds.head())

        # Segment summary
        summary = pd.read_csv("temp_uploads/segment_summary.csv")
        st.subheader("Segment Summary")
        st.dataframe(summary)

    except subprocess.CalledProcessError as e:
        st.error(f"Model run failed: {e}")
else:
    st.info("Upload both CSVs to run the segmentation.")
