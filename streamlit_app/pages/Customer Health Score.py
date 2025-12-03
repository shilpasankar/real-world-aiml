import os
import sys
from datetime import date

import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------------------------------------------------------
# Import health_scoring with a robust path setup
# -----------------------------------------------------------------------------
# Assumed layout:
#   real-world-aiml/
#     streamlit_app/
#       customer_health/
#         health_scoring.py
#       pages/
#         Customer Health Score.py  (this file)
#
# We add streamlit_app/ to sys.path so `customer_health.health_scoring` works.
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))  # streamlit_app

if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

try:
    from customer_health.health_scoring import (
        features_engagement,
        features_service,
        features_value_momentum,
        features_policy,
        build_scores,
    )
except Exception as e:
    st.set_page_config(page_title="Customer Health Score", page_icon="🏦", layout="wide")
    st.error(
        "❌ Could not import `customer_health.health_scoring`.\n\n"
        "Check that:\n"
        "- `streamlit_app/customer_health/health_scoring.py` exists, and\n"
        "- the folder is named exactly `customer_health`.\n\n"
        f"Python error: `{e}`"
    )
    st.stop()

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Health Score",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Customer Health Score")
st.caption("Rule-based 0–100 scoring demo for UAE banking customers")

with st.expander("ℹ️ What is this?", expanded=False):
    st.write(
        """
        This page wraps the **rule-based customer health score** defined in
        `customer_health/health_scoring.py`.

        It lets you:

        - Upload the required input CSVs  
        - Compute per-customer scores and bands (Red / Amber / Green)  
        - Explore score distributions and band mix with **Plotly visuals**  
        - Inspect **feature-level contributions** for a selected customer
        """
    )

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration")

ref_date_input = st.sidebar.date_input(
    "Reference date",
    value=date.today(),
    help="Used for lookback windows, inactivity, tenure, etc.",
)

run_button = st.sidebar.button("🚀 Run scoring", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Input CSVs")

st.sidebar.write(
    "Upload six CSV files with the expected schemas. "
    "Column names must match those used in `health_scoring.py`."
)

# File uploaders
customers_file = st.sidebar.file_uploader("customers.csv", type=["csv"])
social_file = st.sidebar.file_uploader("social_interactions.csv", type=["csv"])
comms_file = st.sidebar.file_uploader("communications.csv", type=["csv"])
complaints_file = st.sidebar.file_uploader("complaints.csv", type=["csv"])
txns_file = st.sidebar.file_uploader("transactions.csv", type=["csv"])
products_file = st.sidebar.file_uploader("products.csv", type=["csv"])

required_cols = {
    "customers.csv": ["customer_id", "join_date", "kyc_risk_tier"],
    "social_interactions.csv": ["customer_id", "date"],
    "communications.csv": ["customer_id", "date", "response_time_hours"],
    "complaints.csv": [
        "customer_id",
        "complaint_id",
        "opened_at",
        "status",
        "first_contact_resolution",
    ],
    "transactions.csv": ["customer_id", "date", "amount", "category"],
    "products.csv": ["customer_id", "event_date", "event"],
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _parse_csv(uploaded, name: str) -> pd.DataFrame:
    if uploaded is None:
        raise ValueError(f"Missing required file: {name}")

    # date parsing per file
    parse_dates = []
    if name == "customers.csv":
        parse_dates = ["join_date"]
    elif name == "social_interactions.csv":
        parse_dates = ["date"]
    elif name == "communications.csv":
        parse_dates = ["date"]
    elif name == "complaints.csv":
        parse_dates = ["opened_at", "closed_at"]
    elif name == "transactions.csv":
        parse_dates = ["date"]
    elif name == "products.csv":
        parse_dates = ["event_date"]

    df = pd.read_csv(uploaded, parse_dates=parse_dates)

    missing = [c for c in required_cols[name] if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

    return df


def _compute_scores(
    customers: pd.DataFrame,
    social: pd.DataFrame,
    comms: pd.DataFrame,
    complaints: pd.DataFrame,
    txns: pd.DataFrame,
    products: pd.DataFrame,
    ref_date: date,
) -> pd.DataFrame:
    # 🔧 Normalize ref_date to pandas Timestamp so comparisons work
    ref_ts = pd.to_datetime(ref_date)

    fe_eng = features_engagement(social, comms, customers, ref_ts)
    fe_svc = features_service(complaints, customers, ref_ts)
    fe_val = features_value_momentum(txns, products, customers, ref_ts)
    fe_pol = features_policy(customers, ref_ts)

    scored = build_scores(fe_eng, fe_svc, fe_val, fe_pol)
    return scored



# -----------------------------------------------------------------------------
# Plotly visual helpers (fixed sizes)
# -----------------------------------------------------------------------------
DEFAULT_HEIGHT = 350  # keep charts neat & non-giant


def score_distribution_chart(scored: pd.DataFrame):
    # Bin scores into ranges for a nice bar chart
    bins = pd.interval_range(start=0, end=100, periods=20)
    binned = pd.cut(scored["score"], bins)
    dist = binned.value_counts().sort_index().reset_index()
    dist.columns = ["score_bin", "count"]
    dist["bin_label"] = dist["score_bin"].astype(str)

    fig = px.bar(
        dist,
        x="bin_label",
        y="count",
        title="Customer Health Score Distribution",
    )
    fig.update_layout(
        height=DEFAULT_HEIGHT,
        xaxis_title="Score bin",
        yaxis_title="Number of customers",
        margin=dict(l=10, r=10, t=60, b=40),
    )
    return fig


def band_breakdown_chart(scored: pd.DataFrame):
    band_counts = (
        scored["band"]
        .value_counts()
        .reindex(["Green", "Amber", "Red"])
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    band_counts.columns = ["band", "count"]

    fig = px.bar(
        band_counts,
        x="band",
        y="count",
        title="Band Breakdown",
        text="count",
        color="band",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        height=DEFAULT_HEIGHT,
        xaxis_title="Band",
        yaxis_title="Number of customers",
        margin=dict(l=10, r=10, t=60, b=40),
    )
    fig.update_traces(textposition="outside")
    return fig


def contributions_chart(contrib_df: pd.DataFrame):
    fig = px.bar(
        contrib_df,
        x="contribution",
        y="feature",
        orientation="h",
        title="Top contributing features (percentage points)",
    )
    fig.update_layout(
        height=DEFAULT_HEIGHT,
        xaxis_title="Contribution to score (approx % pts)",
        yaxis_title="Feature",
        margin=dict(l=10, r=10, t=60, b=40),
    )
    return fig


# -----------------------------------------------------------------------------
# UI sections
# -----------------------------------------------------------------------------
def _show_overview(scored: pd.DataFrame):
    st.subheader("Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Customers scored", f"{scored['customer_id'].nunique():,}")
    with col2:
        st.metric("Average health score", f"{scored['score'].mean():.1f}")
    with col3:
        band_counts = scored["band"].value_counts()
        green = int(band_counts.get("Green", 0))
        amber = int(band_counts.get("Amber", 0))
        red = int(band_counts.get("Red", 0))
        st.metric("Band mix (G / A / R)", f"{green} / {amber} / {red}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Score distribution")
        fig_dist = score_distribution_chart(scored)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_b:
        st.markdown("### Band breakdown")
        fig_band = band_breakdown_chart(scored)
        st.plotly_chart(fig_band, use_container_width=True)


def _show_top_bottom(scored: pd.DataFrame):
    st.subheader("Top & bottom customers")

    num = st.slider(
        "Number of customers to show",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )

    cols = [
        c
        for c in [
            "customer_id",
            "score",
            "band",
            "social_sentiment_mean",
            "social_rate_30d",
            "response_rate",
            "avg_resp_hours",
            "resolution_rate",
            "fcr_rate",
            "reopen_rate",
            "spend_cagr3",
            "product_events_180d",
            "inactive_days",
            "tenure_months",
        ]
        if c in scored.columns
    ]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 🟢 Highest scores")
        top_df = scored[cols].sort_values("score", ascending=False).head(num)
        st.dataframe(top_df, use_container_width=True)

    with c2:
        st.markdown("#### 🔴 Lowest scores")
        bottom_df = scored[cols].sort_values("score", ascending=True).head(num)
        st.dataframe(bottom_df, use_container_width=True)


def _show_customer_explainability(scored: pd.DataFrame):
    st.subheader("Customer-level explainability")

    customer_ids = scored["customer_id"].unique()
    if len(customer_ids) == 0:
        st.info("No customers found in the scored dataset.")
        return

    selected_id = st.selectbox(
        "Select a customer",
        options=customer_ids,
        index=0,
    )

    row = scored.loc[scored["customer_id"] == selected_id].iloc[0]

    st.markdown(
        f"**Customer:** `{selected_id}`  ·  "
        f"**Score:** `{row['score']:.1f}`  ·  "
        f"**Band:** `{row['band']}`"
    )

    contrib_cols = [c for c in scored.columns if c.startswith("contrib_")]
    if not contrib_cols:
        st.info("No contribution columns found in the dataset.")
        return

    contrib_df = (
        row[contrib_cols]
        .to_frame("contribution")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
    contrib_df = contrib_df.sort_values("abs_contribution", ascending=False).head(15)

    st.markdown("Top contributing features")
    st.dataframe(
        contrib_df[["feature", "contribution"]],
        use_container_width=True,
        hide_index=True,
    )

    fig_contrib = contributions_chart(contrib_df)
    st.plotly_chart(fig_contrib, use_container_width=True)


# -----------------------------------------------------------------------------
# Main interaction
# -----------------------------------------------------------------------------
st.markdown("## Run the health score")

if not run_button:
    st.info("Upload the required CSVs on the left, pick a reference date, then click **Run scoring**.")
else:
    try:
        # Read & validate inputs
        customers_df = _parse_csv(customers_file, "customers.csv")
        social_df = _parse_csv(social_file, "social_interactions.csv")
        comms_df = _parse_csv(comms_file, "communications.csv")
        complaints_df = _parse_csv(complaints_file, "complaints.csv")
        txns_df = _parse_csv(txns_file, "transactions.csv")
        products_df = _parse_csv(products_file, "products.csv")

        with st.spinner("Computing features and scores…"):
            scored_df = _compute_scores(
                customers_df,
                social_df,
                comms_df,
                complaints_df,
                txns_df,
                products_df,
                ref_date_input,
            )

        st.success("Scoring complete ✅")

        # Tabs for exploration
        tab_overview, tab_tables, tab_explain = st.tabs(
            ["📊 Overview", "📋 Top/Bottom", "🧠 Explainability"]
        )

        with tab_overview:
            _show_overview(scored_df)

        with tab_tables:
            _show_top_bottom(scored_df)

        with tab_explain:
            _show_customer_explainability(scored_df)

        # Download
        st.markdown("---")
        st.subheader("Download results")
        csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download full scored dataset (CSV)",
            data=csv_bytes,
            file_name="customer_health_scores_streamlit.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except ValueError as e:
        st.error(f"Input error: {e}")
    except Exception as e:
        st.exception(e)
