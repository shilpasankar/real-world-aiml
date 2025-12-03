python
import argparse
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

plt.switch_backend("Agg")  # plotting without GUI

# -------------------------
# Helpers
# -------------------------

def _read_csv(path, parse_dates=None):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)

def _days_between(a, b):
    return (b - a).days

def _safe_div(a, b):
    return 0.0 if b == 0 else a / b

def _clamp01(x):
    return max(0.0, min(1.0, float(x)))

def _cap(series, q_low=0.01, q_high=0.99):
    lo, hi = series.quantile(q_low), series.quantile(q_high)
    return series.clip(lower=lo, upper=hi)

# -------------------------
# Feature engineering
# -------------------------

def features_engagement(social_df, comms_df, customers, ref_date):
    """Compute engagement features per customer."""
    last90 = ref_date - timedelta(days=90)

    # --- Social ---
    soc = social_df[social_df["date"] >= last90].copy()
    # Normalize sentiment to 0..1 from -1..1
    if "sentiment" in soc.columns:
        soc["sentiment01"] = (soc["sentiment"] + 1) / 2.0
    else:
        soc["sentiment01"] = 0.5

    social_stats = soc.groupby("customer_id").agg(
        social_mentions=("customer_id", "count"),
        social_sentiment_mean=("sentiment01", "mean")
    )
    # mentions per 30 days
    days = max(1, (ref_date - last90).days)
    social_stats["social_rate_30d"] = social_stats["social_mentions"] * 30.0 / days

    # --- Communications ---
    comms = comms_df[comms_df["date"] >= last90].copy()
    # Response rate: inbound messages that received a response within window
    # If response_time_hours exists, we treat non-null as "responded"
    responded = (comms["response_time_hours"].notna()).astype(int)
    comms["responded"] = responded
    comm_stats = comms.groupby("customer_id").agg(
        comm_cnt=("customer_id", "count"),
        response_rate=("responded", "mean"),
        avg_resp_hours=("response_time_hours", "mean")
    )
    # Fill NaNs: no comms → response_rate=0, avg_resp_hours large
    comm_stats["response_rate"] = comm_stats["response_rate"].fillna(0.0)
    comm_stats["avg_resp_hours"] = comm_stats["avg_resp_hours"].fillna(72.0)

    # Merge
    out = customers[["customer_id"]].merge(social_stats, how="left", on="customer_id") \
                                    .merge(comm_stats, how="left", on="customer_id")
    out = out.fillna({"social_mentions": 0, "social_sentiment_mean": 0.5,
                      "social_rate_30d": 0.0, "comm_cnt": 0,
                      "response_rate": 0.0, "avg_resp_hours": 72.0})
    return out

def features_service(complaints_df, customers, ref_date):
    """Complaint service quality metrics."""
    last180 = ref_date - timedelta(days=180)
    comp = complaints_df[(complaints_df["opened_at"] <= ref_date) &
                         ((complaints_df["closed_at"].isna()) | (complaints_df["closed_at"] >= last180))].copy()

    comp["is_resolved"] = (comp["status"].fillna("").str.lower() == "resolved").astype(int)
    comp["is_reopened"] = (comp["status"].fillna("").str.lower() == "reopened").astype(int)
    # first contact resolution rate
    fcr = comp["first_contact_resolution"].fillna(0).astype(int)

    svc = comp.groupby("customer_id").agg(
        complaints=("complaint_id", "count"),
        resolved=("is_resolved", "sum"),
        reopened=("is_reopened", "sum"),
        fcr_sum=("first_contact_resolution", "sum")
    )
    svc["resolution_rate"] = svc.apply(lambda r: _safe_div(r["resolved"], r["complaints"]), axis=1)
    svc["reopen_rate"] = svc.apply(lambda r: _safe_div(r["reopened"], r["complaints"]), axis=1)
    svc["fcr_rate"] = svc.apply(lambda r: _safe_div(r["fcr_sum"], r["complaints"]), axis=1)
    svc = svc.drop(columns=["resolved", "reopened", "fcr_sum"])

    out = customers[["customer_id"]].merge(svc, how="left", on="customer_id")
    out = out.fillna({"complaints": 0, "resolution_rate": 1.0, "reopen_rate": 0.0, "fcr_rate": 0.0})
    return out

def features_value_momentum(txns_df, products_df, customers, ref_date):
    """Spend trend, inactivity, and product uptake."""
    # Focus on card & transfer categories as value proxies (customize as needed)
    tx = txns_df.copy()
    tx["month"] = tx["date"].values.astype('datetime64[M]')
    value_tx = tx[tx["category"].isin(["cards", "transfers"])].copy()
    monthly = value_tx.groupby(["customer_id", "month"]).agg(spend=("amount", "sum")).reset_index()

    # 3-month CAGR proxy: (last / prev3)^(1/3) - 1
    def spend_trend(g):
        g = g.sort_values("month")
        if len(g) < 4:
            return 0.0
        last = g.iloc[-1]["spend"]
        prev3 = g.iloc[-4]["spend"]
        if prev3 <= 0:
            return 0.0
        return (last / prev3) ** (1/3) - 1

    trend = monthly.groupby("customer_id").apply(spend_trend).rename("spend_cagr3").reset_index()

    # Inactivity: days since last transaction
    last_tx = tx.groupby("customer_id")["date"].max().rename("last_tx_date").reset_index()
    last_tx["inactive_days"] = (ref_date - last_tx["last_tx_date"]).dt.days
    last_tx["inactive_days"] = last_tx["inactive_days"].fillna(999)

    # Product uptake in last 180 days
    last180 = ref_date - timedelta(days=180)
    pr = products_df[products_df["event_date"] >= last180]
    prod_events = pr.groupby("customer_id").size().rename("product_events_180d").reset_index()

    out = customers[["customer_id"]].merge(trend, how="left", on="customer_id") \
                                   .merge(last_tx[["customer_id", "inactive_days"]], how="left", on="customer_id") \
                                   .merge(prod_events, how="left", on="customer_id")
    out = out.fillna({"spend_cagr3": 0.0, "inactive_days": 999, "product_events_180d": 0})
    return out

def features_policy(customers, ref_date):
    cust = customers.copy()
    cust["join_date"] = pd.to_datetime(cust["join_date"], errors="coerce")
    cust["tenure_months"] = ((ref_date.to_period("M") - cust["join_date"].dt.to_period("M")).apply(lambda x: x.n)) \
                             .fillna(0).astype(int)
    # KYC risk penalty: L=0, M=2, H=4
    tier = cust["kyc_risk_tier"].astype(str).str.upper().map({"L": 0, "LOW": 0, "M": 2, "MEDIUM": 2, "H": 4, "HIGH": 4})
    cust["kyc_penalty"] = tier.fillna(0).astype(int)
    return cust[["customer_id", "tenure_months", "kyc_penalty"]]

# -------------------------
# Scoring
# -------------------------

def minmax01(df, cols, clip=True):
    out = df.copy()
    for c in cols:
        s = _cap(out[c].astype(float)) if clip else out[c].astype(float)
        mn, mx = s.min(), s.max()
        out[c + "_01"] = 0.0 if mn == mx else (s - mn) / (mx - mn)
        out[c + "_01"] = out[c + "_01"].fillna(0.0)
    return out

def build_scores(fe_eng, fe_svc, fe_val, fe_pol):
    df = fe_eng.merge(fe_svc, on="customer_id", how="left") \
               .merge(fe_val, on="customer_id", how="left") \
               .merge(fe_pol, on="customer_id", how="left")

    # Normalize directional features to 0..1
    df = minmax01(df, ["social_sentiment_mean", "social_rate_30d", "response_rate"], clip=True)
    # Lower response time is better → invert after scale
    df = minmax01(df, ["avg_resp_hours"], clip=True)
    df["avg_resp_hours_01"] = 1 - df["avg_resp_hours_01"]

    df = minmax01(df, ["resolution_rate", "fcr_rate"], clip=True)
    # Lower reopen rate is better
    df = minmax01(df, ["reopen_rate"], clip=True)
    df["reopen_rate_01"] = 1 - df["reopen_rate_01"]

    df = minmax01(df, ["spend_cagr3", "product_events_180d"], clip=True)
    # Lower inactivity is better
    df = minmax01(df, ["inactive_days"], clip=True)
    df["inactive_days_01"] = 1 - df["inactive_days_01"]

    # Tenure scaled, KYC penalty mapped later
    df = minmax01(df, ["tenure_months"], clip=True)

    # Weights
    w = {
        # Engagement (30)
        "social_sentiment_mean_01": 12,
        "social_rate_30d_01": 6,
        "response_rate_01": 6,
        "avg_resp_hours_01": 6,
        # Service (25)
        "resolution_rate_01": 10,
        "fcr_rate_01": 8,
        "reopen_rate_01": 7,
        # Value & Momentum (35)
        "spend_cagr3_01": 12,
        "product_events_180d_01": 12,
        "inactive_days_01": 11,
        # Policy Adj. (10) — handled as adjustments below
    }

    # Base score (0..100 before policy)
    total_w = sum(w.values())
    weighted_sum = sum(df[k] * v for k, v in w.items())
    df["base_score"] = (weighted_sum / total_w) * 100.0

    # Policy adjustments
    # tenure bonus up to +6
    df["tenure_bonus"] = df["tenure_months_01"] * 6.0
    # KYC penalty up to -4
    df["kyc_penalty_pts"] = df["kyc_penalty"].clip(0, 4) * 1.0  # already in 0..4

    df["score"] = df["base_score"] + df["tenure_bonus"] - df["kyc_penalty_pts"]
    df["score"] = df["score"].clip(0, 100)

    # Bands
    df["band"] = pd.cut(df["score"],
                        bins=[-0.1, 44, 69, 100],
                        labels=["Red", "Amber", "Green"])

    # Store contributions (for explainability)
    for k, v in w.items():
        df[f"contrib_{k}"] = (df[k] * v / total_w) * 100.0
    df["contrib_tenure"] = (df["tenure_bonus"] / (total_w)) * 100.0
    df["contrib_kyc_penalty"] = -(df["kyc_penalty_pts"] / (total_w)) * 100.0

    return df

# -------------------------
# Plots
# -------------------------

def plot_feature_distributions(df, outdir):
    cols = [
        "social_sentiment_mean", "social_rate_30d", "response_rate", "avg_resp_hours",
        "resolution_rate", "fcr_rate", "reopen_rate",
        "spend_cagr3", "product_events_180d", "inactive_days", "tenure_months"
    ]
    n = len(cols)
    plt.figure(figsize=(12, 10))
    for i, c in enumerate(cols, 1):
        plt.subplot((n + 2) // 3, 3, i)
        df[c].plot(kind="hist", bins=30, alpha=0.8, title=c)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "feature_distributions.png"))
    plt.close()

def plot_score_distribution(df, outdir):
    plt.figure(figsize=(7,5))
    df["score"].plot(kind="hist", bins=30, alpha=0.9, title="Customer Health Score Distribution")
    plt.xlabel("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "score_distribution.png"))
    plt.close()

def plot_band_breakdown(df, outdir):
    plt.figure(figsize=(6,5))
    df["band"].value_counts().sort_index().plot(kind="bar", title="Band Breakdown")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "band_breakdown.png"))
    plt.close()

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Customer Health Score — UAE Banking (Rule-based)")
    ap.add_argument("--customers", required=True)
    ap.add_argument("--social", required=True)
    ap.add_argument("--comms", required=True)
    ap.add_argument("--complaints", required=True)
    ap.add_argument("--txns", required=True)
    ap.add_argument("--products", required=True)
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--ref_date", default=None, help="YYYY-MM-DD; default=today")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ref_date = pd.to_datetime(args.ref_date).to_pydatetime().date() if args.ref_date else datetime.utcnow().date()

    customers = _read_csv(args.customers, parse_dates=["join_date"])
    social = _read_csv(args.social, parse_dates=["date"])
    comms = _read_csv(args.comms, parse_dates=["date"])
    complaints = _read_csv(args.complaints, parse_dates=["opened_at", "closed_at"])
    txns = _read_csv(args.txns, parse_dates=["date"])
    products = _read_csv(args.products, parse_dates=["event_date"])

    # Ensure required columns exist
    for df, name, cols in [
        (customers, "customers.csv", ["customer_id", "join_date", "kyc_risk_tier"]),
        (social, "social_interactions.csv", ["customer_id", "date"]),
        (comms, "communications.csv", ["customer_id", "date", "response_time_hours"]),
        (complaints, "complaints.csv", ["customer_id", "complaint_id", "opened_at", "status", "first_contact_resolution"]),
        (txns, "transactions.csv", ["customer_id", "date", "amount", "category"]),
        (products, "products.csv", ["customer_id", "event_date", "event"]),
    ]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    fe_eng = features_engagement(social, comms, customers, ref_date)
    fe_svc = features_service(complaints, customers, ref_date)
    fe_val = features_value_momentum(txns, products, customers, ref_date)
    fe_pol = features_policy(customers, ref_date)

    scored = build_scores(fe_eng, fe_svc, fe_val, fe_pol)

    # Save outputs
    out_cols = ["customer_id", "score", "band",
                "social_sentiment_mean", "social_rate_30d", "response_rate", "avg_resp_hours",
                "resolution_rate", "fcr_rate", "reopen_rate",
                "spend_cagr3", "product_events_180d", "inactive_days", "tenure_months",
                "contrib_social_sentiment_mean_01", "contrib_social_rate_30d_01",
                "contrib_response_rate_01", "contrib_avg_resp_hours_01",
                "contrib_resolution_rate_01", "contrib_fcr_rate_01", "contrib_reopen_rate_01",
                "contrib_spend_cagr3_01", "contrib_product_events_180d_01",
                "contrib_inactive_days_01", "contrib_tenure", "contrib_kyc_penalty"]
    # Only keep columns that exist (robust to adjustments)
    out_cols = [c for c in out_cols if c in scored.columns]
    scored[out_cols].sort_values("score", ascending=False).to_csv(
        os.path.join(args.output_dir, "customer_health_scores.csv"), index=False
    )

    # Plots
    plot_feature_distributions(scored, args.output_dir)
    plot_score_distribution(scored, args.output_dir)
    plot_band_breakdown(scored, args.output_dir)

    print(f"Saved scores & plots to: {args.output_dir}")

if __name__ == "__main__":
    main()
