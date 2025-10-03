
## app.py
```python
import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from dateutil import parser as dtp

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_csv(path, parse_dates=None):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=parse_dates)

def load_data():
    base = "data"
    scores = load_csv(f"{base}/customer_health_scores.csv")
    customers = load_csv(f"{base}/customers.csv", parse_dates=["join_date"])
    complaints = load_csv(f"{base}/complaints.csv", parse_dates=["opened_at","closed_at"])
    txns = load_csv(f"{base}/transactions.csv", parse_dates=["date"])
    products = load_csv(f"{base}/products.csv", parse_dates=["event_date"])
    return scores, customers, complaints, txns, products

# -----------------------------
# Signals & NBAs
# -----------------------------
def recent_window(df, col, days=90, ref=None):
    if df is None: return df
    ref = ref or datetime.utcnow()
    return df[df[col] >= (ref - timedelta(days=days))]

def compute_signals(scores, customers, complaints, txns, products, ref=None):
    ref = ref or datetime.utcnow()
    # Base frame
    base = customers[["customer_id","segment","emirate"]].copy()
    if scores is not None:
        base = base.merge(scores[["customer_id","score","band"]], on="customer_id", how="left")
    else:
        base["score"] = 60.0
        base["band"] = pd.cut(base["score"], [-1,44,69,100], labels=["Red","Amber","Green"])

    # Spend trend (last 90 vs prior 90)
    if txns is not None and len(txns):
        tx = txns.copy()
        tx["date"] = pd.to_datetime(tx["date"])
        last90 = tx[tx["date"] > ref - timedelta(days=90)].groupby("customer_id")["amount"].sum().rename("spend_90")
        prev90 = tx[(tx["date"] <= ref - timedelta(days=90)) & (tx["date"] > ref - timedelta(days=180))].groupby("customer_id")["amount"].sum().rename("spend_prev90")
        spend = pd.concat([last90, prev90], axis=1).fillna(0.0)
        spend["spend_change"] = (spend["spend_90"] - spend["spend_prev90"]) / (spend["spend_prev90"] + 1e-6)
        base = base.merge(spend, left_on="customer_id", right_index=True, how="left")
    else:
        base["spend_90"] = 0.0; base["spend_prev90"] = 0.0; base["spend_change"] = 0.0

    # Open complaints
    if complaints is not None and len(complaints):
        comp = complaints.copy()
        comp["is_open"] = comp["status"].fillna("").str.lower().ne("resolved").astype(int)
        comp_open = comp.groupby("customer_id")["is_open"].sum().rename("open_complaints")
        fcr = comp.groupby("customer_id")["first_contact_resolution"].mean().rename("fcr_rate")
        base = base.merge(pd.concat([comp_open, fcr], axis=1), left_on="customer_id", right_index=True, how="left")
    else:
        base["open_complaints"] = 0; base["fcr_rate"] = 0.0

    # Product gaps (simple): count of upgrades/new in 180d; if zero for Green → upsell opportunity
    if products is not None and len(products):
        pr = products.copy()
        pr_recent = pr[pr["event_date"] >= (ref - timedelta(days=180))]
        prod_events = pr_recent.groupby("customer_id").size().rename("product_events_180d")
        base = base.merge(prod_events, left_on="customer_id", right_index=True, how="left")
    else:
        base["product_events_180d"] = 0

    # Normalize NaNs
    base = base.fillna({"spend_90":0.0,"spend_prev90":0.0,"spend_change":0.0,"open_complaints":0,"fcr_rate":0.0,"product_events_180d":0})
    return base

def rule_based_nba(row):
    reasons = []
    actions = []
    confidence = 0.6

    band = (row.get("band") or "Amber")
    score = float(row.get("score") or 60)
    spend_change = float(row.get("spend_change") or 0.0)
    open_comp = int(row.get("open_complaints") or 0)
    fcr = float(row.get("fcr_rate") or 0.0)
    prod_ev = int(row.get("product_events_180d") or 0)
    segment = row.get("segment") or "Mass"

    # Service risk first
    if open_comp > 0 and band in ["Red","Amber"]:
        actions.append("Escalate service recovery within 24h; waive fee if applicable")
        reasons.append("Unresolved complaint(s)")
        confidence = 0.8 if open_comp >= 2 else 0.7

    # Retention risk
    if band == "Red" or spend_change < -0.25:
        actions.append("Retention offer: targeted discount / concierge callback")
        reasons.append("Churn risk (low health or sharp spend drop)")
        confidence = max(confidence, 0.7)

    # Growth
    if band == "Green" and prod_ev == 0 and spend_change >= 0:
        actions.append("Cross-sell: propose adjacent product bundle")
        reasons.append("Healthy + no recent product activity")
        confidence = max(confidence, 0.65)

    # Nurture / Education
    if not actions:
        actions.append("Nurture: educational content + light incentive")
        reasons.append("Neutral signals")

    # Cap reasons to top 3
    reasons = reasons[:3]
    nba = actions[0]
    return nba, reasons, confidence

def outreach_template(row, nba, reasons):
    seg = row.get("segment") or "Customer"
    emirate = row.get("emirate") or ""
    why = "; ".join(reasons) if reasons else "your recent activity"
    return (f"Hi there — we noticed {why}. Based on your profile, we recommend: {nba}. "
            f"If you'd like, we can follow up with a quick call to tailor this to you.")

# Optional: call an LLM if OPENAI_API_KEY present (fallback to template otherwise).
def llm_summary_stub(row, nba, reasons):
    # You can wire up OpenAI/Anthropic here; keeping a safe fallback for MVP.
    return outreach_template(row, nba, reasons)

def make_actions(df):
    recs = []
    for _, r in df.iterrows():
        nba, reasons, conf = rule_based_nba(r)
        text = llm_summary_stub(r, nba, reasons)
        recs.append({
            "customer_id": r["customer_id"],
            "score": round(float(r.get("score", 0)), 2),
            "band": r.get("band"),
            "segment": r.get("segment"),
            "emirate": r.get("emirate"),
            "spend_change": round(float(r.get("spend_change", 0.0)), 3),
            "open_complaints": int(r.get("open_complaints", 0)),
            "product_events_180d": int(r.get("product_events_180d", 0)),
            "nba": nba,
            "reason_codes": "|".join(reasons),
            "confidence": round(conf, 2),
            "outreach_text": text
        })
    return pd.DataFrame(recs)

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Customer Health Copilot", layout="wide")
st.title("🩺 AI-based Customer Health Copilot (MVP)")

scores, customers, complaints, txns, products = load_data()
if customers is None:
    st.warning("Place CSVs in ./data (see README).")
    st.stop()

ref_date = st.sidebar.date_input("Reference date", value=datetime.utcnow().date())
df = compute_signals(scores, customers, complaints, txns, products, ref=datetime.combine(ref_date, datetime.min.time()))
actions = make_actions(df)

# Filters
col1, col2, col3 = st.columns(3)
with col1:
    band_filter = st.multiselect("Band", options=sorted(actions["band"].dropna().unique()), default=None)
with col2:
    seg_filter = st.multiselect("Segment", options=sorted(actions["segment"].dropna().unique()), default=None)
with col3:
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.6, 0.05)

view = actions.copy()
if band_filter: view = view[view["band"].isin(band_filter)]
if seg_filter: view = view[view["segment"].isin(seg_filter)]
view = view[view["confidence"] >= min_conf]

st.markdown("### Suggested actions")
st.dataframe(view.sort_values(["band","confidence"], ascending=[True, False]).reset_index(drop=True), use_container_width=True)

os.makedirs("outputs", exist_ok=True)
csv_path = "outputs/copilot_actions.csv"
view.to_csv(csv_path, index=False)
st.download_button("⬇️ Download actions CSV", data=view.to_csv(index=False), file_name="copilot_actions.csv", mime="text/csv")

st.markdown("#### Notes")
st.write("- Recommendations are rule+signals driven; add an API key to enable LLM wording.")
st.write("- Always review suggested actions before activation.")
