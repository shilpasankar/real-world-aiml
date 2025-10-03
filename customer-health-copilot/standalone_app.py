#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-based Customer Health Copilot (standalone)
- No prior project dependencies
- Auto-generates sample data if ./data is missing/empty
- Computes a simple customer health score (0..100)
- Produces Next-Best-Actions (NBA) with reason codes & outreach text
- Saves ./outputs/copilot_actions.csv

Run:
  python customer_health_copilot.py
Optional args:
  python customer_health_copilot.py --data_dir data --output_dir outputs --ref_date 2025-09-30
"""

import os
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------

def ensure_dirs(data_dir: str, output_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


def file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def load_or_generate_sample_data(data_dir: str, now: datetime):
    """
    Loads CSVs if present; else generates small, realistic samples so the script runs OOTB.
    Returns: customers, transactions, complaints, products (DataFrames)
    """
    cust_p = os.path.join(data_dir, "customers.csv")
    txn_p = os.path.join(data_dir, "transactions.csv")
    comp_p = os.path.join(data_dir, "complaints.csv")
    prod_p = os.path.join(data_dir, "products.csv")

    have_any = any(file_exists(p) for p in [cust_p, txn_p, comp_p, prod_p])

    if have_any:
        # Load whatever exists; create empty placeholders for missing
        customers = pd.read_csv(cust_p) if file_exists(cust_p) else pd.DataFrame(columns=["customer_id","segment","join_date","emirate"])
        txns = pd.read_csv(txn_p, parse_dates=["date"]) if file_exists(txn_p) else pd.DataFrame(columns=["customer_id","date","amount","category"])
        complaints = pd.read_csv(comp_p, parse_dates=["opened_at","closed_at"]) if file_exists(comp_p) else pd.DataFrame(columns=["customer_id","opened_at","closed_at","status","first_contact_resolution"])
        products = pd.read_csv(prod_p, parse_dates=["event_date"]) if file_exists(prod_p) else pd.DataFrame(columns=["customer_id","event_date","product","event"])
        return customers, txns, complaints, products

    # --- Generate samples (10 customers, 6 months activity) ---
    rng = np.random.default_rng(42)
    customers = pd.DataFrame({
        "customer_id": [f"C{100+i}" for i in range(10)],
        "segment": rng.choice(["Mass","Affluent"], size=10, p=[0.7,0.3]),
        "join_date": [(now - timedelta(days=int(rng.integers(60, 800)))).date() for _ in range(10)],
        "emirate": rng.choice(["Dubai","Abu Dhabi","Sharjah"], size=10, p=[0.6,0.3,0.1])
    })

    dates = pd.date_range(end=now.date(), periods=180, freq="D")
    tx_rows = []
    for cid in customers["customer_id"]:
        # baseline spend intensity
        mu = rng.uniform(0.2, 0.9)
        for d in dates:
            if rng.random() < mu:  # purchase happens
                amt = float(np.round(rng.uniform(15, 200), 2))
                cat = rng.choice(["Groceries","Fashion","Electronics","Home"], p=[0.55,0.2,0.15,0.1])
                tx_rows.append([cid, d, amt, cat])
    txns = pd.DataFrame(tx_rows, columns=["customer_id","date","amount","category"])

    # Inject some churn signals (recent spend drop) for a few customers
    for cid in customers.sample(3, random_state=1)["customer_id"]:
        mask = (txns["customer_id"] == cid) & (txns["date"] > now - timedelta(days=60))
        txns.loc[mask, "amount"] *= 0.35  # recent drop

    # Complaints
    comp_rows = []
    for cid in customers.sample(4, random_state=2)["customer_id"]:
        opened = now - timedelta(days=int(rng.integers(10, 120)))
        status = rng.choice(["resolved","open","reopened"], p=[0.5,0.35,0.15])
        closed = (opened + timedelta(days=int(rng.integers(1, 14)))) if status == "resolved" else pd.NaT
        fcr = 1 if (status == "resolved" and rng.random() < 0.6) else 0
        comp_rows.append([cid, opened.date(), pd.NaT if pd.isna(closed) else closed.date(), status, fcr])
    complaints = pd.DataFrame(comp_rows, columns=["customer_id","opened_at","closed_at","status","first_contact_resolution"])

    # Products (new/upgrade in last 180d)
    prod_rows = []
    for cid in customers.sample(6, random_state=3)["customer_id"]:
        for _ in range(int(rng.integers(1, 3))):
            evd = now - timedelta(days=int(rng.integers(10, 180)))
            prod = rng.choice(["cc","loan","account","fx","bnpl"])
            ev = rng.choice(["new","upgrade","cross_sell"], p=[0.6,0.2,0.2])
            prod_rows.append([cid, evd.date(), prod, ev])
    products = pd.DataFrame(prod_rows, columns=["customer_id","event_date","product","event"])

    # Save samples so user can inspect
    customers.to_csv(cust_p, index=False)
    txns.to_csv(txn_p, index=False)
    complaints.to_csv(comp_p, index=False)
    products.to_csv(prod_p, index=False)

    return customers, txns, complaints, products


# -----------------------------
# Feature engineering & Health score
# -----------------------------

def compute_rfm(txns: pd.DataFrame, ref: datetime) -> pd.DataFrame:
    if txns.empty:
        return pd.DataFrame(columns=["customer_id","R","F","M"])
    g = txns.groupby("customer_id")
    last_dt = g["date"].max().rename("last_txn")
    recency = (pd.to_datetime(ref.date()) - last_dt.dt.date).dt.days.rename("R")
    freq = g.size().rename("F")
    monetary = g["amount"].sum().rename("M")
    rfm = pd.concat([recency, freq, monetary], axis=1).reset_index()
    # Robust caps
    rfm["R"] = rfm["R"].clip(lower=0, upper=np.nanpercentile(rfm["R"], 99) if len(rfm) else 0)
    rfm["F"] = rfm["F"].clip(upper=np.nanpercentile(rfm["F"], 99) if len(rfm) else 0)
    rfm["M"] = rfm["M"].clip(upper=np.nanpercentile(rfm["M"], 99) if len(rfm) else 0)
    return rfm


def spend_change_90(txns: pd.DataFrame, ref: datetime) -> pd.DataFrame:
    if txns.empty:
        return pd.DataFrame(columns=["customer_id","spend_90","spend_prev90","spend_change"])
    last90 = txns[txns["date"] > ref - timedelta(days=90)].groupby("customer_id")["amount"].sum().rename("spend_90")
    prev90 = txns[(txns["date"] <= ref - timedelta(days=90)) & (txns["date"] > ref - timedelta(days=180))].groupby("customer_id")["amount"].sum().rename("spend_prev90")
    df = pd.concat([last90, prev90], axis=1).fillna(0.0).reset_index()
    df["spend_change"] = (df["spend_90"] - df["spend_prev90"]) / (df["spend_prev90"] + 1e-6)
    return df


def complaints_features(complaints: pd.DataFrame) -> pd.DataFrame:
    if complaints.empty:
        return pd.DataFrame(columns=["customer_id","open_complaints","fcr_rate"])
    comp = complaints.copy()
    comp["status"] = comp["status"].astype(str).str.lower()
    comp["is_open"] = (comp["status"] != "resolved").astype(int)
    grp = comp.groupby("customer_id")
    open_c = grp["is_open"].sum().rename("open_complaints")
    fcr = grp["first_contact_resolution"].mean().rename("fcr_rate")
    return pd.concat([open_c, fcr], axis=1).reset_index()


def product_events_180(products: pd.DataFrame, ref: datetime) -> pd.DataFrame:
    if products.empty:
        return pd.DataFrame(columns=["customer_id","product_events_180d"])
    pr = products[products["event_date"] >= (ref - timedelta(days=180))].copy()
    cnt = pr.groupby("customer_id").size().rename("product_events_180d").reset_index()
    return cnt


def scale_0_1(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    s = series.astype(float)
    mn, mx = s.min(), s.max()
    if np.isclose(mx, mn):
        return pd.Series(0.5, index=s.index)  # neutral if no spread
    return (s - mn) / (mx - mn)


def compute_health(customers: pd.DataFrame,
                   txns: pd.DataFrame,
                   complaints: pd.DataFrame,
                   products: pd.DataFrame,
                   ref: datetime) -> pd.DataFrame:
    """
    Simple transparent health score:
      base = 100*( 0.3*(1-R01) + 0.25*F01 + 0.25*M01 + 0.10*(spend_change01) + 0.10*(1-open01) )
      adjustments: - up to 10 points for high open complaints; + up to 5 for product activity
    """
    base = customers[["customer_id","segment","emirate"]].copy()

    rfm = compute_rfm(txns, ref)
    sc = spend_change_90(txns, ref)
    cf = complaints_features(complaints)
    pe = product_events_180(products, ref)

    df = base.merge(rfm, on="customer_id", how="left") \
             .merge(sc, on="customer_id", how="left") \
             .merge(cf, on="customer_id", how="left") \
             .merge(pe, on="customer_id", how="left")

    df = df.fillna({"R":90, "F":0, "M":0.0, "spend_90":0.0, "spend_prev90":0.0,
                    "spend_change":0.0, "open_complaints":0, "fcr_rate":0.0, "product_events_180d":0})

    # Normalize
    R01 = scale_0_1(df["R"]); F01 = scale_0_1(df["F"]); M01 = scale_0_1(df["M"])
    SC01 = scale_0_1(df["spend_change"])
    OPEN01 = scale_0_1(df["open_complaints"])

    base_score = 100.0 * (
        0.30 * (1 - R01) +   # lower R (recent activity) is healthier
        0.25 * F01 +
        0.25 * M01 +
        0.10 * SC01 +
        0.10 * (1 - OPEN01) # fewer open complaints is healthier
    )

    # Adjustments
    adj = (- np.minimum(df["open_complaints"], 3) * 3.0) + np.minimum(df["product_events_180d"], 3) * 1.5
    score = np.clip(base_score + adj, 0, 100)

    # Bands
    band = pd.cut(score, bins=[-0.1, 44, 69, 100], labels=["Red","Amber","Green"])

    out = df.copy()
    out["score"] = score.round(2)
    out["band"] = band
    return out


# -----------------------------
# NBAs (rules) & Outreach
# -----------------------------

def rule_based_nba(row: pd.Series):
    """
    Produce a Next-Best-Action, reasons, and a confidence score (0..1).
    Rules prioritize service recovery, then retention, then growth, then nurture.
    """
    reasons = []
    actions = []
    conf = 0.6

    band = str(row.get("band") or "Amber")
    score = float(row.get("score") or 60.0)
    spend_change = float(row.get("spend_change") or 0.0)
    open_comp = int(row.get("open_complaints") or 0)
    fcr = float(row.get("fcr_rate") or 0.0)
    prod_ev = int(row.get("product_events_180d") or 0)

    # Service recovery
    if open_comp > 0 and band in ["Red","Amber"]:
        actions.append("Escalate service recovery within 24h; prioritize resolution")
        reasons.append("SERVICE_OPEN")
        conf = max(conf, 0.75 if open_comp >= 2 else 0.7)

    # Retention risk
    if band == "Red" or spend_change < -0.25:
        actions.append("Retention outreach with personalized incentive")
        reasons.append("CHURN_RISK")
        conf = max(conf, 0.7)

    # Growth opportunity
    if band == "Green" and prod_ev == 0 and spend_change >= 0:
        actions.append("Cross-sell adjacent product bundle")
        reasons.append("GROWTH_OPP")
        conf = max(conf, 0.65)

    # Nurture fallback
    if not actions:
        actions.append("Nurture: educational content and light offer")
        reasons.append("NEUTRAL")

    reasons = reasons[:3]
    return actions[0], reasons, round(conf, 2)


def outreach_text(row: pd.Series, nba: str, reasons: list) -> str:
    why_map = {
        "SERVICE_OPEN": "unresolved service issue",
        "CHURN_RISK": "recent activity signals indicating possible churn",
        "GROWTH_OPP": "great recent engagement and a chance to add value",
        "NEUTRAL": "your recent activity"
    }
    whys = [why_map.get(r, r.lower()) for r in reasons]
    why = "; ".join(whys) if whys else "your recent activity"
    return (f"Hi there — we noticed {why}. We recommend: {nba}. "
            f"Reply to this message if you'd like us to tailor this to you.")


def build_actions(health_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in health_df.iterrows():
        nba, reasons, conf = rule_based_nba(r)
        text = outreach_text(r, nba, reasons)
        rows.append({
            "customer_id": r["customer_id"],
            "segment": r.get("segment", ""),
            "emirate": r.get("emirate", ""),
            "score": float(r.get("score", 0.0)),
            "band": r.get("band", ""),
            "spend_change": float(r.get("spend_change", 0.0)),
            "open_complaints": int(r.get("open_complaints", 0)),
            "product_events_180d": int(r.get("product_events_180d", 0)),
            "nba": nba,
            "reason_codes": "|".join(reasons),
            "confidence": conf,
            "outreach_text": text
        })
    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="AI-based Customer Health Copilot (standalone)")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--ref_date", default=None, help="YYYY-MM-DD (default=today)")
    args = ap.parse_args()

    ensure_dirs(args.data_dir, args.output_dir)
    ref = datetime.strptime(args.ref_date, "%Y-%m-%d") if args.ref_date else datetime.utcnow()

    # Load or create data
    customers, txns, complaints, products = load_or_generate_sample_data(args.data_dir, ref)

    # Parse/ensure date types
    if not txns.empty:
        txns["date"] = pd.to_datetime(txns["date"])
    if not complaints.empty:
        complaints["opened_at"] = pd.to_datetime(complaints["opened_at"], errors="coerce")
        complaints["closed_at"] = pd.to_datetime(complaints["closed_at"], errors="coerce")
    if not products.empty:
        products["event_date"] = pd.to_datetime(products["event_date"], errors="coerce")

    # Compute health + signals
    health = compute_health(customers, txns, complaints, products, ref)

    # Create NBAs
    actions = build_actions(health)

    # Save outputs
    out_path = os.path.join(args.output_dir, "copilot_actions.csv")
    actions.sort_values(["band","confidence"], ascending=[True, False]).to_csv(out_path, index=False)

    # Print quick summary
    print("\n=== Customer Health Copilot (MVP) ===")
    print(f"Reference date: {ref.date()}")
    print(f"Customers scored: {len(actions):,}")
    print("Band breakdown:")
    print(actions["band"].value_counts(dropna=False).to_string())
    print("\nTop suggested actions:")
    print(actions["nba"].value_counts().to_string())
    print(f"\nSaved actions to: {out_path}\n")


if __name__ == "__main__":
    main()
