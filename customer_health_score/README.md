# Customer Health Score — UAE Banking (Rule-based Engine)

Calculate a **customer health score (0–100)** that reflects relationship strength and risk, using:
- **Engagement** across social media & direct channels
- **Complaints**: resolution, speed, and reopen rate
- **Financial behavior**: spend trend, product uptake/upsell, inactivity
- **Lifecycle context**: tenure and recent churn signals

Designed for **UAE retail banking** teams to prioritize outreach, retain high-value customers, and spot early churn risk.

---

## 🎯 Objectives
- A transparent, **rule-based** scoring model (auditable, controllable)
- Input from multiple channels, unified at the **customer_id** level
- Outputs:
  - Per-customer **score** (0–100)
  - **Band**: Green / Amber / Red
  - Feature-level contributions for explainability
  - CSVs + diagnostic charts (PNG)

---

## 🗂️ Data (expected CSVs in `data/`)
Minimum required columns are bold.

1) `customers.csv`
   - **customer_id**, segment, join_date, emirate, kyc_risk_tier (L/M/H)

2) `social_interactions.csv`
   - **customer_id**, **date**, channel (`twitter`,`instagram`,`facebook`), sentiment (`-1..1`), is_public (0/1)

3) `communications.csv` (bank-initiated or inbound non-social)
   - **customer_id**, **date**, channel (`email`,`sms`,`call`,`chat`), response_time_hours, is_inbound (0/1)

4) `complaints.csv`
   - **customer_id**, **complaint_id**, **opened_at**, closed_at, status (`resolved`,`open`,`reopened`), first_contact_resolution (0/1)

5) `transactions.csv`
   - **customer_id**, **date**, amount, category (`cards`,`transfers`,`fees`), is_debit (0/1)

6) `products.csv`
   - **customer_id**, **event_date**, product (`cc`,`loan`,`account`,`fx`,`bnpl`), event (`new`,`upgrade`,`cross_sell`)

> **Notes**
> - Dates: `YYYY-MM-DD`
> - Monthly frequency recommended for stability; script aggregates automatically.
> - Missing fields are handled via sensible defaults and caps.

---

## 🧠 Scoring Approach (Rule-based → 0..100)
We compute normalized features, apply **weights**, and sum to a **base score**; then apply **policy adjustments** (KYCRisk, tenure). Finally we map to **bands**.

| Dimension | Feature | Direction | Weight |
|---|---|---:|---:|
| **Engagement (30)** | Social sentiment (last 90d, mean) | ↑ | 12 |
|  | Social interaction rate (posts/mentions per 30d) | ↑ (with cap) | 6 |
|  | Direct comms response rate (responded/initiated) | ↑ | 6 |
|  | Avg response time (hours) | ↓ | 6 |
| **Service (25)** | Complaint resolution rate (last 180d) | ↑ | 10 |
|  | First contact resolution rate | ↑ | 8 |
|  | Reopen rate | ↓ | 7 |
| **Value & Momentum (35)** | Spend trend (3-month CAGR on card/transfers) | ↑ | 12 |
|  | Product events (new/upgrade/cross-sell count, last 180d) | ↑ | 12 |
|  | Inactivity days (no txn last N days) | ↓ | 11 |
| **Policy Adj. (10)** | Tenure (months) | ↑ (cap) | +0..6 |
|  | KYC risk tier (H/M/L) | penalty | −0..4 |

- Feature engineering transparently scales each component to **0..1** via min-max with caps (robust to outliers).
- Final score clipped to **0..100**.

**Bands**
- **Green**: 70–100 → healthy & growing  
- **Amber**: 45–69 → watchlist & nurture  
- **Red**: 0–44 → risk & proactive retention

---

## 🔐 Responsible Use (UAE Context)
- Avoid using sensitive attributes directly (religion, nationality, etc.).  
- Keep explainability: store **feature contributions** per customer.  
- Ensure **opt-outs** for social data and comply with platform policies & UAE privacy law.  
- Human-in-the-loop for outreach decisions.

---

