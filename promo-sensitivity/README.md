# Promotion Sensitivity Model (UAE, Qatar, KSA)  
**K-Means + Decision Tree + Rule-based Engine** to identify customers who are sensitive to price discounts and promotions.

> Use this to prioritize win-back campaigns, tailor discount depth, and protect margin via guardrails.

---

## 🎯 Objectives
- **Cluster** customers by promo sensitivity using behavioral features.
- **Train** a lightweight **Decision Tree** to operationalize cluster labels for new data.
- **Apply business rules** (dominance, margin, abuse/cooldown) to finalize a stable segment.
- Deliver explainable outputs: per-customer label, scores, and profile summaries.

---

## 🗂️ Data (CSV in `data/`)
Minimum required columns in **bold**.

1) `transactions.csv`  
   - **customer_id**, **date**, **amount**, promo_applied (0/1), discount_rate (0..1), channel, region (`UAE`,`Qatar`,`KSA`)
2) `customers.csv`  
   - **customer_id**, join_date, segment (`Mass`,`Affluent`, …)
3) (Optional) `returns.csv`  
   - **customer_id**, **date**, return_amount
4) (Optional) `margins.csv`  
   - **customer_id**, **date**, gross_margin

> **Frequency**: Transaction grain (daily). Script aggregates to customer-level features.  
> **Date format**: `YYYY-MM-DD`.

---

## 🧠 Features (per customer)
- **Promo redemption rate**: share of orders with `promo_applied=1`
- **Avg discount depth** when used: mean `discount_rate` on promo orders
- **Promo lift**: spend rate when on promo vs off-promo (ratio)
- **Elasticity proxy**: Δorder_count around promo windows vs baseline
- **Time since last promo redemption** (days)
- **Basket delta on promo**: avg basket size on promo − off-promo
- **Return rate** (optional) and **avg margin** (optional)
- **Recency/Frequency/Monetary** (RFM-lite)

---

## 🔧 Method
1) **Unsupervised**: K-Means on standardized features → 3 clusters (Low / Medium / High sensitivity) labeled by profile inspection.
2) **Supervised**: Train a **DecisionTreeClassifier** to predict the K-Means label for new customers; export feature importance.
3) **Rules (overrides)**:
   - If `redemption_rate ≥ 0.7` **and** `avg_discount ≥ 0.15` → force **High**
   - If `promo_lift ≤ 1.05` **and** `redemption_rate ≤ 0.2` → force **Low**
   - If `avg_margin ≤ margin_floor` or `return_rate ≥ return_ceiling` → downgrade one level
   - **Cooldown**: if `days_since_last_promo < cooldown_days` → keep current band (optional flag)
