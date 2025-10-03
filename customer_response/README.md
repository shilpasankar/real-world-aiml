# Customer Response Model (UAE, Qatar, KSA Retail)
**Tree-based feature selection (RFE) + Class weighting + Logistic Regression** to:
1) Predict a customer's **propensity to respond** to a promotion (binary).
2) Rank **categories** the customer is most likely to respond to.

This model consumes features (and labels when available) plus the outputs from your **Promotion Sensitivity Model**.

---

## 🎯 Objectives
- Class-imbalance-aware response model with **calibrated probabilities**.
- Transparent feature selection via **RFE** (tree-derived ranking backing).
- Category interest ranking for **right offer, right product category**.
- Reproducible artifacts: metrics, plots, and scored CSVs.

---

## 🗂️ Data (CSV in `data/`)
Minimum columns in **bold**.

1) `transactions.csv`  
   - **customer_id**, **date**, amount, category, promo_applied (0/1), discount_rate (0..1), region (`UAE`,`Qatar`,`KSA`)
2) `customers.csv`  
   - **customer_id**, join_date, segment
3) `campaign_history.csv` *(optional, recommended)*  
   - **customer_id**, **send_date**, campaign_id, category, offer_value, **responded** (0/1)
4) `promo_sensitivity_predictions.csv` *(from previous project)*  
   - **customer_id**, cluster_label (`Low|Medium|High`), final_label
5) `category_map.csv` *(optional, if you aggregate SKUs)*  
   - sku, category

> **Labeling:** If `campaign_history.responded` exists, we supervise on it. If not, we create **weak labels**: a positive if the customer made a purchase in the **7 days** after a campaign in the same category (can adjust via CLI).

---

## 🧠 Features
- RFM-lite (`R` = days since last txn, `F` = order count, `M` = spend)
- Promo behavior: redemption_rate, avg_discount, promo_lift, basket_delta
- Recency features around latest campaign window (±7/14/30d)
- Category affinities: last-90d spend share by category (top-K columns)
- Region dummies (UAE/Qatar/KSA)
- **Promotion Sensitivity label** (Low/Medium/High) as an input feature

---

## 🔧 Method
1) Build customer-level feature table and (if needed) **weak labels**.
2) **Feature selection (RFE)** driven by a tree (GradientBoosting) to rank features, then apply RFE wrapper around Logistic Regression.
3) Train **Logistic Regression** with **class_weight='balanced'** (or custom priors) and **probability calibration** (isotonic by default).
4) Evaluate on a **time-aware split** (past → train, recent → validation):
   - ROC-AUC, PR-AUC, F1, confusion matrix, calibration curve
   - **Decile gains** table for campaign target sizing
5) **Category Ranking:** one-vs-rest logistic models per category (or shared LR with category share features); output top-N categories per customer.
