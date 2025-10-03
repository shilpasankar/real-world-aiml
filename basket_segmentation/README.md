# Cuisine Preference Segmentation (UAE, Qatar, KSA)  
**RFM + XGBoost + Rule-based overrides** to classify customers into 4 cuisine **preference** segments based on their purchase baskets.

> 🔐 **Ethics:** This project **does not** infer or label ethnicity. Segments are cuisine-preference clusters derived from transaction behavior (e.g., ingredient/meal tags). Use for personalization and merchandising—not for sensitive profiling.

---

## 🎯 Problem
Retailers want to personalize offers, assortments, and content by likely **cuisine preference**. We build an interpretable pipeline that:
- Engineers **RFM** features and **basket cuisine mix** from SKU tags
- Trains a **classifier (XGBoost)** to predict 4 preference segments (A/B/C/D)
- Applies **business rules** as transparent overrides (confidence & dominance thresholds)
- Produces customer-level scores, labels, diagnostics, and explainability artifacts

---

## 🗂️ Data (CSV in `data/`)
1) `transactions.csv`
   - **customer_id**, **date**, **sku**, qty, amount
2) `sku_cuisine_map.csv`
   - **sku**, **cuisine_tag** (string; e.g., `arabic`, `south_asian`, `western`, `east_asian`, …)
3) (Optional) `labels.csv` — if you already have known preference labels for training/eval
   - **customer_id**, **pref_segment** in `{A,B,C,D}`

> The script can **self-label** using basket dominance if `labels.csv` is absent.

---

## 🧠 Features & Labels
- **RFM**: recency (days), frequency (txn count), monetary (spend)
- **Basket cuisine mix**: share of spend per cuisine_tag
- **Diversity**: Herfindahl/HHI over cuisine shares
- **Labeling**:
  - **If `labels.csv` is provided** → supervised learning  
  - **Else** → auto label by dominant cuisine cluster:
    - Compute cuisine shares; assign A/B/C/D by top-K mapping with stable sort

---

## 🔧 Model
- **XGBoost** (multi-class) with class weights
- Train/valid split by **time** (past → train, recent → validation)
- Metrics: accuracy, macro-F1; confusion matrix plot; feature importances

---

## 🧩 Rule-based Engine (Overrides)
- If top cuisine share ≥ **0.60** and model confidence < **0.55** → override to dominant cuisine segment
- If customer **low activity** (R < 90 days & F < 3 & M < threshold) → set label = **“Cold/Unknown”** (optional flag)
- Thresholds configurable via CLI

