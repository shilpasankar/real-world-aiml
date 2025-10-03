# Sales & Waste Prediction — Fresh Food (UAE Retail)

Probabilistic daily demand forecasting (P10/P50/P90) + order optimization to reduce waste
while meeting service levels. Incorporates **promotions, festival periods, and seasonality**.

- ⚙️ Models: SARIMAX (with exogenous features) + Quantile Gradient Boosting (P10/P50/P90)
- 🎯 Policy: Newsvendor-style order-up-to target using forecast quantiles
- 📈 Outputs: Per-item daily forecasts, recommended orders, expected waste & service metrics

---

## 📦 Data (CSV in `data/`)
Minimum required columns in **bold**.

1) `sales.csv`  
   - **date**, **item_id**, units_sold
2) `items.csv`  
   - **item_id**, shelf_life_days, min_stock (optional), max_stock (optional)
3) `promos.csv` (optional)  
   - **date**, **item_id**, promo_flag (0/1), discount_rate (0..1)
4) `calendar.csv` (optional)  
   - **date**, is_weekend (0/1), is_festival (0/1), holiday_name (optional)

> Daily frequency. Missing rows are treated as 0 sales (stockouts not encoded by default).

---

## 🧠 Approach

1) **Feature Engineering**
   - Lags & moving averages (1,7,14,28)
   - Calendar features (dow, weekend, festival)
   - Promo features (flag, discount depth)
2) **Probabilistic Forecasting**
   - **SARIMAX** per item (exogenous: promos, calendar)
   - **Quantile GradientBoosting** for P10/P50/P90 (tree model with `loss='quantile'`)
   - Model blending: median = mean of SARIMAX and GBM P50 (can be toggled)
3) **Order Optimization (Newsvendor)**
   - Choose **service level** α (default 85%) → order up to Pα quantile
   - Respect **min_stock**, **max_stock**, and **shelf_life** (cap horizon)
   - Compute **expected waste proxy** from (order − P50)+ subject to shelf life
4) **Validation**
   - Last `N` days holdout (default 14): MAE/MAPE and **Pinball Loss** at τ∈{0.1,0.5,0.9}
