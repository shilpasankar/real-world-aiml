# Price Volatility Analysis (KSA Oil & Gas)  
**Forecast prices of key chemicals/substances using macroeconomic & regional drivers (time series with exogenous variables).**

## 🎯 Problem
KSA oil & gas operators need short– to mid-term visibility into prices of important chemicals (e.g., methanol, ammonia, MEG) to optimize procurement and hedging. Prices are volatile and influenced by macroeconomic indicators (CPI, PPI, FX, freight), energy benchmarks (Brent), and regional signals (GCC PMI, KSA electricity demand, seasonalities/holidays).

## ✨ Goals
- Build **daily/weekly/monthly** forecasts (configurable) for multiple chemicals.
- Use **time-series models with exogenous regressors** (SARIMAX) + a **tree-boosting baseline**.
- Provide metrics (**MAE/MAPE**) and charts, and export **.csv forecasts** for the next N periods.

## 🗂️ Data (expected CSVs in `data/`)
1) `chemical_prices.csv`
   - `date` (YYYY-MM-DD), `chemical`, `price`
   - Example chemicals: `methanol`, `ammonia`, `MEG` (free-text allowed)
2) `macro_factors.csv`
   - `date`, **exogenous columns** such as:
     - `brent_usd` (Brent crude), `usd_sar` (FX), `cpi_ksa`, `ppi_ksa`, `gcc_pmi`,
     - `freight_index`, `natgas_us`, `electricity_demand_ksa`, `holiday` (0/1), …
   - You can include any additional, consistently dated indicators. Missing values are handled.
> **Frequency:** The script will infer and align frequency from your chemical series. Keep both files at the **same frequency** or higher frequency for `macro_factors.csv` (it will be resampled/forward-filled).

## 🧠 Approach
- **Feature engineering**
  - Time features: month/quarter, seasonality dummies (optional), moving averages, lags.
  - Align exogenous regressors to target frequency, forward-fill small gaps.
- **Models**
  - **SARIMAX** per chemical with selected exogenous drivers.
  - **XGBoost-like baseline via GradientBoostingRegressor** on lagged features (no external dependency).
- **Validation**
  - Expanding-window time split with a **holdout horizon** (default 12 periods).
  - Report **MAE/MAPE** and save diagnostic plots.
------------------------------------------
