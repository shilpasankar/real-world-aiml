# AI-based Customer Health Copilot (MVP)

Turn customer health + recent signals into **next-best-actions** with reasons, guardrails, and outreach templates.

## What it does
- Ingests CSVs (customers, transactions, complaints, products, optional precomputed health scores)
- Computes **signals** (risk, opportunity, service issues)
- Generates **Next Best Action (NBA)** with **reason codes** and an **outreach draft**
- Streamlit app for browsing, filtering, and exporting actions

## Data (./data)
- `customer_health_scores.csv` — `customer_id,score,band` (+ optional contributions)
- `customers.csv` — `customer_id,segment,join_date,emirate`
- `complaints.csv` — `customer_id,opened_at,closed_at,status,first_contact_resolution`
- `transactions.csv` — `customer_id,date,amount,category`
- `products.csv` — `customer_id,event_date,product,event`

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py

## requirements.txt
pandas
numpy
matplotlib
scikit-learn
streamlit
python-dateutil
