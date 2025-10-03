import pandas as pd
import numpy as np

# 🎲 Step 1: Simulate a mini customer dataset
np.random.seed(42)
n_customers = 20

df = pd.DataFrame({
    'customer_id': np.arange(1001, 1001 + n_customers),
    'churn_score': np.round(np.random.uniform(0, 1, n_customers), 2),  # 0 to 1
    'promo_sensitivity': np.random.choice(['High', 'Medium', 'Low'], n_customers),
    'customer_lifetime_value': np.round(np.random.uniform(200, 2000, n_customers), 2),
})

# 🎯 Step 2: Rule-based function to assign offers
def assign_offer(row):
    if row['churn_score'] > 0.8:
        if row['promo_sensitivity'] == 'High':
            return "50% Discount"
        elif row['promo_sensitivity'] == 'Medium':
            return "30% Discount"
        else:
            return "15% Discount"
    
    elif row['churn_score'] > 0.6:
        if row['customer_lifetime_value'] > 1500:
            return "Free Upgrade + 20% Discount"
        else:
            return "20% Discount"
    
    elif row['churn_score'] > 0.4:
        if row['promo_sensitivity'] == 'High':
            return "Targeted Promo Code"
        else:
            return "Loyalty Points Top-up"

    else:
        return "No Offer (Healthy)"

# 🧠 Step 3: Apply logic
df['recommended_offer'] = df.apply(assign_offer, axis=1)

# 📊 Show the mapped offers
print(df[['customer_id', 'churn_score', 'promo_sensitivity', 'customer_lifetime_value', 'recommended_offer']])
