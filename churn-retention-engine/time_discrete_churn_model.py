import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import seaborn as sns

# Simulate user data
np.random.seed(42)
n_users = 1000

df = pd.DataFrame({
    'user_id': np.arange(n_users),
    'tenure_months': np.random.randint(1, 36, n_users),
    'monthly_charges': np.round(np.random.uniform(30, 120, n_users), 2),
    'num_complaints': np.random.poisson(1.2, n_users),
    'has_downgraded': np.random.binomial(1, 0.3, n_users),
})

# Simulate churn based on tenure + risk factors
df['churn'] = (
    (df['tenure_months'] < 12).astype(int) * 0.25 +
    (df['num_complaints'] > 2).astype(int) * 0.3 +
    df['has_downgraded'] * 0.3 +
    np.random.normal(0, 0.05, n_users)
) > 0.4
df['churn'] = df['churn'].astype(int)

# 🕒 Time-discrete binning (group users by tenure range)
df['tenure_bin'] = pd.cut(df['tenure_months'], bins=[0, 6, 12, 18, 24, 36], labels=[
    '0-6 mo', '7-12 mo', '13-18 mo', '19-24 mo', '25-36 mo'
])

# 🧪 Train logistic model per bin
results = []
for bin_label in df['tenure_bin'].unique():
    subset = df[df['tenure_bin'] == bin_label]
    if len(subset) < 30:  # avoid small bins
        continue

    X = subset[['monthly_charges', 'num_complaints', 'has_downgraded']]
    y = subset['churn']

    model = LogisticRegression()
    model.fit(X, y)
    y_pred_prob = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, y_pred_prob)
    results.append({
        'tenure_bin': bin_label,
        'roc_auc': auc,
        'coef_monthly_charges': model.coef_[0][0],
        'coef_complaints': model.coef_[0][1],
        'coef_downgrade': model.coef_[0][2]
    })

# 📊 Convert results to DataFrame
df_results = pd.DataFrame(results)

# 📈 Plot AUC and coefficients across tenure
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_results, x='tenure_bin', y='roc_auc', marker='o')
plt.title('ROC AUC per Tenure Group')
plt.ylabel('ROC AUC')
plt.xlabel('Customer Tenure Bin')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
df_results.set_index('tenure_bin')[[
    'coef_monthly_charges', 'coef_complaints', 'coef_downgrade'
]].plot(kind='bar')
plt.title('Feature Coefficients per Tenure Bin')
plt.ylabel('Logistic Coefficient')
plt.grid(True)
plt.tight_layout()
plt.show()
