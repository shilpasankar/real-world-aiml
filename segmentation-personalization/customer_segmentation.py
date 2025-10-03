import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Simulate customer behavioral data
np.random.seed(42)
n_customers = 500

df = pd.DataFrame({
    'customer_id': np.arange(1001, 1001 + n_customers),
    'avg_purchase_value': np.random.gamma(2.0, 50, n_customers),
    'num_transactions': np.random.poisson(10, n_customers),
    'promo_response_rate': np.random.beta(2, 5, n_customers),  # Promo sensitivity
    'visits_per_month': np.random.normal(4, 1.5, n_customers).clip(0.5),
    'days_since_last_visit': np.random.exponential(30, n_customers),
})

# Step 2: Data preprocessing - standardize features
features = ['avg_purchase_value', 'num_transactions', 'promo_response_rate', 'visits_per_month', 'days_since_last_visit']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Step 3: PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 4: K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters
df['pca_x'] = X_pca[:, 0]
df['pca_y'] = X_pca[:, 1]

# Step 5: Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='pca_x', y='pca_y', hue='cluster', palette='Set2', s=50)
plt.title('Customer Segmentation using K-Means + PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 6: Cluster profiling
cluster_summary = df.groupby('cluster')[features].mean().round(2)
print("Cluster Summary:")
print(cluster_summary)

# Step 7: Rule-based Offer Matching Engine
def assign_offer(row):
    if row['promo_response_rate'] > 0.5:
        return "20% Discount Coupon"
    elif row['avg_purchase_value'] > 80:
        return "Loyalty Program Upgrade"
    elif row['days_since_last_visit'] > 60:
        return "Win-back Offer"
    else:
        return "No Offer"

df['recommended_offer'] = df.apply(assign_offer, axis=1)

# Step 8: Show sample with offers
print("\nSample customer offers:")
print(df[['customer_id', 'cluster', 'recommended_offer']].head(10))

# Optional: Save to CSV for further use
df.to_csv('segmentation_personalization_results.csv', index=False)
