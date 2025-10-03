import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 🎲 Simulate customer behavioral data
np.random.seed(42)
n_customers = 1000

df = pd.DataFrame({
    'avg_purchase_value': np.random.gamma(2.0, 50, n_customers),
    'num_transactions': np.random.poisson(10, n_customers),
    'promo_response_rate': np.random.beta(2, 5, n_customers),  # promo sensitivity
    'visits_per_month': np.random.normal(4, 1.5, n_customers).clip(0.5),
    'days_since_last_visit': np.random.exponential(30, n_customers),
})

# ⚙️ Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 🧠 PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 🔍 K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 📌 Add cluster and PCA values to DataFrame
df['cluster'] = clusters
df['pca_x'] = X_pca[:, 0]
df['pca_y'] = X_pca[:, 1]

# 📊 Cluster Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='pca_x', y='pca_y', hue='cluster', palette='Set2', s=60)
plt.title("Customer Segmentation via K-Means + PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()

# 📋 Cluster Profile Summary
cluster_summary = df.groupby('cluster').agg({
    'avg_purchase_value': 'mean',
    'num_transactions': 'mean',
    'promo_response_rate': 'mean',
    'visits_per_month': 'mean',
    'days_since_last_visit': 'mean',
    'cluster': 'count'
}).rename(columns={'cluster': 'count'})

print(cluster_summary)
