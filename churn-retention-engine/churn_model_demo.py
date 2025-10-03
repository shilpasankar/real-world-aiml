# 📦 Dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 🎲 Step 1: Simulate Data
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'tenure_months': np.random.randint(1, 36, n_samples),
    'monthly_charges': np.round(np.random.uniform(20, 120, n_samples), 2),
    'num_complaints': np.random.poisson(1, n_samples),
    'has_downgraded': np.random.binomial(1, 0.3, n_samples),
    'received_offer': np.random.binomial(1, 0.5, n_samples),
    'promo_sensitive': np.random.binomial(1, 0.4, n_samples),
})

# Simulate churn based on risk factors
churn_prob = (
    0.2 * (df['tenure_months'] < 12).astype(int) +
    0.3 * (df['num_complaints'] > 2).astype(int) +
    0.3 * df['has_downgraded'] +
    np.random.normal(0, 0.05, n_samples)
)
df['churn'] = (churn_prob > 0.4).astype(int)

# 📊 Step 2: Train-Test Split
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🤖 Step 3: Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 🧪 Step 4: Predictions + Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# 🔍 Step 5: Feature Importance
feature_importance = pd.Series(model.coef_[0], index=X.columns).sort_values()
print("\nFeature Importances:\n", feature_importance)

# 📉 Step 6: Visualizations
plt.figure(figsize=(12, 5))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Feature Importance
plt.subplot(1, 2, 2)
feature_importance.plot(kind='barh', color='teal')
plt.title("Feature Importance (Logistic Regression)")
plt.tight_layout()
plt.show()
