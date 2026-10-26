import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Ensure output directories exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("docs", exist_ok=True)

# ---------------------------------------------------------
# Part 1: Feature Engineering (Customers)
# ---------------------------------------------------------
print("Loading raw datasets...")
customers = pd.read_csv("data/raw/customers.csv")
orders = pd.read_csv("data/raw/orders.csv")
reviews = pd.read_csv("data/raw/reviews.csv")

# Convert timestamps
orders['order_timestamp'] = pd.to_datetime(orders['order_timestamp'])
customers['signup_date'] = pd.to_datetime(customers['signup_date'])

# Extract order-level behavioral flags
orders['is_weekend'] = orders['order_timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
orders['is_late_night'] = orders['order_timestamp'].dt.hour.isin([22, 23, 0, 1, 2, 3, 4]).astype(int)

# Merge ratings to orders
orders_with_reviews = orders.merge(reviews[['order_id', 'rating']], on='order_id', how='left')

print("Engineering customer behavioral features...")
customer_agg = orders_with_reviews.groupby('customer_id').agg(
    total_orders=('order_id', 'count'),
    avg_order_value=('order_amount', 'mean'),
    total_spending=('order_amount', 'sum'),
    weekend_orders=('is_weekend', 'sum'),
    late_night_orders=('is_late_night', 'sum'),
    avg_rating_given=('rating', 'mean')
).reset_index()

# Merge with demographic data
customer_features = customers[['customer_id', 'age', 'is_premium_member']].merge(
    customer_agg, on='customer_id', how='left'
)

# Impute missing values (for customers with no reviews or orders)
customer_features['total_orders'] = customer_features['total_orders'].fillna(0)
customer_features['avg_order_value'] = customer_features['avg_order_value'].fillna(0)
customer_features['total_spending'] = customer_features['total_spending'].fillna(0)
customer_features['weekend_orders'] = customer_features['weekend_orders'].fillna(0)
customer_features['late_night_orders'] = customer_features['late_night_orders'].fillna(0)
customer_features['avg_rating_given'] = customer_features['avg_rating_given'].fillna(customer_features['avg_rating_given'].median())

# Calculate ordering frequency (orders relative to tenure)
customer_features['ordering_frequency'] = np.round(customer_features['total_orders'] / 12.0, 2)

# Save feature engineered data
customer_features.to_csv("data/processed/customer_features.csv", index=False)
print("Saved feature dataset to data/processed/customer_features.csv")

# ---------------------------------------------------------
# Part 2: Feature Selection & Scaling
# ---------------------------------------------------------
# Separate identifiers and target clustering columns
feature_cols = [
    'total_orders', 'avg_order_value', 'total_spending',
    'ordering_frequency', 'avg_rating_given', 'weekend_orders', 'late_night_orders'
]

X = customer_features[feature_cols].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# Part 3: K-Means (Elbow Method & Silhouette Analysis)
# ---------------------------------------------------------
print("Evaluating K-Means cluster candidates (k=2 to 8)...")
inertia = []
silhouette_scores = []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, km.labels_))

# Plot Elbow & Silhouette
fig, ax1 = plt.subplots(figsize=(8, 4))

color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Elbow Method)', color=color)
ax1.plot(k_range, inertia, 'o--', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Silhouette Score', color=color)
ax2.plot(k_range, silhouette_scores, 's-', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('K-Means: Elbow Method and Silhouette Analysis')
fig.tight_layout()
plt.savefig("docs/elbow_silhouette_plot.png")
plt.close()
print("Saved evaluation plot to docs/elbow_silhouette_plot.png")

# Optimal k selected (k=3 provides clean segmentation without fragmentation)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_features['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

print(f"Optimal K-Means Silhouette Score (k={optimal_k}): {silhouette_score(X_scaled, kmeans.labels_):.3f}")

# ---------------------------------------------------------
# Part 4: DBSCAN (Density-Based Clustering & Outlier Detection)
# ---------------------------------------------------------
# eps: neighborhood radius, min_samples: minimum points to form a dense region
dbscan = DBSCAN(eps=1.5, min_samples=8)
customer_features['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

noise_count = (customer_features['dbscan_cluster'] == -1).sum()
dense_clusters = len(set(customer_features['dbscan_cluster'])) - (1 if -1 in customer_features['dbscan_cluster'] else 0)
print(f"DBSCAN detected {dense_clusters} dense clusters and {noise_count} outlier/noise points.")

# ---------------------------------------------------------
# Part 5: PCA & Visualization
# ---------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
pca_transformed = pca.fit_transform(X_scaled)

customer_features['pca_1'] = pca_transformed[:, 0]
customer_features['pca_2'] = pca_transformed[:, 1]

plt.figure(figsize=(9, 6))
sns.scatterplot(
    x='pca_1', y='pca_2',
    hue='kmeans_cluster',
    palette='Set2',
    data=customer_features,
    s=50,
    alpha=0.8
)
plt.title(f'Customer Segments Visualized via PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.savefig("docs/pca_clusters.png")
plt.close()
print("Saved PCA visualization to docs/pca_clusters.png")

# ---------------------------------------------------------
# Part 6: Cluster Profiling & Business Personas
# ---------------------------------------------------------
profile = customer_features.groupby('kmeans_cluster')[feature_cols].mean().round(2)
print("\n" + "="*70)
print("CLUSTER PROFILES (MEANS PER FEATURE):")
print("="*70)
print(profile.to_string())

# Dynamic business persona mapping
print("\n" + "="*70)
print("ACTIONABLE BUSINESS CLUSTER INTERPRETATION:")
print("="*70)
for c_id, row in profile.iterrows():
    if row['total_spending'] == profile['total_spending'].max():
        persona = "Premium Frequent Diners"
        desc = "High spend, consistent order frequency, highly valuable cohort. Target with Uber One loyalty perks."
    elif row['late_night_orders'] == profile['late_night_orders'].max() or row['weekend_orders'] == profile['weekend_orders'].max():
        persona = "Late-Night & Weekend Convenience Seekers"
        desc = "Concentrated demand around weekend nights. Prime target for late-night promo alerts and bundled offers."
    else:
        persona = "Occasional Budget Customers"
        desc = "Low-to-moderate order counts and lower ticket size. Target with re-engagement coupons and free delivery threshold incentives."
    
    print(f"Cluster {c_id} -> '{persona}'")
    print(f"  Profile: Total Orders={row['total_orders']}, Total Spend=${row['total_spending']}, Avg AOV=${row['avg_order_value']}")
    print(f"  Strategy: {desc}\n")

# Save segmented customers with assignments
customer_features.to_csv("data/processed/customer_segments_final.csv", index=False)
print("Day 2 execution complete. Deliverables saved.")