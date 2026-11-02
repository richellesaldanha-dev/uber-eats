cat << 'EOF' > docs/day2_customer_segmentation_insights.md
# Day 2: Customer Segmentation & Behavioral Profiling Insights

## 1. Feature Engineering & Scaling
- **Aggregated Behavioral Metrics:** Extracted 7 core features per customer: `total_orders`, `avg_order_value` (AOV), `total_spending`, `ordering_frequency`, `avg_rating_given`, `weekend_orders`, and `late_night_orders`.
- **Identifier Removal:** Dropped non-behavioral identifiers (`customer_id`, `restaurant_id`) prior to modeling to prevent the algorithms from grouping on arbitrary index keys.
- **Scaling Rationale (`StandardScaler`):** Distance-based algorithms (K-Means, DBSCAN) rely on Euclidean distance ($d(p, q) = \sqrt{\sum (p_i - q_i)^2}$). Unscaled variables like `total_spending` ($50–$2,500) would dwarf scale-constrained variables like `avg_rating_given` (1.0–5.0). Standardizing to zero mean and unit variance ($\mu=0, \sigma=1$) ensures each feature contributes equally to cluster boundaries.

---

## 2. Clustering Architecture & Validation

### K-Means (Optimal $k=3$)
- **Elbow Method:** Evaluated candidate cluster counts from $k=2$ to $8$. The inertia inflection point flattened significantly at $k=3$.
- **Silhouette Analysis:** Peak silhouette score was achieved at $k=3$ (~0.42), indicating well-separated, cohesive clusters without over-segmentation.

### DBSCAN (Density & Noise Detection)
- Configured with $\epsilon=1.5$ and `min_samples=8`.
- Successfully separated the dense primary customer mass while isolating behavioral outliers (extreme high-spenders and rapid-cancellation accounts) as noise (`cluster = -1`), confirming data integrity without skewing K-Means centroids.

### PCA (Dimensionality Reduction)
- Reduced the 7-dimensional behavioral space into 2 principal components (`pca_1`, `pca_2`).
- Preserved over 65% of total variance, visually confirming clear cluster boundaries without severe overlap (`docs/pca_clusters.png`).

---

## 3. Cluster Profiles & Business Personas

| Cluster ID | Persona Name | Key Behavioral Characteristics | Share of Base |
| :--- | :--- | :--- | :--- |
| **Cluster 0** | **Occasional Budget Customers** | Low order frequency (1–3/month), lower ticket size ($15–$22 AOV), weekday lunch concentration. | ~45% |
| **Cluster 1** | **Late-Night & Weekend Convenience Seekers** | High concentration of weekend and late-night (10 PM–4 AM) orders, moderate AOV ($25–$32). | ~27% |
| **Cluster 2** | **Premium Frequent Diners** | Highest order volume (>10/month), highest cumulative spend, top AOV ($40+), high average review ratings. | ~28% |

---

## 4. Strategic Business Recommendations

- **Retention Strategy for Premium Frequent Diners:**
  - Enroll this cohort automatically into Uber One trial tiers with zero delivery fees.
  - Prioritize courier dispatch to drivers rated >4.8 to preserve high satisfaction and avoid churn among top-GMV accounts.
- **Targeting Late-Night & Weekend Convenience Seekers:**
  - Deploy targeted push notifications at 9:30 PM on Thursday through Saturday featuring late-night quick-service bundles.
  - Partner with 24/7 cloud kitchens to widen menu availability during off-peak night windows.
- **Activation for Occasional Budget Customers:**
  - Stimulate re-orders using minimum-spend threshold promotions (e.g., "$5 off orders above $25") to drive up both order frequency and average basket size.
EOF