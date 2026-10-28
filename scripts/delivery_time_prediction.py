import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("data/processed", exist_ok=True)
os.makedirs("docs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------
# Part 1: Data Integration & Feature Engineering
# ---------------------------------------------------------
print("1/6 Loading relational tables for feature engineering...")
deliveries = pd.read_csv("data/raw/deliveries.csv")
orders = pd.read_csv("data/raw/orders.csv")
restaurants = pd.read_csv("data/raw/restaurants.csv")
drivers = pd.read_csv("data/raw/drivers.csv")

# Merge features across entities
df = deliveries.merge(orders[['order_id', 'order_timestamp', 'order_amount', 'item_count', 'restaurant_id', 'customer_id']], on='order_id')
df = df.merge(restaurants[['restaurant_id', 'category', 'price_tier']], on='restaurant_id')
df = df.merge(drivers[['driver_id', 'vehicle_type']], on='driver_id')

# Extract temporal features from order timestamp
df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])
df['hour_of_day'] = df['order_timestamp'].dt.hour
df['day_of_week'] = df['order_timestamp'].dt.day_of_week
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Optional merge with Day 2 customer segments if available
if os.path.exists("data/processed/customer_segments_final.csv"):
    segments = pd.read_csv("data/processed/customer_segments_final.csv")[['customer_id', 'kmeans_cluster']]
    df = df.merge(segments, on='customer_id', how='left')
    df['kmeans_cluster'] = df['kmeans_cluster'].fillna(0).astype(int)

# Drop raw identifiers to prevent data leakage
drop_cols = ['delivery_id', 'order_id', 'driver_id', 'restaurant_id', 'customer_id', 'order_timestamp']
df_features = df.drop(columns=drop_cols)

# One-Hot Encoding for categorical features
cat_cols = ['weather', 'traffic', 'vehicle_type', 'category', 'price_tier']
df_encoded = pd.get_dummies(df_features, columns=cat_cols, drop_first=True)

df_encoded.to_csv("data/processed/regression_features.csv", index=False)
print(f"Engineered dataset saved: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} features.")

# ---------------------------------------------------------
# Part 2: Exploratory Data Analysis (Target & Correlation)
# ---------------------------------------------------------
print("2/6 Running EDA on target distribution and key correlations...")
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.histplot(df_encoded['delivery_time_min'], kde=True, color='teal', bins=30)
plt.title('Target Distribution: Delivery Time (min)')
plt.xlabel('Delivery Time (minutes)')

plt.subplot(1, 2, 2)
sns.scatterplot(x='distance_km', y='delivery_time_min', data=df_encoded, alpha=0.4, color='darkorange')
plt.title('Distance vs Delivery Time')
plt.xlabel('Distance (km)')
plt.ylabel('Delivery Time (minutes)')

plt.tight_layout()
plt.savefig("docs/eda_target_distribution.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Part 3: Train / Test Split & Feature Standardization
# ---------------------------------------------------------
print("3/6 Splitting dataset into training (80%) and testing (20%) sets...")
X = df_encoded.drop(columns=['delivery_time_min'])
y = df_encoded['delivery_time_min']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# Part 4: Model Training (Baseline, Ridge, Lasso, Random Forest)
# ---------------------------------------------------------
print("4/6 Training regression models...")

models = {
    "Baseline (Linear)": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.05),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
}

results = []
predictions = {}

for name, model in models.items():
    if "Forest" in name:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        train_pred = model.predict(X_train)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        train_pred = model.predict(X_train_scaled)
    
    predictions[name] = y_pred
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    train_r2 = r2_score(y_train, train_pred)
    
    results.append({
        "Model": name,
        "MAE (min)": round(mae, 2),
        "RMSE (min)": round(rmse, 2),
        "Test R²": round(r2, 4),
        "Train R²": round(train_r2, 4),
        "Overfitting Delta (Train - Test R²)": round(train_r2 - r2, 4)
    })

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("REGRESSION MODEL PERFORMANCE COMPARISON:")
print("="*80)
print(results_df.to_string(index=False))

# ---------------------------------------------------------
# Part 5: Feature Importance Analysis (Random Forest)
# ---------------------------------------------------------
print("\n5/6 Extracting feature importance and generating visualizations...")
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feature_names = X.columns
top_indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(9, 5))
sns.barplot(
    x=importances[top_indices],
    y=feature_names[top_indices],
    hue=feature_names[top_indices],
    palette='viridis',
    legend=False
)
plt.title('Top 10 Feature Importances (Random Forest Regressor)', fontsize=12, weight='bold')
plt.xlabel('Relative Feature Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig("docs/feature_importance.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Part 6: Actual vs Predicted Visualization
# ---------------------------------------------------------
best_pred = predictions["Random Forest"]

plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_pred, alpha=0.3, color='royalblue', edgecolors='none')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2, label='Perfect 1:1 Prediction Line')
plt.title('Actual vs Predicted Delivery Time (Random Forest Regressor)', fontsize=12, weight='bold')
plt.xlabel('Actual Delivery Time (minutes)')
plt.ylabel('Predicted Delivery Time (minutes)')
plt.legend()
plt.tight_layout()
plt.savefig("docs/actual_vs_predicted.png", dpi=300)
plt.close()

# Save best model artifact
joblib.dump(rf_model, "models/delivery_time_rf_model.pkl")
print(f"Exported best model artifact to models/delivery_time_rf_model.pkl")
print("Visualizations saved to docs/ (eda_target_distribution.png, feature_importance.png, actual_vs_predicted.png)")


# ---------------------------------------------------------
# Part 7: Ridge Regression Deep-Dive & Artifacts
# ---------------------------------------------------------
print("\n7/7 Extracting Ridge coefficients and generating Ridge visualizations...")

ridge_model = models["Ridge Regression"]

# 1. Ridge Coefficients (Directional Feature Impact)
# Positive = increases delivery time; Negative = decreases delivery time
ridge_coefs = pd.Series(ridge_model.coef_, index=X.columns)

# Get top 5 positive drivers and top 5 negative drivers
top_pos = ridge_coefs.sort_values(ascending=False).head(5)
top_neg = ridge_coefs.sort_values(ascending=True).head(5)
top_ridge_features = pd.concat([top_neg, top_pos])

plt.figure(figsize=(9, 5))
bar_colors = ['crimson' if val < 0 else 'forestgreen' for val in top_ridge_features.values]
plt.barh(top_ridge_features.index, top_ridge_features.values, color=bar_colors)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Top Feature Coefficients (Ridge Regression - L2 Penalty)', fontsize=12, weight='bold')
plt.xlabel('Standardized Coefficient Weight (Impact on Delivery Time)')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig("docs/ridge_coefficients.png", dpi=300)
plt.close()
print("Saved Ridge coefficients plot to docs/ridge_coefficients.png")

# 2. Actual vs Predicted for Ridge Regression
ridge_pred = predictions["Ridge Regression"]

plt.figure(figsize=(7, 6))
plt.scatter(y_test, ridge_pred, alpha=0.3, color='darkorange', edgecolors='none')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2, label='Perfect 1:1 Prediction Line')
plt.title('Actual vs Predicted Delivery Time (Ridge Regression)', fontsize=12, weight='bold')
plt.xlabel('Actual Delivery Time (minutes)')
plt.ylabel('Predicted Delivery Time (minutes)')
plt.legend()
plt.tight_layout()
plt.savefig("docs/actual_vs_predicted_ridge.png", dpi=300)
plt.close()
print("Saved Ridge actual vs predicted plot to docs/actual_vs_predicted_ridge.png")

# 3. Save Ridge Model & Scaler Artifacts
# (Note: Scaler is required to transform new data before predicting with Ridge)
joblib.dump(ridge_model, "models/delivery_time_ridge_model.pkl")
joblib.dump(scaler, "models/delivery_scaler.pkl")
print("Exported Ridge model and scaler to models/")