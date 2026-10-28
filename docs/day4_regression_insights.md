cat << 'EOF' > docs/day4_regression_insights.md
# Day 4: Delivery Time Prediction & Regression Insights

## 1. Model Evaluation & Comparison Summary
- **Baseline (Linear Regression):** Demonstrates strong linear capture between transit distance and time, but cannot model non-linear traffic and weather friction.
- **Ridge ($L_2$) & Lasso ($L_1$):** Effectively regularize feature coefficients, preventing over-reliance on individual restaurant categories while confirming distance and traffic as dominant drivers.
- **Random Forest Regressor (Recommended):** Best overall model with the highest $R^2$ and lowest MAE/RMSE. It successfully captures non-linear interactions (e.g., compound delays during Rainy weather under High traffic).

## 2. Feature Importance Hierarchy
1. **`distance_km`:** Primary driver of physical transit duration.
2. **`traffic_High` & `traffic_Medium`:** Road congestion contributes fixed and variable delays across all vehicle types.
3. **`weather_Rainy` / `weather_Stormy`:** Weather adds systematic friction, dampening two-wheeler speeds.
4. **`prep_time_min`:** Kitchen prep variances impact handoff dispatch times.
5. **`vehicle_type_Scooter` / `vehicle_type_Car`:** Speed differences across courier transportation modes.

## 3. Business Recommendations for Uber Eats Operations
- **Dynamic ETA Calibration:** Integrate the Random Forest model into customer-facing checkout screens to dynamically show real-time ETAs adjusted for rain and traffic, preventing customer expectation mismatch.
- **Courier Dispatch Optimization:** Restrict bicycle assignments for delivery trips exceeding 5 km during rainy periods, routing those orders to cars or scooters to prevent transit time from crossing the 45-minute satisfaction cliff.
- **Smart Staging:** For restaurants with high average prep times, delay courier dispatch by 10 minutes to minimize driver idle wait time at the storefront.
EOF