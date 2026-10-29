import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

os.makedirs("data/processed", exist_ok=True)
os.makedirs("docs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------
# Part 1: Data Aggregation (Hourly Demand Series)
# ---------------------------------------------------------
print("1/7 Loading orders and building hourly demand time series...")
orders = pd.read_csv("data/raw/orders.csv")
orders['order_timestamp'] = pd.to_datetime(orders['order_timestamp'])

# Resample into continuous hourly frequency
hourly_series = (
    orders.set_index('order_timestamp')
    .resample('h')
    .size()
    .rename('order_count')
    .asfreq('h', fill_value=0)
    .reset_index()
)

hourly_series.to_csv("data/processed/hourly_demand.csv", index=False)
print(f"Hourly dataset created: {len(hourly_series)} consecutive hours.")

# ---------------------------------------------------------
# Part 2: Exploratory Time-Series Analysis
# ---------------------------------------------------------
print("2/7 Generating exploratory time-series patterns...")
df_eda = hourly_series.copy()
df_eda['hour'] = df_eda['order_timestamp'].dt.hour
df_eda['day_name'] = df_eda['order_timestamp'].dt.day_name()
df_eda['is_weekend'] = df_eda['order_timestamp'].dt.dayofweek.isin([5, 6])

plt.figure(figsize=(12, 5))
sns.lineplot(
    data=df_eda,
    x='hour',
    y='order_count',
    hue='is_weekend',
    palette={False: 'steelblue', True: 'darkorange'},
    ci=None,
    marker='o'
)
plt.title('Average Hourly Demand: Weekday vs Weekend', fontsize=12, weight='bold')
plt.xlabel('Hour of Day (0-23)')
plt.ylabel('Average Order Volume')
plt.xticks(range(0, 24))
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(['Weekday', 'Weekend'], title='Day Type')
plt.tight_layout()
plt.savefig("docs/eda_hourly_patterns.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Part 3: Time-Series Decomposition
# ---------------------------------------------------------
print("3/7 Decomposing series into Trend, Seasonality, and Residuals...")
decomp_data = hourly_series.set_index('order_timestamp')['order_count']
# 24-hour seasonal cycle
decomposition = seasonal_decompose(decomp_data, model='additive', period=24)

fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.suptitle('Hourly Demand Additive Decomposition (24-Hour Period)', y=1.02, fontsize=12, weight='bold')
plt.tight_layout()
plt.savefig("docs/time_series_decomposition.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Part 4: Stationarity Testing (ADF Test)
# ---------------------------------------------------------
print("4/7 Conducting Augmented Dickey-Fuller (ADF) Test...")
adf_result = adfuller(decomp_data)
adf_stat, adf_pvalue = adf_result[0], adf_result[1]
adf_critical_values = adf_result[4]

print(f"  ADF Statistic: {adf_stat:.4f}")
print(f"  p-value:       {adf_pvalue:.4e}")

stationarity_status = "Stationary" if adf_pvalue < 0.05 else "Non-Stationary (Differencing Required)"
print(f"  Series Stationarity Assessment: {stationarity_status}")

# ---------------------------------------------------------
# Part 5: Chronological Train / Test Split
# ---------------------------------------------------------
# Strict chronological split: Reserve final 20% of hours for testing
split_idx = int(len(hourly_series) * 0.80)
train_df = hourly_series.iloc[:split_idx].copy()
test_df = hourly_series.iloc[split_idx:].copy()

train_series = train_df.set_index('order_timestamp')['order_count']
test_series = test_df.set_index('order_timestamp')['order_count']
test_horizon = len(test_df)

print(f"5/7 Chronological Split: Train ({len(train_df)} hrs) | Test ({test_horizon} hrs)")

# ---------------------------------------------------------
# Part 6: Model Training (ARIMA vs Prophet)
# ---------------------------------------------------------
print("6/7 Fitting ARIMA & Prophet models...")

# Model A: ARIMA (1, 0, 1) or (2, 1, 2)
# Using d=0 if stationary; p=1, q=1 captures primary hourly autoregression
d_param = 0 if adf_pvalue < 0.05 else 1
arima_order = (2, d_param, 2)
arima_model = ARIMA(train_series, order=arima_order)
arima_fitted = arima_model.fit()
arima_forecast = arima_fitted.forecast(steps=test_horizon)
arima_forecast.index = test_series.index

# Model B: Prophet
prophet_train = train_df.rename(columns={'order_timestamp': 'ds', 'order_count': 'y'})
prophet_model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False
)
prophet_model.fit(prophet_train)

future = prophet_model.make_future_dataframe(periods=test_horizon, freq='h')
prophet_pred_full = prophet_model.predict(future)
prophet_forecast = prophet_pred_full.iloc[-test_horizon:]['yhat'].values

# ---------------------------------------------------------
# Part 7: Evaluation, Comparison & Deliverables
# ---------------------------------------------------------
print("7/7 Evaluating forecasting metrics...")

def calculate_mape(y_true, y_pred):
    denom = np.where(y_true == 0, 1e-5, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

y_true = test_series.values
arima_pred = np.clip(arima_forecast.values, 0, None)
prophet_pred = np.clip(prophet_forecast, 0, None)

metrics = [
    {
        "Model": "ARIMA" + str(arima_order),
        "MAE (orders)": round(mean_absolute_error(y_true, arima_pred), 2),
        "RMSE (orders)": round(np.sqrt(mean_squared_error(y_true, arima_pred)), 2),
        "MAPE (%)": round(calculate_mape(y_true, arima_pred), 2)
    },
    {
        "Model": "Prophet (Daily + Weekly)",
        "MAE (orders)": round(mean_absolute_error(y_true, prophet_pred), 2),
        "RMSE (orders)": round(np.sqrt(mean_squared_error(y_true, prophet_pred)), 2),
        "MAPE (%)": round(calculate_mape(y_true, prophet_pred), 2)
    }
]

comparison_df = pd.DataFrame(metrics)
print("\n" + "="*70)
print("TIME-SERIES FORECAST MODEL COMPARISON:")
print("="*70)
print(comparison_df.to_string(index=False))

# Plot Actual vs Forecasts
plt.figure(figsize=(14, 6))
plt.plot(test_series.index, y_true, label='Actual Demand', color='black', alpha=0.75, linewidth=1.5)
plt.plot(test_series.index, arima_pred, label=f'ARIMA{arima_order} Forecast', color='dodgerblue', linestyle='--')
plt.plot(test_series.index, prophet_pred, label='Prophet Forecast', color='crimson', linestyle='-.')
plt.title('Test Period Demand Forecast Comparison: Actual vs ARIMA vs Prophet', fontsize=12, weight='bold')
plt.xlabel('Timestamp')
plt.ylabel('Order Volume')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("docs/demand_forecast_comparison.png", dpi=300)
plt.close()

# Save final predictions table
forecast_table = pd.DataFrame({
    'timestamp': test_series.index,
    'actual_demand': y_true,
    'arima_forecast': arima_pred.round(1),
    'prophet_forecast': prophet_pred.round(1)
})
forecast_table.to_csv("data/processed/final_demand_forecast.csv", index=False)

# Export models
joblib.dump(arima_fitted, "models/demand_arima_model.pkl")
joblib.dump(prophet_model, "models/demand_prophet_model.pkl")

print(f"Saved artifacts:")
print("  - docs/eda_hourly_patterns.png")
print("  - docs/time_series_decomposition.png")
print("  - docs/demand_forecast_comparison.png")
print("  - data/processed/final_demand_forecast.csv")
print("  - models/demand_arima_model.pkl & demand_prophet_model.pkl")