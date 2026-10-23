import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Set deterministic seed
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CUSTOMERS = 1000
N_RESTAURANTS = 100
N_DRIVERS = 150
N_ORDERS = 6000

print("Generating synthetic Uber Eats marketplace data...")

# 1. Customers
customer_ids = [f"C{i:04d}" for i in range(1, N_CUSTOMERS + 1)]
city_zones = ["Downtown", "Midtown", "Suburbs", "Uptown"]
base_signup = datetime(2023, 1, 1)
signup_offsets = np.random.randint(0, 730, size=N_CUSTOMERS)
signup_dates = [base_signup + timedelta(days=int(d)) for d in signup_offsets]

customers_df = pd.DataFrame({
    "customer_id": customer_ids,
    "signup_date": [d.strftime("%Y-%m-%d") for d in signup_dates],
    "age": np.random.randint(18, 71, size=N_CUSTOMERS),
    "gender": np.random.choice(["Male", "Female", "Other"], size=N_CUSTOMERS, p=[0.48, 0.48, 0.04]),
    "city_zone": np.random.choice(city_zones, size=N_CUSTOMERS, p=[0.35, 0.25, 0.25, 0.15]),
    "is_premium_member": np.random.choice([True, False], size=N_CUSTOMERS, p=[0.30, 0.70])
})

# 2. Restaurants
restaurant_ids = [f"R{i:03d}" for i in range(1, N_RESTAURANTS + 1)]
cuisines = ["American", "Italian", "Indian", "Japanese", "Mexican", "Healthy", "Dessert", "Thai"]
price_tiers = ["$", "$$", "$$$"]

restaurants_df = pd.DataFrame({
    "restaurant_id": restaurant_ids,
    "name": [f"Kitchen_{c}_{i}" for i, c in enumerate(np.random.choice(cuisines, size=N_RESTAURANTS), start=1)],
    "category": np.random.choice(cuisines, size=N_RESTAURANTS),
    "price_tier": np.random.choice(price_tiers, size=N_RESTAURANTS, p=[0.45, 0.40, 0.15]),
    "avg_prep_time_min": np.round(np.random.uniform(12.0, 38.0, size=N_RESTAURANTS), 1),
    "city_zone": np.random.choice(city_zones, size=N_RESTAURANTS, p=[0.40, 0.30, 0.15, 0.15])
})

# 3. Drivers
driver_ids = [f"D{i:03d}" for i in range(1, N_DRIVERS + 1)]
drivers_df = pd.DataFrame({
    "driver_id": driver_ids,
    "vehicle_type": np.random.choice(["Bicycle", "Scooter", "Car"], size=N_DRIVERS, p=[0.25, 0.45, 0.30]),
    "rating": np.round(np.random.uniform(4.1, 5.0, size=N_DRIVERS), 2),
    "acceptance_rate": np.round(np.random.uniform(0.70, 0.99, size=N_DRIVERS), 2)
})

# 4. Orders
start_timestamp = datetime(2025, 1, 1, 0, 0, 0)
hourly_weights = np.array([
    0.01, 0.005, 0.005, 0.005, 0.005, 0.01, 0.02, 0.03,
    0.04, 0.04, 0.05, 0.08, 0.11, 0.09, 0.05, 0.04,
    0.04, 0.06, 0.12, 0.11, 0.08, 0.05, 0.03, 0.02
])
hourly_weights = hourly_weights / hourly_weights.sum()

random_days = np.random.randint(0, 365, size=N_ORDERS)
random_hours = np.random.choice(np.arange(24), size=N_ORDERS, p=hourly_weights)
random_minutes = np.random.randint(0, 60, size=N_ORDERS)
random_seconds = np.random.randint(0, 60, size=N_ORDERS)

order_timestamps = [
    start_timestamp + timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))
    for d, h, m, s in zip(random_days, random_hours, random_minutes, random_seconds)
]
order_timestamps.sort()

customer_choice = np.random.choice(customers_df["customer_id"], size=N_ORDERS)
restaurant_choice = np.random.choice(restaurants_df["restaurant_id"], size=N_ORDERS)

order_amounts = []
item_counts = []
for r_id in restaurant_choice:
    tier = restaurants_df.loc[restaurants_df["restaurant_id"] == r_id, "price_tier"].values[0]
    items = np.random.randint(1, 9)
    base_price = 10.0 if tier == "$" else (22.0 if tier == "$$" else 38.0)
    order_amounts.append(round(base_price * items + np.random.uniform(1.0, 5.0), 2))
    item_counts.append(items)

orders_df = pd.DataFrame({
    "order_id": [f"ORD{i:05d}" for i in range(1, N_ORDERS + 1)],
    "customer_id": customer_choice,
    "restaurant_id": restaurant_choice,
    "order_timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in order_timestamps],
    "order_amount": order_amounts,
    "item_count": item_counts,
    "status": np.random.choice(["Completed", "Cancelled"], size=N_ORDERS, p=[0.94, 0.06])
})

# 5. Deliveries
completed_orders = orders_df[orders_df["status"] == "Completed"].copy().reset_index(drop=True)
n_completed = len(completed_orders)

distances = np.round(np.clip(np.random.exponential(scale=3.5, size=n_completed) + 0.8, 0.5, 18.0), 1)
weather_cond = np.random.choice(["Clear", "Rainy", "Stormy"], size=n_completed, p=[0.72, 0.22, 0.06])
traffic_cond = np.random.choice(["Low", "Medium", "High"], size=n_completed, p=[0.40, 0.38, 0.22])

rest_prep_lookup = dict(zip(restaurants_df["restaurant_id"], restaurants_df["avg_prep_time_min"]))
prep_times = [round(max(8.0, rest_prep_lookup[r_id] + np.random.normal(0, 3.5)), 1) for r_id in completed_orders["restaurant_id"]]

assigned_drivers = np.random.choice(drivers_df["driver_id"], size=n_completed)
driver_vehicle_lookup = dict(zip(drivers_df["driver_id"], drivers_df["vehicle_type"]))

actual_delivery_times = []
for dist, weather, traffic, d_id in zip(distances, weather_cond, traffic_cond, assigned_drivers):
    v_type = driver_vehicle_lookup[d_id]
    speed_factor = 3.0 if v_type == "Car" else (3.8 if v_type == "Scooter" else 4.6)
    weather_multiplier = 1.0 if weather == "Clear" else (1.25 if weather == "Rainy" else 1.55)
    traffic_delay = 0.0 if traffic == "Low" else (5.0 if traffic == "Medium" else 12.0)
    transit_time = (dist * speed_factor * weather_multiplier) + traffic_delay + np.random.normal(0, 2.5)
    actual_delivery_times.append(round(max(6.0, transit_time), 1))

deliveries_df = pd.DataFrame({
    "delivery_id": [f"DEL{i:05d}" for i in range(1, n_completed + 1)],
    "order_id": completed_orders["order_id"],
    "driver_id": assigned_drivers,
    "distance_km": distances,
    "prep_time_min": prep_times,
    "delivery_time_min": actual_delivery_times,
    "weather": weather_cond,
    "traffic": traffic_cond
})

# 6. Payments
tips = []
for amt, d_time in zip(completed_orders["order_amount"], deliveries_df["delivery_time_min"]):
    tip_prob = 0.85 if d_time < 30 else 0.40
    if np.random.rand() < tip_prob:
        pct = np.random.uniform(0.10, 0.22) if d_time < 25 else np.random.uniform(0.05, 0.12)
        tips.append(round(amt * pct, 2))
    else:
        tips.append(0.0)

payments_df = pd.DataFrame({
    "payment_id": [f"PAY{i:05d}" for i in range(1, n_completed + 1)],
    "order_id": completed_orders["order_id"],
    "payment_method": np.random.choice(["Credit Card", "Debit Card", "Digital Wallet"], size=n_completed, p=[0.55, 0.25, 0.20]),
    "tip_amount": tips,
    "discount_applied": np.random.choice([0.0, 3.0, 5.0, 10.0], size=n_completed, p=[0.60, 0.20, 0.15, 0.05])
})

# 7. Reviews
review_idx = np.random.choice(np.arange(n_completed), size=int(n_completed * 0.40), replace=False)
review_orders = completed_orders.iloc[review_idx].reset_index(drop=True)
review_deliveries = deliveries_df.iloc[review_idx].reset_index(drop=True)

positive_texts = [
    "Food arrived burning hot and delicious!",
    "Incredible delivery speed, rider was courteous and professional.",
    "Packaged carefully, order was completely accurate. Loved it!",
    "Great service and tasty meal. Will definitely order again."
]
neutral_texts = [
    "Food was okay, delivery time was acceptable.",
    "Portion sizes were smaller than expected, but arrived on time."
]
negative_texts = [
    "Delivery took way too long! Food was freezing cold upon arrival.",
    "Driver got lost, food was completely spilled inside the bag. Terrible.",
    "Missing drinks and cold fries. Extremely disappointed with this store."
]

review_ratings = []
review_corpus = []
for _, del_row in review_deliveries.iterrows():
    total_wait = del_row["prep_time_min"] + del_row["delivery_time_min"]
    if total_wait > 50 or del_row["weather"] == "Stormy":
        rating = np.random.choice([1, 2], p=[0.75, 0.25])
        text = random.choice(negative_texts)
    elif total_wait < 28:
        rating = np.random.choice([4, 5], p=[0.20, 0.80])
        text = random.choice(positive_texts)
    else:
        rating = 3
        text = random.choice(neutral_texts)
    review_ratings.append(rating)
    review_corpus.append(text)

reviews_df = pd.DataFrame({
    "review_id": [f"REV{i:04d}" for i in range(1, len(review_orders) + 1)],
    "order_id": review_orders["order_id"],
    "rating": review_ratings,
    "review_text": review_corpus
})

# Export to CSV
tables = {
    "customers.csv": customers_df,
    "restaurants.csv": restaurants_df,
    "drivers.csv": drivers_df,
    "orders.csv": orders_df,
    "deliveries.csv": deliveries_df,
    "payments.csv": payments_df,
    "reviews.csv": reviews_df
}

for filename, df in tables.items():
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
    print(f"Exported {filename:<16} | Shape: {df.shape}")

print(f"\nAll 7 CSVs successfully saved into '{OUTPUT_DIR}'.")