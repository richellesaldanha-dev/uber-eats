import os
import joblib
import pandas as pd
import numpy as np

# Ensure docs and logs exist
os.makedirs("docs", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("="*75)
print("UBER EATS OPERATIONAL INTELLIGENCE: END-TO-END PIPELINE RUNNER")
print("="*75)

# ---------------------------------------------------------
# Step 1: Verify & Load Serialized Model Artifacts
# ---------------------------------------------------------
print("\n[1/4] Loading trained models and transformation artifacts...")

required_artifacts = {
    "Customer Segments": "data/processed/customer_segments_final.csv",
    "Sentiment Vectorizer": "models/tfidf_vectorizer.pkl",
    "Sentiment Model": "models/sentiment_nb_model.pkl",
    "Delivery Time Model": "models/delivery_time_rf_model.pkl",
    "Demand Forecast": "data/processed/final_demand_forecast.csv"
}

loaded_artifacts = {}
for name, path in required_artifacts.items():
    if os.path.exists(path):
        if path.endswith(".pkl"):
            loaded_artifacts[name] = joblib.load(path)
        elif path.endswith(".csv"):
            loaded_artifacts[name] = pd.read_csv(path)
        print(f"  ✓ Loaded {name} from {path}")
    else:
        print(f"  ✗ Warning: Artifact not found at {path}")

# ---------------------------------------------------------
# Step 2: Simulate Live Incoming Orders Batch
# ---------------------------------------------------------
print("\n[2/4] Ingesting and simulating new incoming orders stream...")

sample_orders = pd.DataFrame([
    {
        "order_id": "ORD-LIVE-9001",
        "customer_id": 105,
        "prep_time_min": 18.0,
        "distance_km": 4.2,
        "order_amount": 42.50,
        "item_count": 3,
        "hour_of_day": 19,
        "day_of_week": 4, # Friday
        "weather": "Rainy",
        "traffic": "High",
        "vehicle_type": "Scooter",
        "category": "Fast Food",
        "price_tier": "$$",
        "review_text": "The delivery was delayed in heavy rain and the food was cold and leaking."
    },
    {
        "order_id": "ORD-LIVE-9002",
        "customer_id": 312,
        "prep_time_min": 12.0,
        "distance_km": 1.8,
        "order_amount": 18.00,
        "item_count": 1,
        "hour_of_day": 13,
        "day_of_week": 1, # Tuesday
        "weather": "Clear",
        "traffic": "Low",
        "vehicle_type": "Bicycle",
        "category": "Cafe",
        "price_tier": "$",
        "review_text": "Super fast delivery and fresh hot coffee! Driver was polite."
    }
])

# ---------------------------------------------------------
# Step 3: End-to-End Operational Pipeline Inference
# ---------------------------------------------------------
print("\n[3/4] Running end-to-end multi-model inference...")

# A. Customer Segmentation Tagging
seg_df = loaded_artifacts.get("Customer Segments")
if seg_df is not None:
    seg_map = dict(zip(seg_df['customer_id'], seg_df['kmeans_cluster']))
    persona_map = {0: "Occasional Budget", 1: "Late-Night / Weekend", 2: "Premium Frequent"}
    sample_orders['cluster_id'] = sample_orders['customer_id'].map(seg_map).fillna(0).astype(int)
    sample_orders['customer_persona'] = sample_orders['cluster_id'].map(persona_map)
else:
    sample_orders['customer_persona'] = "Unsegmented"

# B. NLP Sentiment Scoring
tfidf = loaded_artifacts.get("Sentiment Vectorizer")
nb_model = loaded_artifacts.get("Sentiment Model")
if tfidf and nb_model:
    cleaned_input = sample_orders['review_text'].str.lower()
    features = tfidf.transform(cleaned_input)
    sample_orders['predicted_sentiment'] = nb_model.predict(features)
    sample_orders['sentiment_label'] = sample_orders['predicted_sentiment'].map({1: "Positive", 0: "Negative"})
else:
    sample_orders['sentiment_label'] = "N/A"

# C. Delivery Time Prediction (Random Forest)
rf_model = loaded_artifacts.get("Delivery Time Model")
if rf_model is not None:
    # Build dummy feature vector aligned with training columns
    model_features = rf_model.feature_names_in_
    input_encoded = pd.get_dummies(sample_orders, drop_first=True)
    aligned_input = input_encoded.reindex(columns=model_features, fill_value=0)
    sample_orders['predicted_eta_min'] = np.round(rf_model.predict(aligned_input), 1)
else:
    sample_orders['predicted_eta_min'] = 30.0

# D. Operational SLA Risk Flag
sample_orders['churn_risk_flag'] = (sample_orders['predicted_eta_min'] > 40.0) | (sample_orders['sentiment_label'] == "Negative")

# ---------------------------------------------------------
# Step 4: Export Integrated Pipeline Audit
# ---------------------------------------------------------
print("\n[4/4] Generating pipeline execution audit table...")
cols_to_display = [
    'order_id', 'customer_persona', 'distance_km', 'weather',
    'traffic', 'predicted_eta_min', 'sentiment_label', 'churn_risk_flag'
]
print(sample_orders[cols_to_display].to_string(index=False))

sample_orders.to_csv("data/processed/pipeline_live_inference_audit.csv", index=False)
print("\nInference log saved to data/processed/pipeline_live_inference_audit.csv")