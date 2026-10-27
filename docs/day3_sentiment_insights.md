cat << 'EOF' > docs/day3_sentiment_insights.md
# Day 3: Customer Feedback & Sentiment Analysis Insights

## 1. Sentiment Modeling Insights
- **Operational Latency Trumps Food Quality:** TF-IDF feature importance shows negative sentiment is primarily driven by logistical failures rather than food taste. Negative predictive terms (`cold`, `late`, `hour`, `lost`) strongly correlate with total fulfillment durations exceeding 40 minutes.
- **The High-Satisfaction Baseline:** Positive reviews focus heavily on delivery speed and packaging integrity (`hot`, `fast`, `accurate`, `steaming`), proving prompt delivery is the core prerequisite for 5-star ratings.
- **Model Comparison:** While VADER works well for generic text, the domain-trained Naive Bayes model captures delivery-specific operational friction (`spilled`, `missing`, `wrong route`) with higher precision.

---

## 2. Top Customer Complaints & Themes
1. **Thermal Degradation (`cold`, `freezing`, `soggy`):** Long transit times and two-wheeler deliveries in bad weather lead directly to 1-star ratings.
2. **Transit Delays (`late`, `hour`, `delay`):** Compound bottlenecks when kitchen preparation delays (>30 mins) overlap with peak traffic or rain.
3. **Order Incompleteness (`missing`, `wrong`):** Merchant handoff errors at pickup (especially missing beverages and combo sides).
4. **Transit Packaging Damage (`spilled`, `smashed`):** Inadequate restaurant container seals and lack of compartment dividers in courier bags.
5. **Drop-off Navigation Friction (`lost`):** Difficulties navigating multi-story complexes and dense urban housing.

---

## 3. Actionable Business Recommendations
- **Dynamic Delivery Radii:** Reduce store delivery radius by 20–25% during rain or high traffic to prevent trips exceeding the 40-minute satisfaction threshold.
- **Merchant Item Verification:** Require merchant staff to confirm bundled items (beverages, sides) on their tablet and apply tamper-evident seals before courier handoff.
- **Proactive ETA Alerts & Micro-Credits:** Send automated notifications and small credits (e.g., ₹50) if an order exceeds its ETA by 15+ minutes to soften negative review intent before delivery completes.
- **Courier Equipment Standards:** Ensure two-wheeler couriers use insulated thermal bags with internal beverage dividers to eliminate temperature loss and spills.
EOF
