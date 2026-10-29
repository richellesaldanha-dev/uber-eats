cat << 'EOF' > docs/day5_demand_forecasting_insights.md
# Day 5: Hourly Demand Forecasting & Operational Insights

## 1. Demand Patterns & Decomposition Analysis
- **Intraday Peaks:** Consistent bi-modal demand volume peaking at Lunch (12:00 PM – 2:00 PM) and Dinner (7:00 PM – 9:30 PM). 
- **Weekend Shift:** Friday evening through Sunday night exhibits higher dinner peaks and elevated late-night ordering volumes compared to weekdays.
- **Decomposition:** The 24-hour seasonal component accounts for the majority of variance, with stable baseline trends across cycles.

## 2. Stationarity Assessment (ADF Test)
- The raw hourly aggregated series was evaluated using the Augmented Dickey-Fuller test.
- The resulting test statistic rejected the null hypothesis ($p < 0.05$), confirming the series possesses mean-reverting stationarity without an unbounded macro trend, enabling stable ARIMA parameterization.

## 3. Model Comparison: ARIMA vs. Prophet
- **ARIMA (2, d, 2):** Reacts rapidly to short-term immediate lags, performing well on immediate 1–3 hour lookaheads, but regresses toward the historical mean over multi-day test horizons.
- **Prophet:** Better captures recurring multi-period seasonality (daily lunch/dinner spikes overlaid with weekend effects). Provides smoother operational capacity planning estimates.

## 4. Operational Business Strategy
- **Driver Shift Scheduling:** Pre-schedule courier incentives (surge multipliers) starting 45 minutes prior to identified lunch and dinner peaks to guarantee fleet availability.
- **Merchant Kitchen Pacing:** Provide forecasted order volumes to high-volume restaurants 2 hours in advance to optimize kitchen food prep staging.
- **Off-Peak Demand Stimulation:** Deploy automated push notifications and targeted discounts between 2:30 PM – 5:30 PM to smooth out platform capacity utilization.
EOF