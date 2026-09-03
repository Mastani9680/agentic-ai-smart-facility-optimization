import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression


# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "data/energy/energy_data.csv"

df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# ==========================================
# 2. CREATE TIME FEATURE
# ==========================================

df["time_index"] = range(len(df))


# ==========================================
# 3. PREPARE DATA FOR FORECASTING
# ==========================================

X = df[["time_index"]]

y = df["total_energy"]


# ==========================================
# 4. CREATE MODEL
# ==========================================

model = LinearRegression()


# ==========================================
# 5. TRAIN MODEL
# ==========================================

model.fit(X, y)


# ==========================================
# 6. PREDICT EXISTING ENERGY
# ==========================================

df["predicted_energy"] = model.predict(X)


# ==========================================
# 7. FORECAST NEXT 24 HOURS
# ==========================================

future_indices = range(
    len(df),
    len(df) + 24
)

future_indices = list(future_indices)

future_X = pd.DataFrame({
    "time_index": future_indices
})

future_predictions = model.predict(future_X)


# ==========================================
# 8. CREATE FORECAST DATAFRAME
# ==========================================

last_timestamp = df["timestamp"].iloc[-1]

future_timestamps = pd.date_range(
    start=last_timestamp + pd.Timedelta(hours=1),
    periods=24,
    freq="h"
)

forecast_df = pd.DataFrame({
    "timestamp": future_timestamps,
    "predicted_energy": future_predictions
})


# ==========================================
# 9. CREATE RESULTS FOLDER
# ==========================================

os.makedirs("results/forecast", exist_ok=True)


# ==========================================
# 10. SAVE FORECAST
# ==========================================

forecast_df.to_csv(
    "results/forecast/energy_forecast.csv",
    index=False
)


# ==========================================
# 11. DISPLAY FORECAST
# ==========================================

print("=" * 60)
print("ENERGY DEMAND FORECAST")
print("=" * 60)

print("\nForecast for next 24 hours:")

print(forecast_df.to_string(index=False))


# ==========================================
# 12. CREATE FORECAST GRAPH
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["timestamp"].tail(100),
    df["total_energy"].tail(100),
    label="Actual Energy"
)

plt.plot(
    df["timestamp"].tail(100),
    df["predicted_energy"].tail(100),
    label="Predicted Energy"
)

plt.plot(
    forecast_df["timestamp"],
    forecast_df["predicted_energy"],
    linestyle="--",
    label="24 Hour Forecast"
)

plt.xlabel("Time")
plt.ylabel("Energy Consumption")

plt.title("Energy Consumption Forecast")

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "results/forecast/energy_forecast.png"
)

plt.close()


# ==========================================
# 13. COMPLETION MESSAGE
# ==========================================

print("\n" + "=" * 60)
print("FORECASTING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nForecast saved in:")
print("results/forecast/")