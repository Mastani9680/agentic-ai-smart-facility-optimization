import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Number of records
n = 1000

# Generate timestamps
timestamps = pd.date_range(
    start="2026-01-01 00:00:00",
    periods=n,
    freq="h"
)

# Building IDs
building_ids = np.random.choice(
    ["B001", "B002", "B003", "B004"],
    size=n
)

# Time information
hours = timestamps.hour.to_numpy()
days = timestamps.dayofweek.to_numpy()

# Weekend indicator
is_weekend = (days >= 5).astype(int)

# Occupancy pattern
occupancy = []

for hour, weekend in zip(hours, is_weekend):

    if weekend:
        base = 20
    elif 9 <= hour <= 18:
        base = 120
    elif 7 <= hour < 9 or 18 < hour <= 21:
        base = 60
    else:
        base = 10

    value = max(0, int(np.random.normal(base, 15)))
    occupancy.append(value)

occupancy = np.array(occupancy)

# Temperature
temperature = (
    24
    + 5 * np.sin((hours - 6) * np.pi / 12)
    + np.random.normal(0, 1.5, n)
)

# Humidity
humidity = (
    55
    - 0.5 * (temperature - 24)
    + np.random.normal(0, 3, n)
)

# HVAC energy
hvac_energy = (
    150
    + occupancy * 2.2
    + np.maximum(temperature - 24, 0) * 35
    + np.random.normal(0, 25, n)
)

# Lighting energy
lighting_energy = (
    30
    + occupancy * 0.7
    + np.random.normal(0, 8, n)
)

# Equipment energy
equipment_energy = (
    50
    + occupancy * 0.5
    + np.random.normal(0, 10, n)
)

# Water consumption
water_consumption = (
    20
    + occupancy * 0.15
    + np.random.normal(0, 3, n)
)

# Total energy
total_energy = (
    hvac_energy
    + lighting_energy
    + equipment_energy
)

# Create some abnormal energy readings
anomaly_indices = np.random.choice(
    n,
    size=30,
    replace=False
)

total_energy[anomaly_indices] *= np.random.uniform(
    1.8,
    3.0,
    size=len(anomaly_indices)
)

# Create DataFrame
df = pd.DataFrame({
    "timestamp": timestamps,
    "building_id": building_ids,
    "temperature": np.round(temperature, 2),
    "humidity": np.round(humidity, 2),
    "occupancy": occupancy,
    "hvac_energy": np.round(hvac_energy, 2),
    "lighting_energy": np.round(lighting_energy, 2),
    "equipment_energy": np.round(equipment_energy, 2),
    "water_consumption": np.round(water_consumption, 2),
    "total_energy": np.round(total_energy, 2),
    "day_of_week": days,
    "is_weekend": is_weekend
})

# Save dataset
output_file = "data/energy/energy_data.csv"

df.to_csv(output_file, index=False)

print("Energy dataset generated successfully!")
print("--------------------------------------")
print("Number of records:", len(df))
print("Number of columns:", len(df.columns))
print("Saved to:", output_file)

print("\nFirst 5 records:")
print(df.head())