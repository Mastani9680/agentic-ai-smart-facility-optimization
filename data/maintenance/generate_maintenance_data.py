import pandas as pd
import numpy as np

np.random.seed(42)

# Number of equipment records
n = 1000

asset_ids = [f"A{i:04d}" for i in range(1, n + 1)]

building_ids = np.random.choice(
    ["B001", "B002", "B003", "B004"],
    n
)

equipment_types = np.random.choice(
    ["HVAC", "Generator", "Pump", "Chiller", "Elevator"],
    n
)

temperature = np.random.normal(65, 10, n)
vibration = np.random.normal(3, 1, n)
pressure = np.random.normal(50, 8, n)

operating_hours = np.random.randint(100, 10000, n)

last_maintenance_days = np.random.randint(1, 365, n)

failure_count = np.random.poisson(1, n)

energy_consumption = np.random.normal(500, 100, n)

# Calculate initial health score
health_score = (
    100
    - (temperature - 65) * 0.5
    - vibration * 3
    - failure_count * 5
    - last_maintenance_days * 0.05
)

# Keep score between 0 and 100
health_score = np.clip(health_score, 0, 100)

# Maintenance requirement
maintenance_required = (
    (health_score < 60) |
    (failure_count >= 3) |
    (last_maintenance_days > 300)
).astype(int)

# Create dataframe
df = pd.DataFrame({
    "asset_id": asset_ids,
    "building_id": building_ids,
    "equipment_type": equipment_types,
    "temperature": temperature.round(2),
    "vibration": vibration.round(2),
    "pressure": pressure.round(2),
    "operating_hours": operating_hours,
    "last_maintenance_days": last_maintenance_days,
    "failure_count": failure_count,
    "energy_consumption": energy_consumption.round(2),
    "health_score": health_score.round(2),
    "maintenance_required": maintenance_required
})

# Save dataset
df.to_csv(
    "data/maintenance/maintenance_data.csv",
    index=False
)

print("Maintenance dataset created successfully!")
print("Records:", len(df))
print("Columns:", len(df.columns))
print("\nColumns:")
print(df.columns.tolist())

print("\nMaintenance requirement:")
print(df["maintenance_required"].value_counts())

print("\nFirst 5 records:")
print(df.head())
