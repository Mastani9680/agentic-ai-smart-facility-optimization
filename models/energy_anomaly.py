import pandas as pd
import numpy as np
import os

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "data/energy/energy_data.csv"

df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])


# ==========================================
# 2. SELECT FEATURES
# ==========================================

features = [
    "temperature",
    "humidity",
    "occupancy",
    "hvac_energy",
    "lighting_energy",
    "equipment_energy",
    "water_consumption",
    "total_energy"
]

X = df[features]


# ==========================================
# 3. SCALE DATA
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================
# 4. CREATE ISOLATION FOREST MODEL
# ==========================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    random_state=42
)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

model.fit(X_scaled)


# ==========================================
# 6. PREDICT ANOMALIES
# ==========================================

predictions = model.predict(X_scaled)

# Isolation Forest:
#  1  = Normal
# -1  = Anomaly

df["anomaly"] = predictions

df["anomaly_status"] = np.where(
    df["anomaly"] == -1,
    "Anomaly",
    "Normal"
)


# ==========================================
# 7. GET ANOMALY RECORDS
# ==========================================

anomalies = df[df["anomaly"] == -1]


# ==========================================
# 8. CREATE RESULTS FOLDER
# ==========================================

os.makedirs("results/anomaly", exist_ok=True)


# ==========================================
# 9. SAVE RESULTS
# ==========================================

df.to_csv(
    "results/anomaly/energy_anomaly_results.csv",
    index=False
)

anomalies.to_csv(
    "results/anomaly/detected_anomalies.csv",
    index=False
)


# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

print("=" * 60)
print("ENERGY ANOMALY DETECTION")
print("=" * 60)

print("\nTotal records:", len(df))

print("Normal records:", len(df) - len(anomalies))

print("Anomaly records:", len(anomalies))

print(
    "Anomaly percentage:",
    round((len(anomalies) / len(df)) * 100, 2),
    "%"
)


# ==========================================
# 11. DISPLAY ANOMALIES
# ==========================================

print("\n" + "=" * 60)
print("DETECTED ENERGY ANOMALIES")
print("=" * 60)

print(
    anomalies[
        [
            "timestamp",
            "building_id",
            "occupancy",
            "hvac_energy",
            "total_energy",
            "anomaly_status"
        ]
    ].head(20)
)


print("\n" + "=" * 60)
print("ANOMALY DETECTION COMPLETED")
print("=" * 60)

print("\nResults saved in:")
print("results/anomaly/")