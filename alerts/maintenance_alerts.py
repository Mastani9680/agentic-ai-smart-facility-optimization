import pandas as pd
import os

# Load prediction results
df = pd.read_csv(
    "results/maintenance/maintenance_predictions.csv"
)

# Select assets requiring maintenance
alerts = df[df["predicted_maintenance"] == 1].copy()

# Create alert message
def create_alert(row):
    if row["health_score"] < 50:
        priority = "CRITICAL"
    elif row["health_score"] < 75:
        priority = "HIGH"
    else:
        priority = "MEDIUM"

    return (
        f"{priority} ALERT: Asset {row['asset_id']} "
        f"({row['equipment_type']}) in {row['building_id']} "
        f"requires maintenance. "
        f"Health Score: {row['health_score']}"
    )

alerts["alert_message"] = alerts.apply(create_alert, axis=1)

# Save alerts
os.makedirs("results/maintenance", exist_ok=True)

alerts[
    [
        "asset_id",
        "building_id",
        "equipment_type",
        "health_score",
        "predicted_maintenance",
        "alert_message"
    ]
].to_csv(
    "results/maintenance/maintenance_alerts.csv",
    index=False
)

print("Maintenance Alerts Generated Successfully!")
print("-" * 40)

print("Total Assets Requiring Maintenance:", len(alerts))

print("\nAlert Priority:")
print(
    alerts["alert_message"]
    .str.extract(r"^(.*?) ALERT")[0]
    .value_counts()
)

print("\nSample Alerts:")
print(
    alerts[
        [
            "asset_id",
            "equipment_type",
            "health_score",
            "alert_message"
        ]
    ].head(10)
)

print("\nAlerts saved to:")
print("results/maintenance/maintenance_alerts.csv")
