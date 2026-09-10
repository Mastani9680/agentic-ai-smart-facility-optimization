import pandas as pd

health_file = "results/maintenance/equipment_health.csv"
prediction_file = "results/maintenance/maintenance_predictions.csv"
alerts_file = "results/maintenance/maintenance_alerts.csv"

health_df = pd.read_csv(health_file)
prediction_df = pd.read_csv(prediction_file)
alerts_df = pd.read_csv(alerts_file)

total_assets = len(health_df)

average_health = health_df["health_score"].mean()

maintenance_required = prediction_df[
    prediction_df["predicted_maintenance"] == 1
]

maintenance_count = len(maintenance_required)

critical_count = len(
    health_df[health_df["health_category"] == "Critical"]
)

if critical_count > 0:
    status = "CRITICAL MAINTENANCE RISK"
elif maintenance_count >= total_assets * 0.20:
    status = "HIGH MAINTENANCE RISK"
elif maintenance_count >= total_assets * 0.10:
    status = "MODERATE MAINTENANCE RISK"
else:
    status = "NORMAL"

print("MAINTENANCE AGENT")
print("-" * 40)

print("Total Assets:", total_assets)
print("Average Equipment Health:",
      round(average_health, 2))
print("Maintenance Required:", maintenance_count)
print("Critical Assets:", critical_count)
print("Total Alerts:", len(alerts_df))

print("\nMaintenance Status:", status)

print("\nMaintenance Agent Recommendations:")

if critical_count > 0:
    print("1. Immediately inspect critical equipment.")

if maintenance_count > 0:
    print("2. Schedule maintenance for predicted assets.")

print("3. Monitor equipment health regularly.")
print("4. Review assets with repeated failures or abnormal readings.")

print("\nMAINTENANCE AGENT COMPLETED")