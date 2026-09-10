import pandas as pd
import os

# Load maintenance dataset
file_path = "data/maintenance/maintenance_data.csv"

df = pd.read_csv(file_path)

# Calculate equipment health score
health_score = (
    100
    - (df["temperature"] - 65).abs() * 0.5
    - df["vibration"] * 3
    - df["failure_count"] * 5
    - df["last_maintenance_days"] * 0.05
)

# Keep score between 0 and 100
df["health_score"] = health_score.clip(0, 100).round(2)

# Health category
def get_health_category(score):
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 50:
        return "Warning"
    else:
        return "Critical"

df["health_category"] = df["health_score"].apply(get_health_category)

# Create results folder
os.makedirs("results/maintenance", exist_ok=True)

# Save health results
output_file = "results/maintenance/equipment_health.csv"
df.to_csv(output_file, index=False)

print("Equipment Health Score completed successfully!")
print("Total Assets:", len(df))

print("\nHealth Category Distribution:")
print(df["health_category"].value_counts())

print("\nAverage Health Score:",
      round(df["health_score"].mean(), 2))

print("\nHealth Score Range:")
print("Minimum:", df["health_score"].min())
print("Maximum:", df["health_score"].max())

print("\nSample Results:")
print(
    df[
        ["asset_id", "equipment_type",
         "health_score", "health_category"]
    ].head()
)

print("\nResults saved to:", output_file)
