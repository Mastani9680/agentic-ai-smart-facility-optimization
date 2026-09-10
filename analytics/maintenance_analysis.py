import pandas as pd
import os

# Load maintenance dataset
file_path = "data/maintenance/maintenance_data.csv"

df = pd.read_csv(file_path)

print("Maintenance Data Analysis")
print("-" * 40)

# Basic information
print("Number of records:", len(df))
print("Number of columns:", len(df.columns))

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# Equipment types
print("\nEquipment Type Distribution:")
print(df["equipment_type"].value_counts())

# Building distribution
print("\nBuilding Distribution:")
print(df["building_id"].value_counts())

# Health score statistics
print("\nHealth Score Statistics:")
print(df["health_score"].describe())

# Maintenance requirement
print("\nMaintenance Requirement:")
print(df["maintenance_required"].value_counts())

# Average values
print("\nAverage Temperature:", round(df["temperature"].mean(), 2))
print("Average Vibration:", round(df["vibration"].mean(), 2))
print("Average Pressure:", round(df["pressure"].mean(), 2))
print("Average Operating Hours:", round(df["operating_hours"].mean(), 2))
print("Average Health Score:", round(df["health_score"].mean(), 2))

# Create results folder
os.makedirs("results/maintenance", exist_ok=True)

# Save summary
summary = pd.DataFrame({
    "Metric": [
        "Total Records",
        "Total Columns",
        "Missing Values",
        "Duplicate Records",
        "Average Temperature",
        "Average Vibration",
        "Average Pressure",
        "Average Operating Hours",
        "Average Health Score",
        "Maintenance Required"
    ],
    "Value": [
        len(df),
        len(df.columns),
        df.isnull().sum().sum(),
        df.duplicated().sum(),
        round(df["temperature"].mean(), 2),
        round(df["vibration"].mean(), 2),
        round(df["pressure"].mean(), 2),
        round(df["operating_hours"].mean(), 2),
        round(df["health_score"].mean(), 2),
        int(df["maintenance_required"].sum())
    ]
})

summary.to_csv(
    "results/maintenance/maintenance_summary.csv",
    index=False
)

print("\nAnalysis completed successfully!")
print("Summary saved to results/maintenance/maintenance_summary.csv")
