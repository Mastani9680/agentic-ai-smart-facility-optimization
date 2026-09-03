import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "data/energy/energy_data.csv"

df = pd.read_csv(file_path)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Remove duplicates
df = df.drop_duplicates()

# Sort by timestamp
df = df.sort_values("timestamp")

# Reset index
df = df.reset_index(drop=True)


# ==========================================
# 2. CREATE RESULTS FOLDER
# ==========================================

os.makedirs("results/energy", exist_ok=True)


# ==========================================
# 3. BASIC ENERGY STATISTICS
# ==========================================

print("=" * 60)
print("ENERGY CONSUMPTION ANALYTICS")
print("=" * 60)

print("\nNumber of records:", len(df))

print("\nTotal Energy Consumption:")
print(round(df["total_energy"].sum(), 2))

print("\nAverage Energy Consumption:")
print(round(df["total_energy"].mean(), 2))

print("\nMinimum Energy Consumption:")
print(round(df["total_energy"].min(), 2))

print("\nMaximum Energy Consumption:")
print(round(df["total_energy"].max(), 2))


# ==========================================
# 4. ENERGY COMPONENT ANALYSIS
# ==========================================

print("\n" + "=" * 60)
print("ENERGY COMPONENT ANALYSIS")
print("=" * 60)

print("\nTotal HVAC Energy:")
print(round(df["hvac_energy"].sum(), 2))

print("\nTotal Lighting Energy:")
print(round(df["lighting_energy"].sum(), 2))

print("\nTotal Equipment Energy:")
print(round(df["equipment_energy"].sum(), 2))

print("\nTotal Water Consumption:")
print(round(df["water_consumption"].sum(), 2))


# ==========================================
# 5. BUILDING-WISE ENERGY
# ==========================================

building_energy = (
    df.groupby("building_id")["total_energy"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("BUILDING-WISE ENERGY CONSUMPTION")
print("=" * 60)

print(building_energy)


# ==========================================
# 6. HOURLY ENERGY ANALYSIS
# ==========================================

df["hour"] = df["timestamp"].dt.hour

hourly_energy = df.groupby("hour")["total_energy"].mean()

print("\n" + "=" * 60)
print("AVERAGE ENERGY BY HOUR")
print("=" * 60)

print(hourly_energy)


# ==========================================
# 7. OCCUPANCY VS ENERGY
# ==========================================

occupancy_energy = df[
    ["occupancy", "total_energy"]
].corr()

print("\n" + "=" * 60)
print("OCCUPANCY VS ENERGY CORRELATION")
print("=" * 60)

print(occupancy_energy)


# ==========================================
# 8. SAVE ANALYTICS RESULTS
# ==========================================

building_energy.to_csv(
    "results/energy/building_energy.csv"
)

hourly_energy.to_csv(
    "results/energy/hourly_energy.csv"
)


# ==========================================
# 9. GRAPH 1 - ENERGY OVER TIME
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["timestamp"],
    df["total_energy"]
)

plt.xlabel("Time")
plt.ylabel("Total Energy")
plt.title("Energy Consumption Over Time")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "results/energy/energy_over_time.png"
)

plt.close()


# ==========================================
# 10. GRAPH 2 - ENERGY BY HOUR
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    hourly_energy.index,
    hourly_energy.values,
    marker="o"
)

plt.xlabel("Hour of Day")
plt.ylabel("Average Energy Consumption")
plt.title("Average Energy Consumption by Hour")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/energy/energy_by_hour.png"
)

plt.close()


# ==========================================
# 11. GRAPH 3 - BUILDING ENERGY
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    building_energy.index,
    building_energy.values
)

plt.xlabel("Building")
plt.ylabel("Total Energy Consumption")
plt.title("Energy Consumption by Building")

plt.tight_layout()

plt.savefig(
    "results/energy/building_energy.png"
)

plt.close()


# ==========================================
# 12. GRAPH 4 - OCCUPANCY VS ENERGY
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["occupancy"],
    df["total_energy"]
)

plt.xlabel("Occupancy")
plt.ylabel("Total Energy")
plt.title("Occupancy vs Energy Consumption")

plt.tight_layout()

plt.savefig(
    "results/energy/occupancy_vs_energy.png"
)

plt.close()


print("\n" + "=" * 60)
print("ANALYTICS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGraphs saved in:")
print("results/energy/")