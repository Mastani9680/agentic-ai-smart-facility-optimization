import streamlit as st
import pandas as pd
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="FacilityOps AI - Energy Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ FacilityOps AI - Energy Intelligence Dashboard")
st.write("AI-powered energy monitoring and facility intelligence")

# -----------------------------
# File Paths
# -----------------------------
energy_file = "data/energy/energy_data.csv"
anomaly_file = "results/anomaly/energy_anomaly_results.csv"
forecast_file = "results/forecast/energy_forecast.csv"
evaluation_file="results/anomaly/anomaly_evaluation.csv"

# -----------------------------
# Load Energy Data
# -----------------------------
if not os.path.exists(energy_file):
    st.error("Energy dataset not found.")
    st.stop()

df = pd.read_csv(energy_file)

# -----------------------------
# Load Anomaly Results
# -----------------------------
if os.path.exists(anomaly_file):
    anomaly_df = pd.read_csv(anomaly_file)
else:
    anomaly_df = pd.DataFrame()

# -----------------------------
# Load Forecast Results
# -----------------------------
if os.path.exists(forecast_file):
    forecast_df = pd.read_csv(forecast_file)
else:
    forecast_df = pd.DataFrame()

# -----------------------------
# Load Evaluation Results
# -----------------------------
if os.path.exists(evaluation_file):
    evaluation_df=pd.read_csv(evaluation_file)
else:
    evaluation_df=pd.DataFrame()

# -----------------------------
# KPI Calculations
# -----------------------------
total_energy = df["total_energy"].sum()
average_energy = df["total_energy"].mean()
maximum_energy = df["total_energy"].max()

if not anomaly_df.empty:
    anomaly_count = (
        anomaly_df["anomaly"]
        .astype(str)
        .str.lower()
        .isin(["-1", "true", "anomaly"])
        .sum()
    )
else:
    anomaly_count = 0

anomaly_percentage = (anomaly_count / len(df)) * 100

if not forecast_df.empty:
    average_forecast = forecast_df["predicted_energy"].mean()
else:
    average_forecast = 0

# -----------------------------
# KPI Section
# -----------------------------
st.subheader("📊 Energy Overview")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Energy",
    f"{total_energy:,.2f}"
)

col2.metric(
    "Average Energy",
    f"{average_energy:,.2f}"
)

col3.metric(
    "Maximum Energy",
    f"{maximum_energy:,.2f}"
)

col4.metric(
    "Energy Anomalies",
    f"{anomaly_count}"
)

col5.metric(
    "Forecasted Energy",
    f"{average_forecast:,.2f}"
)

# -----------------------------
# Energy Trend
# -----------------------------
st.subheader("📈 Energy Consumption Trend")

df["timestamp"] = pd.to_datetime(df["timestamp"])

hourly_energy = (
    df.groupby("timestamp")["total_energy"]
    .sum()
    .reset_index()
)

st.line_chart(
    hourly_energy.set_index("timestamp")
)

# -----------------------------
# Building-wise Energy
# -----------------------------
st.subheader("🏢 Building-wise Energy Consumption")

building_energy = (
    df.groupby("building_id")["total_energy"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(building_energy)

# -----------------------------
# Energy Distribution by System
# -----------------------------
st.subheader("⚡ Energy Distribution by System")

hvac_total = df["hvac_energy"].sum()
lighting_total = df["lighting_energy"].sum()
equipment_total = df["equipment_energy"].sum()

system_total = (
    hvac_total +
    lighting_total +
    equipment_total
)

hvac_percentage = (hvac_total / system_total) * 100
lighting_percentage = (lighting_total / system_total) * 100
equipment_percentage = (equipment_total / system_total) * 100

col1, col2, col3 = st.columns(3)

col1.metric(
    "HVAC",
    f"{hvac_percentage:.2f}%"
)

col2.metric(
    "Lighting",
    f"{lighting_percentage:.2f}%"
)

col3.metric(
    "Equipment",
    f"{equipment_percentage:.2f}%"
)

component_df = pd.DataFrame(
    {
        "System": [
            "HVAC",
            "Lighting",
            "Equipment"
        ],
        "Percentage": [
            hvac_percentage,
            lighting_percentage,
            equipment_percentage
        ]
    }
)

st.bar_chart(
    component_df.set_index("System")
)


# -----------------------------
# Occupancy vs Energy
# -----------------------------
st.subheader("👥 Occupancy vs Energy")

occupancy_energy = (
    df.groupby("occupancy")["total_energy"]
    .mean()
    .reset_index()
)

st.line_chart(
    occupancy_energy.set_index("occupancy")
)

# -----------------------------
# Anomaly Section
# -----------------------------
st.subheader("🚨 Energy Anomalies")

if not anomaly_df.empty:

    anomaly_records = anomaly_df[
        anomaly_df["anomaly"].astype(str).isin(["-1", "True", "true"])
    ]

    if len(anomaly_records) > 0:
        st.dataframe(
            anomaly_records,
            use_container_width=True
        )
    else:
        st.success("No energy anomalies detected.")

else:
    st.info("Anomaly results are not available.")

# -----------------------------
# Forecast Section
# -----------------------------
st.subheader("🔮 Energy Forecast")

if not forecast_df.empty:

    forecast_df["timestamp"] = pd.to_datetime(
        forecast_df["timestamp"]
    )

    st.line_chart(
        forecast_df.set_index("timestamp")[
            ["predicted_energy"]
        ]
    )

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

else:
    st.info("Forecast results are not available.")

# -----------------------------
# Energy Status
# -----------------------------
st.subheader("🤖 AI Energy Status")

if anomaly_percentage >= 5:
    status = "🔴 HIGH ENERGY RISK"
elif anomaly_percentage >= 3:
    status = "🟠 MODERATE ENERGY RISK"
else:
    status = "🟢 NORMAL"

st.info(status)

# -----------------------------
# Anomaly Detection Performance
# -----------------------------
st.subheader("📊 Anomaly Detection Performance")

if not evaluation_df.empty:

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    actual = evaluation_df["actual_anomaly"]
    predicted = evaluation_df["predicted_anomaly"]

    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    f1 = f1_score(actual, predicted, zero_division=0)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )

    if accuracy >= 0.85:
        st.success(
            "✅ Accuracy target achieved (Target: ≥85%)"
        )
    else:
        st.warning(
            "⚠️ Accuracy target not achieved (Target: ≥85%)"
        )

else:
    st.info("Evaluation results are not available.")


# -----------------------------
# AI Energy Recommendations
# -----------------------------
st.subheader("💡 AI-Generated Energy Efficiency Recommendations")

recommendations = []

avg_occupancy = df["occupancy"].mean()
avg_hvac = df["hvac_energy"].mean()
avg_lighting = df["lighting_energy"].mean()
avg_equipment = df["equipment_energy"].mean()

if avg_hvac > 500 and avg_occupancy < 100:
    recommendations.append(
        "Review HVAC schedules during low-occupancy periods."
    )
else:
    recommendations.append(
        "HVAC consumption is within the expected operating range."
    )

if avg_lighting > 100:
    recommendations.append(
        "Optimize lighting schedules in low-occupancy areas."
    )
else:
    recommendations.append(
        "Lighting consumption is within the expected range."
    )

if avg_equipment > 100:
    recommendations.append(
        "Check equipment operating during unnecessary or low-usage periods."
    )
else:
    recommendations.append(
        "Equipment energy consumption is within the expected range."
    )

recommendations.append(
    "Continue monitoring repeated high-energy anomalies."
)

for i, recommendation in enumerate(
    recommendations,
    start=1
):
    st.write(
        f"**{i}.** {recommendation}"
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.write(
    "FacilityOps AI | Energy Agent | Milestone 1"
)