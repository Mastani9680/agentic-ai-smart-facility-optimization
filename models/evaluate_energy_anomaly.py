import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load anomaly results
# -----------------------------
file_path = "results/anomaly/energy_anomaly_results.csv"

df = pd.read_csv(file_path)

print("Columns in anomaly results:")
print(df.columns.tolist())

# -----------------------------
# Create ground truth
# -----------------------------
# The dataset was generated with 30 intentionally
# abnormal energy records.
#
# We identify the 30 highest total_energy values
# as the known abnormal records for this simulated dataset.

df["actual_anomaly"] = 0

top_anomaly_indices = (
    df["total_energy"]
    .nlargest(30)
    .index
)

df.loc[top_anomaly_indices, "actual_anomaly"] = 1

# -----------------------------
# Convert model prediction
# -----------------------------
df["predicted_anomaly"] = (
    df["anomaly"]
    .astype(str)
    .isin(["-1", "True", "true"])
    .astype(int)
)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(
    df["actual_anomaly"],
    df["predicted_anomaly"]
)

print("\n" + "=" * 50)
print("ENERGY ANOMALY MODEL EVALUATION")
print("=" * 50)

print(f"Total Records       : {len(df)}")
print(f"Actual Anomalies    : {df['actual_anomaly'].sum()}")
print(f"Predicted Anomalies : {df['predicted_anomaly'].sum()}")
print(f"Accuracy            : {accuracy * 100:.2f}%")

# -----------------------------
# Confusion Matrix
# -----------------------------
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        df["actual_anomaly"],
        df["predicted_anomaly"]
    )
)

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report:")
print(
    classification_report(
        df["actual_anomaly"],
        df["predicted_anomaly"],
        target_names=["Normal", "Anomaly"]
    )
)

# -----------------------------
# Save evaluation results
# -----------------------------
output_file = "results/anomaly/anomaly_evaluation.csv"

df.to_csv(output_file, index=False)

print("\nEvaluation results saved to:")
print(output_file)

print("\nENERGY ANOMALY EVALUATION COMPLETED")