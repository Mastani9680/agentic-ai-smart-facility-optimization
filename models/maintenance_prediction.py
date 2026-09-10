import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("results/maintenance/equipment_health.csv")

# Features
features = [
    "temperature",
    "vibration",
    "pressure",
    "operating_hours",
    "last_maintenance_days",
    "failure_count",
    "energy_consumption",
    "health_score"
]

X = df[features]
y = df["maintenance_required"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("Maintenance Prediction Model")
print("-" * 40)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

print("\nModel Performance:")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

# Predict all assets
df["predicted_maintenance"] = model.predict(X)

# Save results
os.makedirs("results/maintenance", exist_ok=True)

df.to_csv(
    "results/maintenance/maintenance_predictions.csv",
    index=False
)

print("\nPredicted Maintenance:")
print(df["predicted_maintenance"].value_counts())

print("\nResults saved to:")
print("results/maintenance/maintenance_predictions.csv")