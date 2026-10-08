import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("traffic_data.csv")

# Input features
X = data[
    [
        "Speed",
        "Density",
        "Congestion",
        "Delay",
        "Signal_Strength",
        "Bandwidth"
    ]
]

# -------------------------------
# MODEL 1: Traffic Condition
# -------------------------------

y_condition = data["Traffic_Condition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_condition,
    test_size=0.2,
    random_state=42,
    stratify=y_condition
)

condition_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

condition_model.fit(X_train, y_train)

condition_predictions = condition_model.predict(X_test)

condition_accuracy = accuracy_score(
    y_test,
    condition_predictions
)

print("Traffic Condition Model")
print("Accuracy:", condition_accuracy)


# -------------------------------
# MODEL 2: Traffic Priority
# -------------------------------

y_priority = data["Priority"]

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X,
    y_priority,
    test_size=0.25,
    random_state=42,
    stratify=y_priority
)

priority_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

priority_model.fit(X_train_p, y_train_p)

priority_predictions = priority_model.predict(X_test_p)

priority_accuracy = accuracy_score(
    y_test_p,
    priority_predictions
)

print("\nTraffic Priority Model")
print("Accuracy:", priority_accuracy)


# -------------------------------
# Save both models
# -------------------------------

with open("traffic_condition_model.pkl", "wb") as file:
    pickle.dump(condition_model, file)

with open("traffic_priority_model.pkl", "wb") as file:
    pickle.dump(priority_model, file)

print("\nBoth models saved successfully!")