from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

with open("traffic_condition_model.pkl", "rb") as file:
    condition_model = pickle.load(file)

with open("traffic_priority_model.pkl", "rb") as file:
    priority_model = pickle.load(file)


# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv("traffic_data.csv")


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURES = [
    "Speed",
    "Density",
    "Congestion",
    "Delay",
    "Signal_Strength",
    "Bandwidth"
]


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = dict(
    zip(
        FEATURES,
        condition_model.feature_importances_
    )
)


# ============================================================
# HOME / DASHBOARD
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    condition = None
    priority = None
    error = None

    # Traffic condition counts
    condition_counts = (
        data["Traffic_Condition"]
        .value_counts()
        .to_dict()
    )

    # Traffic priority counts
    priority_counts = (
        data["Priority"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # ========================================================
    # REAL-TIME PREDICTION
    # ========================================================

    if request.method == "POST":

        try:

            speed = float(
                request.form["speed"]
            )

            density = float(
                request.form["density"]
            )

            congestion = float(
                request.form["congestion"]
            )

            delay = float(
                request.form["delay"]
            )

            signal_strength = float(
                request.form["signal_strength"]
            )

            bandwidth = float(
                request.form["bandwidth"]
            )

            # Create DataFrame with exact feature names
            traffic_input = pd.DataFrame([{

                "Speed": speed,

                "Density": density,

                "Congestion": congestion,

                "Delay": delay,

                "Signal_Strength": signal_strength,

                "Bandwidth": bandwidth

            }])

            # Predict traffic condition
            condition = condition_model.predict(
                traffic_input
            )[0]

            # Predict traffic priority
            priority = priority_model.predict(
                traffic_input
            )[0]

        except Exception as e:

            error = str(e)

    # ========================================================
    # DATASET
    # ========================================================

    dataset = data.to_dict(
        orient="records"
    )

    # ========================================================
    # RENDER DASHBOARD
    # ========================================================

    return render_template(
        "index.html",

        condition=condition,

        priority=priority,

        error=error,

        condition_counts=condition_counts,

        priority_counts=priority_counts,

        dataset=dataset,

        feature_importance=feature_importance
    )


# ============================================================
# EDA PAGE
# ============================================================

@app.route("/eda")
def eda():

    total_records = len(data)

    total_features = len(FEATURES)

    # Traffic condition distribution
    condition_counts = (
        data["Traffic_Condition"]
        .value_counts()
        .to_dict()
    )

    # Traffic priority distribution
    priority_counts = (
        data["Priority"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    statistics = {}

    for column in FEATURES:

        statistics[column] = {

            "mean": round(
                data[column].mean(),
                2
            ),

            "min": round(
                data[column].min(),
                2
            ),

            "max": round(
                data[column].max(),
                2
            )

        }

    # ========================================================
    # CORRELATION MATRIX
    # ========================================================

    correlation = (
        data[FEATURES]
        .corr()
        .round(2)
        .to_dict()
    )

    # ========================================================
    # DATASET
    # ========================================================

    dataset = data.to_dict(
        orient="records"
    )

    return render_template(
        "eda.html",

        total_records=total_records,

        total_features=total_features,

        condition_counts=condition_counts,

        priority_counts=priority_counts,

        statistics=statistics,

        correlation=correlation,

        dataset=dataset
    )


# ============================================================
# CLASSIFICATION PAGE
# ============================================================

@app.route("/classification")
def classification():

    total_records = len(data)

    # Traffic condition distribution
    condition_counts = (
        data["Traffic_Condition"]
        .value_counts()
        .to_dict()
    )

    # Traffic priority distribution
    priority_counts = (
        data["Priority"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return render_template(
        "classification.html",

        total_records=total_records,

        condition_counts=condition_counts,

        priority_counts=priority_counts
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    condition = None

    priority = None

    error = None

    # ========================================================
    # PREDICTION
    # ========================================================

    if request.method == "POST":

        try:

            speed = float(
                request.form["speed"]
            )

            density = float(
                request.form["density"]
            )

            congestion = float(
                request.form["congestion"]
            )

            delay = float(
                request.form["delay"]
            )

            signal_strength = float(
                request.form["signal_strength"]
            )

            bandwidth = float(
                request.form["bandwidth"]
            )

            traffic_input = pd.DataFrame([{

                "Speed": speed,

                "Density": density,

                "Congestion": congestion,

                "Delay": delay,

                "Signal_Strength": signal_strength,

                "Bandwidth": bandwidth

            }])

            # Traffic condition prediction
            condition = condition_model.predict(
                traffic_input
            )[0]

            # Traffic priority prediction
            priority = priority_model.predict(
                traffic_input
            )[0]

        except Exception as e:

            error = str(e)

    return render_template(
        "prediction.html",

        condition=condition,

        priority=priority,

        error=error
    )


# ============================================================
# DATASET PAGE
# ============================================================

@app.route("/dataset")
def dataset_page():

    dataset = data.to_dict(
        orient="records"
    )

    return render_template(
        "dataset.html",

        dataset=dataset
    )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )