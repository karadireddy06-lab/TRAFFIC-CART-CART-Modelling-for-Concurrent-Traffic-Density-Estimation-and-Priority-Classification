import pickle

# Load Traffic Condition model
with open("traffic_condition_model.pkl", "rb") as file:
    condition_model = pickle.load(file)

# Load Traffic Priority model
with open("traffic_priority_model.pkl", "rb") as file:
    priority_model = pickle.load(file)

# Get traffic information
speed = float(input("Enter Speed: "))
density = float(input("Enter Density: "))
congestion = float(input("Enter Congestion Level: "))
delay = float(input("Enter Delay: "))
signal_strength = float(input("Enter Signal Strength: "))
bandwidth = float(input("Enter Bandwidth: "))

# Create input data
traffic_data = [[
    speed,
    density,
    congestion,
    delay,
    signal_strength,
    bandwidth
]]

# Make predictions
condition_prediction = condition_model.predict(traffic_data)
priority_prediction = priority_model.predict(traffic_data)

# Display results
print("\n==============================")
print("      TRAFFIC CART RESULT")
print("==============================")

print("Traffic Condition:", condition_prediction[0])
print("Traffic Priority:", priority_prediction[0])

print("==============================")