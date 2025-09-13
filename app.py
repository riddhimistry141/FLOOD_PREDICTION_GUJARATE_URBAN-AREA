from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ---------------- Load Models ----------------
rf_model = joblib.load("flood_rf_model.pkl")  # Random Forest
lstm_model = load_model("my_model.keras")     # LSTM

# Keep prediction history
history = []

# ---------------- Encoding Maps ----------------
land_cover_map = {
    "Forest": 0, "Urban": 1, "Agriculture": 2, "Water": 3, "Barren": 4
}
soil_type_map = {
    "Clay": 0, "Sandy": 1, "Loamy": 2, "Silty": 3, "Peaty": 4, "Chalky": 5
}

# Final feature order (must match training!)
FEATURES = [
    "rainfall", "temperature", "humidity", "river_discharge",
    "water_level", "elevation", "land_cover", "soil_type",
    "population_density", "historical_floods"
]

@app.route("/")
def home():
    return render_template("index.html", final_result=None, history=history)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form inputs
        rainfall = float(request.form["rainfall"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        river_discharge = float(request.form["river_discharge"])
        water_level = float(request.form["water_level"])
        elevation = float(request.form["elevation"])
        land_cover = request.form["land_cover"]
        soil_type = request.form["soil_type"]
        population_density = float(request.form["population_density"])
        historical_floods = int(request.form["historical_floods"])

        # Encode categorical values
        land_cover_enc = land_cover_map.get(land_cover, -1)
        soil_type_enc = soil_type_map.get(soil_type, -1)

        # Prepare input row
        input_data = np.array([[
            rainfall, temperature, humidity, river_discharge,
            water_level, elevation, land_cover_enc, soil_type_enc,
            population_density, historical_floods
        ]])

        # ---------------- RF Prediction ----------------
        rf_prob = rf_model.predict_proba(input_data)[0][1]

        # ---------------- LSTM Prediction ----------------
        lstm_input = input_data.reshape((1, 1, input_data.shape[1]))
        lstm_prob = float(lstm_model.predict(lstm_input)[0][0])

        # ---------------- Ensemble Decision ----------------
        avg_prob = (rf_prob + lstm_prob) / 2
        if avg_prob > 0.5:
            final_result = f"🔴 High Flood Risk ({avg_prob:.2%} confidence)"
        elif avg_prob > 0.4:
            final_result = f"🟡 Moderate Flood Risk ({avg_prob:.2%} confidence)"
        else:
            final_result = f"🟢 Safe ({(1-avg_prob):.2%} confidence)"
            #final_result = f"🔴 High Flood Risk ({avg_prob:.2%} confidence)"

        # Save to history
        history.append({
            "rainfall": rainfall,
            "temperature": temperature,
            "humidity": humidity,
            "result": final_result
        })

        return render_template("index.html", final_result=final_result, history=history)

    except Exception as e:
        return render_template("index.html", final_result="Error: " + str(e))

@app.route("/gujarat_flood_map1")
def gujarat_flood_map1():
    return render_template("gujarat_flood_map1.html")

@app.route("/compare")
def compare():
    return render_template("compare.html")

if __name__ == "__main__":
    app.run(debug=True)
