import os
import time
import psutil
import threading
import numpy as np
from scipy.signal import savgol_filter
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import requests
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load model and scaler
model_path = r"C:\Users\danny\OneDrive - Johns Hopkins\2025 Summer Semester\Class Project\Training Data\Models\RandomForest_model.bundle.pkl"
model = joblib.load(model_path)

# Discord webhook
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1392979421979349083/ItHV6qipj-ub8Fp8ims6eq1NGNQmpNkMj2SGPyi20xHcoMmM2RxqmlVIwBFyL5sUltnB'

# Shared state
shared = {
    "system_online": False,
    "last_alert_time": 0,
    "data_log": []
}

# Sensor data model
class SensorData(BaseModel):
    timestamp: float
    pulse: float
    gsr_slope: float
    temperature: float
    humidity: float
    sound_db: float
    light_lux: float
    label: int = 0

# Discord alert function with memory and rate limiting
def send_discord_alert(message: str):
    try:
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        now = time.time()
        if available_memory < 50:
            print("Skipping alert due to low memory.")
            return
        if now - shared["last_alert_time"] < 30:
            print("Skipping alert due to rate limit.")
            return
        payload = {"content": message}
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        print("Discord alert sent:", response.status_code)
        shared["last_alert_time"] = now
    except Exception as e:
        print("Discord alert failed:", e)

# Prediction endpoint
@app.post("/predict")
async def predict(data: SensorData):
    if not shared["system_online"]:
        return {"prediction": 0, "probability": 0.0}

    # Smooth GSR slope using Savitzky-Golay filter
    shared["data_log"].append([
        data.timestamp, data.pulse, data.gsr_slope, data.temperature,
        data.humidity, data.sound_db, data.light_lux, data.label
    ])
    if len(shared["data_log"]) >= 5:
        gsr_values = [row[2] for row in shared["data_log"][-5:]]
        smoothed_gsr = savgol_filter(gsr_values, window_length=5, polyorder=2)
        data.gsr_slope = float(smoothed_gsr[-1])

    # Prepare features
    features = np.array([[data.pulse, data.gsr_slope, data.temperature,
                          data.humidity, data.sound_db, data.light_lux]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][int(prediction)]

    print(f"Prediction: {prediction}, Probability: {probability:.2f}")

    if prediction == 1 and probability > 0.9:
        send_discord_alert("🚨 Headache Imminent Find Safety!")

    return {"prediction": int(prediction), "probability": round(float(probability), 2)}

# System ON trigger
@app.post("/system_on")
def system_on():
    shared["system_online"] = True
    send_discord_alert("✅ System Online and Monitoring Vitals")
    print("System ON triggered")
    return {"status": "System Online"}

# System OFF trigger
@app.post("/system_off")
def system_off():
    shared["system_online"] = False
    send_discord_alert("🛑 System Offline. Monitoring Halted.")
    print("System OFF triggered")
    return {"status": "System Offline"}

# Background thread to save data every 10 minutes
def save_data_periodically():
    while True:
        time.sleep(600)
        if shared["data_log"]:
            df = pd.DataFrame(shared["data_log"], columns=[
                "Timestamp", "Pulse", "GSR_Slope", "Temperature",
                "Humidity", "Sound_dB", "Light_Lux", "Label"
            ])
            date_str = time.strftime("%m.%d")
            save_path = os.path.join(
                r"C:\Users\danny\OneDrive - Johns Hopkins\2025 Summer Semester\Class Project\Raw Data",
                f"{date_str}.HeadacheData.csv"
            )
            df.to_csv(save_path, index=False)
            print(f"Saved data to {save_path}")

# Start background thread
threading.Thread(target=save_data_periodically, daemon=True).start()