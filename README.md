# Portable-Headache-Migrane-Monitor
---

# Headache Prediction System Using TinyML

## Overview

This project implements a real-time headache prediction system using physiological sensor data and a machine learning model deployed on a Raspberry Pi Pico 2 W. It combines embedded sensing, cloud-based inference, and alerting mechanisms to detect early signs of headaches and notify users.

---

## Project Structure

- `main.py`  
  MicroPython script running on the Raspberry Pi Pico W. It:
  - Connects to Wi-Fi
  - Reads data from sensors (pulse, GSR, temperature, humidity, sound, light)
  - Sends data to a FastAPI server for prediction
  - Toggles system state via a push button
  - Sends Discord alerts when a headache is predicted

- `client.py`  
  FastAPI server script that:
  - Hosts endpoints for prediction and system control
  - Loads a trained Random Forest model
  - Smooths incoming data and performs inference
  - Sends alerts via Discord webhook
  - Periodically saves incoming data to CSV

- `model_trainer.py`  
  Python script for training and evaluating ML models. It:
  - Loads and preprocesses sensor data
  - Trains Decision Tree, Logistic Regression, Random Forest, and Neural Network models
  - Evaluates models using confusion matrices, ROC curves, and k-fold cross-validation
  - Saves the best-performing model as a `.pkl` bundle

---

## Setup Instructions

### Hardware
- Raspberry Pi Pico 2 W
- Sensors: Pulse, GSR, Microphone, Temperature/Humidity (CHT8305), Light
- KS0029 Push Button

### Software
- MicroPython (for Pico 2 W)
- Python 3.x (for training and server)
- Required Python packages: `scikit-learn`, `joblib`, `fastapi`, `pydantic`, `psutil`, `requests`, `matplotlib`, `pandas`, `numpy`

---

## How It Works

1. **Data Collection**  
   Sensor readings are gathered on the Pico W and sent to the FastAPI server.

2. **Prediction**  
   The server uses a trained Random Forest model to predict headache likelihood.

3. **Alerting**  
   If a headache is predicted with high confidence, a Discord alert is triggered.

4. **Logging**  
   Sensor data is saved periodically for future analysis and model retraining.

---

## Model Evaluation

- Models compared: Decision Tree, Logistic Regression, Random Forest, Neural Network
- Metrics: Accuracy, ROC AUC, Confusion Matrix
- Best model saved as `{Best_model}_model.bundle.pkl`

---

## Connectivity

- Wi-Fi SSID: `Your SSID`
- Password: `Your SSID PW`
- Server IP: `Your PC IP`
- Discord Webhook: Configured for real-time alerts

---

## Output Files

- CSV logs of sensor data
- Confusion matrix and ROC curve plots
- Trained model bundle (`.pkl`)

---

Would you like me to format this into a downloadable `README.md` file or add badges and licensing info?
