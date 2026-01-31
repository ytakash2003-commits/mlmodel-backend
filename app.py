from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("fatigue_model.pkl")
scaler = joblib.load("scaler.pkl")
latest_data = {}

@app.post("/fatigue")
def fatigue(data: dict):
  global latest_data
  x = [[data['alpha_rms'], data['beta_rms'], data['gamma_rms'], data['delta_rms']]]
  x_scaled = scaler.transform(x)
  fatigue_prediction = int(model.predict(x_scaled)[0])
  latest_data = {
      "time": time.time(),
      "alpha_rms": data['alpha_rms'],
      "beta_rms": data['beta_rms'],
      "gamma_rms": data['gamma_rms'],
      "delta_rms": data['delta_rms'],
      "prediction": fatigue_prediction
  }
  return {"status": "ok", "prediction": fatigue_prediction}
@app.get("/latest_data")
def get_latest_data():
    return latest_data

