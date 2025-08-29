from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

FEATURES = [
  "temperature_2m_max","temperature_2m_min","windspeed_10m_max","precipitation_sum",
  "temperature_2m_max_ma3","temperature_2m_min_ma3","windspeed_10m_max_ma3","precipitation_sum_ma3",
  "temperature_2m_max_ma7","temperature_2m_min_ma7","windspeed_10m_max_ma7","precipitation_sum_ma7",
]

model = joblib.load("models/rf.joblib")  # or logit.joblib

class WeatherRow(BaseModel):
    temperature_2m_max: float
    temperature_2m_min: float
    windspeed_10m_max: float
    precipitation_sum: float
    temperature_2m_max_ma3: float
    temperature_2m_min_ma3: float
    windspeed_10m_max_ma3: float
    precipitation_sum_ma3: float
    temperature_2m_max_ma7: float
    temperature_2m_min_ma7: float
    windspeed_10m_max_ma7: float
    precipitation_sum_ma7: float

app = FastAPI()



@app.post("/predict")
def predict(row: WeatherRow):
    X = pd.DataFrame([row.dict()])[FEATURES]
    proba = float(model.predict_proba(X)[:,1][0])
    return {"rain_tomorrow_prob": proba, "rain_tomorrow": int(proba >= 0.5)}