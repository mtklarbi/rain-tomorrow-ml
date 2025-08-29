import pandas as pd, numpy as np, pathlib
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/weather_daily.csv", parse_dates=["time"])
df = df.sort_values("time")

# Create target variable: whether it will rain (precipitation > 0) the next day

df["rain_today"] = (df["precipitation_sum"] >= 1.0).astype(int)
df["rain_tomorrow"] = df["rain_today"].shift(-1)

# simple features + rolling stats
for col in ["temperature_2m_max","temperature_2m_min","windspeed_10m_max","precipitation_sum"]:
    df[f"{col}_ma3"] = df[col].rolling(3, min_periods=1).mean()
    df[f"{col}_ma7"] = df[col].rolling(7, min_periods=1).mean()

df = df.dropna(subset=["rain_tomorrow"]).reset_index(drop=True)


FEATURES = [
  "temperature_2m_max","temperature_2m_min","windspeed_10m_max","precipitation_sum",
  "temperature_2m_max_ma3","temperature_2m_min_ma3","windspeed_10m_max_ma3","precipitation_sum_ma3",
  "temperature_2m_max_ma7","temperature_2m_min_ma7","windspeed_10m_max_ma7","precipitation_sum_ma7",
]

X, y = df[FEATURES], df["rain_tomorrow"].astype(int)

# time-aware split: last 20% as test (no shuffle)
split = int(len(df)*0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]


pathlib.Path("data").mkdir(exist_ok=True)
X_train.to_parquet("data/X_train.parquet"); X_test.to_parquet("data/X_test.parquet")
y_train.to_csv("data/y_train.csv", index=False); y_test.to_csv("data/y_test.csv", index=False)
print("Train/Test:", X_train.shape, X_test.shape)