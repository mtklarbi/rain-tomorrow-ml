import requests, pandas as pd, datetime as dt, pathlib 

LAT, LON = 29.7, -9.7
start = "2023-01-01"
end = dt.datetime.today().strftime("%Y-%m-%d")
url = (
  "https://archive-api.open-meteo.com/v1/archive"
    f"?latitude={LAT}&longitude={LON}"
    f"&start_date={start}&end_date={end}"
    "&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,windspeed_10m_max"
    "&timezone=Africa%2FCasablanca"
)

print("Requesting URL:", url)
r = requests.get(url, timeout=60); r.raise_for_status()
j = r.json()["daily"]
df = pd.DataFrame(j)
df["time"] = pd.to_datetime(df["time"])
df.to_csv("data/weather_daily.csv", index=False)
print(f"Saved -> data/weather_daily.csv", df.shape)