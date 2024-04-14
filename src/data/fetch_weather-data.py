import requests
import os
import json
from datetime import timedelta

hourly_variables = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability", "rain", "surface_pressure"]

def hour_rounder(t):
  # Rounds to nearest hour by adding a timedelta hour if minute >= 30
  return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+timedelta(hours=t.minute//30)).strftime('%Y-%m-%dT%H:%M')

def main(latitudes, longitudes):
  # get latitudes and longitudes from file

  weather_url = "https://api.open-meteo.com/v1/forecast"
  params = {
    "latitude": latitudes,
    "longitude": longitudes,
    "hourly": hourly_variables,
    "forecast_hours": 1
  }
  response = requests.get(weather_url, params=params)
  data = response.json()
  save_data_to_json(data)

def save_data_to_json(data):
  script_dir = os.path.dirname(os.path.realpath(__file__))
  data_output_dir = os.path.join(script_dir, '..', '..', 'data', 'raw', 'weather')
  os.makedirs(data_output_dir, exist_ok=True)
  raw_output_file_path = os.path.join(data_output_dir, f'fetched_data_weather.json')
  with open(raw_output_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
  main()