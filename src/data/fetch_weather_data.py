import requests
import os
import json
from datetime import timedelta

hourly_variables = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability", "rain", "surface_pressure"]

def extract_coordinates(station_data):
  latitudes = [station['position']['lat'] for station in station_data]
  longitudes = [station['position']['lng'] for station in station_data]
  return latitudes, longitudes

def read_json(file_path):
  with open(file_path, 'r') as file:
    return json.load(file)

def save_data_to_json(data):
  data_output_dir = os.path.join('data', 'raw', 'weather')
  os.makedirs(data_output_dir, exist_ok=True)
  raw_output_file_path = os.path.join(data_output_dir, f'fetched_data_weather.json')
  with open(raw_output_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

def main():
  # get latitudes and longitudes from file
  station_data_path = os.path.join('data', 'preprocessed', 'bike', 'preprocessed_data_bike.json')
  station_data = read_json(station_data_path)
  latitudes, longitudes = extract_coordinates(station_data)

  weather_url = "https://api.open-meteo.com/v1/forecast"
  params = {
    "latitude": latitudes,
    "longitude": longitudes,
    "hourly": hourly_variables,
    "timezone": "Europe/Berlin",
    "forecast_days": 1
  }
  response = requests.get(weather_url, params=params)
  data = response.json()
  save_data_to_json(data)

if __name__ == "__main__":
  main()