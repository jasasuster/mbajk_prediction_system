import requests
import csv
import os
import json
from datetime import datetime
import pandas as pd

def fetch_weather_forecast(latitudes, longitudes, forecast_days, hourly_variables):
  weather_url = "https://api.open-meteo.com/v1/forecast"
  params = {
    "latitude": latitudes,
    "longitude": longitudes,
    "hourly": hourly_variables,
    "forecast_days": forecast_days
  }
  response = requests.get(weather_url, params=params)
  return response.json()

def extract_coordinates(station_data):
    latitudes = [station['position']['lat'] for station in station_data]
    longitudes = [station['position']['lng'] for station in station_data]
    return latitudes, longitudes

def fetch_bike_station_data(contract, api_key):
  url = f"https://api.jcdecaux.com/vls/v1/stations?contract={contract}&apiKey={api_key}"
  response = requests.get(url)
  data = response.json()
  return data

def save_data_to_json(data, timestamp, dict):
  script_dir = os.path.dirname(os.path.realpath(__file__))
  data_output_dir = os.path.join(script_dir, '..', '..', 'data', 'raw', dict)
  os.makedirs(data_output_dir, exist_ok=True)
  raw_output_file_path = os.path.join(data_output_dir, f'fetched_data_{timestamp}.json')
  with open(raw_output_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
  return raw_output_file_path

def save_bike_data_to_csv(data, timestamp, dict):
  script_dir = os.path.dirname(os.path.realpath(__file__))
  data_output_dir = os.path.join(script_dir, '..', '..', 'data', 'processed', dict)
  os.makedirs(data_output_dir, exist_ok=True)
  for station in data:
    station_name = station['name']
    filename = f"{station_name.replace(' ', '_')}.csv"
    filepath = os.path.join(data_output_dir, filename)
    file_exists = os.path.isfile(filepath)
    mode = 'a' if file_exists else 'w'
    with open(filepath, mode, newline='', encoding='utf-8') as file:
      writer = csv.writer(file)
      if not file_exists:
        header = station.keys()
        writer.writerow(header)
      writer.writerow(station.values())

def save_weather_data_to_csv(data, timestamp, dict):
  script_dir = os.path.dirname(os.path.realpath(__file__))
  data_output_dir = os.path.join(script_dir, '..', '..', 'data', 'processed', dict)
  os.makedirs(data_output_dir, exist_ok=True)
  filename = f"weather_{timestamp}.csv"
  filepath = os.path.join(data_output_dir, filename)
  file_exists = os.path.isfile(filepath)
  mode = 'a' if file_exists else 'w'
  with open(filepath, mode, newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    if not file_exists:
      header = data[0].keys()
      writer.writerow(header)
    for entry in data:
      writer.writerow(entry.values())

def main():
  forecast_days = 1
  hourly_variables = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability", "rain", "surface_pressure"]
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

  bike_station_data = fetch_bike_station_data("maribor", "5e150537116dbc1786ce5bec6975a8603286526b")
  latitudes, longitudes = extract_coordinates(bike_station_data)
  save_data_to_json(bike_station_data, timestamp, 'mbajk')
  save_bike_data_to_csv(bike_station_data, timestamp, 'mbajk')

  weather_data = fetch_weather_forecast(latitudes, longitudes, forecast_days, hourly_variables)
  save_data_to_json(weather_data, timestamp, 'weather')
  save_weather_data_to_csv(weather_data, timestamp, 'weather')

if __name__ == "__main__":
  main()
