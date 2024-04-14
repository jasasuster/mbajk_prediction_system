import requests
import os
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

hourly_variables = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability", "rain", "surface_pressure"]

def extract_coordinates(station_data):
  latitudes = [station['position']['lat'] for station in station_data]
  longitudes = [station['position']['lng'] for station in station_data]
  return latitudes, longitudes

def get_timestamp(file_name):
  return (file_name.split('_')[2] + '_' + file_name.split('_')[3]).split('.')[0]

def read_json(bike_directory_path):
  files = os.listdir(bike_directory_path)
  pattern = re.compile(r'preprocessed_data_\d{8}_\d{6}')
  matching_files = [file for file in files if pattern.match(file)]

  last_timestamp = None

  matching_files.sort(key=lambda x: datetime.strptime(get_timestamp(x), '%Y%m%d_%H%M%S'))

  last_file = matching_files[-1] if matching_files else None
  if last_file:
    file_path = os.path.join(bike_directory_path, last_file)
    last_timestamp = get_timestamp(last_file)
    with open(file_path, 'r') as file:
      return json.load(file), last_timestamp
  else:
    print("No matching files found.")
    return None

def save_data_to_json(data, timestamp):
  data_output_dir = os.path.join('data', 'raw', 'weather')
  os.makedirs(data_output_dir, exist_ok=True)
  raw_output_file_path = os.path.join(data_output_dir, f'fetched_data_{timestamp}.json')
  with open(raw_output_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

def main():
  # get latitudes and longitudes from file
  station_data_path = os.path.join('data', 'preprocessed', 'mbajk')
  station_data, timestamp = read_json(station_data_path)
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
  save_data_to_json(data, timestamp)

if __name__ == "__main__":
  main()