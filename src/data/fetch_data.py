import requests
import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from src.data.process_data import process_data

def hour_rounder(t):
  # Rounds to nearest hour by adding a timedelta hour if minute >= 30
  return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+timedelta(hours=t.minute//30)).strftime('%Y-%m-%dT%H:%M')

def fetch_weather_forecast(latitudes, longitudes, forecast_days, hourly_variables):
  weather_url = "https://api.open-meteo.com/v1/forecast"
  params = {
    "latitude": latitudes,
    "longitude": longitudes,
    "hourly": hourly_variables,
    "forecast_days": forecast_days
  }
  response = requests.get(weather_url, params=params)
  data = response.json()

  rounded_time = hour_rounder(datetime.now(ZoneInfo("Europe/Ljubljana")))
  closest_weather_data_list = []

  forecast_times = data[0]['hourly']['time']
  if rounded_time not in forecast_times:
    rounded_time = forecast_times[0]

  for forecast_object in data:
    if forecast_object:
      weather_data = forecast_object['hourly']
      closest_weather_data = {key: value[forecast_times.index(rounded_time)] for key, value in weather_data.items() if key != 'time'}
      renamed_weather_data = {k.replace('_2m', ''): v for k, v in closest_weather_data.items()}
      closest_weather_data_list.append(renamed_weather_data)

  return closest_weather_data_list

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

def main():
  forecast_days = 1
  hourly_variables = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability", "rain", "surface_pressure"]
  timestamp = datetime.now(ZoneInfo("Europe/Ljubljana")).strftime('%Y%m%d_%H%M%S')

  bike_station_data = fetch_bike_station_data("maribor", "5e150537116dbc1786ce5bec6975a8603286526b")
  latitudes, longitudes = extract_coordinates(bike_station_data)
  save_data_to_json(bike_station_data, timestamp, 'mbajk')

  weather_data = fetch_weather_forecast(latitudes, longitudes, forecast_days, hourly_variables)
  save_data_to_json(weather_data, timestamp, 'weather')

  process_data(bike_station_data, weather_data)

if __name__ == "__main__":
  main()