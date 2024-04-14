import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

def read_json(file_path):
  with open(file_path, 'r') as file:
    return json.load(file)
  
def save_data_to_json(data):
  data_output_dir = os.path.join('data', 'preprocessed', 'weather')
  os.makedirs(data_output_dir, exist_ok=True)
  raw_output_file_path = os.path.join(data_output_dir, f'preprocessed_data_weather.json')
  with open(raw_output_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

def hour_rounder(t):
  # Rounds to nearest hour by adding a timedelta hour if minute >= 30
  return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+timedelta(hours=t.minute//30)).strftime('%Y-%m-%dT%H:%M')

def extract_data(data):
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

def main():
  weather_data_path = os.path.join('data', 'raw', 'weather', 'fetched_data_weather.json')
  weather_data = read_json(weather_data_path)
  preprocessed_data = extract_data(weather_data)
  save_data_to_json(preprocessed_data)

  # delete raw data
  if os.path.isfile(weather_data_path):
    os.remove(weather_data_path)
    print(f"{weather_data_path} has been deleted.")
  else:
    print(f"{weather_data_path} does not exist.")
  pass

if __name__ == "__main__":
  main()