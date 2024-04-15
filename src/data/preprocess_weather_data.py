import os
import json
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

def get_timestamp(file_name):
  return (file_name.split('_')[2] + '_' + file_name.split('_')[3]).split('.')[0]

def read_json(weather_directory_path):
  files = os.listdir(weather_directory_path)
  pattern = re.compile(r'fetched_data_\d{8}_\d{6}')
  matching_files = [file for file in files if pattern.match(file)]

  last_timestamp = None

  matching_files.sort(key=lambda x: datetime.strptime(get_timestamp(x), '%Y%m%d_%H%M%S'))

  last_file = matching_files[-1] if matching_files else None
  if last_file:
    file_path = os.path.join(weather_directory_path, last_file)
    last_timestamp = get_timestamp(last_file)
    with open(file_path, 'r') as file:
      return json.load(file), last_timestamp
  else:
    print("No matching files found.")
    return None

def save_data_to_json(data, timestamp):
  data_output_dir = os.path.join('data', 'preprocessed', 'weather')
  os.makedirs(data_output_dir, exist_ok=True)
  raw_output_file_path = os.path.join(data_output_dir, f'preprocessed_data_{timestamp}.json')
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
  weather_data_path = os.path.join('data', 'raw', 'weather')
  weather_data, timestamp = read_json(weather_data_path)
  preprocessed_data = extract_data(weather_data)
  save_data_to_json(preprocessed_data, timestamp)

if __name__ == "__main__":
  main()