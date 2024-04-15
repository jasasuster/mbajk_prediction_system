import os
import csv
import json
import re
from datetime import datetime

def get_timestamp(file_name):
  return (file_name.split('_')[2] + '_' + file_name.split('_')[3]).split('.')[0]

def read_json(weather_directory_path):
  files = os.listdir(weather_directory_path)
  pattern = re.compile(r'preprocessed_data_\d{8}_\d{6}')
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

def main():
  mbajk_path = os.path.join('data', 'preprocessed', 'mbajk')
  weather_path = os.path.join('data', 'preprocessed', 'weather')

  bike_data, bike_timestamp = read_json(mbajk_path)
  weather_data, weather_timestamp = read_json(weather_path)

  if (bike_data and weather_data and bike_timestamp == weather_timestamp):
    process_data(bike_data, weather_data)
  else:
    print('Error reading json files')

def process_data(bike_data, weather_data):
  merged_data = [
    {**bike_station, **weather_forecast}
    for bike_station, weather_forecast in zip(bike_data, weather_data)
  ]
  data_output_dir = os.path.join('data', 'processed')
  os.makedirs(data_output_dir, exist_ok=True)
  for station in merged_data:
    station_number = station['number']
    filename = f"{station_number}.csv"
    filepath = os.path.join(data_output_dir, filename)
    file_exists = os.path.isfile(filepath)
    mode = 'a' if file_exists else 'w'
    with open(filepath, mode, newline='', encoding='utf-8') as file:
      writer = csv.writer(file)
      if not file_exists:
        header = station.keys()
        writer.writerow(header)
      writer.writerow(station.values())

if __name__ == "__main__":
  main()