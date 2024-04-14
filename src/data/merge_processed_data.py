import os
import csv
import json

script_dir = os.path.dirname(os.path.realpath(__file__))
mbajk_path = os.path.join(script_dir, '..', '..', 'data', 'raw', 'mbajk', 'fetched_data_mbike.json')
weather_path = os.path.join(script_dir, '..', '..', 'data', 'raw', 'weather', 'fetched_data_weather.json')

def main():
  with open(mbajk_path, 'r') as file:
    bike_data = json.load(file)

  with open(weather_path, 'r') as file:
    weather_data = json.load(file)

  if (bike_data and weather_data):
    process_data(bike_data, weather_data)
  else:
    print('Error reading json files')

def process_data(bike_data, weather_data):
  merged_data = [
    {**bike_station, **weather_forecast}
    for bike_station, weather_forecast in zip(bike_data, weather_data)
  ]
  script_dir = os.path.dirname(os.path.realpath(__file__))
  data_output_dir = os.path.join(script_dir, '..', '..', 'data', 'processed')
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

      # delete files
      json_file_paths = [mbajk_path, weather_path]
      for json_file_path in json_file_paths:
        if os.path.isfile(json_file_path):
          os.remove(json_file_path)
          print(f"{json_file_path} has been deleted.")
        else:
          print(f"{json_file_path} does not exist.")

if __name__ == "__main__":
  main()