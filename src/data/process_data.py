import os
import csv

def process_data(bike_data, weather_data):
  merged_data = [
    {**bike_station, **weather_forecast}
    for bike_station, weather_forecast in zip(bike_data, weather_data)
  ]
  script_dir = os.path.dirname(os.path.realpath(__file__))
  data_output_dir = os.path.join(script_dir, '..', '..', 'data', 'processed')
  os.makedirs(data_output_dir, exist_ok=True)
  for station in merged_data:
    # station_name = station['name']
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