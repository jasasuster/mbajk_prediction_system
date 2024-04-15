import json
import os
import re
from datetime import datetime

def get_timestamp(file_name):
  return (file_name.split('_')[2] + '_' + file_name.split('_')[3]).split('.')[0]

def read_json(bike_directory_path):
  files = os.listdir(bike_directory_path)
  pattern = re.compile(r'fetched_data_\d{8}_\d{6}')
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

def extract_data(data):
  extracted_data = []
  for item in data:
    extracted_item = {
      'number': item['number'],
      'last_update': item['last_update'],
      'position': item['position'],
      'bike_stands': item['bike_stands'],
      'available_bike_stands': item['available_bike_stands'],
      'available_bikes': item['available_bikes']
    }
    extracted_data.append(extracted_item)
  return extracted_data

def save_data_to_json(data, timestamp):
  data_output_dir = os.path.join('data', 'preprocessed', 'mbajk')
  os.makedirs(data_output_dir, exist_ok=True)
  raw_output_file_path = os.path.join(data_output_dir, f'preprocessed_data_{timestamp}.json')
  with open(raw_output_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

def main():
  bike_directory_path = os.path.join('data', 'raw', 'mbajk')
  data, timestamp = read_json(bike_directory_path)
  if data:
    extracted_data = extract_data(data)
    save_data_to_json(extracted_data, timestamp)

if __name__ == "__main__":
  main()