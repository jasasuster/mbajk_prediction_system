import json
import os

def read_json(file_path):
  with open(file_path, 'r') as file:
    return json.load(file)

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

def save_data_to_json(data):
  data_output_dir = os.path.join('data', 'preprocessed', 'bike')
  os.makedirs(data_output_dir, exist_ok=True)
  raw_output_file_path = os.path.join(data_output_dir, f'preprocessed_data_bike.json')
  with open(raw_output_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

def main():
  bike_data_path = os.path.join('data', 'raw', 'bike', 'fetched_data_bike.json')
  data = read_json(bike_data_path)
  extracted_data = extract_data(data)
  save_data_to_json(extracted_data)

  # delete raw data
  if os.path.isfile(bike_data_path):
    os.remove(bike_data_path)
    print(f"{bike_data_path} has been deleted.")
  else:
    print(f"{bike_data_path} does not exist.")

if __name__ == "__main__":
  main()