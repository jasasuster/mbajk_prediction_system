import requests
import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

def main():
  contract = "maribor"
  api_key = "5e150537116dbc1786ce5bec6975a8603286526b"
  url = f"https://api.jcdecaux.com/vls/v1/stations?contract={contract}&apiKey={api_key}"
  response = requests.get(url)
  data = response.json()
  save_data_to_json(data)

def save_data_to_json(data):
  timestamp = datetime.now(ZoneInfo("Europe/Ljubljana")).strftime('%Y%m%d_%H%M%S')
  data_output_dir = os.path.join('data', 'raw', 'mbajk')
  os.makedirs(data_output_dir, exist_ok=True)
  raw_output_file_path = os.path.join(data_output_dir, f'fetched_data_{timestamp}.json')
  with open(raw_output_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
  main()