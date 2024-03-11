import requests
import csv
import os
import json

# Fetch data
url = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"
response = requests.get(url)
data = response.json()

# Get directory of the current script and define the output file path
script_dir = os.path.dirname(os.path.realpath(__file__)) # Get the directory of the current script
data_output_dir = os.path.join(script_dir, '..', '..', 'data', 'raw') # Adjust the path as necessary
raw_output_file_path = os.path.join(data_output_dir, 'fetched_data.json')
os.makedirs(data_output_dir, exist_ok=True)

# Write the data to JSON file
with open(raw_output_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

data_output_dir = os.path.join(script_dir, '..', '..', 'data', 'processed')
processed_output_file_path = os.path.join(data_output_dir, 'fetched_data.csv')
os.makedirs(data_output_dir, exist_ok=True)

# Write the data to a CSV file
with open(processed_output_file_path, 'w', newline='', encoding='utf-8') as file:
    fieldnames = ['number', 'contract_name', 'name', 'address', 'lat', 'lng', 'banking', 'bonus', 'bike_stands', 'available_bike_stands', 'available_bikes', 'status', 'last_update']
    
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    
    for item in data:
        position = item['position']
        item['lat'] = position['lat']
        item['lng'] = position['lng']
        
        del item['position']
        
        # Write the item to the CSV file
        writer.writerow(item)

print(f'Data has been written to {raw_output_file_path}')
print(f'Data has been written to {processed_output_file_path}')
