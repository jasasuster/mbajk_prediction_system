import requests
import csv
import os
import json
from datetime import datetime

# Fetch data
url = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"
response = requests.get(url)
data = response.json()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Get directory of the current script and define the output file path
script_dir = os.path.dirname(os.path.realpath(__file__)) # Get the directory of the current script
data_output_dir = os.path.join(script_dir, '..', '..', 'data', 'raw') # Adjust the path as necessary
raw_output_file_path = os.path.join(data_output_dir, f'fetched_data_{timestamp}.json')
os.makedirs(data_output_dir, exist_ok=True)

# Write the data to JSON file
with open(raw_output_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

data_output_dir = os.path.join(script_dir, '..', '..', 'data', 'processed')
os.makedirs(data_output_dir, exist_ok=True)

# Write the data to a CSV file
for station in data:
    # Extract the station's unique name attribute
    station_name = station['name']
    
    # Create a filename for the CSV file based on the station's name
    filename = f"{station_name.replace(' ', '_')}.csv"
    filepath = os.path.join(data_output_dir, filename)
    
    # Check if the file already exists
    file_exists = os.path.isfile(filepath)
    
    # Open the file in append mode if it exists, otherwise create a new file and write the header
    mode = 'a' if file_exists else 'w'
    with open(filepath, mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header if the file is new
        if not file_exists:
            header = station.keys()
            writer.writerow(header)
        
        # Write the station's data
        writer.writerow(station.values())

print(f'Data has been written to {raw_output_file_path}')
print("Data has been written to separate CSV files.")
