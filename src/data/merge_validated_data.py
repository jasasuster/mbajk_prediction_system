import os
import pandas as pd

def main():
  validated_path = os.path.join('data', 'validate', 'current_data.csv')
  
  # Read the CSV file
  df = pd.read_csv(validated_path)
  
  # Define the path for the reference data CSV file
  reference_path = os.path.join('data', 'processed', 'reference_data.csv')
  
  if not os.path.exists(reference_path):
    os.makedirs(os.path.dirname(reference_path), exist_ok=True)
  
  # Write the data to the reference data CSV file
  df.to_csv(reference_path, index=False)
