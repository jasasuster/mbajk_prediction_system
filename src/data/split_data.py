import os
import pandas as pd

def split_data(df):
  test_size = int(0.9 * len(df))
  train_df = df.head(test_size)
  test_df = df.iloc[test_size:]

  return train_df, test_df

def main():
  # Station 1
  df = pd.read_csv('data/processed/reference_data.csv')

  train_df, test_df = split_data(df)

  train_df.to_csv('data/processed/1_train.csv', index=False)
  test_df.to_csv('data/processed/1_test.csv', index=False)

  # Stations 2-29
  data_path = "data/processed/"
  for station_number in range(2, 30):
    file_path = os.path.join(data_path, f"{station_number}.csv")

    df = pd.read_csv(file_path)

    train_df, test_df = split_data(df)

    train_df.to_csv(f'data/processed/{station_number}_train.csv', index=False)
    test_df.to_csv(f'data/processed/{station_number}_test.csv', index=False)