from tensorflow.keras.models import load_model

import joblib
import ast
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from src.data.fetch_weather_data import fetch_weather_forecast, hour_rounder

def check_missing_features(data, expected_features):
  for feature in expected_features:
    if feature not in data:
      return {'error': f'Missing feature: {feature}'}, 400
  return None

def create_multi_array(df, loaded_bk_scaler, loaded_fo_scaler):
  df_multi = df[['available_bike_stands', 'apparent_temperature', 'dew_point', 'precipitation_probability', 'surface_pressure']]

  multi_array = df_multi.values
  
  bike_stands = multi_array[:, -1]
  bike_stands_normalized = loaded_bk_scaler.transform(bike_stands.reshape(-1, 1))

  other_features = multi_array[:, :-1]
  other_features_normalized = loaded_fo_scaler.transform(other_features)

  multi_array_scaled = np.column.stack([bike_stands_normalized, other_features_normalized])

  multi_array_scaled = multi_array_scaled.reshape(1, multi_array_scaled.shape[1], multi_array_scaled.shape[0])

  return multi_array_scaled

def preprocess_data(data, bk_scaler, fo_scaler):  
  expected_features = ['available_bike_stands', 'apparent_temperature', 'dew_point', 'precipitation_probability', 'surface_pressure']

  for obj in data:
    missing_feature_error = check_missing_features(obj, expected_features)
    if missing_feature_error:
      return missing_feature_error
    
  df = pd.DataFrame(data)
  df['date'] = pd.to_datetime(df['date'])
  df = df.sort_values(by='date')

  df_multi = df['available_bike_stands']
  multi_array = create_multi_array(df_multi, bk_scaler, fo_scaler)

  return multi_array

def predict(station_name):
  df = pd.read_csv(f'./data/processed/{station_name}.csv')

  position = df['position'][0]
  position = ast.literal_eval(position)
  latitude = position['lat']
  longitude = position['lng']
  hourly_variables = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability", "rain", "surface_pressure"]
  data = df.tail(20)  # get last 20 for prediction

  weather_data = fetch_weather_forecast(latitude, longitude, 3, hourly_variables)
  weather_data = pd.DataFrame(weather_data)

  rounded_time = hour_rounder(datetime.now(ZoneInfo("Europe/Ljubljana")))

  index = weather_data['hourly']['time'].index(rounded_time)

  # new_df = pd.DataFrame(columns=['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature', 'precipitation_probability', 'rain', 'surface_pressure'])
  new_df = pd.DataFrame()
  for i, row in weather_data.iterrows():
    hourly = row['hourly'][index + 1:index + 8]
    # new_df.loc[row.name] = hourly
    new_df[row.name] = hourly

  # row_names = {'temperature_2m':'temperature', 'relative_humidity_2m':'relative_humidity', 'dew_point_2m':'dew_point'}
  # new_df = new_df.rename(index=row_names)
  
  print('new_df:', new_df)
  print('new_df.t:', new_df.transpose())

  weather_data = weather_data.head(7) # get first 7 hours for prediction

  model_dir = f'./models/{station_name}'
  loaded_model = load_model(f'{model_dir}/multi_gru_model.h5')
  loaded_bk_scaler = joblib.load(f'{model_dir}/multi_gru_bk_scaler.pkl')
  loaded_fo_scaler = joblib.load(f'{model_dir}/multi_gru_fo_scaler.pkl')

  predictions = []
  for i in range(7):
    print(i,':')
    multi_array = preprocess_data(data, loaded_bk_scaler, loaded_fo_scaler)
    prediction = loaded_model.predict(multi_array)
    prediction = loaded_bk_scaler.inverse_transform(prediction)
    predictions.append(prediction.tolist()[0][0])

    print('\tprediction:', prediction)

    forecast_data = new_df.iloc[i]
    forecast_data['available_bike_stands'] = prediction

    print('\tforecast_data:', forecast_data)

    data = data.append(forecast_data, ignore_index=True)

    data = data.iloc[1:]

  return predictions