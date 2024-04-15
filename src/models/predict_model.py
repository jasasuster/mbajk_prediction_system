from tensorflow.keras.models import load_model

import joblib
import ast
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.data.fetch_weather_data import fetch_weather_forecast

def hour_rounder(t):
  # Rounds to nearest hour by adding a timedelta hour if minute >= 30
  return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+timedelta(hours=t.minute//30)).strftime('%Y-%m-%dT%H:%M')

def fetch_weather_forecast(latitudes, longitudes, forecast_days, hourly_variables):
  weather_url = "https://api.open-meteo.com/v1/forecast"
  params = {
    "latitude": latitudes,
    "longitude": longitudes,
    "hourly": hourly_variables,
    "forecast_days": forecast_days,
    "timezone": "Europe/Berlin",
    "forecast_days": 1
  }
  response = requests.get(weather_url, params=params)
  data = response.json()
  return data

def check_missing_features(data, expected_features):
  for feature in expected_features:
    if feature not in data:
      return {'error': f'Missing feature: {feature}'}, 400
  return None

def create_multi_array(df, loaded_bk_scaler, loaded_fo_scaler):
  df_multi = df[['temperature', 'apparent_temperature', 'dew_point', 'precipitation_probability', 'surface_pressure', 'relative_humidity']]
  multi_array = df_multi.values
  
  bike_stands = multi_array[:, -1]
  bike_stands_normalized = loaded_bk_scaler.transform(bike_stands.reshape(-1, 1))

  other_features = multi_array[:,1:]
  other_features_normalized = loaded_fo_scaler.transform(other_features)

  multi_array_scaled = np.column_stack([bike_stands_normalized, other_features_normalized])

  multi_array_scaled = multi_array_scaled.reshape(1, multi_array_scaled.shape[1], multi_array_scaled.shape[0])

  return multi_array_scaled

def preprocess_data(data, bk_scaler, fo_scaler):
  expected_features = ['temperature', 'apparent_temperature', 'dew_point', 'precipitation_probability', 'surface_pressure', 'relative_humidity']

  missing_feature_error = check_missing_features(data, expected_features)
  if missing_feature_error:
    return missing_feature_error
    
  df = data
  df['date'] = pd.to_datetime(df['last_update'], unit='ms')
  df = df.sort_values(by='date')

  multi_array = create_multi_array(df, bk_scaler, fo_scaler)

  return multi_array

def predict(station_name):
  df = pd.read_csv(f'./data/processed/{station_name}.csv')

  position = df['position'][0]
  position = ast.literal_eval(position)
  latitude = position['lat']
  longitude = position['lng']
  hourly_variables = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability", "rain", "surface_pressure"]
  data = df.tail(30)  # get last 30 for prediction

  weather_data = fetch_weather_forecast(latitude, longitude, 3, hourly_variables)
  weather_data = pd.DataFrame(weather_data)
  rounded_time = hour_rounder(datetime.now(ZoneInfo("Europe/Ljubljana")))
  index = weather_data['hourly']['time'].index(rounded_time)

  weather_df = pd.DataFrame()
  for i, row in weather_data.iterrows():
    hourly = row['hourly'][index + 1:index + 8]
    weather_df[row.name] = hourly

  row_names = {'temperature_2m':'temperature', 'relative_humidity_2m':'relative_humidity', 'dew_point_2m':'dew_point'}
  weather_df = weather_df.rename(columns=row_names)

  model_dir = f'./models/{station_name}'
  loaded_model = load_model(f'{model_dir}/multi_gru_model.h5')
  loaded_bk_scaler = joblib.load(f'{model_dir}/multi_gru_bk_scaler.pkl')
  loaded_fo_scaler = joblib.load(f'{model_dir}/multi_gru_fo_scaler.pkl')

  predictions = []
  for i in range(7):
    multi_array = preprocess_data(data, loaded_bk_scaler, loaded_fo_scaler)
    prediction = loaded_model.predict(multi_array)
    prediction = loaded_bk_scaler.inverse_transform(prediction).tolist()[0][0]
    predictions.append(math.floor(prediction))

    forecast_data = weather_df.iloc[i]
    forecast_data['available_bike_stands'] = math.floor(prediction)

    new_row_df = pd.DataFrame([forecast_data])
    data = pd.concat([data, new_row_df], ignore_index=True)
    data = data.iloc[1:]

  return predictions