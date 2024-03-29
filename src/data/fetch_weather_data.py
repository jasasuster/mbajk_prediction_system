import requests
import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from src.data.process_data import process_data


def hour_rounder(t):
  # Rounds to nearest hour by adding a timedelta hour if minute >= 30
  return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+timedelta(hours=t.minute//30)).strftime('%Y-%m-%dT%H:%M')

def fetch_weather_forecast(latitudes, longitudes, forecast_days, hourly_variables):
  print('fetch_weather_forecast')
  weather_url = "https://api.open-meteo.com/v1/forecast"
  params = {
    "latitude": latitudes,
    "longitude": longitudes,
    "hourly": hourly_variables,
    "forecast_days": forecast_days
  }
  print('params:', params)
  response = requests.get(weather_url, params=params)
  data = response.json()
  print('data:', data)

  rounded_time = hour_rounder(datetime.now(ZoneInfo("Europe/Ljubljana")))
  closest_weather_data_list = []

  forecast_times = data[0]['hourly']['time']
  if rounded_time not in forecast_times:
    rounded_time = forecast_times[0]
  print('rounded_time:', rounded_time)

  for forecast_object in data:
    if forecast_object:
      weather_data = forecast_object['hourly']
      closest_weather_data = {key: value[forecast_times.index(rounded_time)] for key, value in weather_data.items() if key != 'time'}
      renamed_weather_data = {k.replace('_2m', ''): v for k, v in closest_weather_data.items()}
      closest_weather_data_list.append(renamed_weather_data)

  print('closest_weather_data_list:', closest_weather_data_list)
  return closest_weather_data_list