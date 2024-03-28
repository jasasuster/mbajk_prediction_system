from tensorflow.keras.models import load_model

import joblib
import pandas as pd
import numpy as np

loaded_model = load_model('./models/multi_gru_model.h5')
loaded_bk_scaler = joblib.load('./models/multi_gru_bk_scaler.pkl')
loaded_fo_scaler = joblib.load('./models/multi_gru_fo_scaler.pkl')

def check_missing_features(data, expected_features):
  for feature in expected_features:
    if feature not in data:
      return {'error': f'Missing feature: {feature}'}, 400
  return None

def create_multi_array(df):
  df_multi = df[['available_bike_stands', 'apparent_temperature', 'dew_point', 'precipitation_probability', 'surface_pressure']]

  multi_array = df_multi.values
  
  bike_stands = multi_array[:, -1]
  bike_stands_normalized = loaded_bk_scaler.transform(bike_stands.reshape(-1, 1))

  other_features = multi_array[:, :-1]
  other_features_normalized = loaded_fo_scaler.transform(other_features)

  multi_array_scaled = np.column.stack([bike_stands_normalized, other_features_normalized])

  multi_array_scaled = multi_array_scaled.reshape(1, multi_array_scaled.shape[1], multi_array_scaled.shape[0])

  return multi_array_scaled

def preprocess_data(data):  
  expected_features = ['available_bike_stands', 'apparent_temperature', 'dew_point', 'precipitation_probability', 'surface_pressure']

  for obj in data:
    missing_feature_error = check_missing_features(obj, expected_features)
    if missing_feature_error:
      return missing_feature_error
    
  df = pd.DataFrame(data)
  df['date'] = pd.to_datetime(df['date'])
  df = df.sort_values(by='date')

  df_multi = df['available_bike_stands']
  multi_array = create_multi_array(df_multi)

  return multi_array

def predict(data):
  if len(data) != 45:
    return {'error': 'Invalid data length'}, 400

  multi_array = preprocess_data(data)

  prediction = loaded_model.predict(multi_array)
  prediction = loaded_bk_scaler.inverse_transform(prediction)

  return prediction.tolist()[0][0]