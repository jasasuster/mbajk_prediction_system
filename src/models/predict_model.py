from tensorflow.keras.models import load_model

import joblib
import pandas as pd

loaded_model = load_model('./models/uni_gru_model.h5')
loaded_scaler = joblib.load('./models/uni_gru_scaler.pkl')

def check_missing_features(data, expected_features):
  for feature in expected_features:
    if feature not in data:
      return {'error': f'Missing feature: {feature}'}, 400
  return None

def preprocess_data(data):
  if len(data) != 45:
    return {'error': 'Invalid data length'}, 400
  
  expected_features = ['date', 'available_bike_stands']

  for obj in data:
    missing_feature_error = check_missing_features(obj, expected_features)
    if missing_feature_error:
      return missing_feature_error
    
  df = pd.DataFrame(data)
  df['date'] = pd.to_datetime(df['date'])
  df = df.sort_values(by='date')

  df_uni = df['available_bike_stands']
  uni_array = df_uni.values.reshape(-1, 1)
  uni_array = loaded_scaler.transform(uni_array)

  uni_array = uni_array.reshape(uni_array.shape[1], 1, uni_array.shape[0])

  return uni_array

def predict(data):
  uni_array = preprocess_data(data)

  prediction = loaded_model.predict(uni_array)
  prediction = loaded_scaler.inverse_transform(prediction)

  return prediction.tolist()[0][0]