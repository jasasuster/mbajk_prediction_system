from tensorflow.keras.models import load_model
from flask import Flask, request

import pandas as pd
import numpy as np
import joblib

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

def make_predictions(uni_array):
  print('making predictions')
  prediction = loaded_model.predict(uni_array)
  prediction = loaded_scaler.inverse_transform(prediction)
  print('prediction', prediction)
  return {'prediction': prediction.tolist()[0][0]}

# API
app = Flask(__name__)

@app.route('/mbajk/predict', methods=['POST'])
def predict():
  try:
    data = request.get_json()

    uni_array = preprocess_data(data)

    result = make_predictions(uni_array)

    return result, 200
  except Exception as e:
    return {'error': str(e)}, 400
  
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=3000)

# [
#   '{{repeat(45)}}',
#   {
#     "date": '{{date(new Date(2023, 11, 4), new Date(2023, 11, 10), "YYYY-MM-dd HH:mm:ss+00:00")}}',
#     "temperature":'{{floating(9.0, 36.0)}}',
#     "relative_humidity":'{{integer(27, 100)}}',
#     "dew_point":'{{floating(6.0, 23.5)}}',
#     "apparent_temperature": '{{floating(9.0, 38.0)}}',
#     "precipitation_probability": '{{integer(0, 100)}}',
#     "rain":'{{floating(1.0, 25.0)}}',
#     "surface_pressure": '{{floating(950.0, 1000.0)}}',
#     'bike_stands':'22',
#     'available_bike_stands':'{{integer(1, 22)}}'
#   }
# ]