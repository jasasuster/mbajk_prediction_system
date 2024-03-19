import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

scaler = StandardScaler()

def create_uni_array(df):
  df_uni = df['available_bike_stands']
  uni_array = df_uni.values.reshape(-1, 1)

  return uni_array

def split_and_scale(uni_array):
  train_size = len(uni_array) - 1488
  train, test = uni_array[0:train_size], uni_array[train_size:]

  train_scaled = scaler.fit_transform(train)
  test_scaled = scaler.transform(test)

  return train_scaled, test_scaled

def create_dataset(dataset, window_size=186):
  X, y = [], []
  for i in range(len(dataset) - window_size):
      window = dataset[i:i+window_size, 0]
      target = dataset[i+window_size, 0]
      X.append(window)
      y.append(target)
  return np.array(X), np.array(y)

def reshape_for_model(train_scaled, test_scaled):
  window_size = 45

  X_train, y_train = create_dataset(train_scaled, window_size)
  X_test, y_test = create_dataset(test_scaled, window_size)

  X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
  X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

  return X_train, y_train, X_test, y_test

def build_gru_model(input_shape):
  model = Sequential()
  model.add(GRU(32, activation='relu', input_shape=input_shape, return_sequences=True))
  model.add(GRU(32, activation='relu'))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  return model

def calculate_and_save_metrics(y_test, gru_predictions_inv, model_history):
  y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

  gru_mae = mean_absolute_error(y_test_inv, gru_predictions_inv)
  gru_mse = mean_squared_error(y_test_inv, gru_predictions_inv)
  gru_evs = explained_variance_score(y_test_inv, gru_predictions_inv)

  model_history_df = pd.DataFrame(model_history.history)
  model_history_df.to_csv('../reports/train_metrics.txt', sep='\t', index=False)

  test_metrics = pd.DataFrame({
    'mae': [gru_mae],
    'mse': [gru_mse],
    'evs': [gru_evs]
  })

  test_metrics.to_csv('../reports/test_metrics.txt', sep='\t', index=False)

def train():
  df = pd.read_csv('../mbajk_dataset.csv')

  df['date'] = pd.to_datetime(df['date'])
  df = df.sort_values(by='date')

  df.set_index('date', inplace=True)
  df_hourly = df.resample('H').mean()
  df_hourly.reset_index(inplace=True)

  uni_array = create_uni_array(df_hourly)

  train_scaled, test_scaled = split_and_scale(uni_array)

  X_train, y_train, X_test, y_test = reshape_for_model(train_scaled, test_scaled)

  input_shape = (X_train.shape[1], X_train.shape[2])

  model = build_gru_model(input_shape)
  model_history = model.fit(X_train, y_train, epochs=15, validation_split=0.2)

  predictions = model.predict(X_test)
  predictions_inv = scaler.inverse_transform(predictions)

  calculate_and_save_metrics(y_test, predictions_inv, model_history)

  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

  model.save(f'../models/uni_gru_model_{timestamp}.h5')
  joblib.dump(scaler, f'../models/uni_gru_scaler{timestamp}.pkl')