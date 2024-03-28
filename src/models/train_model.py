import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

bk_scaler = StandardScaler()
fo_scaler = StandardScaler()

def fill_missing_values(df):
  missing_values_df = df[df.isnull().any(axis=1)]
  complete_data_df = df.dropna()
  missing_values_columns = df.columns[df.isnull().any()].tolist()

  for column in missing_values_columns:
    X = complete_data_df[['available_bike_stands']]
    y = complete_data_df[column]

    model = RandomForestRegressor()
    model.fit(X, y)

    predictions = model.predict(missing_values_df[['available_bike_stands']])
    df.loc[missing_values_df.index, column] = predictions

  missing_values = df.isna().sum()
  print('Missing values after:\n', missing_values)

def create_multi_array(df):
  df_multi = df[['available_bike_stands', 'apparent_temperature', 'dew_point', 'precipitation_probability', 'surface_pressure']]
  multi_array = df_multi.values
  return multi_array

def split_and_scale_bk(train, test):
  train_bike_stands = np.array(train[:,0])
  test_bike_stands = np.array(test[:,0])

  train_bike_stands_scaled = bk_scaler.fit_transform(train_bike_stands.reshape(-1, 1))
  test_bike_stands_scaled = bk_scaler.transform(test_bike_stands.reshape(-1, 1))

  return train_bike_stands_scaled, test_bike_stands_scaled

def split_and_scale_fo(train, test):
  train_features_other = np.array(train[:,1:])
  test_features_other = np.array(test[:,1:])

  train_features_other_scaled = fo_scaler.fit_transform(train_features_other)
  test_features_other_scaled = fo_scaler.transform(test_features_other)

  return train_features_other_scaled, test_features_other_scaled

def combine_features(train_bike_stands_scaled, test_bike_stands_scaled, train_features_other_scaled, test_features_other_scaled):
  train_scaled = np.column_stack([train_bike_stands_scaled, train_features_other_scaled])
  test_scaled = np.column_stack([test_bike_stands_scaled, test_features_other_scaled])

  return train_scaled, test_scaled

def split(multi_array):
  train_size = len(multi_array) - 300
  train, test = multi_array[0:train_size], multi_array[train_size:]

  return train, test

def create_dataset(dataset, window_size=45):
  X, y = [], []
  for i in range(len(dataset) - window_size):
      window = dataset[i:i+window_size, :]
      target = dataset[i+window_size, 0]
      X.append(window)
      y.append(target)
  return np.array(X), np.array(y)

def reshape_for_model(train_scaled, test_scaled):
  window_size = 45

  X_train, y_train = create_dataset(train_scaled, window_size)
  X_test, y_test = create_dataset(test_scaled, window_size)

  print('Oblika učnih podatkov: X_train: ', X_train.shape, ', y_train:', y_train.shape)
  print('Oblika testnih podatkov: X_test: ', X_test.shape, ', y_test:', y_test.shape)

  X_train = np.reshape(X_train, (X_train.shape[0], train_scaled.shape[1], X_train.shape[1]))
  X_test = np.reshape(X_test, (X_test.shape[0], test_scaled.shape[1], X_test.shape[1]))

  print('Preoblikovani X_train:', X_train.shape)
  print('Preoblikovani X_test:', X_test.shape)

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
  print('Calculating and saving metrics...')

  y_test_inv = bk_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

  gru_mae = mean_absolute_error(y_test_inv, gru_predictions_inv)
  gru_mse = mean_squared_error(y_test_inv, gru_predictions_inv)
  gru_evs = explained_variance_score(y_test_inv, gru_predictions_inv)

  model_history_df = pd.DataFrame(model_history.history)
  model_history_df.to_csv('./reports/train_metrics.txt', sep='\t', index=False)

  test_metrics = pd.DataFrame({
    'mae': [gru_mae],
    'mse': [gru_mse],
    'evs': [gru_evs]
  })

  test_metrics.to_csv('./reports/test_metrics.txt', sep='\t', index=False)

def train():
  df = pd.read_csv('./mbajk_dataset.csv')

  df['date'] = pd.to_datetime(df['date'])
  df = df.sort_values(by='date')

  df.set_index('date', inplace=True)
  df_hourly = df.resample('h').mean()
  df_hourly.reset_index(inplace=True)
  df_hourly = df_hourly.dropna()

  multi_array = create_multi_array(df_hourly)

  train, test = split(multi_array)
  train_bike_stands_scaled, test_bike_stands_scaled = split_and_scale_bk(train, test)
  train_features_other_scaled, test_features_other_scaled = split_and_scale_fo(train, test)
  train_scaled, test_scaled = combine_features(train_bike_stands_scaled, test_bike_stands_scaled, train_features_other_scaled, test_features_other_scaled)

  print('train shape', train_scaled.shape)
  print('test shape', test_scaled.shape)

  X_train, y_train, X_test, y_test = reshape_for_model(train_scaled, test_scaled)

  input_shape = (X_train.shape[1], X_train.shape[2])

  model = build_gru_model(input_shape)
  print('Training model...')
  model_history = model.fit(X_train, y_train, epochs=15, validation_split=0.2)

  predictions = model.predict(X_test)
  predictions_inv = bk_scaler.inverse_transform(predictions)

  calculate_and_save_metrics(y_test, predictions_inv, model_history)

  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

  model.save(f'./models/multi_gru_model_{timestamp}.h5')
  joblib.dump(bk_scaler, f'./models/multi_gru_bk_scaler{timestamp}.pkl')
  joblib.dump(fo_scaler, f'./models/multi_gru_fo_scaler{timestamp}.pkl')