import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import mlflow
import dagshub.auth
import dagshub
import src.models.train_model as train_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

import src.settings as settings

def evaluate_model(data_path, station_number):
  with mlflow.start_run(run_name=f"station_{station_number}", experiment_id="1", nested=True):
    mlflow.tensorflow.autolog()

    model = tf.keras.models.load_model('./models/'+station_number+'/multi_gru_model.h5')
    loaded_bk_scaler = joblib.load('./models/'+station_number+'/multi_gru_bk_scaler.pkl')
    loaded_fo_scaler = joblib.load('./models/'+station_number+'/multi_gru_fo_scaler.pkl')
    print("Model and scalers loaded")

    df = train_model.load_df(data_path)
    df_multi = df[['temperature', 'apparent_temperature', 'dew_point', 'precipitation_probability', 'surface_pressure', 'relative_humidity']]
    multi_array = df_multi.values
    print(f"Data length: {len(multi_array)}")
    
    bike_stands = multi_array[:, -1]
    bike_stands_normalized = loaded_bk_scaler.transform(bike_stands.reshape(-1, 1))

    other_features = multi_array[:,1:]
    other_features_normalized = loaded_fo_scaler.transform(other_features)

    multi_array_scaled = np.column_stack([bike_stands_normalized, other_features_normalized])

    window_size = 30

    X_final, y_final = train_model.create_dataset(multi_array_scaled, window_size)
    X_final = np.reshape(X_final, (X_final.shape[0], multi_array_scaled.shape[1], X_final.shape[1]))

    y_pred = model.predict(X_final)
    y_test = loaded_bk_scaler.inverse_transform(y_final.reshape(-1, 1)).flatten()
    y_pred = loaded_bk_scaler.inverse_transform(y_pred)
    print(f"Predictions: {y_pred}")

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    print(f"MAE: {mae}, MSE: {mse}, EVS: {evs}")

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("EVS", evs)

    # Save
    test_metrics = pd.DataFrame({
      'mae': [mae],
      'mse': [mse],
      'evs': [evs]
    })
    os.makedirs('./reports/'+station_number, exist_ok=True)
    test_metrics.to_csv(f'./reports/{station_number}/test_metrics.txt', sep='\t', index=False)
    
    mlflow.end_run()

def main():
  dagshub.auth.add_app_token(token=settings.MLFLOW_TRACKING_PASSWORD)
  dagshub.init("mbajk_prediction_system", settings.MLFLOW_TRACKING_USERNAME, mlflow=True)
  mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

  data_path = os.path.join('data', 'processed', '1_test.csv')  
  evaluate_model(data_path, str(1))

if __name__ == '__main__':
  main()