import mlflow
import dagshub
import pandas as pd

from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

import src.db.db as db
import src.settings as settings

def test_predictions(station_number, station_data):
  station_predictions = db.predictions_today(station_number)

  if not station_predictions:
    print(f"No predictions for today for station {station_number}")
    mlflow.end_run()
    return
  
  mlflow.start_run(run_name=f"station_{station_number}", experiment_id=3)

  mapped_predictions = []
  for prediction in station_predictions:
    predictions_hourly = prediction['predictions']
    date = pd.to_datetime(prediction['date'])
    station_data.reset_index(inplace=True)
    station_data = station_data.set_index('date')
    for i, pred in enumerate(predictions_hourly):
      target_time = date + timedelta(hours=i)
      nearest_timestamp_index = station_data.index.get_indexer([target_time], method='nearest')
      nearest_timestamp_bike_data = station_data.iloc[nearest_timestamp_index].to_dict()
      mapped_predictions.append({
        'date': target_time,
        'prediction': pred,
        'true': nearest_timestamp_bike_data['available_bike_stands']
      })

  y_true = [pred['true'] for pred in mapped_predictions]
  y_pred = [pred['prediction'] for pred in mapped_predictions]

  mse = mean_squared_error(y_true, y_pred)
  mae = mean_absolute_error(y_true, y_pred)
  evs = explained_variance_score(y_true, y_pred)

  mlflow.log_metric('mse', mse)
  mlflow.log_metric('mae', mae)
  mlflow.log_metric('evs', evs)

  mlflow.end_run()

def main():
  dagshub.auth.add_app_token(token=settings.MLFLOW_TRACKING_PASSWORD)
  dagshub.init("mbajk_prediction_system", settings.MLFLOW_TRACKING_USERNAME, mlflow=True)
  mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

  for station_number in range(1, 30):
    station_data = pd.read_csv(f"./data/processed/{station_number}.csv")
    station_data['date'] = pd.to_datetime(station_data['last_update'], unit='ms')
    station_data.sort_values(by='date', inplace=True)
    station_data.drop(columns=['last_update'], inplace=True)
    station_data.reset_index(inplace=True)
    station_data.set_index('date')

    test_predictions(f"station_{station_number}", station_data)

if __name__ == '__main__':
  main()