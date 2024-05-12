import os
import onnx
import joblib
import mlflow
import dagshub
import src.settings as settings

from mlflow.onnx import load_model as load_onnx
from mlflow.sklearn import load_model as load_sklearn

def download_latest_model_onnx(station_name, stage):
  model_name = f"model_{station_name}"

  try:
    client = mlflow.MlflowClient()
    model = load_onnx( client.get_latest_versions(name=model_name, stages=[stage])[0].source)

    onnx.save_model(model, f"./models/{station_name}/model_{stage}.onnx")

    return f"./models/{station_name}/model_{stage}.onnx"
  except IndexError:
    print(f"Error downloading {stage}, {model_name}")
    return None
  
def download_latest_scaler(station_name, scaler_type, stage):
  scaler_name = f"{scaler_type}_{station_name}"

  try:
    client = mlflow.MlflowClient()
    scaler = load_sklearn(client.get_latest_versions(name=scaler_name, stages=[stage])[0].source)
    return scaler
  except IndexError:
    print(f"Error downloading {stage}, {scaler_name}")
    return None

def save_production_model_and_scalers(station_name):
  try:
    client = mlflow.MlflowClient()

    model_version = client.get_latest_versions(name= f"model_{station_name}", stages=["staging"])[0].version
    client.transition_model_version_stage(f"model_{station_name}", model_version, "production")

    bk_scaler_version = client.get_latest_versions(name=f"bk_scaler_{station_name}", stages=["staging"])[0].version
    client.transition_model_version_stage(f"bk_scaler_{station_name}", bk_scaler_version, "production")
    
    fo_scaler_version = client.get_latest_versions(name= f"fo_scaler_{station_name}", stages=["staging"])[0].version
    client.transition_model_version_stage(f"fo_scaler_{station_name}", fo_scaler_version, "production")

  except IndexError:
    print(f"#####error##### \n replace_prod_model {station_name}")
    return

def download_all_models():
  dagshub.auth.add_app_token(token=settings.MLFLOW_TRACKING_PASSWORD)
  dagshub.init("mbajk_prediction_system", settings.MLFLOW_TRACKING_USERNAME, mlflow=True)
  mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

  try:
    for i in range(1, 30):
        station_dir = f"models/{i}/"
        os.makedirs(station_dir, exist_ok=True)
        model = download_latest_model_onnx(str(i), "production")
        stands_scaler = download_latest_scaler(str(i), "bk_scaler", "production")
        other_scaler = download_latest_scaler(str(i), "fo_scaler", "production")

        joblib.dump(stands_scaler, os.path.join(station_dir, 'bk_scaler.joblib'))
        joblib.dump(other_scaler, os.path.join(station_dir, 'fo_scaler.joblib'))

  except:
    print("Error getting models")
    return None