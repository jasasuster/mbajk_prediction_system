import mlflow
from mlflow.onnx import load_model as load_onnx
from mlflow.sklearn import load_model as load_sklearn
import onnx

def download_model_onnx(station_name, stage):
  model_name = f"model_{station_name}"

  try:
    client = mlflow.MlflowClient()
    model = load_onnx( client.get_latest_versions(name=model_name, stages=[stage])[0].source)

    onnx.save_model(model, f"./models/{station_name}/model_{stage}.onnx")

    return f"./models/{station_name}/model_{stage}.onnx"
  except IndexError:
    print(f"Error downloading {stage}, {model_name}")
    return None
  
def download_scaler(station_name, scaler_type, stage):
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
    client.transition_model_version_stage(f"model={station_name}", model_version, "production")

    bk_scaler_version = client.get_latest_versions(name=f"bk_scaler_{station_name}", stages=["staging"])[0].version
    client.transition_model_version_stage(f"bk_scaler_{station_name}", bk_scaler_version, "production")
    
    fo_scaler_version = client.get_latest_versions(name= f"fo_scaler_{station_name}", stages=["staging"])[0].version
    client.transition_model_version_stage(f"fo_scaler_{station_name}", fo_scaler_version, "production")

  except IndexError:
    print(f"#####error##### \n replace_prod_model {station_name}")
    return