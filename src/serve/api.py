from flask import Flask, request
from flask_cors import CORS

from src.models.predict_model import predict
from src.models.mlflow_client import download_all_models

def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route('/mbajk/predict', methods=['POST'])
    def predict_val():
        try:
            data = request.get_json()
            station_number = data['station_number']
            print(station_number)
            predictions = predict(station_number)

            return {'predictions': predictions}, 200
        except Exception as e:
            return {'error': str(e)}, 400

    return app

def main():
    app = create_app()

    download_all_models()

    app.run(host='0.0.0.0', port=3000)

if __name__ == '__main__':
    main()

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