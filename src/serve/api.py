from flask import Flask, request
import requests

def make_predictions(multi_array):
  return {'prediction': 12}

# API
app = Flask(__name__)

@app.route('/mbajk/predict', methods=['POST'])
def predict():
  try:
    data = request.get_json()

    print('data sent', data)

    result = make_predictions(data)

    return result, 200
  except Exception as e:
    return {'error': str(e)}, 400
  
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=3000)