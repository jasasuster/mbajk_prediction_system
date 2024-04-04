import pytest
from flask_testing import TestCase

from src.serve.api import create_app

class TestAPI(TestCase):
  def create_app(self):
    app = create_app()
    app.config['TESTING'] = True
    return app
  
  def test_predict(self):
    data = {
      "station_name": "DVORANA TABOR"
    }

    response = self.client.post('/mbajk/predict', json=data)

    assert response.status_code == 200

    assert isinstance(response.json['prediction'], list)
    assert len(response.json['prediction']) == 7
    for element in response.json['prediction']:
      assert isinstance(element, float)

if __name__ == '__main__':
  pytest.main()
