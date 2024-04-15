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
      "station_number": 5
    }

    response = self.client.post('/mbajk/predict', json=data)

    assert response.status_code == 200

    assert isinstance(response.json['predictions'], list)
    assert len(response.json['predictions']) == 7
    for element in response.json['predictions']:
      assert isinstance(element, int)

if __name__ == '__main__':
  pytest.main()
