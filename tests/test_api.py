import pytest
import json
from flask_testing import TestCase

from src.serve.api import app

class TestAPI(TestCase):
  def create_app(self):
    app.config['TESTING'] = True
    return app
  
  def test_predict(self):
    with open('./tests/test_input.json', 'r') as file:
      data = json.load(file)

    response = self.client.post('/mbajk/predict', json=data)

    assert response.status_code == 200

    assert isinstance(response.json['prediction'], float)

if __name__ == '__main__':
  pytest.main()