import pytest
from flask import Flask
from flask_testing import TestCase

from src.serve.api import app

class TestAPI(TestCase):
  def create_app(self):
    app.config['TESTING'] = True
    return app
  
  def test_predict(self):
    data = {'input': [1, 2, 3, 4]}
    response = self.client.post('/mbajk/predict', json=data)

    assert response.status_code == 200

    assert response.json == {'prediction': 12}

if __name__ == '__main__':
  pytest.main()