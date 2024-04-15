import requests

def test_api_returns_200():
    url = "https://api.open-meteo.com/v1/forecast"
    response = requests.get(url)
    assert response.status_code == 200
