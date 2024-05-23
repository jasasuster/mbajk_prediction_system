import src.settings as settings

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import DuplicateKeyError

from datetime import datetime, date

url = f"mongodb+srv://{settings.MONGO_USERNAME}:{settings.MONGO_PASSWORD}@{settings.MONGO_HOST}/?retryWrites=true&w=majority&appName=Cluster0"

def insert_prediction(collection_name, data):
  try:
    client = MongoClient(url, server_api=ServerApi('1'))
    if client:
      collection = client.get_database('station_predictions').get_collection(collection_name)
      collection.insert_one(data)
  
  except DuplicateKeyError:
    print("Data with the same _id already exists!")
  except Exception as e:
    print(f"Error: {e}")

def get_predictions_by_date(collection, start_date, end_date):
  try:
    predictions = collection.find({
      "date": {
        "$gte": datetime.combine(start_date, datetime.min.time()),
        "$lte": datetime.combine(end_date, datetime.max.time())
      }
    })
    return list(predictions)
  
  except Exception as e:
    print(f"Error: {e}")

def predictions_today(station_name):
  try:
    client = MongoClient(url, server_api=ServerApi('1'))
    if client:
      collection = client.get_database('station_predictions').get_collection(station_name)
      today = date.today()
      start_date = datetime.combine(today, datetime.min.time())
      end_date = datetime.combine(today, datetime.max.time())
      predictions = get_predictions_by_date(collection, start_date, end_date)
      return predictions
    
  except Exception as e:
    print(f"Error: {e}")