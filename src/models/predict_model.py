from tensorflow.keras.models import load_model

import joblib

loaded_model = load_model('./models/uni_gru_model.h5')
loaded_scaler = joblib.load('./models/uni_gru_scaler.pkl')

