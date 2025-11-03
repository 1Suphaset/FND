from tensorflow.keras.models import load_model
import joblib
import numpy as np
from app.config.settings import MODEL_PATH, VECTORIZER_PATH

model = None
vectorizer = None

def get_model():
    global model
    if model is None:
        print("🔹 Loading model...")
        model = load_model(MODEL_PATH)
        print("✅ Model loaded.")
    return model

def get_vectorizer():
    global vectorizer
    if vectorizer is None:
        print("🔹 Loading vectorizer...")
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ Vectorizer loaded.")
    return vectorizer

def predict_text(model, vectorizer, text):
    X_input = vectorizer.transform([text]).toarray().astype(np.float32)
    proba = model.predict(X_input)[0][0]
    prediction = int(proba > 0.5)
    return prediction, proba
