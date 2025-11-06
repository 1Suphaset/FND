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
    # """Predict a single text and return a flexible result.

    # Supports models that return a single probability (old behavior) or
    # multiple outputs (list/tuple) as implemented in the multi-output model.

    # Returns a dict with at least:
    #   - prediction: int (0 or 1)
    #   - probability: float (main output probability)

    # If the model returns multiple outputs, the dict will also include:
    #   - source: float
    #   - language: float
    #   - factual: float
    #   - cross_source: float
    # """
    # Prepare input (try to be permissive about sparse/dense expectations)
    X_input = vectorizer.transform([text])
    try:
        X_input_arr = X_input.toarray().astype(np.float32)
    except Exception:
        # If transform already returns dense
        import numpy as _np
        X_input_arr = _np.asarray(X_input, dtype=_np.float32)

    preds = model.predict(X_input_arr)

    # If model.predict returns a list/tuple of arrays (multi-output Keras model)
    if isinstance(preds, (list, tuple)):
        # Convert each output to scalar probability
        probs = []
        for p in preds:
            try:
                probs.append(float(p.ravel()[0]))
            except Exception:
                probs.append(float(p))

        result = {
            'prediction': int(probs[0] > 0.5),
            'probability': probs[0],
        }

        # Map remaining outputs if present
        names = ['source', 'language', 'factual', 'cross_source']
        for name, val in zip(names, probs[1:]):
            result[name] = val

        return result

    # Single-output model (legacy)
    try:
        proba = float(preds.ravel()[0])
    except Exception:
        proba = float(preds[0])
    prediction = int(proba > 0.5)
    return {'prediction': prediction, 'probability': proba}
