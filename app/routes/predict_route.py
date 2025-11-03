from flask import Blueprint, request, jsonify
from app.services.ml_service import get_model, get_vectorizer, predict_text

predict_bp = Blueprint("predict", __name__)

@predict_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" in request'}), 400

        input_text = data['text']
        model = get_model()
        vectorizer = get_vectorizer()

        prediction, proba = predict_text(model, vectorizer, input_text)
        return jsonify({
            'prediction': prediction,
            'probability': float(proba)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
