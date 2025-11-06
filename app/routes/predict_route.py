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

        result = predict_text(model, vectorizer, input_text)

        # If the service returned a tuple (legacy), handle it
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            prediction, proba = result[0], result[1]
            resp = {
                'prediction': int(prediction),
                'probability': float(proba)
            }
            return jsonify(resp)

        # Expecting a dict result from updated ml_service.predict_text
        if isinstance(result, dict):
            # Base response
            resp = {
                'prediction': int(result.get('prediction', 0)),
                'probability': float(result.get('probability', 0.0))
            }

            # Optional detailed scores
            # source, language, factual, cross_source
            if 'source' in result:
                resp['source_score'] = float(result.get('source', 0.0))
            if 'language' in result:
                resp['language_score'] = float(result.get('language', 0.0))
            if 'factual' in result:
                resp['factual_score'] = float(result.get('factual', 0.0))
            if 'cross_source' in result:
                resp['cross_source_score'] = float(result.get('cross_source', 0.0))

            return jsonify(resp)

        # Fallback
        return jsonify({'error': 'Unexpected prediction result format'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500
