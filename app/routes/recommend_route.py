from flask import Blueprint, request, jsonify
from Core.Recomendation.news_recommender import get_recommender

recommend_bp = Blueprint("recommend", __name__)

@recommend_bp.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" in request'}), 400

        input_text = data['text']
        n_recommendations = data.get('n_recommendations', 5)  # Optional param

        # Get recommendations
        recommender = get_recommender()
        recommendations = recommender.get_recommendations(
            input_text, 
            n_recommendations=n_recommendations
        )

        return jsonify({
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500