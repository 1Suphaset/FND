from flask import Flask, jsonify
from flask_cors import CORS
from app.routes.predict_route import predict_bp
from app.routes.recommend_route import recommend_bp

def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route('/')
    def home():
        return jsonify({'message': 'Fake News Detection API is running 🚀'})

    # Register blueprints
    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(recommend_bp, url_prefix="/api")

    return app
