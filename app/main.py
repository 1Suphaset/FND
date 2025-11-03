from flask import Flask, jsonify
from flask_cors import CORS
from app.routes.predict_route import predict_bp

def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route('/')
    def home():
        return jsonify({'message': 'Fake News Detection API is running 🚀'})

    # Register blueprint
    app.register_blueprint(predict_bp, url_prefix="/api")

    return app
