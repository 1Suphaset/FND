import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from app.config.settings import VECTORIZER_PATH,X_UNDER_PATH
class NewsRecommender:
    def __init__(self, vectorizer_path, news_data_path):
        """Initialize recommender with paths to required files.
        
        Args:
            vectorizer_path: Path to saved TF-IDF vectorizer
            news_data_path: Path to saved news dataset
        """
        self.vectorizer = joblib.load(vectorizer_path)
        self.news_data = joblib.load(news_data_path)
        
        # Pre-compute TF-IDF for all news
        self.news_tfidf = self.vectorizer.transform(
            self.news_data["Title"].fillna("") + " " + self.news_data["Body"].fillna("")
        )
    
    def get_recommendations(self, input_text, n_recommendations=5):
        """Get similar news recommendations for input text.
        
        Args:
            input_text: Text to find recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of dicts with news details and similarity scores
        """
        # Vectorize input text
        input_vector = self.vectorizer.transform([input_text])
        
        # Calculate similarity scores
        similarities = cosine_similarity(input_vector, self.news_tfidf).flatten()
        
        # Get top N similar indices (excluding exact matches)
        similar_indices = np.argsort(similarities)[::-1]
        
        # Filter out exact matches and get top N
        recommendations = []
        added = 0
        
        for idx in similar_indices:
            if similarities[idx] < 0.99:  # Skip exact matches
                news = self.news_data.iloc[idx]
                recommendations.append({
                    'title': str(news['Title']),
                    'body': str(news['Body']),
                    'category': str(news['category']),
                    'viewers': int(news['Viewers']),  # Convert np.int64 to standard int
                    'hashtags': str(news['Hashtag']),
                    'Verify_Department': str(news['Verify_Department']),
                    'URL': str(news['URL']),
                    'similarity_score': float(similarities[idx])  # Already converting float
                })
                added += 1
                
            if added >= n_recommendations:
                break
        
        return recommendations

# Singleton instance
_recommender = None

def get_recommender():
    """Get or create singleton recommender instance."""
    global _recommender
    if _recommender is None:
        # Adjust paths as needed
        vectorizer_path = VECTORIZER_PATH
        news_data_path = X_UNDER_PATH
        _recommender = NewsRecommender(vectorizer_path, news_data_path)
    return _recommender