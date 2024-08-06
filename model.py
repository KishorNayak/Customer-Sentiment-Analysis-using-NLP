import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Train a simple sentiment analysis model (for demonstration purposes)
def train_model():
    # Example training data
    data = pd.DataFrame({
        'review': ['I love this product', 'This is the worst', 'Not bad', 'Amazing', 'Terrible'],
        'sentiment': [1, 0, 1, 1, 0]
    })
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['review'])
    y = data['sentiment']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    joblib.dump((vectorizer, model), 'model.pkl')

def predict_sentiment(review):
    vectorizer, model = joblib.load('model.pkl')
    data = vectorizer.transform([review])
    prediction = model.predict(data)
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Train the model when script is run
if __name__ == '__main__':
    train_model()
