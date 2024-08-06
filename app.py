from flask import Flask, request, render_template, jsonify
from model import train_model, predict_sentiment

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.json['review']
    prediction = predict_sentiment(review)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
