from flask import Flask, render_template, request, jsonify
from flask_restx import Api, Resource, reqparse, fields
import numpy as np
import pandas as pd
import re
import os
import pickle
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import scipy.sparse
import praw
import requests
import gdown

IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER


FILE_ID = "1H7as1SVEhFj7J_YIxUBBSXLib5YFkZkV"
MODEL_PATH = "lstm_m.pkl"

def download_model():
    print("Downloading model from Google Drive...")
    
    # Google Drive URL for direct download
    download_url = f"https://drive.google.com/uc?id={FILE_ID}"
    
    # Using gdown to download the file
    gdown.download(download_url, MODEL_PATH, quiet=False)
    
    print("Model downloaded successfully!")

def load_model():
    try:
        # Check if model exists and is correct
        if not os.path.exists(MODEL_PATH):
            download_model()

        # Open the file in binary read mode
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)  # Try to load the model
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

lstm_model = load_model()

def init():
    global lstm_model, tokenizer, lr_model, nb_model, tfidf

    # Loading the LSTM model via drive since size is > 100MB

    # Load LSTM tokenizer
    with open('tokenizer_lstm.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Load the Logistic Regression model
    with open('logr_m.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    # Load the LR TF-IDF vectorizer
    with open('tfidf_vectorizer.pickle', 'rb') as f:
        tfidf = pickle.load(f)

    # Load the Naive Bayes model
    with open('naivebern_m.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    # TF-IDF vectorizer is same as LR

def get_top_posts(subreddit, limit=20):
    # Fetch the top 20 posts from the given subreddit
    posts = []
    for submission in reddit.subreddit(subreddit).top(limit=limit):
        posts.append({
            'title': submission.title,
            'url': submission.url,
            'text': submission.selftext
        })
    return posts

# Function to clean the text
def clean_text(text):
    pat1 = r'@[^ ]+'  # @ signs and value
    pat2 = r'https?://[A-Za-z0-9./]+'  # links
    pat3 = r'\'s'  # floating s's
    pat4 = r'\#\w+'  # hashtags and value
    pat5 = r'&amp '  # & and
    pat6 = r'[^A-Za-z\s]'  # remove non-alphabet
    combined_pat = r'|'.join((pat1, pat2, pat3, pat4, pat5, pat6))
    text = re.sub(combined_pat, "", text).lower()
    return text.strip()

# Function to tokenize and lemmatize the text
def tokenize_lem(sentence):
    lem = WordNetLemmatizer()
    outlist = []
    token = sentence.split()
    for tok in token:
        outlist.append(lem.lemmatize(tok))
    return " ".join(outlist)

######################### Code for Sentiment Analysis ##########################
@app.route('/', methods=['GET', 'POST'])
def home():
    print("Home route accessed")  # Debugging line
    sentiment = ''
    img_filename = ''
    text = ''
    model_type = request.form.get('model_type', 'lstm')  # Default model

    if request.method == 'POST':
        text = request.form['text']
        model_type = request.form['model_type']

        text = clean_text(text)
        text = tokenize_lem(text)

        if model_type == 'lstm':
            print("LSTM")
            # Tokenize the text using the tokenizer
            sequence = tokenizer.texts_to_sequences([text])

            # Padding the sequence to the required length (30 tokens)
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post', maxlen=30)

            # Predicting with the LSTM model
            prediction = lstm_model.predict(padded_sequence)
            sentiment_score = 0

            if (prediction <= 0.3):
                sentiment_score = -1
            elif (0.3 < prediction <= 0.6):  
                sentiment_score = 0
            elif (prediction > 0.6):
                sentiment_score = 1
        
        elif model_type == 'logistic_regression':
            print("LR")
            # Create TF-IDF vector
            tweet_tfidf = tfidf.transform([text])
            print(f"TF-IDF shape: {tweet_tfidf.shape}")

            # Manual scaling based on data obtained from colab notebook
            min_len = 6
            max_len = 375
            single_tweet_len_scaled = (len(text) - min_len) / (max_len - min_len)
            single_tweet_len_scaled = np.array([[single_tweet_len_scaled]])

            # Combine features
            tweet_features = scipy.sparse.hstack([tweet_tfidf, single_tweet_len_scaled], format="csr")
            print(f"Combined feature shape: {tweet_features.shape}")

            # Predicting with model
            sentiment_score = lr_model.predict(tweet_features)
            print(sentiment_score)  # Output: [-1] or [0] or [1]

        elif model_type == 'naive_bayes':
            print("NR")
            # Create TF-IDF vector
            tweet_tfidf = tfidf.transform([text])

            # Manual scaling based on data obtained from colab notebook
            min_len = 6
            max_len = 375
            single_tweet_len_scaled = (len(text) - min_len) / (max_len - min_len)
            single_tweet_len_scaled = np.array([[single_tweet_len_scaled]])

            # Combine features
            tweet_features = scipy.sparse.hstack([tweet_tfidf, single_tweet_len_scaled], format="csr")

            # Predicting with model
            sentiment_score = nb_model.predict(tweet_features)
            print(sentiment_score)  # Output: [-1] or [0] or [1]

        # Set sentiment and image based on the prediction
        if sentiment_score == -1:
            sentiment = 'Negative'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.png')
        elif sentiment_score == 0:
            sentiment = 'Neutral'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'neutral.png')
        elif sentiment_score == 1:
            sentiment = 'Positive'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'smiling.png')
        else:
            sentiment = 'Error: None'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'neutral.png')

    # Rendering the home.html template with the results
    return render_template('home.html', text=text, sentiment=sentiment, image=img_filename, model_type=model_type)

######################### Code for Sentiment Analysis ##########################

######################### Code for Reddit Sentiment Analysis ##########################

@app.route('/reddit-sentiment', methods=['GET', 'POST'])
def reddit_sentiment():
    subreddit_name = ''
    sentiment_results = []
    if request.method == 'POST':
        subreddit_name = request.form['subreddit_name']
        
        # Fetch Reddit posts using Reddit API (praw)
        reddit = praw.Reddit(client_id='HLiUhFewQ6kY3mXdTCGFdg', 
                     client_secret='Ga48TayinWGv3KmQgzT6OgFpzlmlSA', 
                     user_agent='Project PG DAI v1.0')

        # Get the top 20 posts from the subreddit
        subreddit = reddit.subreddit(subreddit_name)
        posts = subreddit.new(limit=20)
        
        for post in posts:
            post_text = post.title + ' ' + post.selftext  # Combine title and text
            cleaned_text = clean_text(post_text)
            lemmatized_text = tokenize_lem(cleaned_text)

            # Predict sentiment for each post using LSTM model (or any model of your choice)
            # For example, using LSTM model:
            sequence = tokenizer.texts_to_sequences([lemmatized_text])
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post', maxlen=30)
            prediction = lstm_model.predict(padded_sequence)

            sentiment_score = 0
            if prediction <= 0.3:
                sentiment_score = -1  # Negative sentiment
            elif 0.3 < prediction <= 0.6:
                sentiment_score = 0  # Neutral sentiment
            elif prediction > 0.6:
                sentiment_score = 1  # Positive sentiment

            sentiment = 'Negative' if sentiment_score == -1 else 'Neutral' if sentiment_score == 0 else 'Positive'
            sentiment_results.append({'post': post.title, 'sentiment': sentiment})
    
    return render_template('reddit_sentiment.html', sentiment_results=sentiment_results, subreddit_name=subreddit_name)

######################### Code for Reddit Sentiment Analysis ##########################

# Initialize the Swagger UI with flask-restplus (Swagger UI at /swagger)
api = Api(app, doc='/swagger', version='1.0', title='Sentiment Analysis API', description='API for Sentiment Analysis and Reddit Sentiment Analysis')


# Define input model for Sentiment Analysis
sentiment_model = api.model('SentimentModel', {
    'text': fields.String(required=True, description='Text to analyze sentiment'),
    'model_type': fields.String(required=True, description='The model to use (lstm, logistic_regression, naive_bayes)')
})

# Define input model for Reddit Sentiment Analysis
reddit_model = api.model('RedditModel', {
    'subreddit_name': fields.String(required=True, description='Subreddit name to analyze posts')
})


######################### Code for Swagger UI - Sentiment Analysis ##########################

# Swagger UI (using flask-restplus)
@api.route('/api/sentiment-analysis')
class SentimentAnalysis(Resource):
    @api.doc(description="Sentiment analysis on the given text")
    @api.expect(sentiment_model, validate=True)
    def post(self):
        data = api.payload
        text = data['text']
        model_type = data['model_type']
        
        # Clean and preprocess text
        text = clean_text(text)
        text = tokenize_lem(text)
        
        sentiment = ''
        img_filename = ''
        sentiment_score = 0

        # Sentiment Analysis Logic as in the original code
        if model_type == 'lstm':
            sequence = tokenizer.texts_to_sequences([text])
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post', maxlen=30)
            prediction = lstm_model.predict(padded_sequence)
            if prediction <= 0.3:
                sentiment_score = -1
            elif 0.3 < prediction <= 0.6:
                sentiment_score = 0
            elif prediction > 0.6:
                sentiment_score = 1
        elif model_type == 'logistic_regression':
            tweet_tfidf = tfidf.transform([text])
            single_tweet_len_scaled = (len(text) - 6) / (375 - 6)
            tweet_features = scipy.sparse.hstack([tweet_tfidf, np.array([[single_tweet_len_scaled]])], format="csr")
            sentiment_score = lr_model.predict(tweet_features)
        elif model_type == 'naive_bayes':
            tweet_tfidf = tfidf.transform([text])
            single_tweet_len_scaled = (len(text) - 6) / (375 - 6)
            tweet_features = scipy.sparse.hstack([tweet_tfidf, np.array([[single_tweet_len_scaled]])], format="csr")
            sentiment_score = nb_model.predict(tweet_features)

        if sentiment_score == -1:
            sentiment = 'Negative'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.png')
        elif sentiment_score == 0:
            sentiment = 'Neutral'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'neutral.png')
        elif sentiment_score == 1:
            sentiment = 'Positive'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'smiling.png')

        return jsonify({'sentiment': sentiment, 'image': img_filename})
    
######################### Code for Swagger UI - Sentiment Analysis ##########################

######################### Code for Swagger UI - Reddit Sentiment Analysis ##########################

@api.route('/api/reddit-sentiment')
class RedditSentimentAnalysis(Resource):
    @api.doc(description="Sentiment analysis for Reddit posts")
    @api.expect(reddit_model, validate=True)
    def post(self):
        data = api.payload
        subreddit_name = data['subreddit_name']

        # Fetch Reddit posts and analyze sentiment
        reddit = praw.Reddit(client_id='HLiUhFewQ6kY3mXdTCGFdg', 
                             client_secret='Ga48TayinWGv3KmQgzT6OgFpzlmlSA', 
                             user_agent='Project PG DAI v1.0')

        subreddit = reddit.subreddit(subreddit_name)
        posts = subreddit.new(limit=20)
        sentiment_results = []

        for post in posts:
            post_text = post.title + ' ' + post.selftext
            cleaned_text = clean_text(post_text)
            lemmatized_text = tokenize_lem(cleaned_text)

            sequence = tokenizer.texts_to_sequences([lemmatized_text])
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post', maxlen=30)
            prediction = lstm_model.predict(padded_sequence)

            sentiment_score = 0
            if prediction <= 0.3:
                sentiment_score = -1  # Negative sentiment
            elif 0.3 < prediction <= 0.6:
                sentiment_score = 0  # Neutral sentiment
            elif prediction > 0.6:
                sentiment_score = 1  # Positive sentiment

            sentiment = 'Negative' if sentiment_score == -1 else 'Neutral' if sentiment_score == 0 else 'Positive'
            sentiment_results.append({'post': post.title, 'sentiment': sentiment})

        return jsonify({'subreddit': subreddit_name, 'sentiments': sentiment_results})

######################### Code for Swagger UI - Reddit Sentiment Analysis ##########################

if __name__ == "__main__":
    init()
    app.run(debug=True)
