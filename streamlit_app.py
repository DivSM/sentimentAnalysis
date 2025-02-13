import os
import re
import pickle
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
import scipy.sparse
import praw
import gdown
import streamlit as st

# Download necessary NLTK data
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# Image Folder path (for sentiment images)
IMAGE_FOLDER = os.path.join('static', 'img_pool')

# Load models
with open('logr_m.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('tfidf_vectorizer.pickle', 'rb') as f:
    tfidf = pickle.load(f)

with open('naivebern_m.pkl', 'rb') as f:
    nb_model = pickle.load(f)

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

# Function for Sentiment Prediction
def predict_sentiment(text, model_type):
    text = clean_text(text)
    text = tokenize_lem(text)

    if model_type == 'logistic_regression':
        tweet_tfidf = tfidf.transform([text])
        min_len = 6
        max_len = 375
        single_tweet_len_scaled = (len(text) - min_len) / (max_len - min_len)
        single_tweet_len_scaled = np.array([[single_tweet_len_scaled]])

        tweet_features = scipy.sparse.hstack([tweet_tfidf, single_tweet_len_scaled], format="csr")
        sentiment_score = lr_model.predict(tweet_features)

    elif model_type == 'naive_bayes':
        tweet_tfidf = tfidf.transform([text])
        min_len = 6
        max_len = 375
        single_tweet_len_scaled = (len(text) - min_len) / (max_len - min_len)
        single_tweet_len_scaled = np.array([[single_tweet_len_scaled]])

        tweet_features = scipy.sparse.hstack([tweet_tfidf, single_tweet_len_scaled], format="csr")
        sentiment_score = nb_model.predict(tweet_features)

    # Determine the sentiment from the score
    if sentiment_score == -1:
        sentiment = 'Negative'
        img_filename = os.path.join(IMAGE_FOLDER, 'sad.png')
    elif sentiment_score == 0:
        sentiment = 'Neutral'
        img_filename = os.path.join(IMAGE_FOLDER, 'neutral.png')
    elif sentiment_score == 1:
        sentiment = 'Positive'
        img_filename = os.path.join(IMAGE_FOLDER, 'smiling.png')
    else:
        sentiment = 'Error: None'
        img_filename = os.path.join(IMAGE_FOLDER, 'neutral.png')

    return sentiment, img_filename

# Streamlit Interface for Sentiment Analysis
def sentiment_analysis_ui():
    st.title('Sentiment Analysis Tool')

    # Text input and model selection
    text = st.text_area("Enter Text for Sentiment Analysis")
    model_type = st.selectbox("Select Sentiment Model", ['logistic_regression', 'naive_bayes'])

    # Perform sentiment analysis when the button is clicked
    if st.button('Predict Sentiment'):
        if text:
            sentiment, img_filename = predict_sentiment(text, model_type)
            st.write(f"Sentiment: {sentiment}")
            st.image(img_filename)

# Streamlit Interface for Reddit Sentiment Analysis
def reddit_sentiment_analysis_ui():
    st.title('Reddit Sentiment Analysis')

    # Subreddit input field
    subreddit_name = st.text_input("Enter Subreddit Name:")

    # Fetch Reddit posts and predict sentiment
    if st.button('Get Top Posts and Predict Sentiment'):
        if subreddit_name:
            reddit = praw.Reddit(client_id='HLiUhFewQ6kY3mXdTCGFdg', 
                                 client_secret='Ga48TayinWGv3KmQgzT6OgFpzlmlSA', 
                                 user_agent='Project PG DAI v1.0')

            posts = reddit.subreddit(subreddit_name).top(limit=20)
            sentiment_results = []

            for post in posts:
                post_text = post.title + ' ' + post.selftext
                sentiment, _ = predict_sentiment(post_text, 'logistic_regression')
                sentiment_results.append({'post': post.title, 'sentiment': sentiment})

            st.write(f"Sentiment results for top 20 posts from r/{subreddit_name}:")
            for result in sentiment_results:
                st.write(f"Post: {result['post']}")
                st.write(f"Sentiment: {result['sentiment']}")
        else:
            st.error("Please enter a valid subreddit name.")

# Main function to display Streamlit UI
def main():
    option = st.sidebar.selectbox('Choose an option', ['Sentiment Analysis', 'Reddit Sentiment Analysis'])

    if option == 'Sentiment Analysis':
        sentiment_analysis_ui()
    elif option == 'Reddit Sentiment Analysis':
        reddit_sentiment_analysis_ui()

if __name__ == '__main__':
    main()
