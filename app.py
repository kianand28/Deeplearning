import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
model = tf.keras.models.load_model('sentiment_lstm_model.h5')

# Load tokenizer
import pickle
with open('sentiment_lstm_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Function to predict sentiment
def predict_sentiment(review):
    review_seq = tokenizer.texts_to_sequences([review])
    review_pad = pad_sequences(review_seq, maxlen=100)  # Adjust maxlen as per your model
    prediction = model.predict(review_pad)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    return sentiment

# Streamlit interface
st.title('Sentiment Analysis on IMDB Reviews')

user_input = st.text_area("Enter your movie review")

if st.button('Analyze'):
    sentiment = predict_sentiment(user_input)
    # Display the result
    st.write(f'Sentiment: {sentiment}')
else:
    st.write('Please enter a movie review.')
