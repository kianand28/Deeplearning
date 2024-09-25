import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle

# Load the model and tokenizer
model = load_model('nextword_lstm_model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

# Function to predict the next word
def predict_word(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    preds = np.argmax(model.predict(sequence), axis=-1)
    
    predicted_word = ""
    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break
    return predicted_word

# Function to predict the next n words
def predict_next_n_words(model, tokenizer, text, n):
    predicted_words = []
    for _ in range(n):
        next_word = predict_word(model, tokenizer, text)
        predicted_words.append(next_word)
        text.append(next_word)  
        text = text[-3:]  
    return predicted_words


# Streamlit app
st.title("Next Word Predictor")

# Input field for text
input_text = st.text_input("Enter your line (last 3 words):", value="Type here")
n_words = st.number_input("Number of words to predict:", min_value=1, max_value=10, value=5)


if st.button("Predict"):
    if input_text:
        text = input_text.split()[-3:]  # Get the last 3 words
        predicted_words = predict_next_n_words(model, tokenizer, text, n_words)
        st.write(f"Predicted next {n_words} words: {input_text + ' ' + ' '.join(predicted_words)}")
    else:
        st.write("Please enter a valid input.")
