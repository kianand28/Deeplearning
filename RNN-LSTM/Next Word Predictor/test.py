import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle

model = load_model('nextword_lstm_model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

def predict_word(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    preds=np.argmax(model.predict(sequence))
    

# Function to predict the next 5 words
def predict_next_n_words(model, tokenizer, text, n):
    for _ in range(n):
        next_word = predict_word(model, tokenizer, text)
        text.append(next_word)  # Add the predicted word to the input sequence
        text = text[-3:]  # Keep only the last 3 words as input for the next prediction
        print(f"Predicted word: {next_word}")
    return text

while True:
    text = input("Enter your line: ")
    if text=="1":
        break
    else:
        text = text.split(" ")
        text = text[-3:]
        predict_word(model, tokenizer, text, 5)
