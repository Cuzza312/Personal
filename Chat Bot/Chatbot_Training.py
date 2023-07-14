import io
import json
import pickle
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, Model

model = load_model('NextWordPredictionModel4.h5')

with open('tokenizer4.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

tokenizer: Tokenizer
model: Sequential

# seed_text = "<ss> Are you ok <es>"


while True:
    seed_text = input("You: ")

    if seed_text.lower() == "quit":
        break
    seed_text = "<ss> " + seed_text + " <es>"

    predicted_word = ""
    output_sentence = ""
    i = 0
    while predicted_word != "es" and i < 50:
        i += 1
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=36, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                predicted_word = word
                seed_text = seed_text + " " + predicted_word
                output_sentence = output_sentence + " " + predicted_word
                break

    output_sentence = output_sentence.replace("ss", "")
    output_sentence = output_sentence.replace("es", "")

    print("Chatbot: " + output_sentence)