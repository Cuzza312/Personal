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

tokenizer = Tokenizer()

data = pd.read_csv('Conversation.csv')

data = data[['question', 'answer']]

i = 0
for column in data.columns:
    i = 0
    for sentence in data[column]:
        data[column][i] = "<SS> " + sentence + " <ES>"
        i += 1

data["Together"] = data['question']+ " " + data["answer"]

tokenizer.fit_on_texts(data["Together"])
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)

input_sequences = []
for line in data["Together"]:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
max_sequence_len = max([len(x) for x in input_sequences])
print(max_sequence_len)

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]

ys = to_categorical(labels, num_classes=total_words)

with open('tokenizer4.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len - 1))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xs, ys, epochs=20, verbose=1, batch_size=128)

model.save('NextWordPredictionModel4.h5')
