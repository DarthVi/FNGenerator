import os
from tensorflow import keras
from tensorflow.python.keras import layers

import csv
import numpy as np
import random


class Generator:
    def __init__(self, maxlen=40, step=3):
        self._maxlen = 40
        self._step = 3
        self._data_preprocessing()
        self._model = keras.Sequential(
            [
                keras.Input(shape=(self._maxlen, self._charlen)),
                layers.LSTM(128),
                layers.Dense(self._charlen, activation="softmax"),
            ]
        )
        optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
        self._model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    def _data_preprocessing(self):
        names = []
        chars = None
        self._char_indices = None
        self._indices_char = None
        with open("namedb.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            next(csv_reader)  # skip first line with column names
            for row in csv_reader:
                for name in row:
                    if name != "":
                        names.append(name)
        text = " ".join(names)

        chars = sorted(list(set(text)))
        self._charlen = len(chars)
        print("Total chars:", self._charlen)
        self._char_indices = dict((c, i) for i, c in enumerate(chars))
        self._indices_char = dict((i, c) for i, c in enumerate(chars))

        # cut the text in semi-redundant sequences of maxlen characters
        sentences = []
        next_chars = []
        for i in range(0, len(text) - self._maxlen, self._step):
            sentences.append(text[i : i + self._maxlen])
            next_chars.append(text[i + self._maxlen])
        print("Number of sequences:", len(sentences))

        self._x = np.zeros((len(sentences), self._maxlen, len(chars)), dtype=bool)
        self._y = np.zeros((len(sentences), len(chars)), dtype=bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                self._x[i, t, self._char_indices[char]] = 1
            self._y[i, self._char_indices[next_chars[i]]] = 1

    def _sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def trainModel(self, epochs=40, batch_size=128):
        for epoch in range(epochs):
            self._model.fit(self._x, self._y, batch_size=batch_size, epochs=1)

    def saveModel(self, filepath="SimpleModel"):
        self._model.save(filepath)

    def loadModel(self, filepath="SimpleModel"):
        self._model = keras.models.load_model(filepath)

    def generate(self, seed="a", diversity=0.5, length=5):
        generated = ""
        for i in range(length):
            x_pred = np.zeros((1, self._maxlen, self._charlen))
            for t, char in enumerate(seed):
                x_pred[0, t, self._char_indices[char]] = 1.0
            preds = self._model.predict(x_pred, verbose=0)[0]
            next_index = self._sample(preds, diversity)
            next_char = self._indices_char[next_index]
            seed = seed[1:] + next_char
            generated += next_char
        return generated


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    generator = Generator()
    # generator.trainModel(epochs=100, batch_size=32)
    generator.loadModel()
    # generator.saveModel()
    for i in range(10):
        print(generator.generate("al", diversity=10, length=7))
