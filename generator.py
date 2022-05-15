import pandas as pd
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
import numpy as np
import random
import os
import csv


class Model:
    def __init__(
        self,
        step_length=1,
        epochs=50,
        batch_size=32,
        latent_dim=64,
        dropout_rate=0.2,
        verbosity=1,
        input_path="names.txt",
    ):
        self.step_length = step_length  # The step length we take to get our samples from our corpus
        self.epochs = epochs  # Number of times we train on our full data
        self.batch_size = batch_size  # Data samples in each training step
        self.latent_dim = latent_dim  # Size of our LSTM
        self.dropout_rate = dropout_rate  # Regularization with dropout
        self.model_path = os.path.realpath("./model.h5")  # Location for the model
        self.verbosity = verbosity  # Print result for each epoch
        self.input_path = input_path

    def init_model_on_data(self):
        self.names = []
        with open(self.input_path, "r") as f:
            for line in f:
                name = f.readline().removesuffix("\n")
                self.names.append(name)
        # print(names)
        self.c_names = "\n".join(self.names).lower()
        # print(c_names)
        # Find all unique characters by using set()
        self.chars = sorted(list(set(self.c_names)))
        self.num_chars = len(self.chars)

        # Build translation dictionaries, 'a' -> 0, 0 -> 'a'
        self.char2idx = dict((c, i) for i, c in enumerate(self.chars))
        self.idx2char = dict((i, c) for i, c in enumerate(self.chars))

        # Use longest name length as our sequence window
        self.max_sequence_length = max([len(name) for name in self.names])

        print("Total chars: {}".format(self.num_chars))
        print("Corpus length:", len(self.c_names))
        print("Number of names: ", len(self.names))
        print("Longest name: ", self.max_sequence_length)
        self.sequences = []
        next_chars = []

        # Loop over our data and extract pairs of sequances and next chars
        for i in range(0, len(self.c_names) - self.max_sequence_length, self.step_length):
            self.sequences.append(self.c_names[i : i + self.max_sequence_length])
            next_chars.append(self.c_names[i + self.max_sequence_length])

        num_sequences = len(self.sequences)

        print("Number of sequences:", num_sequences)
        print("First 10 sequences and next chars:")
        for i in range(10):
            print("X=[{}] y=[{}]".replace("\n", " ").format(self.sequences[i], next_chars[i]).replace("\n", " "))

        self.X = np.zeros((num_sequences, self.max_sequence_length, self.num_chars), dtype=bool)
        self.Y = np.zeros((num_sequences, self.num_chars), dtype=bool)

        for i, sequence in enumerate(self.sequences):
            for j, char in enumerate(sequence):
                self.X[i, j, self.char2idx[char]] = 1
                self.Y[i, self.char2idx[next_chars[i]]] = 1

        print("X shape: {}".format(self.X.shape))
        print("Y shape: {}".format(self.Y.shape))

        self.model = Sequential()
        self.model.add(
            LSTM(
                self.latent_dim,
                input_shape=(self.max_sequence_length, self.num_chars),
                recurrent_dropout=self.dropout_rate,
            )
        )
        self.model.add(Dense(units=self.num_chars, activation="softmax"))

        optimizer = RMSprop(lr=0.01)
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer)

        self.model.summary()

    def saveModel(self):
        self.model.save_weights(self.model_path)

    def loadModel(self):
        self.model.load_weights(self.model_path)

    def train(self):
        start = time.time()
        print("Start training for {} epochs".format(self.epochs))
        history = self.model.fit(self.X, self.Y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbosity)
        end = time.time()
        print("Finished training - time elapsed:", (end - start) / 60, "min")

    def generate(self, gen_amount=10, seed=42):
        # Start sequence generation from end of the input sequence
        sequence = self.c_names[-(self.max_sequence_length - 1) :] + "\n"

        new_names = []
        print("{} new names are being generated".format(gen_amount))

        while len(new_names) < gen_amount:
            # Vectorize sequence for prediction
            x = np.zeros((1, self.max_sequence_length, self.num_chars))
            for i, char in enumerate(sequence):
                x[0, i, self.char2idx[char]] = 1

            # Sample next char from predicted probabilities
            probs = self.model.predict(x, verbose=0)[0]
            probs /= probs.sum()
            np.random.seed(seed)
            next_idx = np.random.choice(len(probs), p=probs)
            next_char = self.idx2char[next_idx]
            sequence = sequence[1:] + next_char

            # New line means we have a new name
            if next_char == "\n":
                gen_name = [name for name in sequence.split("\n")][1]

                # Never start name with two identical chars, could probably also
                if len(gen_name) > 2 and gen_name[0] == gen_name[1]:
                    gen_name = gen_name[1:]

                # Discard all names that are too short
                if len(gen_name) > 2:
                    # Only allow new and unique names
                    if gen_name.capitalize() not in (self.names + new_names):
                        new_names.append(gen_name.capitalize())

                if 0 == (len(new_names) % (gen_amount / 10)):
                    print("Generated {}".format(len(new_names)))

        return new_names


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = Model()
    model.init_model_on_data()
    # model.train()
    # model.saveModel()
    model.loadModel()
    print(model.generate(seed=10))
