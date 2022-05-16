# Fantasy Name Generator: a collection of experiments

This is a collection of experiments about generating new fantasy names.
Currently there are two approach used here:

1. RNN using a character-based LSTM (Long Short Term Memory), by customizing the code from https://livecodestream.dev/post/lstm-based-name-generator-first-dive-into-nlp/
2. an approach based on genetic algorithm with an existing initial population

In both cases I used the Kismet's Fantasy Name compendium (found here: https://docs.google.com/spreadsheets/d/1JNukIS9NThOusuWXDQvi5FHnkmd0SB9UGIG-3Jos2OQ/) as a starting point (to train the LSTM and as initial population for the genetic algorithm).

## Genetic algorithm

The genetic approach uses PyGAD (https://pygad.readthedocs.io/en/latest/index.html) as the main engine.
The fitness function follows this main idea: new names should have a balanced number of vowels and constants and be of an appropriate length.
This is the formula currently used: `fitness = 1 / (abs(num_vowels - num_consonants) + bias) + length_bias * word_length`.
A small bias of 0.000000001 has been used to avoid division by 0.
