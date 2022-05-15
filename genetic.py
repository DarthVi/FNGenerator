# Using genetic algorithms to generate new names.
# Main idea for fitness function: names should have a good balance of vowels and consonants
# without too many consecutive consonants.
# Maybe don't make the rule too rigid, apply probabilities or soft constraints.
# Maybe use a wheighted function?

# Use roulette wheel + fitness function for parents selection, than apply crossing over and mutation.
# Execute for n-generations
import numpy as np
import random
import pygad
import string

alphabet = string.ascii_lowercase + "'"
vowels = ["a", "e", "i", "o", "u"]
consonants = list(filter(lambda c: c not in vowels, alphabet))
bias = 0.000000001

num_genes = len(alphabet)

# taken from https://stackoverflow.com/a/32043366/1262118
def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = np.concatenate(data)
    return out


def fitness_function_factory(length_bias=1):
    vowels_ord = [ord(c) for c in vowels]
    consonants_ord = [ord(c) for c in consonants]
    alphabet_ord = [ord(c) for c in alphabet]

    def fitness_function(solution, solution_idx):
        num_vowels = len(list(filter(lambda c: c in vowels_ord, solution)))
        num_consonants = len(list(filter(lambda c: c in consonants_ord, solution)))
        word_length = len(solution)
        zeros_length = len(list(filter(lambda c: c == 0, solution)))
        # maximize values for good equilibrium of consonants and vowels and takes into account word length
        fitness = 1 / (abs(num_vowels - num_consonants) + bias) + length_bias * word_length
        return fitness

    return fitness_function


def read_data(filename):
    words = []
    with open(filename, "r") as f:
        for line in f:
            word = line.removesuffix("\n")
            words.append(word)
    return words


def convert_string_to_integers(s):
    removed_zeros = filter(lambda i: i != 0, s)
    return [ord(c) for c in removed_zeros]


def convert_integers_to_string(i):
    return "".join([chr(v) for v in i])


def get_initial_population(filename):
    words = read_data(filename)
    words_ord = [convert_string_to_integers(word) for word in words]
    filled = numpy_fillna(words_ord)
    return filled


def get_random_solutions(num, solutions, seed=42):
    random.seed(seed)
    length, i = solutions.shape
    idx = np.random.randint(length, size=num)
    selection = solutions[idx, :]
    words = [convert_integers_to_string(solution) for solution in selection]
    return words


initial_pop = get_initial_population("names.txt")

ga_instance = pygad.GA(
    num_generations=10,
    fitness_func=fitness_function_factory(length_bias=0),
    num_parents_mating=10,
    mutation_type="adaptive",
    initial_population=initial_pop,
    crossover_type="single_point",
    crossover_probability=0.7,
    mutation_percent_genes=(12, 8),
    gene_type=int,
)

ga_instance.run()
sols = get_random_solutions(10, ga_instance.population)
print("\n".join(sols))
