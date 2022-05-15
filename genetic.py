# Using genetic algorithms to generate new names.
# Main idea for fitness function: names should have a good balance of vowels and consonants
# without too many consecutive consonants.
# Maybe don't make the rule too rigid, apply probabilities or soft constraints.
# Maybe use a wheighted function?

# Use roulette wheel + fitness function for parents selection, than apply crossing over and mutation.
# Execute for n-generations
from numpy import char
import pygad
import string

alphabet = string.ascii_lowercase + "'"
vowels = ["a", "e", "i", "o", "u"]
consonants = list(filter(lambda c: c not in vowels, alphabet))
bias = 0.000000001

num_genes = len(alphabet)


def fitness_function_factory(length_bias=1):
    def fitness_function(solution, solution_idx):
        num_vowels = len(list(filter(lambda c: c in vowels, solution)))
        num_consonants = len(list(filter(lambda c: c in consonants, solution)))
        word_length = len(solution)
        # maximize values for good equilibrium of consonants and vowels and takes into account word length
        fitness = 1 / (abs(num_vowels - num_consonants) + bias) + length_bias * word_length
        return fitness

    return fitness_function


ga_instance = pygad.GA(
    num_generations=200,
    fitness_func=fitness_function_factory,
    num_parents_mating=10,
    sol_per_pop=20,
    num_genes=num_genes,
    mutation_type="adaptive",
    mutation_num_genes=(3, 1),
    gene_type=str,
)
