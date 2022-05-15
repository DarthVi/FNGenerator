# Using genetic algorithms to generate new names.
# Main idea for fitness function: names should have a good balance of vowels and consonants
# without too many consecutive consonants.
# Maybe don't make the rule too rigid, apply probabilities or soft constraints.

# Use roulette wheel + fitness function for parents selection, than apply crossing over and mutation.
# Execute for n-generations
import pygad
import string

alphabet = string.ascii_lowercase + "'"
vowels = ["a", "e", "i", "o", "u"]
consonants = list(filter(lambda c: c not in vowels, alphabet))
