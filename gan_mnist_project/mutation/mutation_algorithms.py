# mutation/mutation_algorithms.py
import numpy as np


def gaussian_mutation(vector, sigma=0.1):
    return vector + np.random.normal(0, sigma, vector.shape)


def uniform_mutation(vector, low=-0.1, high=0.1):
    return vector + np.random.uniform(low, high, vector.shape)


def boundary_mutation(vector, boundary_range=0.1):
    random_values = np.random.uniform(-boundary_range, boundary_range, vector.shape)
    mutated_vector = np.where(np.random.rand(*vector.shape) > 0.5, vector + random_values, vector - random_values)
    return np.clip(mutated_vector, 0, 1)


def polynomial_mutation(vector, eta=20):
    delta = np.random.randn(*vector.shape)
    mutated_vector = vector + delta * (1 - vector) ** eta
    return np.clip(mutated_vector, 0, 1)
