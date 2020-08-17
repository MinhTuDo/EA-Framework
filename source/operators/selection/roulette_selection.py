from model.operation import Operation
import numpy as np

class RouletteSelection(Operation):
    def __init__(self):
        super().__init__()

    def _do(self, ga):
        selected_indices = []
        fitnesses = np.unique(ga.fitness_pop)
        probabilities = np.zeros(fitnesses.shape)
        sum_of_fitness = fitnesses.sum()
        prev_prob = 0
        for prob, fitness in zip(probabilities, fitnesses):
            prob = prev_prob + (fitness/sum_of_fitness)
            prev_prob = prob
        
        r = np.random.rand()
