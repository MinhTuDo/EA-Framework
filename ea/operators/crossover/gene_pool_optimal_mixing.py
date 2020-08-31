from model.operation import Operation
import numpy as np

class GOM(Operation):
    def __init__(self):
        super().__init__()
        self.trial_pop = None
        self.model = None
    
    def _do(self, ga):
        self.trial_pop = ga.pop.copy()
        self.model = ga.model
        
        N = ga.pop_size
        random_indices = np.arange(N)
        np.random.shuffle(random_indices)
        for i in range(N):
            np.random.shuffle(self.model)
            for group in self.model:
                d = self.trial_pop[random_indices[i]]
                x_dash = self.trial_pop[i].copy()
                x_dash[group] = d[group]
                y_dash = ga.problem._f(x_dash)
                ga.n_evals += 1
                if ga.problem._sol_compare(y_dash, ga.fitness_pop[i]):
                    self.trial_pop[i] = x_dash
        return self.trial_pop