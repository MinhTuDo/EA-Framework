from model.operation import Operation
import numpy as np

class GOM(Operation):
    def __init__(self):
        super().__init__()
        self.trial_pop = None
        self.model = None
    
    def _do(self, ga):
        self.trial_pop = ga.pop.copy()
        self.model = ga.model.copy()
        N = ga.pop_size
        for i in range(N):
            np.random.shuffle(self.model)
            for group in self.model:
                r = np.random.randint(low=0, high=N)
                d = self.trial_pop[r]
                x_dash = self.trial_pop[i].copy()
                x_dash[group] = d[group]
                y_dash = ga.problem._function(x_dash)
                if ga.problem._f_comparer(y_dash, ga.f_pop[i]):
                    self.trial_pop[i] = x_dash
                # x = x_dash if ga.problem._f_comparer(y_dash, ga.f_pop[i]) else x
        return self.trial_pop