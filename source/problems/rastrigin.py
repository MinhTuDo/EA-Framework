from model.problem import Problem
import numpy as np
from numpy import cos, pi

class Rastrigin(Problem):
    def __init__(self, n_params=2):
        super().__init__(n_params,
        
                         n_obj=-1,
                         n_constraints=0,
                         domain=(-5.12, 5.12),
                         param_type=np.double,
                         multi_dims=True)

        self._pareto_set = np.zeros((1, n_params), dtype=self.param_type)
        self._pareto_front = 0
        self._optimum = min
        self._argopt = np.argmin
        self.A = 10

    ## Overide Methods ##
    def _function(self, X):
        f = self.A*len(X) + (X**2 - self.A*cos(2*pi*X)).sum(axis=0)
        return f

    def _f_comparer(self, y1, y2):
        return y1 <= y2