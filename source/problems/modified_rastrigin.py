from model.problem import Problem
import numpy as np
from numpy import cos, pi

class ModifiedRastrigin(Problem):
    def __init__(self, n_params=2):
        super().__init__(n_params,
        
                         n_obj=-1,
                         n_constraints=0,
                         domain=(0, 1),
                         param_type=np.double,
                         multi_dims=True)

        # self._pareto_set = np.zeros((1, n_params), dtype=self.param_type)
        # self._pareto_front = 0
        self._optimum = min
        self._argopt = np.argmin
        self.A = 10
        self.k = np.array([1, 1, 1, 2, 1, 1, 1, 2,
                           1, 1, 1, 3, 1, 1, 1, 4])
        if self.n_params == 2:
            self.k = np.array([3, 4])
        else:
            self.k = np.ones((self.n_params, 1))
        self.step = 0.01

    ## Overide Methods ##
    def _function(self, X):
        try:
            f = -(10 + 9*cos(2*pi*X*self.k)).sum(axis=0)
        except:
            f = -(10 + 9*cos(2*pi*X*self.k[:, np.newaxis])).sum(axis=0)
        return f

    def _f_comparer(self, y1, y2):
        return y1 <= y2