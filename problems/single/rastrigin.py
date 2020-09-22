from .so_problem import SingleObjectiveProblem
import numpy as np
from numpy import cos, pi

class Rastrigin(SingleObjectiveProblem):
    def __init__(self, n_params=2, **kwargs):
        super().__init__(n_params,
                         n_constraints=0,
                         param_type=np.double,
                         multi_dims=True)
        xl = np.ones((self.n_params,)) * -5.12
        xu = np.ones((self.n_params,)) * 5.12
        self.domain = (xl, xu)
        self._pareto_set = np.zeros((1, n_params), dtype=self.param_type)
        self._pareto_front = 0
        self._optimum = min
        self._argopt = np.argmin
        self.A = 10

    ## Overide Methods ##
    def _f(self, X):
        f = self.A*len(X) + (X**2 - self.A*cos(2*pi*X)).sum(axis=0)
        return f

    def _sol_compare(self, y1, y2):
        return y1 <= y2

class ModifiedRastrigin(SingleObjectiveProblem):
    def __init__(self, n_params=2, **kwargs):
        super().__init__(n_params,
                         n_constraints=0,
                         param_type=np.double,
                         multi_dims=True)
        xl = np.ones((self.n_params,)) * 0
        xu = np.ones((self.n_params,)) * 1
        self.domain = (xl, xu)
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

    ## Overide Methods ##
    def _f(self, X):
        try:
            f = -(10 + 9*cos(2*pi*X*self.k)).sum()
        except:
            f = -(10 + 9*cos(2*pi*X*self.k[:, np.newaxis])).sum(axis=0)
        return f

    def _sol_compare(self, y1, y2):
        return y1 <= y2