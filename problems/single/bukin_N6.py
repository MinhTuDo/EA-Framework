from .so_problem import SingleObjectiveProblem
import numpy as np
from numpy import sin, exp, sqrt, pi, cos, e, abs

class BukinN6(SingleObjectiveProblem):
    def __init__(self, **kwargs):
        super().__init__(n_params=2,
        
                         n_constraints=0,
                         param_type=np.double,
                         multi_dims=False)
        xl = np.ones((self.n_params,)) * [-15, -3]
        xu = np.ones((self.n_params,)) * [-5, 3]
        self.domain = (xl, xu)
        self._pareto_set = np.array([[-10, 1]])
        self._pareto_front = 0
        self._optimum = min
        self._argopt = np.argmin

    ## Overide Methods ##
    def _f(self, X):
        f = 100 * sqrt(abs(X[1] - 0.01*X[0]**2)) + 0.001 * abs(X[0] + 10)
        return f

    def _sol_compare(self, y1, y2):
        return y1 <= y2