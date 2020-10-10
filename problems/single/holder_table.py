from .so_problem import SingleObjectiveProblem
import numpy as np
from numpy import sin, exp, sqrt, pi, cos, e, abs

class HolderTable(SingleObjectiveProblem):
    def __init__(self, **kwargs):
        super().__init__(n_params=2,
        
                         n_constraints=0,
                         param_type=np.double,
                         multi_dims=False)
        xl = np.ones((self.n_params,)) * -10
        xu = np.ones((self.n_params,)) * 10
        self.domain = (xl, xu)
        self._pareto_set = np.array([[8.05502, 9.66459],
                                     [-8.05502, 9.66459],
                                     [8.05502, -9.66459],
                                     [-8.05502, -9.66459]])
        self._pareto_front = -19.2085
        self._optimum = min
        self._argopt = np.argmin

    ## Overide Methods ##
    def _f(self, X):
        f = - abs(sin(X[0]) * cos(X[1]) * exp(abs(1 - sqrt(X[0]**2 + X[1]**2) / pi)))
        return f

    def _sol_compare(self, y1, y2):
        return y1 <= y2