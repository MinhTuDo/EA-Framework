from .so_problem import SingleObjectiveProblem
import numpy as np
from numpy import sin, exp, sqrt, pi, cos, e, abs

class GoldsteinPrice(SingleObjectiveProblem):
    def __init__(self, **kwargs):
        super().__init__(n_params=2,
        
                         n_constraints=0,
                         param_type=np.double,
                         multi_dims=False)
        xl = np.ones((self.n_params,)) * -2
        xu = np.ones((self.n_params,)) * 2
        self.domain = (xl, xu)
        self._pareto_set = np.array([[0, -1]])
        self._pareto_front = 3
        self._optimum = min
        self._argopt = np.argmin

    ## Overide Methods ##
    def _f(self, X):
        f = (1 + (X[0] + X[1] + 1)**2 * (19 - 14*X[0] + 3*X[0]**2 - 14*X[1] + 6*X[0]*X[1] + 3*X[1]**2))*\
            (30 + (2*X[0] - 3*X[1])**2 * (18 - 32*X[0] + 12*X[0]**2 + 48*X[1] - 36*X[0]*X[1] + 27*X[1]**2))
        return f

    def _sol_compare(self, y1, y2):
        return y1 <= y2