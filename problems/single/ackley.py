from .so_problem import SingleObjectiveProblem
import numpy as np
from numpy import sin, exp, sqrt, pi, cos, e

class Ackley(SingleObjectiveProblem):
    def __init__(self, **kwargs):
        super().__init__(n_params=2,
        
                         n_constraints=0,
                         param_type=np.double,
                         multi_dims=False)
        xl = np.ones((self.n_params,)) * -5
        xu = np.ones((self.n_params,)) * 5
        self.domain = (xl, xu)
        self._pareto_set = np.array([[0, 0]])
        self._pareto_front = 0
        self._optimum = min
        self._argopt = np.argmin

    ## Overide Methods ##
    def _f(self, X):
        f = -20 * exp(-0.2 * sqrt(0.5 * (X[0]**2 + X[1]**2))) - exp(0.5 * (cos(2*pi*X[0]) + cos(2*pi*X[1]))) + e + 20
        return f

    def _sol_compare(self, y1, y2):
        return y1 <= y2