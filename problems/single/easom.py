from .so_problem import SingleObjectiveProblem
import numpy as np
from numpy import sin, exp, sqrt, pi, cos, e, abs

class Easom(SingleObjectiveProblem):
    def __init__(self, **kwargs):
        super().__init__(n_params=2,
        
                         n_constraints=0,
                         param_type=np.double,
                         multi_dims=False)
        xl = np.ones((self.n_params,)) * -100
        xu = np.ones((self.n_params,)) * 100
        self.domain = (xl, xu)
        self._pareto_set = np.array([[pi, pi]])
        self._pareto_front = -1
        self._optimum = min
        self._argopt = np.argmin

    ## Overide Methods ##
    def _f(self, X):
        f = -cos(X[0])*cos(X[1])*exp(-((X[0] - pi)**2 + (X[1] - pi)**2))
        return f

    def _sol_compare(self, y1, y2):
        return y1 <= y2