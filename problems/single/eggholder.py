from .so_problem import SingleObjectiveProblem
import numpy as np
from numpy import sin, exp, sqrt, pi, cos, e, abs

class EggHolder(SingleObjectiveProblem):
    def __init__(self, **kwargs):
        super().__init__(n_params=2,
        
                         n_constraints=0,
                         param_type=np.double,
                         multi_dims=False)
        xl = np.ones((self.n_params,)) * -512
        xu = np.ones((self.n_params,)) * 512
        self.domain = (xl, xu)
        self._pareto_set = np.array([[512, 404.2319]])
        self._pareto_front = -959.6407
        self._optimum = min
        self._argopt = np.argmin

    ## Overide Methods ##
    def _f(self, X):
        f = -(X[1] + 47)*sin(sqrt(abs(X[0]/2 + (X[1] + 47)))) - X[0]*sin(sqrt(abs(X[0] - (X[1] + 47))))
        return f

    def _sol_compare(self, y1, y2):
        return y1 <= y2