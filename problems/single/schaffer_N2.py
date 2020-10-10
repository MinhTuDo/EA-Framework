from .so_problem import SingleObjectiveProblem
import numpy as np
from numpy import sin, exp, sqrt, pi, cos, e, abs

class SchafferN2(SingleObjectiveProblem):
    def __init__(self, **kwargs):
        super().__init__(n_params=2,
        
                         n_constraints=0,
                         param_type=np.double,
                         multi_dims=False)
        xl = np.ones((self.n_params,)) * -100
        xu = np.ones((self.n_params,)) * 100
        self.domain = (xl, xu)
        self._pareto_set = np.array([[0, 0]])
        self._pareto_front = 0
        self._optimum = min
        self._argopt = np.argmin

    ## Overide Methods ##
    def _f(self, X):
        f = 0.5 + (sin(X[0]**2 - X[1]**2)**2 - 0.5) / abs(1 + 0.001 * (X[0]**2 + X[1]**2))**2
        return f

    def _sol_compare(self, y1, y2):
        return y1 <= y2