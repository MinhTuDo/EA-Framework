from .so_problem import SingleObjectiveProblem
import numpy as np
from numpy import sin, exp, sqrt, pi

class CrossInTray(SingleObjectiveProblem):
    def __init__(self, **kwargs):
        super().__init__(n_params=2,
        
                         n_constraints=0,
                         param_type=np.double,
                         multi_dims=False)
        xl = np.ones((self.n_params,)) * -10
        xu = np.ones((self.n_params,)) * 10
        self.domain = (xl, xu)
        self._pareto_set = np.array([[1.34941, -1.34941],
                                    [1.34941, 1.34941],
                                    [-1.34941, -1.34941],
                                    [-1.34941, 1.34941]])
        self._pareto_front = -2.06261
        self._optimum = min
        self._argopt = np.argmin

    ## Overide Methods ##
    def _f(self, X):
        f = -0.0001 * (abs(sin(X[0]) * sin(X[1]) * \
            exp(abs(100 - sqrt(X[0]**2 + X[1]**2)/pi))) + 1)**0.1
        return f

    def _sol_compare(self, y1, y2):
        return y1 <= y2