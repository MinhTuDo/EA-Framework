import numpy as np
from .so_problem import SingleObjectiveProblem

class Booth(SingleObjectiveProblem):
    def __init__(self, **kwargs):
        super().__init__(n_params=2,
        
                         n_constraints=0,
                         param_type=np.double,
                         multi_dims=False)
        xl = np.ones((self.n_params,)) * -10
        xu = np.ones((self.n_params,)) * 10
        self.domain = (xl, xu)
        self._pareto_front = 0
        self._pareto_set = np.array([1, 3]).reshape((-1, self.n_params))
        self._optimum = min
        self._argopt = np.argmin
    
    ## Overide Methods ##
    def _f(self, X):
        # f = (X[0]**2 _fX[1] - 11)**2 + (X[0] + X[1]**2 - 7)**2
        f = (X[0] + 2*X[1] - 7)**2 + (2*X[0] + X[1] - 5)**2
        return f

    def _sol_compare(self, y1, y2):
        return y1 <= y2