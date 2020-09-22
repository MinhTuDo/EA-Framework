import numpy as np
from .so_problem import SingleObjectiveProblem

class OneMax(SingleObjectiveProblem):
    def __init__(self, n_params=2, **kwargs):
        super().__init__(n_params,
                         n_constraints=0, 
                         param_type=np.int, 
                         multi_dims=True)
        xl = np.ones((self.n_params,)) * 0
        xu = np.ones((self.n_params,)) * 1
        self.domain = (xl, xu)
        self._pareto_front = n_params
        self._pareto_set = np.ones((1, n_params), dtype=self.param_type)
        self._optimum = max
        self._argopt = np.argmax
    
    ## Overide Methods ##
    def _f(self, X):
        f = sum(X)
        return f

    def _sol_compare(self, y1, y2):
        return y1 >= y2

class ZeroMax(SingleObjectiveProblem):
    def __init__(self, n_params=2, **kwargs):
        super().__init__(n_params,
                         n_constraints=0, 
                         param_type=np.int, 
                         multi_dims=True)
        xl = np.ones((self.n_params,)) * 0
        xu = np.ones((self.n_params,)) * 1
        self.domain = (xl, xu)
        self._pareto_front = n_params
        self._pareto_set = np.zeros((1, n_params), dtype=self.param_type)
        self._optimum = min
        self._argopt = np.argmin
    
    ## Overide Methods ##
    def _f(self, X):
        f = sum(X)
        return f

    def _sol_compare(self, y1, y2):
        return y1 <= y2

    