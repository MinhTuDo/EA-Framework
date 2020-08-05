import numpy as np
from model.problem import Problem

class Booth(Problem):
    def __init__(self, **kwargs):
        super().__init__(n_params=2,
        
                         n_obj=1,
                         n_constraints=0,
                         domain=(-10, 10),
                         param_type=np.double,
                         multi_dims=False)
        
        self._pareto_front = 0
        self._pareto_set = np.array([1, 3]).reshape((-1, self.n_params))
        self._optimum = min
        self._argopt = np.argmin
    
    ## Overide Methods ##
    def _function(self, X):
        # f = (X[0]**2 _functionX[1] - 11)**2 + (X[0] + X[1]**2 - 7)**2
        f = (X[0] + 2*X[1] - 7)**2 + (2*X[0] + X[1] - 5)**2
        return f

    def _f_comparer(self, y1, y2):
        return y1 <= y2