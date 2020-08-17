import numpy as np
from model.problem import Problem
class OneMax(Problem):
    def __init__(self, n_params=2):
        super().__init__(n_params,

                         n_obj=1, 
                         n_constraints=0, 
                         domain=(0, 1), 
                         param_type=np.int, 
                         multi_dims=True)
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


    