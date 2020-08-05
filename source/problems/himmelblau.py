from model.problem import Problem
import numpy as np

class Himmelblau(Problem):
    def __init__(self):
        super().__init__(n_params=2,
                         
                         n_obj=-1,
                         n_constraints=0,
                         domain=(-5, 5),
                         param_type=np.double,
                         multi_dims=False)
        
        self._pareto_set = np.array([[3, 2],
                                    [-2.805118, 3.131312],
                                    [-3.779310, -3.283186],
                                    [3.584428, -1.848126]])
        self._pareto_front = 0
        self._optimum = min
        self._argopt = np.argmin
    
    ## Overide Methods ##
    def _function(self, X):
        f = (X[0]**2 + X[1] - 11)**2 + \
            (X[0] + X[1]**2 - 7)**2
        return f

    def _f_comparer(self, y1, y2):
        return y1 <= y2