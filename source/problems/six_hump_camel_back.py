from model.problem import Problem
import numpy as np

class SixHumpCamelBack(Problem):
    def __init__(self):
        super().__init__(n_params=2,
                         
                         n_obj=-1,
                         n_constraints=0,
                         domain=(-1.9, 1.9),
                         param_type=np.double,
                         multi_dims=False)
        
        self._optimum = min
        self._argopt = np.argmin
        self.step = 0.1
        
        
    ## Overide Methods ##
    def _function(self, X):
        f = -4 * ((4 - 2.1*X[0]**2 + (X[0]**4)/3) * X[0]**2 + \
                  X[0]*X[1] + (4*X[1]**2 - 4) * X[1]**2)
        return f

    def _f_comparer(self, y1, y2):
        return y1 <= y2