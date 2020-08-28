from .so_problem import SingleObjectiveProblem
import numpy as np

class SixHumpCamelBack(SingleObjectiveProblem):
    def __init__(self, **kwargs):
        super().__init__(n_params=2,
                         n_constraints=0,
                         param_type=np.double,
                         multi_dims=False)
        xl = np.ones((self.n_params,)) * -1.9
        xu = np.ones((self.n_params,)) * 1.9
        self.domain = (xl, xu)
        self._optimum = min
        self._argopt = np.argmin
        self.step = 0.1
        
        
    ## Overide Methods ##
    def _f(self, X):
        f = -4 * ((4 - 2.1*X[0]**2 + (X[0]**4)/3) * X[0]**2 + \
                  X[0]*X[1] + (4*X[1]**2 - 4) * X[1]**2)
        return f

    def _sol_compare(self, y1, y2):
        return y1 <= y2