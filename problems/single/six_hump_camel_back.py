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
        xl[1] = -1.1
        xu[1] = 1.1
        self.domain = (xl, xu)
        self._optimum = min
        self._argopt = np.argmin
        
        
    ## Overide Methods ##
    def _f(self, X):
        x = X[0]
        y = X[1]
        f = -4 * ((4 - 2.1*x**2 + (x**4)/3) * x**2 + \
                   x*y + (4*y**2 - 4)*y**2)
        return f

    def _sol_compare(self, y1, y2):
        return y1 <= y2