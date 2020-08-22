from .so_problem import SingleObjectiveProblem
import numpy as np

class Beale(SingleObjectiveProblem):
    def __init__(self):
        super().__init__(n_params=2,
                         n_obj=-1,
                         n_constraints=0,
                         domain=(-4.5, 4.5),
                         param_type=np.double,
                         multi_dims=False)
        self._pareto_set = np.array([3, 0.5]).reshape((-1, self.n_params))
        self._pareto_front = 0
        self._optimum = min
        self._argopt = np.argmin

    ## Overide Methods ##
    def _f(self, X):
        f = (1.5 - X[0] + X[0]*X[1])**2 + \
            (2.25 - X[0] + X[0]*(X[1]**2))**2 + \
            (2.265 - X[0] + X[0]*(X[1]**3))**2
            
        return f

    def _sol_compare(self, y1, y2):
        return y1 <= y2