from .so_problem import SingleObjectiveProblem
import numpy as np
from numpy import cos

class Shubert(SingleObjectiveProblem):
    def __init__(self, n_params=2, **kwargs):
        super().__init__(n_params,
                         n_constraints=0,
                         param_type=np.double,
                         multi_dims=True)
        xl = np.ones((self.n_params,)) * -10
        xu = np.ones((self.n_params,)) * 10
        self.domain = (xl, xu)
        self._optimum = min
        self._argopt = np.argmin
        self.j = np.arange(1, 6)


    def _f(self, X):
        try:
            f = -np.multiply.reduce(cos(X[:, np.newaxis].dot(self.j+1) + self.j).dot(self.j.T))
        except:
            f = -np.multiply.reduce(cos(X.dot(self.j+1) + self.j).dot(self.j.T))
        return f

    def _sol_compare(self, y1, y2):
        return y1 <= y2