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
        x1 = X[0]
        x2 = X[1]
        sum1 = 0
        sum2 = 0
        for i in range(5):
            new1 = i * cos((i+1) * x1 + i)
            new2 = i * cos((i+1) * x2 + i)
            sum1 += new1
            sum2 += new2
        return sum1 * sum2

    def _sol_compare(self, y1, y2):
        return y1 <= y2