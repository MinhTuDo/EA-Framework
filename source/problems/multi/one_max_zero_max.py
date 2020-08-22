from .mo_problem import MultiObjectiveProblem
import numpy as np
from problems.single import OneMax, ZeroMax

class OneMaxZeroMax(MultiObjectiveProblem):
    def __init__(self, n_params=2):
        super().__init__(n_params,
                         n_obj=2,
                         n_constraints=0,
                         domain=(0, 1),
                         param_type=np.int)
        self.__f1 = OneMax(n_params)
        self.__f2 = ZeroMax(n_params)
        self._argopt = self._get_best_idx
        self._optimum = self._get_best

    def _f(self, X):
        f1 = self.__f1._f(X)
        f2 = self.__f2._f(X)
        return f1, f2


    def _is_dominated(self, y1, y2):
        return (y1[0] >= y2[0] and y1[1] <= y2[1]) and \
               (y1[0] > y2[0] or y1[1] < y2[1])
