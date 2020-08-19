from model.problem import Problem
import numpy as np

class ZDT1(Problem):
    def __init__(self, n_params=2):
        super().__init__(n_params,
                         n_obj=2,
                         n_constraints=0,
                         domain=(0, 1),
                         param_type=np.double)
        self.objectives = [self.__f1, self.__f2]
        self._argopt = self._get_best_idx
        self._optimum = self._get_best
    
    def _f(self, X):
        f1 = self.__f1(X)
        f2 = self.__f2(X)
        return f1, f2
    
    def __f1(self, X):
        return X[0]

    def __f2(self, X):
        f1 = self.__f1(X)
        g = self.__g(X)
        return g * self.__h(f1, g)

    def __g(self, X):
        g = 1 + (9/(self.n_params-1) * X[1:].sum())
        return g

    def __h(self, f1, g):
        return 1 - np.power(f1/g, 0.5)

    def _sol_compare(self, s1, s2):
        r1, r2 = s1[0], s2[0]
        cd1, cd2 = s1[1], s2[1]
        if r1 < r2:
            return True
        if r1 > r2:
            return False
        if r1 == r2:
            if cd1 > cd2:
                return True
            if cd1 < cd2:
                return False
        return True

    def _is_dominated(self, y1, y2):
        return (y1[0] <= y2[0] and y1[1] <= y2[1]) and \
               (y1[0] < y2[0] or y1[1] < y2[1])

    def _get_best(self, Y):
        opt = Y[0]
        for y in Y:
            opt = opt if self._sol_compare(opt, y) else y
        return opt

    def _get_best_idx(self, Y):
        argopt = 0
        for i, y_i in enumerate(Y):
            argopt = argopt if self._sol_compare(Y[argopt], y_i) else i
        return argopt