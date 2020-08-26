from .mo_problem import MultiObjectiveProblem
import numpy as np

class ZDT1(MultiObjectiveProblem):
    def __init__(self, n_params=30):
        super().__init__(n_params,
                         n_obj=2,
                         n_constraints=0,
                         domain=(0, 1),
                         param_type=np.double)
    
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


    def _is_dominated(self, y1, y2):
        return (y1[0] <= y2[0] and y1[1] <= y2[1]) and \
               (y1[0] < y2[0] or y1[1] < y2[1])

    def _calc_pareto_front(self):
        X1 = np.linspace(self.domain[0], self.domain[1], self._points)
        X2 = 1 - np.sqrt(X1)
        return (X1, X2)

class ZDT2(MultiObjectiveProblem):
    def __init__(self, n_params=30):
        super().__init__(n_params,
                         n_obj=2,
                         n_constraints=0,
                         domain=(0, 1),
                         param_type=np.double)
    
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
        return 1 - np.power(f1/g, 2)


    def _is_dominated(self, y1, y2):
        return (y1[0] <= y2[0] and y1[1] <= y2[1]) and \
               (y1[0] < y2[0] or y1[1] < y2[1])

    def _calc_pareto_front(self):
        X1 = np.linspace(self.domain[0], self.domain[1], self._points)
        X2 = 1 - X1**2
        return (X1, X2)