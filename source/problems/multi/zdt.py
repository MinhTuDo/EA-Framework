from .mo_problem import MultiObjectiveProblem
import numpy as np
from numpy import sin, pi, cos, exp

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


class ZDT3(MultiObjectiveProblem):
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
        return 1 - np.power(f1/g, 0.5) - (f1/g) * sin(10 * pi * f1)


    def _is_dominated(self, y1, y2):
        return (y1[0] <= y2[0] and y1[1] <= y2[1]) and \
               (y1[0] < y2[0] or y1[1] < y2[1])

    def _calc_pareto_front(self):
        regions = [[0, 0.0830015349],
                   [0.182228780, 0.2577623634],
                   [0.4093136748, 0.4538821041],
                   [0.6183967944, 0.6525117038],
                   [0.8233317983, 0.8518328654]]

        X1, X2 = [], []

        for r in regions:
            x1 = np.linspace(r[0], r[1], self._points // len(regions))
            x2 = 1 - np.sqrt(x1) - x1 * sin(10 * pi * x1)
            X1.append(x1), X2.append(x2)
        
        X1 = np.array(X1).flatten()
        X2 = np.array(X2).flatten()
        return (np.array(X1), np.array(X2))

class ZDT4(MultiObjectiveProblem):
    def __init__(self, n_params=10):
        super().__init__(n_params,
                         n_obj=2,
                         n_constraints=0,
                         domain=(-5, 5),
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
        g = (1 + (10*(self.n_params-1)) + (X[1:]**2 - 10*cos(4*pi*X[1:])).sum())
        # g = 1 + (9/(self.n_params-1) * X[1:].sum())
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

class ZDT6(MultiObjectiveProblem):
    def __init__(self, n_params=10):
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
        return 1 - exp(-4*X[0]) * sin(6*pi*X[0])**6

    def __f2(self, X):
        f1 = self.__f1(X)
        g = self.__g(X)
        return g * self.__h(f1, g)

    def __g(self, X):
        g = 1 + 9 * (X[1:].sum()/9)**0.25
        return g

    def __h(self, f1, g):
        return 1 - np.power(f1/g, 2)

    def _is_dominated(self, y1, y2):
        return (y1[0] <= y2[0] and y1[1] <= y2[1]) and \
               (y1[0] < y2[0] or y1[1] < y2[1])

    def _calc_pareto_front(self):
        X1 = np.linspace(0.2807753191, self.domain[1], self._points)
        X2 = 1 - X1**2
        return (X1, X2)


