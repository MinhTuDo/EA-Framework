from .mo_problem import MultiObjectiveProblem
import numpy as np

class SchafferN1(MultiObjectiveProblem):
    def __init__(self, 
                 A=5):
        super().__init__(n_params=1,
                         n_obj=2,
                         n_constraints=0,
                         param_type=np.double)
        xl = np.ones((self.n_params,)) * -A
        xu = np.ones((self.n_params,)) * A
        self.domain = (xl, xu)

    @staticmethod
    def __f1(X):
        return X**2
    
    @staticmethod
    def __f2(X):
        return (X - 2)**2

    ## Overide Methods ##
    def _f(self, X):
        f1 = self.__f1(X)
        f2 = self.__f2(X)
        return f1, f2

    @staticmethod
    def _is_dominated(y1, y2):
        return (y1[0] <= y2[0] and y1[1] <= y2[1]) and \
               (y1[0] < y2[0] or y1[1] < y2[1])