from .so_problem import SingleObjectiveProblem
import numpy as np

class TrapMax(SingleObjectiveProblem):
    def __init__(self,
                 n_params,
                 trap_size,
                 **kwargs):
        super().__init__(n_params,
                         n_constraints=0,
                         param_type=np.int, 
                         multi_dims=True)
        if n_params % trap_size != 0:
            raise Exception('Parameters length must be divisible by trap size')
        xl = np.ones((self.n_params,)) * 0
        xu = np.ones((self.n_params,)) * 1
        self.domain = (xl, xu)
        self._pareto_front = n_params
        self._pareto_set = np.ones((1, n_params), dtype=self.param_type)
        self._optimum = max
        self._argopt = np.argmax

        self.trap_size = trap_size
    
    ## Overide Methods ##
    def _f(self, X):
        f = 0
        k = self.trap_size
        for i in range(0, self.n_params, k):
            u = (X[i : i+k]).sum()
            f += u if u == k else k-u-1
            
        return f

    def _sol_compare(self, y1, y2):
        return y1 >= y2