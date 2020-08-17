from model.problem import Problem
import numpy as np

class TrapMax(Problem):
    def __init__(self,
                 n_params,
                 trap_size):
        super().__init__(n_params,

                         n_obj=1, 
                         n_constraints=0,
                         domain=(0, 1), 
                         param_type=np.int, 
                         multi_dims=True)
        if n_params % trap_size != 0:
            raise Exception('Parameters length must be divisible by trap size')
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