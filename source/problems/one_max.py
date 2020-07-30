import numpy as np
from model.problem import Problem
class OneMax(Problem):
    def __init__(self, 
                 n_params=2, 
                 n_obj=1, 
                 n_constraints=0,
                 domain=(0, 1),
                 param_type=np.int,
                 multi_dims=True):
        super().__init__(n_params, n_obj, n_constraints, 
                         domain, param_type, multi_dims)
        self._pareto_front = n_params
        self._pareto_set = np.ones((n_params,), dtype=param_type)
        self._comparer = max
        self._arg_comparer = np.argmax
        
    def _evaluate(self, X):
        f = sum(X)
        return f

    