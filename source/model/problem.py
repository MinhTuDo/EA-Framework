import numpy as np
class Problem:
    def __init__(self,
                 n_params=-1,
                 n_obj=-1,
                 n_constraints=0,
                 domain=(0, 0),
                 param_type=None,
                 multi_dims=False):
        self.n_params = n_params
        self.n_obj = n_obj
        self.n_constraints = n_constraints
        self.domain = domain
        self.type = param_type
        self.multi_dims = multi_dims
        self.param_type = param_type

        self._comparer = None
        self._f_comparer = None

    def evaluate(self, pop):
        f_pop = np.array(list(map(self._evaluate, pop)))
        return f_pop

    def _evaluate(self, X):
        pass

    def _pareto_front(self):
        pass

    def _pareto_set(self):
        pass