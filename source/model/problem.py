import numpy as np
class Problem:
    def __init__(self,
                 n_params=-1,
                 n_obj=-1,
                 n_constraints=0,
                 domain=(0, 0),
                 param_type=None,
                 multi_dims=False):
        if n_params <= 0:
            raise Exception('Parameters length must be greater than zero')
        self.n_params = n_params
        self.n_obj = n_obj
        self.n_constraints = n_constraints
        self.domain = domain
        self.type = param_type
        self.multi_dims = multi_dims
        self.param_type = param_type

        self._pareto_front = None
        self._pareto_set = None
        self._optimum = None
        self._argopt = None

    def evaluate(self, pop):
        f_pop = np.array(list(map(self._evaluate, pop)))
        return f_pop


    ## Protected Methods ##
    def _evaluate(self, X):
        pass

    def _pareto_front(self):
        pass

    def _pareto_set(self):
        pass

    def _comparer(self, x1, x2):
        pass
    def _f_comparer(self, y1, y2):
        pass