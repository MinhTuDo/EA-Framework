import numpy as np

class Problem:
    def __init__(self,
                 n_params=-1,
                 n_obj=-1,
                 n_constraints=0,
                 domain=(0, 0),
                 param_type=None):
        if n_params <= 0:
            raise Exception('Parameters length must be greater than zero')
        self.n_params = n_params
        self.n_obj = n_obj
        self.n_constraints = n_constraints
        self.domain = domain
        self.type = param_type
        # self.multi_dims = multi_dims
        self.param_type = param_type

        self._pareto_front = None
        self._pareto_set = None
        self._optimum = None
        self._argopt = None

    def evaluate_all(self, pop):
        fitness_pop = np.reshape(list(map(self._f, pop)), (pop.shape[0], -1))
        return fitness_pop


    ## Protected Methods ##
    def _plot(self, **kwargs):
        pass

    def _make_data(self):
        pass

    def _f(self, X):
        pass

    def _sol_compare(self, y1, y2):
        pass