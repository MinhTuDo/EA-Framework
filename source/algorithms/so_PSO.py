from model.GA import GA
from utils.termination.max_time import MaxTime
from operators.selection.historical_best_selection import HistoricalBestSelection
import numpy as np
from numpy.random import uniform

class PSO(GA):
    def __init__(self,
                 pop_size,
                 initialization=RandomInitialization(),
                 selection=HistoricalBestSelection(),
                 topology='star',

                 iw=0.7298,
                 ac=1.49618,

                 **kwargs):
        super().__init__(pop_size, initialization, selection, mutation, **kwargs)
        self.default_termination = MaxTime(5)
        if not topology in ['star', 'ring']:
            raise Exception('Population Topology not found!')
        self.selection = selection
        self.initialization = initialization
        self.intertia_weight = iw
        self.accel_const = ac

    def _initialize(self):
        self.pop_prev = self.pop.copy()
        self.f_pop_prev = self.f_pop.copy()
        (xl, xu) = self.problem.domain
        self.velocity = uniform(low=-abs(xu-xl),
                                high=-abs(xu-xl),
                                size=self.pop.shape).astype(self.problem.param_type)

    def _next(self):
        self.f_pop = self.evaluate(self.pop)
        self.f_pop_prev = self.evaluate(self.pop_prev)

        prev_best = self.selection._do(self)
        indices = np.arange(self.pop_size)






        