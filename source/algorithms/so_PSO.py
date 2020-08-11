from model.GA import GA
from terminations.max_time import MaxTimeTermination
from operators.selection.historical_best_selection import HistoricalBestSelection
from operators.initialization.random_initialization import RandomInitialization
import numpy as np
from numpy.random import uniform
from numpy.random import rand

class PSO(GA):
    def __init__(self,
                 pop_size=50,
                 initialization=RandomInitialization(),
                 topology='star',
                 iw=0.7298,
                 ac=1.49618,
                 **kwargs):
        super().__init__(pop_size, initialization, **kwargs)
        self.default_termination = MaxTimeTermination(5)
        if not topology in ['star', 'ring']:
            raise Exception('Population Topology not found!')
        self.global_best_selection = self.star_topology_selection
        if topology != 'star':
            self.global_best_selection = self.ring_topology_selection
        self.intertia_weight = iw
        self.accel_const = ac
        self.global_best = None
        self.pop_prev = None
        self.f_pop_prev = None

    def star_topology_selection(self):
        argopt = self.problem._argopt
        return argopt(self.f_pop_prev, axis=0)

    def ring_topology_selection(self):
        optimum = self.problem._optimum
        indices = []
        n = self.pop_size
        f_p = self.f_pop_prev
        f = self.f_pop
        for i in range(n-1):
            neighbors = (f[i-1], f[i], f_p[i+1])
            elite_idx = neighbors.index(optimum(neighbors))
            indices.append(elite_idx + (i-1))

        neighbors_last = (f_p[n-2], f_p[n-1], f_p[0])
        idx_last = neighbors_last.index(optimum(neighbors_last))
        if idx_last != 2:
            indices.append(idx_last + n-2)
        else:
            indices.append(0)
        return indices

    def compute_velocity(self):
        r_p, r_g = rand(), rand()
        c1, c2 = self.accel_const, self.accel_const
        w = self.intertia_weight
        p = self.pop_prev
        P = self.pop
        g = self.global_best
        v = self.velocity
        new_v = w*v + c1*r_p * (p-P) + c2*r_g * (g-P)
        return new_v

    def _initialize(self):
        self.pop_prev = self.pop.copy()
        (xl, xu) = self.problem.domain
        self.velocity = uniform(low=-abs(xu-xl),
                                high=-abs(xu-xl),
                                size=self.pop.shape).astype(self.problem.param_type)

    def _next(self):
        self.f_pop = self.evaluate(self.pop)
        self.f_pop_prev = self.evaluate(self.pop_prev)

        prev_best_idx = set(self.selection._do(self))
        indices = set(np.arange(self.pop_size))
        current_best_idx = list(indices - prev_best_idx)
        if len(current_best_idx) != 0:
            self.pop_prev[current_best_idx] = self.pop[current_best_idx]
            self.f_pop_prev[current_best_idx] = self.f_pop[current_best_idx]

        global_best_idx = self.global_best_selection()
        self.global_best = self.pop_prev[global_best_idx]

        self.velocity = self.compute_velocity()
        self.pop = (self.pop + self.velocity).astype(self.problem.param_type)








        