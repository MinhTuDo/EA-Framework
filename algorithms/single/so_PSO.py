from algorithms.GA import GA
from terminations.max_time import MaxTimeTermination
from operators.selection.historical_best_selection import HistoricalBestSelection
from operators.initialization.random_initialization import RandomInitialization
from operators.repairer.out_of_bounds_repair import OutOfBoundsRepair, repair_out_of_bounds_manually
import numpy as np
from numpy.random import random, rand
from utils import denormalize

class PSO(GA):
    def __init__(self,
                 pop_size=50,
                 initialization=RandomInitialization(),
                 topology='star',
                 iw=0.7298,
                 ac=1.49618,
                 max_velocity_rate=0.2,
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
        self.max_velocity_rate = max_velocity_rate
        self.global_best = None
        self.offs = None
        self.fitness_offs = None
        self.selection = HistoricalBestSelection()
        self.repairer = OutOfBoundsRepair()

    def star_topology_selection(self):
        argopt = self.problem._argopt
        return argopt(self.fitness_offs, axis=0)

    def ring_topology_selection(self):
        optimum = self.problem._optimum
        indices = []
        n = self.pop_size
        f_p = self.fitness_offs
        f = self.fitness_pop
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
        p = self.offs
        P = self.pop
        g = self.global_best
        v = self.velocity
        new_v = w*v + c1*r_p * (p-P) + c2*r_g * (g-P)
        return repair_out_of_bounds_manually(new_v, -self.v_max, self.v_max)

    def initialize_velocity(self, xl, xu):
        xl = -abs(xu - xl)
        xu = abs(xu - xl)
        velocity = random(self.pop.shape)
        velocity = denormalize(velocity, xl, xu)
        return velocity

    def _initialize(self):
        self.offs = self.pop.copy()
        (xl, xu) = self.problem.domain
        self.velocity = self.initialize_velocity(xl, xu)
        self.v_max = self.max_velocity_rate * (xu - xl)

    def _next(self):
        self.fitness_pop = self.evaluate(self.pop)
        self.fitness_offs = self.evaluate(self.offs)

        selected_indices = set(self.selection._do(self))
        indices = set(np.arange(self.pop_size))
        current_best_idx = list(indices - selected_indices)
        if len(current_best_idx) != 0:
            self.offs[current_best_idx] = self.pop[current_best_idx]
            self.fitness_offs[current_best_idx] = self.fitness_pop[current_best_idx]

        global_best_idx = self.global_best_selection()
        self.global_best = self.offs[global_best_idx]

        self.velocity = self.compute_velocity()
        self.pop = (self.pop + self.velocity).astype(self.problem.param_type)
        self.pop = self.repairer._do(self.problem, self.pop)

    def _save_history(self):
        res = {'P': self.pop.copy(), 
               'F': self.fitness_pop.copy()}
        self.history.append(res)








        