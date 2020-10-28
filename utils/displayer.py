import time
from model.display import Display

class NSGANetDisplay(Display):
    def _do(self, ga):
        self.display_top = -1
        self.add_attributes('n_gens', ga.n_gens) 
        self.add_attributes('n_evals', ga.n_evals) 
        self.add_attributes('error_rate', ga.F_pop[ga.elite_idx])
        self.add_attributes('flops', ga.F_pop[ga.elite_idx])
        now = time.time()
        self.add_attributes('time', now - ga.start_time)
        self.add_attributes('architecture', ga.pop[ga.elite_idx])

class SODisplay(Display):
    def _do(self, algorithm):
        self.display_top = 10
        self.add_attributes('mean', algorithm.fitness_pop.mean())
        self.add_attributes('std', algorithm.fitness_pop.std())
        self.add_attributes('n_gens', algorithm.n_gens)
        self.add_attributes('n_evals', algorithm.n_evals)

        self.add_attributes('elite', algorithm.pop[algorithm.elite_idx])