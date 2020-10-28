import time
from model.log_saver import LogSaver

class NSGANetLogSaver(LogSaver):
    def _do(self, ga):
        self.add_attributes('architecture', ga.pop[ga.elite_idx])
        self.add_attributes('error_rate', ga.F_pop[ga.elite_idx])
        self.add_attributes('flops', ga.F_pop[ga.elite_idx])
        self.add_attributes('n_gens', ga.n_gens) 
        self.add_attributes('n_evals', ga.n_evals) 
        
        now = time.time()
        self.add_attributes('time', now - ga.start_time)

class SOLogSaver(LogSaver):
    def _do(self, algorithm):
        self.add_attributes('mean', algorithm.fitness_pop.mean())
        self.add_attributes('std', algorithm.fitness_pop.std())
        self.add_attributes('n_gens', algorithm.n_gens)
        self.add_attributes('n_evals', algorithm.n_evals)

        self.add_attributes('elite', algorithm.pop[algorithm.elite_idx])