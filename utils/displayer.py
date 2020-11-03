import time
from model.display import Display
from utils.flops_benchmark import add_flops_counting_methods
import torch
import numpy as np

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

def print_info(self):
    n_params = (np.sum(np.prod(v.size()) for v in filter(lambda p: p.requires_grad, self.model.parameters())) / 1e6)

    self.model = add_flops_counting_methods(self.model)
    self.model.eval()
    self.model.start_flops_count()
    random_data = torch.randn(1, *self.data_info['input_size'])
    self.model(torch.autograd.Variable(random_data).to(self.device))
    n_flops = np.round(self.model.compute_average_flops_cost() / 1e6, 4)

    print('{} Million of parameters | {} MFLOPS'.format(n_params, n_flops))