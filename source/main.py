from algorithms.so_sGA import SGA
from problems.one_max import OneMax
from optimize import optimize


problem = OneMax(n_params=10)
algorithm = SGA(pop_size=10, n_offs=10)

result = optimize(problem, algorithm, verbose=True, save_history=True, seed=2)

