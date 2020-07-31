from algorithms.so_MBEA import MBEA
from algorithms.so_sGA import SGA
from algorithms.so_PSO import PSO

from utils.termination.max_eval import MaxEvalTermination
from utils.termination.convergence import Convergence

from problems.one_max import OneMax
from problems.trap import TrapMax
from problems.rastrigin import Rastrigin

from optimize import optimize
from operators.crossover.model_based_ux import ModelBasedUniformCrossover
from operators.crossover.uniform_crossover import UniformCrossover


# problem = TrapMax(n_params=20, trap_size=5)
problem = Rastrigin(n_params=10)

termination = Convergence()
crossover = ModelBasedUniformCrossover()
algorithm = PSO(pop_size=200, topology='star')

result = optimize(problem, algorithm, verbose=True, save_history=True, seed=1)
print(result.model)
print(result.exec_time)
print(result.n_evals)
