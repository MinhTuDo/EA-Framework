from algorithms.so_MBEA import MBEA
from utils.termination.max_eval import MaxEvalTermination
from utils.termination.convergence import Convergence
from problems.one_max import OneMax
from optimize import optimize
from operators.crossover.model_based_ux import ModelBasedUniformCrossover


problem = OneMax(n_params=20)

termination = Convergence()
crossover = ModelBasedUniformCrossover()
algorithm = MBEA(pop_size=500, elitist_archive=2, termination=termination, crossover=crossover)

result = optimize(problem, algorithm, verbose=True, save_history=True, seed=1)
print(result.model)
