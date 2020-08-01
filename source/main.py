from algorithms.so_MBEA import MBEA
from algorithms.so_sGA import SGA
from algorithms.so_PSO import PSO

from utils.termination.max_eval import MaxEvalTermination
from utils.termination.convergence import Convergence
from utils.termination.max_gen import MaxGenTermination

from utils.gif_saver import GifSaver

from problems.one_max import OneMax
from problems.trap import TrapMax
from problems.rastrigin import Rastrigin

from optimize import optimize
from operators.crossover.model_based_ux import ModelBasedUniformCrossover
from operators.crossover.uniform_crossover import UniformCrossover


# problem = TrapMax(n_params=20, trap_size=5)
problem = Rastrigin(n_params=2)

termination = MaxGenTermination(50)
crossover = ModelBasedUniformCrossover()
algorithm = PSO(pop_size=100, topology='ring')

result = optimize(problem, algorithm, termination=termination, verbose=True, save_history=True, seed=1)
print(result.model)
print(result.exec_time)
print(result.n_evals)

gif_saver = GifSaver(problem, './gif', 'ring-topology', contour_density=10)
gif_saver.make(result, display_optimum=True)
