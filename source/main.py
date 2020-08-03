from algorithms.so_MBEA import MBEA
from algorithms.so_sGA import SGA
from algorithms.so_PSO import PSO

from terminations.max_eval import MaxEvalTermination
from terminations.convergence import Convergence
from terminations.max_gen import MaxGenTermination

from utils.gif_saver import GifSaver

from problems.one_max import OneMax
from problems.trap import TrapMax
from problems.rastrigin import Rastrigin
from problems.booth import Booth

from optimize import optimize
from operators.crossover.model_based_ux import ModelBasedUniformCrossover
from operators.crossover.uniform_crossover import UniformCrossover
from operators.crossover.model_based_1x import ModelBasedOnePointCrossover

from model.display import Display

import numpy as np

class MyDisplay(Display):
    def _do(self, algorithm):
        self.display_top = 5
        self.add_attributes('n_gens', algorithm.n_gens)
        self.add_attributes('n_evals', algorithm.n_evals)
        self.add_attributes('min', algorithm.f_pop.std(), width=5)
        self.add_attributes('mean', algorithm.f_pop.mean(), width=5)
        # self.add_attributes('elite', algorithm.opt, width=3)
        

display = MyDisplay()


problem = TrapMax(n_params=20, trap_size=5)
# problem = Rastrigin(n_params=2)
# problem.plot(plot_3D=True, contour_density=25, colorbar=True)

# termination = MaxGenTermination(5000)
termination = Convergence()
# crossover = ModelBasedUniformCrossover()
crossover = ModelBasedOnePointCrossover()
# algorithm = PSO(pop_size=500, topology='star')
algorithm = MBEA(pop_size=5000, elitist_archive=4, termination=termination, crossover=crossover)

result = optimize(problem, algorithm, termination=termination, verbose=True, save_history=True, seed=1, display=None)
print(result.model)
print(result.exec_time)
print(result.n_evals)

# gif_saver = GifSaver(problem, './gif', 'test', contour_density=15)
# gif_saver.make(result, display_optimum=True)


# from pymoo.algorithms.nsga2 import NSGA2
# from pymoo.factory import get_problem
# from pymoo.optimize import minimize
# from pymoo.util.display import Display
# import numpy as np


# class MyDisplay(Display):

#     def _do(self, problem, evaluator, algorithm):
#         super()._do(problem, evaluator, algorithm)
#         self.output.append("metric_a", np.mean(algorithm.pop.get("X")))
#         self.output.append("metric_b", np.mean(algorithm.pop.get("F")))


# problem = get_problem("zdt2")

# algorithm = NSGA2(pop_size=100)

# res = minimize(problem,
#                algorithm,
#                ('n_gen', 100),
#                seed=1,
#                display=MyDisplay(),
#                verbose=True)
