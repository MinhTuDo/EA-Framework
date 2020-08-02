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
from problems.booth import Booth

from optimize import optimize
from operators.crossover.model_based_ux import ModelBasedUniformCrossover
from operators.crossover.uniform_crossover import UniformCrossover

from model.display import Display

import numpy as np

class MyDisplay(Display):
    def _do(self, algorithm):
        self.add_attributes('n_gens', algorithm.n_gens)
        self.add_attributes('n_evals', algorithm.n_evals)
        self.add_attributes('min', algorithm.f_pop.std(), width=5)
        self.add_attributes('mean', algorithm.f_pop.mean(), width=5)
        #self.add_attributes('elite', algorithm.opt, width=4)
        

display = MyDisplay()


# problem = TrapMax(n_params=20, trap_size=5)
problem = Rastrigin(n_params=5)
# problem.plot(plot_3D=True, contour_density=25, colorbar=True)

#termination = MaxGenTermination(10)
termination = Convergence()
# crossover = ModelBasedUniformCrossover()
algorithm = PSO(pop_size=100, topology='star')
# algorithm = MBEA(pop_size=500, elitist_archive=4, termination=termination)

result = optimize(problem, algorithm, termination=termination, verbose=True, save_history=True, seed=1, display=display)
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
