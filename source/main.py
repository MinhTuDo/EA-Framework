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


# problem = TrapMax(n_params=20, trap_size=5)
problem = Rastrigin()

termination = MaxGenTermination(50)
crossover = ModelBasedUniformCrossover()
algorithm = PSO(pop_size=100, topology='star')
# algorithm = MBEA(pop_size=500, elitist_archive=4, termination=termination)

result = optimize(problem, algorithm, termination=termination, verbose=True, save_history=True, seed=1)
print(result.model)
print(result.exec_time)
print(result.n_evals)

gif_saver = GifSaver(problem, './gif', 'test', contour_density=15)
gif_saver.make(result, display_optimum=True)
# import numpy as np
# from numpy import cos
# from numpy import pi
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# import matplotlib.pyplot as plt

# def rastrigin(X):
#     f = 10*len(X) + (X**2 - 10*cos(2*pi*X)).sum(axis=0)
#     return f

# fig = plt.figure()
# ax = Axes3D(fig, azim = -29, elev = 50)
# X = np.arange(-5, 5, 0.1)
# Y = np.arange(-5, 5, 0.1)
# X, Y = np.meshgrid(X, Y)
# Z = np.array([rastrigin(np.vstack((X[i], Y[i]))) for i in range(X.shape[0])])

# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
 
# plt.xlabel("x")
# plt.ylabel("y")

# plt.show()
