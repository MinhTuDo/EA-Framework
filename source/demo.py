# from model.display import Display
# from factory import GAFactory
# from optimize import optimize
# from utils.gif_saver import GifSaver
# from model.log_saver import LogSaver

# import numpy as np

# class MySaver(LogSaver):
#     def _do(self, algorithm):
#         self.add_attributes('n_gens', algorithm.n_gens) 
#         self.add_attributes('n_evals', algorithm.n_evals) 
#         self.add_attributes('F', algorithm.f_opt)

# class MyDisplay(Display):
#     def _do(self, algorithm):
#         self.display_top = -1
#         self.add_attributes('n_gens', algorithm.n_gens)
#         self.add_attributes('n_evals', algorithm.n_evals)
#         self.add_attributes('min', algorithm.f_pop.std(), width=5)
#         self.add_attributes('mean', algorithm.f_pop.mean(), width=5)
#         self.add_attributes('F', algorithm.f_opt)
        

# display = MyDisplay()
# log_saver = MySaver()
# factory = GAFactory()
# problem = factory.get_problem('Rastrigin')()
# # problem.plot(plot_3D=True, contour_density=20, colorbar=True)

# termination = factory.get_termination('MaxGenTermination')(100)

# crossover = factory.get_crossover('ModelBasedUniformCrossover')()

# algorithm = factory.get_algorithm('DE')(pop_size=500)

# result = optimize(problem, 
#                   algorithm, 
#                   termination=termination, 
#                   verbose=True,
#                   log=False, 
#                   save_history=True, 
#                   seed=18521578, 
#                   display=display,
#                   log_saver=log_saver)

# gif_saver = GifSaver(problem, 'gif', 'Rastrigin-DE', contour_density=20)
# gif_saver.make(result, display_optimum=True, loop=False)
