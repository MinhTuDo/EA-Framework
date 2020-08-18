from model.display import Display
from factory import GAFactory
from optimize import optimize
from utils.gif_saver import GifSaver
from model.log_saver import LogSaver

import numpy as np

class MySaver(LogSaver):
    def _do(self, ga):
        self.add_attributes('n_gens', ga.n_gens) 
        self.add_attributes('n_evals', ga.n_evals) 
        self.add_attributes('F', ga.F_pop[self.elite_idx])

class MyDisplay(Display):
    def _do(self, ga):
        self.display_top = -1
        self.add_attributes('n_gens', ga.n_gens)
        self.add_attributes('n_evals', ga.n_evals)
        # self.add_attributes('min', ga.F_pop.min(), width=5)
        # self.add_attributes('max', ga.F_pop.max(), width=5)
        # self.add_attributes('mean', ga.F_pop.mean(), width=5)
        self.add_attributes('F', ga.F_pop[ga.elite_idx])
        self.add_attributes('rank', ga.rank[ga.elite_idx])
        

display = MyDisplay()
log_saver = MySaver()
factory = GAFactory()
problem = factory.get_problem('SchafferN1')(A=4)
# problem.plot(plot_3D=True, contour_density=20, colorbar=True)

termination = factory.get_termination('MaxGenTermination')(5)

crossover = factory.get_crossover('GOM')()

algorithm = factory.get_algorithm('NSGAII')(pop_size=300, elitist_archive=4)

result = optimize(problem, 
                  algorithm, 
                  termination=termination, 
                  verbose=True,
                  log=False, 
                  save_history=True, 
                  seed=2, 
                  display=display,
                  log_saver=log_saver)
print(result.model)
print(result.exec_time)

gen1 = result.history[0]['F']
gen5 = result.history[1]['F']
gen10 = result.history[2]['F']

import matplotlib.pyplot as plt

f1 = result.problem.objectives[0]
f2 = result.problem.objectives[1]


# plt.plot(gen10[:, 0], gen10[:, 1], 'bo', label='gen 1')
plt.plot(gen1[:, 0], gen1[:, 1], 'g.', label='gen 0')
plt.plot(gen5[:, 0], gen5[:, 1], 'r.', label='gen 1')
plt.plot(gen10[:, 0], gen10[:, 1], 'b.', label='gen 2')
plt.xlabel('f1')
plt.ylabel('f2')
plt.xlim((0, 4))
plt.ylim((0, 4))
plt.legend(loc='upper right')
plt.grid(linestyle='--')
plt.title('Schaffer function N1.')
plt.show()

# gif_saver = GifSaver(problem, 'gif', 'Rastrigin-DE', contour_density=20)
# gif_saver.make(result, display_optimum=True, loop=False)
