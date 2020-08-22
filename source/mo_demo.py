import algorithms.multi_objective as mo

from terminations import *

from model import LogSaver, Display

from optimize import optimize

import operators.crossover as cx
import operators.initialization as init
import operators.mutation as mut
import operators.selection as sel
import operators.model_builder as mb

import problems.multi as mp

import matplotlib.pyplot as plt

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
        self.add_attributes('F', ga.F_pop[ga.elite_idx])
        

display = MyDisplay()
log_saver = MySaver()
problem = mp.ZDT1(n_params=30)
# problem.plot(plot_3D=True, contour_density=20, colorbar=True)

termination = MaxGenTermination(200)

algorithm = mo.NSGAII(pop_size=100, elitist_archive=2)

result = optimize(problem, 
                  algorithm, 
                  termination=termination, 
                  verbose=True,
                  log=False, 
                  save_history=True, 
                  seed=18521578, 
                  display=display,
                  log_saver=log_saver)
print(result.model)
print(result.exec_time)

gen20 = result.history[19]['F']
gen50 = result.history[49]['F']
gen100 = result.history[99]['F']
gen200 = result.history[199]['F']



plt.plot(gen20[:, 0], gen20[:, 1], 'g.', label='gen 20')
plt.plot(gen50[:, 0], gen50[:, 1], 'r.', label='gen 50')
plt.plot(gen100[:, 0], gen100[:, 1], 'c.', label='gen 100')
plt.plot(gen200[:, 0], gen200[:, 1], 'b.', label='gen 200')
plt.xlabel('f1')
plt.ylabel('f2')
# plt.xlim((0, 1))
# plt.ylim((0, 1))
plt.legend(loc='upper right')
plt.grid(linestyle='--')
plt.title('zdt1 (Bounded SBX)')
plt.show()

# gif_saver = GifSaver(problem, 'gif', 'Rastrigin-DE', contour_density=20)
# gif_saver.make(result, display_optimum=True, loop=False)
