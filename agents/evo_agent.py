from model.agent import Agent
from algorithms import *
from problems import *

import operators.crossover as cx
import operators.initialization as init
import operators.mutation as mut
import operators.selection as sel
import operators.model_builder as mb

from terminations import *

from utils.displayer import *
from utils.logger import *

class EvoAgent(Agent):
    def __init__(self, 
                 algorithm, 
                 algorithm_args,
                 problem,
                 problem_args, 
                 termination, 
                 termination_args,
                 setup_args,
                 display=None, 
                 log_saver=None, 
                 **kwargs):

        operations = ['crossover', 'mutation', 'selection', 'model_builder', 'initialization']
        modules = [cx, mut, sel, mb, init]
        for op, module in zip(operations, modules):
            if op in algorithm_args:
                name = algorithm_args[op]
                algorithm_args[op] = getattr(module, name, None)(**algorithm_args['{}_args'.format(op)])
        self.algorithm = globals()[algorithm](**algorithm_args)
        self.problem = globals()[problem](**problem_args)
        self.termination = globals()[termination](**termination_args)

        self.display = globals()[display]() if display else display
        self.log_saver = globals()[log_saver]() if log_saver else log_saver
        
        self.setup_args = setup_args

    def run(self):
        self.algorithm.set_up_problem(problem=self.problem, 
                                      termination=self.termination, 
                                      log_saver=self.log_saver,
                                      display=self.display,
                                      **self.setup_args)
        result = self.algorithm.run()
        return result

    def finalize(self):
        pass