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
    def __init__(self, config):
        super().__init__(config)
        operations = ['crossover', 'mutation', 'selection', 'model_builder', 'initialization']
        modules = [cx, mut, sel, mb, init]
        for op, module in zip(operations, modules):
            if op in config['algorithm_args']:
                name = config['algorithm_args'][op]
                config['algorithm_args'][op] = getattr(module, name, None)(**config['algorithm_args']['{}_args'.format(op)])
        self.algorithm = globals()[config['algorithm']](**config['algorithm_args'])
        self.problem = globals()[config['problem']](**config['problem_args'])
        self.termination = globals()[config['termination']](**config['termination_args'])

        self.display = globals()[config['display']]() if 'display' in config else None
        self.log_saver = globals()[config['log_saver']]() if 'log_saver' in config else None

    def run(self):
        self.algorithm.set_up_problem(problem=self.problem, 
                                      termination=self.termination, 
                                      log_saver=self.log_saver,
                                      display=self.display,
                                      **self.config['setup_args'])
        result = self.algorithm.run()
        return result

    def finalize(self):
        pass