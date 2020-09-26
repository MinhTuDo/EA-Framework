from model.agent import Agent
from algorithms import *
from problems import *

import operators.crossover as cx
import operators.initialization as init
import operators.mutation as mut
import operators.selection as sel
import operators.model_builder as mb

from terminations import *

class EvoAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        operations = ['crossover', 'mutation', 'selection', 'model_builder', 'initialization']
        modules = [cx, mut, sel, mb, init]
        for op, module in zip(operations, modules):
            name = config['algorithm'][op]
            config['algorithm'][op] = getattr(module, name, None)(**config['{}_args'.format(name)])
        self.algorithm = globals()[config['algorithm']](**config['algorithm_args'])
        self.problem = globals()[config['problem']](**config['problem_args'])