from abc import abstractmethod
from model.result import Result
import time
import numpy as np

class Algorithm:
    def __init__(self, **kwargs):
        self.termination = kwargs.get('termination')

        self.epsilon = 10**-5

        self.problem = None
        self.save_history = None
        self.history = None
        self.verbose = None
        self.seed = None
        self.pop = None
        self.f_calls = None
        self.n_gens = None
        self.pf = None
        self.evaluator = None
        self.optimum = None
        pass

    ### Public Methods
    def initialize(self, 
                   problem,
                   pf=True,
                   evaluator=None,

                   seed=1,
                   verbose=False,
                   save_history=False, 
                   epsilon=None, **kwargs):
        self.n_gens = 1
        self.history = []

        if epsilon is not None:
            self.epsilon = epsilon
        
        np.random.seed(self.seed)
        pass

    def run(self):
        result = Result()
        result.start_time = time.time()

        self._run()

        result.end_time = time.time()
        result.exec_time = result.start_time - result.end_time
        
        pass

    def next_gen(self):
        self.n_gens += 1
        pass

    def finalize(self):
        return self._finalize()

    ### Public Methods ###

    ### Protected Methods ###
    def _run(self):
        pass

    def _sub_tasks_each_gen(self):
        pass

    def _finalize(self):
        pass
    ### Protected Methods ###

    ### Abstract Methods ###
    @abstractmethod
    def _initialize(self):
        pass
    ### Abstract Methods ###
    pass