from abc import abstractmethod
from model.result import Result
import time
import numpy as np

class Algorithm:
    def __init__(self, **kwargs):
        self.termination = kwargs.get('termination')

        self.epsilon = None
        self.problem = None
        self.save_history = None
        self.history = None
        self.verbose = None
        self.seed = None
        self.pop = None
        self.f_pop = None
        self.n_evals = None
        self.n_gens = None
        self.opt = None
        self.f_opt = None
        self.default_termination = None
        self.result = None
        self.success = None

    ### Public Methods
    def initialize(self, 
                   problem,

                   seed=1,
                   verbose=False,
                   save_history=False, 
                   epsilon=10**-5,

                   termination=None,
                   **kwargs):
        
        self.problem = problem
        self.verbose = verbose
        self.save_history = save_history
        self.epsilon = epsilon
        self.seed = seed

        if termination is not None:
            self.termination = termination
        else:
            self.termination = self.default_termination

        self.n_gens = 1
        self.history = []
        
        np.random.seed(seed)

    def run(self):
        self.result = Result()
        self.result.start_time = time.time()

        self._run()

        self.result.end_time = time.time()
        self.result.exec_time = result.start_time - result.end_time
        self.result.pop = self.pop
        self.result.success = self.success
        pass

    def next_gen(self):
        self.n_gens += 1
        self.next_gen()
        pass

    def finalize(self):
        return self._finalize()

    ### Public Methods ###

    ### Protected Methods ###
    def _run(self):

        self._finalize()
        pass

    def _sub_tasks_each_gen(self):
        if verbose:
            print('## Gen {}: Best: {} - F: {}'.format(self.n_gens, self.pop.max()))
        pass

    def _finalize(self):
        diff = np.abs(self.problem._pareto_front - self.f_opt)
        if diff <= self.epsilon:
            self.success = True

    ### Protected Methods ###

    ### Abstract Methods ###
    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def _next_gen(self):
        pass
    ### Abstract Methods ###
    pass