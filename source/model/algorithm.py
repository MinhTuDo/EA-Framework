from abc import abstractmethod
from model.result import Result
import time
import numpy as np
from model.display import Display

class Algorithm:
    def __init__(self, **kwargs):
        self.termination = kwargs.get('termination')

        self.epsilon = None
        self.problem = None
        self.save_history = None
        self.history = None
        self.verbose = None
        self.seed = None
        # self.pop = None
        # self.f_pop = None
        self.n_evals = None
        # self.n_gens = None
        self.opt = None
        self.f_opt = None
        self.default_termination = None
        self.result = None
        self.success = None
        self.display = None
        self.default_display = Display()

    ### Public Methods
    def set_up_problem(self, 
                       problem,

                       seed=1,
                       verbose=False,
                       save_history=False, 
                       epsilon=10**-5,

                       termination=None,
                       display=None,
                       **kwargs):
        
        self.problem = problem
        self.verbose = verbose
        self.save_history = save_history
        self.epsilon = epsilon
        self.seed = seed

        self.termination = termination
        if self.termination is None:
            self.termination = self.default_termination

        self.display = display
        if display is None:
            self.display = self.default_display

        # self.n_gens = 1
        self.history = []
        
        np.random.seed(seed)

    def run(self):
        self.result = Result()

        self.start_time = time.time()
        self._run()
        self.end_time = time.time()
        
        self.save_result()
        return self.result

    def sub_tasks_each_gen(self):
        self._sub_tasks_each_gen()

    def save_result(self):
        self._save_result()
        
        self.result.exec_time = self.end_time - self.start_time
        self.result.success = self.success
        self.result.solution = self.opt
        self.result.problem = self.problem
        self.result.algorithm = self
        # self.result.gens = self.n_gens
        self.result.n_evals = self.n_evals
        self.result.history = self.history
        # self.result.pop = self.pop

    def finalize(self):
        self._finalize()
        diff = np.abs(self.problem._pareto_set - self.opt).sum(axis=1, keepdims=False)
        min_diff = min(diff)
        if min_diff <= self.epsilon * self.problem.n_params:
            self.success = True

    ### Public Methods ###

    ### Protected Methods ###
    def _finalize(self):
        pass

    def _save_result(self):
        pass
    
    ### Protected Methods ###

    ### Abstract Methods ###
    @abstractmethod
    def _run(self):
        pass
    ### Abstract Methods ###