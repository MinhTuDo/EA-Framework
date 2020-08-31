from abc import abstractmethod
from .result import Result
import time
import numpy as np
from .display import Display
from .log_saver import LogSaver
import os
import random

class Algorithm:
    def __init__(self, **kwargs):
        self.termination = kwargs.get('termination')

        self.epsilon = None
        self.problem = None
        self.save_history = None
        self.history = None
        self.verbose = None
        self.log = None
        self.seed = None
        # self.pop = None
        # self.fitness_pop = None
        self.n_evals = None
        # self.n_gens = None
        # self.opt = None
        # self.fitness_opt = None
        self.elite_idx = None
        self.default_termination = None
        self.result = None
        self.success = None
        self.display = None
        self.default_display = Display()
        self.log_saver = None
        self.default_log_saver = LogSaver()
        # self.log_dir = 'log/'

    ### Public Methods
    def set_up_problem(self, 
                       problem,

                       seed=1,
                       verbose=False,
                       log=False,
                       log_dir=None,
                       save_history=False, 
                       epsilon=10**-5,

                       termination=None,
                       display=None,
                       log_saver=None,
                       **kwargs):
        
        self.problem = problem
        self.verbose = verbose
        self.log = log
        self.save_history = save_history
        self.epsilon = epsilon
        self.seed = seed


        self.termination = termination
        if self.termination is None:
            self.termination = self.default_termination

        self.display = display
        if display is None:
            self.display = self.default_display
        
        self.log_saver = log_saver
        if log_saver is None:
            self.log_saver = self.default_log_saver

        if log_dir is not None:
            self.log_dir = log_dir

        if log:
            log_saver.log_dir = log_dir if log_dir is not None else log_saver.log_dir
            if not os.path.exists(log_saver.log_dir):
                os.makedirs(log_saver.log_dir)
        # self.n_gens = 1
        self.history = []
        
        np.random.seed(seed)
        random.seed(seed)

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
        self.result.solution = self.pop[self.elite_idx]
        self.result.problem = self.problem
        self.result.algorithm = self
        # self.result.gens = self.n_gens
        self.result.n_evals = self.n_evals
        self.result.history = self.history
        # self.result.pop = self.pop

    def finalize(self):
        if self.log:
            self.log_saver.save(self)
        if self.problem._pareto_set is None or \
           self.problem._pareto_front is None:
            return
        self._finalize()

        

    ### Public Methods ###

    ### Protected Methods ###
    def _finalize(self):
        diff = abs(self.problem._pareto_front - self.fitness_pop[self.elite_idx])
        if diff <= self.epsilon:
            self.success = True
        else:
            self.success = False

    def _save_result(self):
        pass
    
    ### Protected Methods ###

    ### Abstract Methods ###
    @abstractmethod
    def _run(self):
        pass
    ### Abstract Methods ###