from model import Algorithm, Display, LogSaver

class GADisplay(Display):
    def _do(self, ga):
        self.display_top = 5
        self.add_attributes('n_gens', ga.n_gens)
        self.add_attributes('n_evals', ga.n_evals)
        self.add_attributes('F', ga.fitness_opt)

class GALogSaver(LogSaver):
    def _do(self, ga):
        self.add_attributes('n_gens', ga.n_gens)
        self.add_attributes('n_evals', ga.n_evals)
        self.add_attributes('F', ga.fitness_opt)

class GA(Algorithm):
    def __init__(self,
                 pop_size,
                 initialization=None,
                 selection=None,
                 crossover=None,
                 mutation=None,
                 n_offs=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.pop_size = pop_size
        self.initialization = initialization
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.n_offs = n_offs
        self.pop = None
        self.fitness_pop = None
        self.n_gens = None
        self.offs = None
        self.fitness_offs = None
        self.default_display = GADisplay()
        self.default_log_saver = GALogSaver()

    def _run(self):
        self.initialize()

        while not self.termination._criteria_met(self):
            self.next()

        self.finalize()

    def initialize(self):
        self.n_gens = 1
        self.n_evals = 0
        self.pop = self.initialization._do(self)
        self._initialize()

    def evaluate(self, X):
        f_X = self.problem.evaluate_all(X)
        self.n_evals += len(f_X)
        return f_X

    def next(self):
        self.n_gens += 1
        self._next()
        self.sub_tasks_each_gen()

    def sub_tasks_each_gen(self):
        self.elite_idx = self.problem._argopt(self.fitness_pop)
        self._sub_tasks_each_gen()
        # self.opt = self.pop[elite_idx]
        # self.fitness_opt = self.fitness_pop[elite_idx]
        if self.save_history:
            # res = {'P': self.pop.copy(), 
            #        'F': self.fitness_pop.copy()}
            # self.history.append(res)
            self._save_history()
        if self.verbose:
            # print('## Gen {}: Best: {} - F: {}'.format(self.n_gens, self.opt, self.fitness_opt))
            self.display.do(self)
        if self.log:
            self.log_saver.do(self)

    ## Overide Methods ##
    def _save_result(self):
        self._save_result_ga()
        
        self.result.pop = self.pop
        self.result.gens = self.n_gens

    ## Protected Methods ##
    def _initialize(self):
        pass
    def _next(self):
        pass
    def _sub_tasks_each_gen(self):
        pass
    def _save_result_ga(self):
        pass
    def _save_history(self):
        pass
    ## Protected Methods ##

