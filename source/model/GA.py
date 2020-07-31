from model.algorithm import Algorithm

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
        self.f_pop = None
        self.n_gens = None
        self.pop_prev = None
        self.f_pop_prev = None

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
        pass

    def evaluate(self, X):
        f_X = self.problem.evaluate(X)
        self.n_evals += len(f_X)
        return f_X

    def next(self):
        self.n_gens += 1
        self._next()
        self.sub_tasks_each_gen()

    def sub_tasks_each_gen(self):
        self._sub_tasks_each_gen()
        elite_idx = self.problem._argopt(self.f_pop)
        self.opt = self.pop[elite_idx]
        self.f_opt = self.f_pop[elite_idx]
        if self.save_history:
            self.history.append(self.pop.copy())
        if self.verbose:
            print('## Gen {}: Best: {} - F: {}'.format(self.n_gens, self.opt, self.f_opt))

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
    ## Protected Methods ##

