from model.problem import Problem

class MultiObjectiveProblem(Problem):
    def __init__(self, 
                 n_params=2, 
                 n_obj=2,
                 n_constraints=0,
                 domain=(0, 0),
                 param_type=None):
        super().__init__(n_params=n_params,
                         n_obj=n_obj,
                         n_constraints=n_constraints,
                         domain=domain,
                         param_type=param_type)
        
        self._argopt = self.arg_optimum
        self._optimum = self.optimum

    ## Protected Methods ##
    def _is_dominated(self, y1, y2):
        pass

    ## Overide Methods ##
    def _sol_compare(self, s1, s2):
        r1, r2 = s1[0], s2[0]
        cd1, cd2 = s1[1], s2[1]
        return (r1 < r2) or \
               (r1 == r2 and cd1 > cd2)

    ## Private Methods ##
    def optimum(self, Y):
        opt = Y[0]
        for y in Y:
            opt = opt if self._sol_compare(opt, y) else y
        return opt

    def arg_optimum(self, Y):
        argopt = 0
        for i, y_i in enumerate(Y):
            argopt = argopt if self._sol_compare(Y[argopt], y_i) else i
        return argopt
    