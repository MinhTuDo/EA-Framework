from model.problem import Problem

class NAS(Problem):
    def __init__(self,
                 n_params,
                 model_maker,
                 
                 **kwargs):
        super().__init__(n_params, **kwargs)
        self.model_maker = model_maker
        pass