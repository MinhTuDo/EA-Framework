class MaxGenTermination(Termination):
    def __init__(self, max_gen):
        super().__init__()
        self.max_gen = max_gen

    def _criteria_met(self, algorithm):
        return algorithm.n_gens == self.max_gen