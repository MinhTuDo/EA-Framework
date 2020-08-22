from model import Termination
class MaxEvalTermination(Termination):
    def __init__(self, max_eval):
        super().__init__()
        self.max_eval = max_eval

    def _criteria_met(self, algorithm):
        return algorithm.n_evals >= self.max_eval 