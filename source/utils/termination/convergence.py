from numpy import unique
from model.termination import Termination
class Convergence(Termination):
    def __init__(self):
        super().__init__()

    def _criteria_met(self, algorithm):
        return len(unique(algorithm.pop, axis=0)) == 1