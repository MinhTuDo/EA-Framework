from numpy import unique
from model import Termination
class Convergence(Termination):
    def __init__(self):
        super().__init__()

    def _criteria_met(self, ga):
        return len(unique(ga.pop, axis=0)) == 1
