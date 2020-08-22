from numpy import unique
from model import Termination
class RankConvergence(Termination):
    def __init__(self):
        super().__init__()

    def _criteria_met(self, ga):
        return len(unique(ga.rank)) == 1
