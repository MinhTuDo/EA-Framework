import time
from model.termination import Termination
class MaxTimeTermination(Termination):
    def __init__(self, max_time):
        super().__init__()
        self.max_time = max_time

    def _criteria_met(self, algorithm):
        current_time = time.time()
        return current_time >= algorithm.start_time + self.max_time