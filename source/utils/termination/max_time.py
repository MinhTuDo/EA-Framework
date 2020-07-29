import time
class MaxTimeTermination(Termination):
    def __init__(self, max_time):
        super().__init__()
        self.max_time = max_time

    def _criteria_met(self, algorithm):
        current_time = time.time()
        return current_time >= algorithm.result.start_time + self.max_time*60