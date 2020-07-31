from model.operation import Operation
import numpy as np

class HistoricalBestSelection(Operation):
    def __init__(self):
        super().__init__()
    
    def _do(self, ga):
        comparer = ga.problem._f_comparer
        results = comparer(ga.f_pop_prev, ga.f_pop)
        better_idx = np.where(results)[0]
        return better_idx
