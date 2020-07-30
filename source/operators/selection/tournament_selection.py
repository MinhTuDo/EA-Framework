from model.operation import Operation
import numpy as np
class TournamentSelection(Operation):
    def __init__(self, tournament_size):
        super().__init__()
        self.tournament_size = tournament_size

    def _do(self, ga):
        n_inds = len(ga.f_pop)
        # assert(n_inds % self.tournament_size != 0)
        indices = np.arange(n_inds)
        selection_size = ga.n_offs
        selected_indices = []
        comparer = ga.problem._comparer
        f_pop = ga.f_pop

        np.random.seed(ga.seed)

        while len(selected_indices) < selection_size:
            np.random.shuffle(indices)

            for i in range(0, n_inds, self.tournament_size):
                idx_tournament = indices[i : i+self.tournament_size] 
                elite_idx = list(filter(lambda idx : f_pop[idx] == comparer(f_pop[idx_tournament]), idx_tournament))
                selected_indices.append(np.random.choice(elite_idx))
        
        return selected_indices