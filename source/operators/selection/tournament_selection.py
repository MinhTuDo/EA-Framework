from model.operation import Operation
import numpy as np
class TournamentSelection(Operation):
    def __init__(self, tournament_size):
        super().__init__()
        self.tournament_size = tournament_size

    def _do(self, ga):
        n_inds = len(ga.fitness_pop)
        # assert(n_inds % self.tournament_size != 0)
        indices = np.arange(n_inds)
        selection_size = ga.n_offs
        selected_indices = []
        optimum = ga.problem._optimum
        fitness = ga.fitness_pop[:, np.newaxis] if len(ga.fitness_pop.shape) == 1 else ga.fitness_pop


        while len(selected_indices) < selection_size:
            np.random.shuffle(indices)

            for i in range(0, n_inds, self.tournament_size):
                idx_tournament = indices[i : i+self.tournament_size] 
                # elite_idx = list(filter(lambda idx : fitness[idx] == optimum(fitness[idx_tournament]), idx_tournament))
                elite_idx = np.where((fitness[idx_tournament] == optimum(fitness[idx_tournament])).sum(axis=1) == fitness.shape[1])[0]
                selected_indices.append(np.random.choice(idx_tournament[elite_idx]))
        
        return selected_indices