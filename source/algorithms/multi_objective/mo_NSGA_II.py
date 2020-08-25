from algorithms.GA import GA
from operators.initialization.random_initialization import RandomInitialization
from operators.mutation.polynomial_mutation import PolynomialMutation
from operators.crossover.simulated_binary_crossover import SBX
from operators.selection.tournament_selection import TournamentSelection
import numpy as np
from terminations import Convergence


class NSGAII(GA):
    def __init__(self,
                 pop_size=50,
                 n_offs=None,
                 initialization=RandomInitialization(),
                 selection=None,
                 crossover=SBX(eta=15, prob=0.9),
                 elitist_archive=2,
                 mutation=PolynomialMutation(eta=20),
                 **kwargs):
        super().__init__(pop_size, 
                         initialization, 
                         selection,
                         crossover, 
                         mutation,
                         n_offs, 
                         **kwargs)
        self.default_termination = Convergence()
        self.elitist_archive = elitist_archive
        if selection is None:
            self.selection = TournamentSelection(elitist_archive)
        if n_offs is None:
            self.n_offs = self.pop_size
        
        self.CD = None
        self.F_pop = None
        self.F_offs = None
        self.ranks = {}
        self.ranks_F = {}
        self.rank = None

    def __domination_count(self, pop, F_pop):
        count = np.empty((pop.shape[0],))
        for i in range(F_pop.shape[0]):
            count[i] = sum([self.problem._is_dominated(f_j, F_pop[i]) for f_j in F_pop])
        return count

    def __non_dominated_rank(self):
        pop, F_pop = self.pop, self.F_pop
        ranks, ranks_F = {}, {}
        i = 0
        while len(pop) != 0:
            domination_count = self.__domination_count(pop, F_pop)

            ranks[i] = pop[domination_count == 0]
            ranks_F[i] = F_pop[domination_count == 0]

            pop = pop[domination_count != 0]
            F_pop = F_pop[domination_count != 0]
            i += 1

        return ranks, ranks_F

    def __non_dominated_sort(self):
        rank = np.hstack([np.ones(self.ranks[i].shape[0]) * i for i in range(len(self.ranks))])
        pop = np.vstack(list(self.ranks.values()))
        F_pop = np.vstack(list(self.ranks_F.values()))
        return rank, pop, F_pop
    
    def __calc_crowding_distances(self, P_r):
        P, F = P_r
        CD = np.empty((P.shape[0],))
        for f_i in range(F.shape[1]):
            f = F[:, f_i]
            Q = f.argsort()
            CD[Q[0]] = CD[Q[-1]] = np.inf
            for i in range(1, CD.shape[0]-1):
                CD[Q[i]] += f[Q[i + 1]] - f[Q[i - 1]]
        return CD

    def _initialize(self):
        self.F_pop = self.evaluate(self.pop)
        self.ranks, self.ranks_F = self.__non_dominated_rank()
        self.rank, self.pop, self.F_pop = self.__non_dominated_sort()
        self.CD = np.hstack(list(map(self.__calc_crowding_distances, zip(self.ranks.values(), 
                                                                         self.ranks_F.values()))))
        self.fitness_pop = np.hstack((self.rank[:, np.newaxis], 
                                      self.CD[:, np.newaxis]))
        self.sub_tasks_each_gen()

    def _next(self):
        selected_indices = self.selection._do(self)
        self.pop = self.pop[selected_indices]
        self.F_pop = self.F_pop[selected_indices]

        self.offs = self.crossover._do(self)
        self.offs = self.mutation._do(self)
        self.F_offs = self.evaluate(self.offs)

        self.pop = np.vstack((self.pop, self.offs))
        self.F_pop = np.vstack((self.F_pop, self.F_offs))

        self.ranks, self.ranks_F = self.__non_dominated_rank()
        
        n = 0
        ranks, ranks_F = {}, {}
        for i in range(len(self.ranks)):
            if n + len(self.ranks[i]) >= self.pop_size:
                n_takes = self.pop_size - n
                CD_i = self.__calc_crowding_distances((self.ranks[i], self.ranks_F[i]))
                sorted_CD_i = CD_i.argsort()[::-1]
                ranks[i] = self.ranks[i][sorted_CD_i][:n_takes]
                ranks_F[i] = self.ranks_F[i][sorted_CD_i][:n_takes]
                break
            n += self.ranks[i].shape[0]
            ranks[i] = self.ranks[i]
            ranks_F[i] = self.ranks_F[i]

        self.ranks, self.ranks_F = ranks, ranks_F    
        self.rank, self.pop, self.F_pop = self.__non_dominated_sort()
        self.CD = np.hstack(list(map(self.__calc_crowding_distances, zip(self.ranks.values(), 
                                                                         self.ranks_F.values()))))
        self.fitness_pop = np.hstack((self.rank[:, np.newaxis], 
                                      self.CD[:, np.newaxis]))

    def _save_history(self):
        res = {'P': self.pop.copy(), 
               'F': self.F_pop.copy()}
        self.history.append(res)


        

        