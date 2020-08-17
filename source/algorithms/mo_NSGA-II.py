from model.GA import GA
from operators.initialization.random_initialization import RandomInitialization
from operators.crossover.uniform_crossover import UX
from operators.selection.tournament_selection import TournamentSelection
import numpy as np
from terminations.convergence import Convergence


class NSGAII(GA):
    def __init__(self,
                 pop_size=50,
                 n_offs=None,
                 initialization=RandomInitialization(),
                 selection=None,
                 crossover=UX(),
                 elitist_archive=2,
                 **kwargs):
        super().__init__(pop_size, initialization, selection,
                         crossover, n_offs, **kwargs)
        self.default_termination = Convergence()
        self.elitist_archive = elitist_archive
        if selection is None:
            self.selection = TournamentSelection(elitist_archive)
        if n_offs is None:
            self.n_offs = self.pop_size
        
        # self.domination_count = None
        self.CD = None
        self.F_pop = None
        self.F_offs = None
        self.ranks = {}
        self.ranks_F = {}
        self.rank = None

    def __domination_count(self, pop, F_pop):
        count = np.zeros((pop.shape[0],))
        for i, f_i in enumerate(F_pop):
            count[i] = sum([self.problem._is_dominated(f_j, f_i) for f_j in F_pop])
        
        return count

    def __non_dominated_rank(self):
        pop = self.pop.copy()
        F_pop = self.F_pop.copy()
        i = 0
        ranks, ranks_F = {}, {}
        
        while len(pop) != 0:
            domination_count = self.__domination_count(pop, F_pop)
            ranks[i] = pop[domination_count == 0]
            ranks_F[i] = F_pop[domination_count == 0]
            pop = pop[domination_count != 0]
            F_pop = F_pop[domination_count != 0]
            i += 1

        return ranks, ranks_F

        

    def __non_dominated_sort(self):
        rank, pop, F_pop = [], [], []
        for i in range(len(self.ranks)):
            rank.append(np.ones((self.ranks[i].shape)) * i)
            pop.append(self.ranks[i])
            F_pop.append(self.ranks_F[i])

        return np.vstack(rank).flatten(), np.vstack(pop), np.vstack(F_pop).flatten()
        # return np.vstack(self.ranks.keys()), \
        #        np.vstack(self.ranks.values()), \
        #        np.vstack(self.ranks_F.values())


    def __crowding_distance_sort(self, CD):
        return CD.argsort()
    
    def __calc_crowding_distances(self, P_r):
        # R = list(range(max(self.rank)))
        # objectives = list(range(self.F_pop.shape[1])) 
        # for r in R:
            # indices = np.where(self.rank == r)
            # CD_r = CD[indices]
            # P_r = self.rank[self.rank == r]
        n = len(P_r)
        CD = np.zeros((n,))
        for obj in self.problem.objectives:
            f = np.array(list(map(obj, P_r))).flatten()
            Q = f.argsort()
            CD[Q[0]] = CD[Q[-1]] = np.inf
            for i in range(1, n-1):
                CD[Q[i]] += f[Q[i + 1]] - f[Q[i - 1]]
        return CD

    def _initialize(self):
        self.F_pop = self.evaluate(self.pop).reshape((self.pop_size, self.problem.n_obj))
        self.ranks, self.ranks_F = self.__non_dominated_rank()
        self.rank, self.pop, self.F_pop = self.__non_dominated_sort()
        self.CD = np.hstack(list(map(self.__calc_crowding_distances, self.ranks.values())))
        self.fitness_pop = np.vstack((self.rank, self.CD)).T
        # self.sub_tasks_each_gen()

    def _next(self):
        selected_indices = self.selection._do(self)
        self.pop = self.pop[selected_indices]
        self.fitness_pop = self.fitness_pop[selected_indices]
        self.offs = self.crossover._do(self)
        self.F_offs = self.evaluate(self.offs).reshape((self.pop_size, self.problem.n_obj))

        self.pop = np.vstack((self.pop, self.offs))
        self.F_pop = np.vstack((self.F_pop, self.F_offs))

        self.__non_dominated_rank()
        sort_by_rank = self.__non_dominated_sort()


        

        