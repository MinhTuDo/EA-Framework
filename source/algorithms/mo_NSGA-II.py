from model.GA import GA
from operators.initialization.random_initialization import RandomInitialization
from operators.crossover.simulated_binary_crossover import SBX
from operators.selection.tournament_selection import TournamentSelection
import numpy as np
from terminations.convergence import Convergence


class NSGAII(GA):
    def __init__(self,
                 pop_size=50,
                 n_offs=None,
                 initialization=RandomInitialization(),
                 selection=None,
                 crossover=SBX(15),
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
        pop = self.pop
        F_pop = self.F_pop
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
        # rank = np.zeros((self.ranks[0].shape[0],))
        # # pop = self.ranks[0]
        # # F_pop = self.ranks_F[0]
        # n_ranks = len(self.ranks)
        # for i in range(1, n_ranks):
        #     # pop = np.vstack((pop, self.ranks[i]))
        #     # F_pop = np.vstack((F_pop, self.ranks_F[i]))
        #     rank = np.concatenate((rank, np.ones(self.ranks[i].shape[0],)*i))
        rank = np.hstack([np.ones(self.ranks[i].shape[0]) * i for i in range(len(self.ranks))])
        pop = np.vstack(list(self.ranks.values()))
        F_pop = np.vstack(list(self.ranks_F.values()))
        return rank, pop, F_pop
    
    def __calc_crowding_distances(self, P_r):
        CD = np.zeros((P_r.shape[0],))
        for obj in self.problem.objectives:
            f = np.array(list(map(obj, P_r))).flatten()
            Q = f.argsort()
            CD[Q[0]] = CD[Q[-1]] = np.inf
            for i in range(1, CD.shape[0]-1):
                CD[Q[i]] += f[Q[i + 1]] - f[Q[i - 1]]
        return CD

    def _initialize(self):
        self.F_pop = self.evaluate(self.pop)
        self.ranks, self.ranks_F = self.__non_dominated_rank()
        self.rank, self.pop, self.F_pop = self.__non_dominated_sort()
        self.CD = np.hstack(list(map(self.__calc_crowding_distances, self.ranks.values())))
        self.fitness_pop = np.vstack((self.rank, self.CD)).T
        self.sub_tasks_each_gen()

    def _next(self):
        selected_indices = self.selection._do(self)
        self.pop = self.pop[selected_indices]
        self.F_pop = self.F_pop[selected_indices]

        self.offs = self.crossover._do(self)
        self.F_offs = self.evaluate(self.offs)

        self.pop = np.vstack((self.pop, self.offs))
        self.F_pop = np.vstack((self.F_pop, self.F_offs))

        self.ranks, self.ranks_F = self.__non_dominated_rank()
        
        n = 0
        for i in range(len(self.ranks)):
            if n + len(self.ranks[i]) >= self.pop_size:
                n_takes = self.pop_size - n
                # rank.append(np.ones((n_takes,)) * i)
                CD_i = self.__calc_crowding_distances(self.ranks[i])
                sorted_CD_i = CD_i.argsort()[::-1]
                self.ranks[i] = self.ranks[i][sorted_CD_i][:n_takes]
                self.ranks_F[i] = self.ranks_F[i][sorted_CD_i][:n_takes]
                n_pop = len(self.ranks_F) - i
                for j in range(len(self.ranks)-1, i, -1):
                    self.ranks.pop(j)
                    self.ranks_F.pop(j)
                break
            n += self.ranks[i].shape[0]
            
        
        self.rank, self.pop, self.F_pop = self.__non_dominated_sort()
        self.CD = np.hstack(list(map(self.__calc_crowding_distances, self.ranks.values())))
        self.fitness_pop = np.vstack((self.rank, self.CD)).T
        # self.pop = np.vstack(pop)
        # self.F_pop = np.vstack(F_pop)

    def _save_history(self):
        res = {'P': self.pop.copy(), 
               'F': self.F_pop.copy()}
        self.history.append(res)


        

        