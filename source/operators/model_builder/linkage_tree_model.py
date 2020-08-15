from model.operation import Operation
import numpy as np
from itertools import combinations, chain

class LinkageTreeModel(Operation):
    def __init__(self):
        super().__init__()
        self.N = None
        self.pop = None
        self.done = None
        self.root_model = None

    def build(self, ga):
        self.N = len(ga.pop)
        self.epsilon = ga.epsilon
        self.pop = ga.pop
        tree_model = np.arange(ga.problem.n_params)[:, np.newaxis].tolist()
        tree_builder = np.arange(ga.problem.n_params)[:, np.newaxis].tolist()
        self.root_model = np.arange(ga.problem.n_params).tolist()
        self.done = False
        while True:
            combinations = self.__make_combinations(tree_builder)
            I_per_group = np.array(list(map(self.__calc_mutual_information, combinations)))
            max_idx = np.argmax(I_per_group)
            tree_builder.remove(combinations[max_idx][0]), tree_builder.remove(combinations[max_idx][1])
            
            new_group = sorted(list(chain.from_iterable(combinations[max_idx])))
            if len(new_group) == ga.problem.n_params:
                return tree_model
            tree_model.append(new_group)
            tree_builder.append(new_group)
        
        return tree_model

    def __calc_mutual_information(self, combination):
        XY = list(chain.from_iterable(combination))
        (X, Y) = combination
        avg_I = 0
        xy, xy_count = np.unique(self.pop[:, XY], axis=0, return_counts=True)
        P_XY = xy_count / (self.N)

        x, x_count = np.unique(self.pop[:, X], axis=0, return_counts=True)
        P_X = x_count / (self.N)
        # P_X = zip(x, x_count)
        y, y_count = np.unique(self.pop[:, Y], axis=0, return_counts=True)
        P_Y = y_count / (self.N)
        # P_Y = zip(y, y_count)

        
        # for e_x, p_x in zip(x, P_X):
        #     for e_y, p_y in zip(y, P_Y):
        #         e_xy = np.hstack((e_x, e_y))
        #         n_occurrence = np.where((xy == e_xy).all(axis=1))[0]
        #         p_xy = 0 if len(n_occurrence) == 0 else P_XY[n_occurrence[0]]
        #         avg_I += p_xy * np.log2(p_xy / (p_x*p_y + self.epsilon))

        
        # group = self.pop[:, XY]
        # for x_i, y_i in zip(P_X, P_Y):
        #     xy_i = np.hstack((x_i[0], y_i[0]))
        #     match = (group == xy_i).sum(axis=1)
        #     p_xy = np.count_nonzero(match == len(XY)) / (self.N+1)
        #     p_x = x_i[1] / (self.N+1)
        #     p_y = y_i[1] / (self.N+1)
        #     avg_I += p_xy * np.log2(p_xy / (p_x*p_y))
        for event, count in zip(xy, xy_count):
            e_x = event[: len(X)]
            e_y = event[len(X):]
            p_x = np.count_nonzero((self.pop[:, X] == e_x).sum(axis=1) == len(X)) / (self.N)
            p_y = np.count_nonzero((self.pop[:, Y] == e_y).sum(axis=1) == len(Y)) / (self.N)
            p_xy = count / (self.N)
            avg_I += p_xy * np.log2(p_xy / (p_x*p_y))
            
        return avg_I


    def __calc_probability(self, group):
        _, event_counts = np.unique(self.pop[:, group],
                                    axis=0,
                                    return_counts=True)
        prob_per_event =  event_counts / (self.N+1)
        return prob_per_event

    def __make_combinations(self, model):
        combs = list(combinations(model, 2))
        return combs

