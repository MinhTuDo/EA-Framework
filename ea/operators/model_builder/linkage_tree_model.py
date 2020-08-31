from model.operation import Operation
import numpy as np
from itertools import combinations, chain, product
np.errstate(divide='ignore')

class LinkageTreeModel(Operation):
    def __init__(self):
        super().__init__()
        self.N = None
        self.pop = None
        self.done = None
        self.root_model = None
        self.epsilon = None

    def build(self, ga):
        self.N = len(ga.pop)
        self.epsilon = ga.epsilon
        self.pop = ga.pop
        tree_model = np.arange(ga.problem.n_params)[:, np.newaxis].tolist()
        tree_builder = tree_model.copy()
        self.root_model = list(range(ga.problem.n_params))
        while True:
            combinations = self.__make_combinations(tree_builder)
            I_per_group = np.array(list(map(self.__calc_mutual_information, combinations)))
            max_idx = np.argmax(I_per_group)
            tree_builder.remove(combinations[max_idx][0]), tree_builder.remove(combinations[max_idx][1])
            
            new_group = list(chain.from_iterable(combinations[max_idx]))
            if len(new_group) == ga.problem.n_params:
                return tree_model
            tree_model.append(new_group), tree_builder.append(new_group)

    def __calc_mutual_information(self, combination):
        XY = list(chain.from_iterable(combination))
        (X, Y) = combination
        avg_I = 0

        x, x_count = np.unique(self.pop[:, X], axis=0, return_counts=True)
        px = x_count / (self.N+1)
        P_X = zip(x, px)
        y, y_count = np.unique(self.pop[:, Y], axis=0, return_counts=True)
        py = y_count / (self.N+1)
        P_Y = zip(y, py)
        
        group = self.pop[:, XY]
        for x_i, y_i in zip(P_X, P_Y):
            xy_i = np.concatenate((x_i[0], y_i[0]))
            p_xy = np.count_nonzero((group == xy_i).sum(axis=1) == len(XY)) / (self.N+1)
            if p_xy == 0:
                p_xy = self.epsilon
            avg_I += p_xy * np.log2(p_xy / (x_i[1]*y_i[1]))
        return avg_I


    def __make_combinations(self, model):
        combs = list(combinations(model, 2))
        return combs


# for event, p_xy in zip(xy, P_XY):
#     e_x = event[: len(X)]
#     e_y = event[len(X) :]
#     p_x = np.count_nonzero((self.pop[:, X] == e_x).sum(axis=1) == len(X)) / (self.N+1)
#     p_y = np.count_nonzero((self.pop[:, Y] == e_y).sum(axis=1) == len(Y)) / (self.N+1)
#     # p_xy = count / (self.N)
#     avg_I += p_xy * np.log2(p_xy / (p_x*p_y))

# for e_x, p_x in zip(x, P_X):
#     for e_y, p_y in zip(y, P_Y):
#         e_xy = np.hstack((e_x, e_y))
#         n_occurrence = np.where((xy == e_xy).all(axis=1))[0]
#         p_xy = self.epsilon if len(n_occurrence) == 0 else P_XY[n_occurrence[0]]
#         avg_I += p_xy * np.log2(p_xy / (p_x*p_y))

# px_py = np.array(np.meshgrid(P_X, P_Y)).T.reshape(-1, 2)
# # n_occurrences = np.where((xy == ex_ey).all(axis=1))[0]
# # p_xy = (P_XY[n_occurrences])
# P_XY = np.resize(P_XY, (len(px_py,)))
# avg_I = (P_XY * np.log2(P_XY / (np.multiply.reduce(px_py, axis=1)))).sum()

