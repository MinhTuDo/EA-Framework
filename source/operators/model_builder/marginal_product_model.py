from model.operation import Operation
import numpy as np
from numpy import arange, newaxis

class Operation:
    def __init__(self):
        pass
    def _do(self, algorithm):
        pass

class MarginalProductModel(Operation):
    def __init__(self):
        super().__init__()
    
    def build(self, ga):
        model = ga.model.copy()
        while len(model) != 1:
            temp_model = self.__calc_MPM(ga.pop, model)
            if self.__model_converged(temp_model, model):
                break
            else:
                model = temp_model
        
        return model

    def __model_converged(self, current, new):
        for group in current:
            if group not in new:
                return False
        return True

    def __calc_MPM(self, pop, model):
        current_MDL = self.__calc_MDL(pop, model)
        new_models = self.__get_new_models(model)
        new_MDLs = np.array([self.__calc_MDL(pop, model) for model in new_models])
        return model if current_MDL < new_MDLs.min() else new_models[np.argmin(new_MDLs)]

    def __calc_MDL(self, pop, model):
        # Compute model complexity
        N = len(pop)
        S = np.array(list(map(len, model)))
        MC = np.log2(N + 1) * (2**S - 1).sum()

        # Compute compressed population complexity
        entropy = 0
        epsilon = 10**-5
        n_groups = len(model)
        events = [arange(2**S[i])[:, newaxis] >> np.arange(S[i])[::-1] & 1 for i in range(n_groups)]
        for i in range(n_groups):
            for event in events[i]:
                group = pop[:, model[i]]
                match = (group == event).sum(axis=1)
                prob = np.count_nonzero(match == len(event)) / (N+1)
                entropy += prob * np.log2(1 / (prob+epsilon))

        CPC = N * entropy
        return CPC + MC

    def __get_new_models(self, current_model):
        new_models = []
        for i in range(len(current_model) - 1):
            for j in range(i+1, len(current_model)):
                new_group = current_model.copy()
                del new_group[i]
                del new_group[j-1]
                new_group.append(current_model[i] + current_model[j])
                new_models.append(new_group)
                
        return new_models