from model.operation import Operation
import numpy as np
from numpy import arange, newaxis
from itertools import combinations, chain

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
        new_MDLs = [self.__calc_MDL(pop, model) for model in new_models]
        min_idx = np.argmin(new_MDLs)
        return model if current_MDL < new_MDLs[min_idx] else new_models[min_idx]

    def __calc_MDL(self, pop, model):
        # Compute model complexity
        N = len(pop)
        S = np.array(list(map(len, model)))
        MC = np.log2(N + 1) * (2**S - 1).sum()

        # Compute compressed population complexity
        sum_H = 0
        for group in model:
            _, event_counts = np.unique(pop[:, group], 
                                        axis=0, 
                                        return_counts=True)
            prob_per_event = event_counts / (N+1)
            sum_entropy = (prob_per_event * np.log2(1/prob_per_event)).sum()
            sum_H += sum_entropy
        CPC = N * sum_H
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

    