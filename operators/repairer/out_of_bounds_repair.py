import numpy as np
from model.operation import Operation


def repair_out_of_bounds_manually(X, xl, xu):
    if xl is not None:
        xl = np.repeat(xl[None, :], X.shape[0], axis=0)
        X[X < xl] = xl[X < xl]

    if xu is not None:
        xu = np.repeat(xu[None, :], X.shape[0], axis=0)
        X[X > xu] = xu[X > xu]
        
    return X


def repair_out_of_bounds(problem, X):
    return repair_out_of_bounds_manually(X, *problem.domain)


class OutOfBoundsRepair(Operation):
    def _do(self, problem, pop, **kwargs):
        X = pop
        repaired_X = repair_out_of_bounds(problem ,X)
        return repaired_X
