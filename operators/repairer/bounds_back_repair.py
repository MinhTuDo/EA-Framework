import numpy as np
from model.operation import Operation


def bounds_back(problem, X):
    if problem.xl is not None and problem.xu is not None:
        (xl, xu) = problem.domain
        xl = np.repeat(problem.xl[None, :], X.shape[0], axis=0)
        xu = np.repeat(problem.xu[None, :], X.shape[0], axis=0)

        # otherwise bounds back into the feasible space
        _range = xu - xl
        X[X < xl] = (xl + np.mod((xl - X), _range))[X < xl]
        X[X > xu] = (xu - np.mod((X - xu), _range))[X > xu]

        return X


class BoundsBackRepair(Operation):

    def _do(self, problem, pop, **kwargs):
        # bring back to bounds if violated through crossover - bounce back strategy
        X = pop
        (xl, xu) = problem.domain
        xl = np.repeat(xl[None, :], X.shape[0], axis=0)
        xu = np.repeat(xu[None, :], X.shape[0], axis=0)

        # otherwise bounds back into the feasible space
        _range = xu - xl
        X[X < xl] = (xl + np.mod((xl - X), _range))[X < xl]
        X[X > xu] = (xu - np.mod((X - xu), _range))[X > xu]

        return X
