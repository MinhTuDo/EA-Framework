def optimize(algorithm, problem, **kwargs):
    algorithm.initialize(problem, **kwargs)
    result = algorithm.run()
    return result