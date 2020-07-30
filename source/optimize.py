def optimize(problem, algorithm, **kwargs):
    algorithm.set_up_problem(problem, **kwargs)
    result = algorithm.run()
    return result