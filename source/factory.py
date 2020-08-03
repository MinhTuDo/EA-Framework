import importlib
import os

# my_modules = []
# for file in os.listdir(ALGORITHMS_DIR):
#     module = { file.replace('.py', '') : [] }
#     my_modules.append(module)

# def get_module(module_name, class_name, **kwargs):
#     my_class = None
#     for file in os.listdir(module_getter[module_name]):
#         if my_class is not None:
#             return my_class(**kwargs)
#         if file.endswith('.py'):
#             module = file.replace('.py', '')
#             module_path = '{}.{}'.format(module_getter[module_name],
#                                         module)
#             try:
#                 module = importlib.import_module(module_path)
#                 my_class = getattr(module, class_name)
#             except AttributeError:
#                 my_class = None
    
#     raise Exception('Class name not found!')

# booth = get_module('problem', 'Booth')
# booth.plot()

class GAFactory:
    def __init__(self):
        self.ALGORITHMS_DIR = 'algorithms'
        self.PROBLEMS_DIR = 'problems'
        self.OPERATORS_DIR = 'operators'
        self.TERMINATIONS_DIR = 'terminations'

        self.module_getter = {  'algorithm': self.ALGORITHMS_DIR,
                                'problem': self.PROBLEMS_DIR,
                                'operator': self.OPERATORS_DIR,
                                'termination': self.TERMINATIONS_DIR}

        self.my_class = None

    def get_problem(self, problem_name, **kwargs):
        return self.__get_module('problem', problem_name, **kwargs)
        pass

    def get_termination(self, termination_name, **kwargs):
        return self.__get_module('termination', termination_name, **kwargs)

    def get_algorithm(self, algorithm_name, **kwargs):
        return self.__get_module('algorithm', algorithm_name, **kwargs)

    def get_selection(self, selection_name, **kwargs):
        selection_name = 'selection.' + selection_name
        return self.__get_module('operations', selection_name, **kwargs)
    
    def get_crossover(self, crossover_name, **kwargs):
        crossover_name = 'crossover.' + crossover_name
        return self.__get_module('operations', crossover_name, **kwargs)

    def get_model_builder(self, model_builder_name, **kwargs):
        model_builder_name = 'model_builder' + model_builder_name
        return self.__get_module('operations')

    def get_mutation(self, mutation_name, **kwargs):
        mutation_name = 'mutation.' + mutation_name
        return self.__get_module('operations', mutation_name, **kwargs)

    def get_initializer(self, initialization_name, **kwargs):
        initialization_name = 'initialization.' + initialization_name
        return self.__get_module('operations', initialization_name, **kwargs)
    
    
    def __get_module(self, module_name, class_name, **kwargs):
        my_class = None
        for file in os.listdir(self.module_getter[module_name]):
            if my_class is not None:
                return my_class(**kwargs)
            if file.endswith('.py'):
                module = file.replace('.py', '')
                module_path = '{}.{}'.format(self.module_getter[module_name],
                                             module)
                try:
                    module = importlib.import_module(module_path)
                    my_class = getattr(module, class_name)
                except AttributeError:
                    my_class = None
        
        raise Exception('Class name not found!')

factory = GAFactory()

problem = factory.get_problem('Rastrigin')

problem.plot(plot_3D=True)