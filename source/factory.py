import importlib
import os

class GAFactory:
    def __init__(self):
        self.ALGORITHMS_DIR = 'algorithms'
        self.PROBLEMS_DIR = 'problems'
        self.CROSSOVER_DIR = 'operators/crossover'
        self.SELECTION_DIR = 'operators/selection'
        self.INITIALIZATION_DIR = 'operators/initialization'
        self.MODELBUILDER_DIR = 'operators/model_builder'
        self.TERMINATIONS_DIR = 'terminations'

        self.module_getter = {  'algorithm': self.ALGORITHMS_DIR,
                                'problem': self.PROBLEMS_DIR,
                                'crossover': self.CROSSOVER_DIR,
                                'selection': self.SELECTION_DIR,
                                'model_builder': self.MODELBUILDER_DIR,
                                'init': self.INITIALIZATION_DIR,
                                'termination': self.TERMINATIONS_DIR}

        self.my_class = None

    def get_problem(self, problem_name):
        return self.__get_module('problem', problem_name)

    def get_termination(self, termination_name):
        return self.__get_module('termination', termination_name)

    def get_algorithm(self, algorithm_name):
        return self.__get_module('algorithm', algorithm_name,)

    def get_selection(self, selection_name):
        return self.__get_module('selection', selection_name)
    
    def get_crossover(self, crossover_name):
        return self.__get_module('crossover', crossover_name)

    def get_model_builder(self, model_builder_name):
        return self.__get_module('model_builder')

    def get_mutation(self, mutation_name):
        return self.__get_module('mutation', mutation_name)

    def get_initializer(self, initialization_name):
        return self.__get_module('init', initialization_name, **kwargs)
    
    
    def __get_module(self, module_name, class_name):
        my_class = None
        for file in os.listdir(self.module_getter[module_name]):
            if my_class is not None:
                return my_class
            if file.endswith('.py'):
                module = file.replace('.py', '')
                module_path = '{}.{}'.format(self.module_getter[module_name].replace('/', '.'),
                                             module)
                try:
                    module = importlib.import_module(module_path)
                    my_class = getattr(module, class_name)
                except AttributeError:
                    my_class = None
        if my_class is not None:
            return my_class
        raise Exception('Class name not found!')


