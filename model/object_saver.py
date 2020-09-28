import datetime
import os
import pickle
from copy import copy

class ObjectSaver:
    def __init__(self, save_dir='./logs'):
        self.save_dir = save_dir
        self.obj = None
        # self.now = datetime.datetime.now()

    def save(self, algorithm):
        # temp = algorithm.problem
        temp1 = algorithm.log_saver
        temp2 = algorithm.display
        # algorithm.problem = None
        algorithm.log_saver = None
        algorithm.display = None
        self.obj = copy(algorithm)
        # algorithm.problem = temp
        algorithm.log_saver = temp1
        algorithm.display = temp2
        filename = os.path.join(self.save_dir, 
                                "obj_" + \
                                datetime.datetime.now().strftime("%Y%m%d-%H%M") + \
                                '_gen-{}_'.format(algorithm.n_gens) + \
                                '_{}-{}_'.format(algorithm.__class__.__name__,
                                               algorithm.problem.__class__.__name__) +\
                                '.pickle' )
        with open(filename, 'wb') as handle:
            pickle.dump(self.obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(self, filename):
        with open(filename, 'rb') as handle:
            obj = pickle.load(handle)
        return obj