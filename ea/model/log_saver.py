import pandas as pd
import datetime
import os

class LogSaver:
    def __init__(self, log_dir='./log'):
        self.attributes = {}
        self.data_frame = pd.DataFrame()
        self.now = datetime.datetime.now()
        self.log_dir = log_dir
    
    def add_attributes(self, name, value):
        self.attributes[name] = value

    def do(self, algorithm):
        self._do(algorithm)
        self.data_frame = self.data_frame.append(self.attributes, ignore_index=True)

    def save(self, algorithm):
        filename = os.path.join(self.log_dir, 
                                "log_" + \
                                self.now.strftime("%Y%m%d-%H%M") + \
                                '_{}-{}_'.format(algorithm.__class__.__name__, 
                                               algorithm.problem.__class__.__name__) + \
                                ".csv")
        self.data_frame.to_csv(filename)

    ## Overide Methods ##
    def _do(self, algorithm):
        pass