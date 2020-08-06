import pandas as pd
import datetime

class LogSaver():
    def __init__(self):
        self.attributes = {}
        self.data_frame = None
        self.now = datetime.datetime.now()
    
    def add_attributes(self, name, value):
        self.attributes[name] = value

    def do(self, algorithm):
        self._do(algorithm)
        if self.data_frame is None:
            self.data_frame = pd.DataFrame(self.attributes)
        else:
            self.data_frame.append(self.attributes)

    def save(self, algorithm):
        filename = os.path.join(algorithm.log_dir, "data_" + self.now.strftime("%Y%m%d-%H%M") + ".csv")
        self.data_frame.to_csv(filename)

    ## Overide Methods ##
    def _do(self, algorithm):
        pass