import pickle5 as pickle
from utils.displayer import NSGANetDisplay
from utils.logger import NSGANetLogSaver

filename = 'logs/obj_20210323-1403_gen-24__NSGAII-NSGANet_.pickle'

with open(filename, 'rb') as handle:
    algorithm = pickle.load(handle)

algorithm.display = NSGANetDisplay()
algorithm.log_saver = NSGANetLogSaver()

algorithm.run_continue()
print('Debug')