import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
from model import Display, LogSaver
import time
from utils.displayer import NSGANetDisplay
from utils.logger import NSGANetLogSaver


filename = './logs/obj_20201010-0726_gen-17__NSGAII-NSGANet_.pickle'
gen17 = None
with open(filename, 'rb') as handle:
    gen17 = pickle.load(handle)

display = NSGANetDisplay()
logger = NSGANetLogSaver()

gen17.log_saver = logger
gen17.display = display

gen17.problem.input_size = [3, 32, 32]
gen17.problem.arch_config['model_args']['input_size'] = [3, 32, 32]
gen17.termination.max_gen = 25
gen17.run_continue()