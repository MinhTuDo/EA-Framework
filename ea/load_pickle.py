import pickle5 as pickle
import numpy as np
np.set_printoptions(suppress=True)

filename = './log/obj_20200904-1404_NSGAII-CustomizedNSGANET_.pickle'
obj = None
with open(filename, 'rb') as handle:
    obj = pickle.load(handle)

print('Debug')
error_rates = obj.F_pop[:, 0] * 100
print(error_rates)