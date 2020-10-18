from graphs.models.evo_net import EvoNet
from mpl_toolkits.mplot3d import Axes3D
import pickle5 as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model_decoder = EvoNet.setup_model_args

filename = './logs/obj_20201010-0726_gen-17__NSGAII-NSGANet_.pickle'
with open(filename, 'rb') as handle:
    last_gen = pickle.load(handle)

def to_numpy(text):
    return np.array([int(bit) for bit in text], dtype=np.int)

data = np.array(list(last_gen.problem.hash_dict.values()))

error_rates = data[:, 0]
n_flops = data[:, 1]

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(error_rates, bins=50)
axs[1].hist(n_flops, bins=100)
plt.show()

plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('Error rate')
plt.ylabel('Flops')
plt.show()


